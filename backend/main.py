"""
██╗  ██╗████████╗███╗   ███╗
██║  ██║╚══██╔══╝████╗ ████║
███████║   ██║   ██╔████╔██║
██╔══██║   ██║   ██║╚██╔╝██║
██║  ██║   ██║   ██║ ╚═╝ ██║
╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚═╝
                                                        
███████╗████████╗███████╗██████╗     ███████╗██╗██╗     ███████╗
██╔════╝╚══██╔══╝██╔════╝██╔══██╗    ██╔════╝██║██║     ██╔════╝
███████╗   ██║   █████╗  ██████╔╝    █████╗  ██║██║     █████╗  
╚════██║   ██║   ██╔══╝  ██╔═══╝     ██╔══╝  ██║██║     ██╔══╝  
███████║   ██║   ███████╗██║         ██║     ██║███████╗███████╗
╚══════╝   ╚═╝   ╚══════╝╚═╝         ╚═╝     ╚═╝╚══════╝╚══════╝
                                                        
 █████╗ ███╗   ██╗ █████╗ ██╗  ██╗   ██╗███████╗███████╗██████╗ 
██╔══██╗████╗  ██║██╔══██╗██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝██╔══██╗
███████║██╔██╗ ██║███████║██║   ╚████╔╝   ███╔╝ █████╗  ██████╔╝
██╔══██║██║╚██╗██║██╔══██║██║    ╚██╔╝   ███╔╝  ██╔══╝  ██╔══██╗
██║  ██║██║ ╚████║██║  ██║███████╗██║   ███████╗███████╗██║  ██║
╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝╚═╝  ╚═╝
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tempfile
import os
import logging
from steputils import p21
import math
import re
import json
from typing import Optional
import datetime

# Configure logging to show debug messages
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HTM - Step File Analyzer",
    description="Advanced STEP File Analysis and Machining Time Estimation",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Material properties
MATERIAL_PARAMS = {
    "aluminum": {
        "density": 2.7,  # g/cm³
        "base_removal_rate": 25000,  # cubic mm per minute
        "complexity_factor": 0.8,    # multiplier for complex geometries
        "setup_time": 180,          # minutes (3 hours setup)
        "programming_time": 120,    # minutes (2 hours programming)
        "tool_change_time": 5,      # minutes per tool change
        "finishing_factor": 0.3,    # multiplier for finishing operations
        "target_machining_time": 140,  # target machining time in minutes
        "stock_margin": 5,  # mm to add to each dimension for raw stock
        "calibration_data": []  # List to store calibration data
    }
}

def save_material_params():
    """Save material parameters to a JSON file."""
    try:
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Save to data directory
        file_path = os.path.join(data_dir, 'material_params.json')
        with open(file_path, 'w') as f:
            json.dump(MATERIAL_PARAMS, f, indent=2)
        logger.info(f"Material parameters saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving material parameters: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save calibration data: {str(e)}")

def load_material_params():
    """Load material parameters from JSON file."""
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'data', 'material_params.json')
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                global MATERIAL_PARAMS
                MATERIAL_PARAMS = json.load(f)
            logger.info("Material parameters loaded successfully")
    except Exception as e:
        logger.error(f"Error loading material parameters: {e}")

# Load saved parameters on startup
load_material_params()

def calculate_raw_stock(bounding_box, material):
    """Calculate raw stock dimensions and volume."""
    margin = MATERIAL_PARAMS[material]["stock_margin"]
    dimensions = bounding_box["dimensions"]
    
    raw_dimensions = [d + (2 * margin) for d in dimensions]
    raw_volume = raw_dimensions[0] * raw_dimensions[1] * raw_dimensions[2]
    
    return {
        "dimensions": raw_dimensions,
        "volume_mm3": raw_volume,
        "weight_kg": (raw_volume / 1000000) * MATERIAL_PARAMS[material]["density"]
    }

def calculate_material_removal(part_volume, raw_stock_volume):
    """Calculate material removal volume and percentage."""
    removed_volume = raw_stock_volume - part_volume
    removal_percentage = (removed_volume / raw_stock_volume) * 100
    
    return {
        "removed_volume_mm3": removed_volume,
        "removal_percentage": removal_percentage
    }

def calculate_similarity_score(part1, part2):
    """Calculate similarity score between two parts based on volume and complexity."""
    volume_ratio = min(part1["volume_mm3"], part2["volume_mm3"]) / max(part1["volume_mm3"], part2["volume_mm3"])
    complexity_ratio = min(part1["complexity_score"], part2["complexity_score"]) / max(part1["complexity_score"], part2["complexity_score"])
    
    # Weight volume similarity more heavily than complexity
    return (volume_ratio * 0.7) + (complexity_ratio * 0.3)

def estimate_time_from_calibration(current_part, calibration_data, time_type):
    """Estimate time based on weighted average of similar calibrated parts."""
    if not calibration_data:
        return None
        
    total_weight = 0
    weighted_time = 0
    
    # Calculate similarity scores and weights
    for calibrated_part in calibration_data:
        similarity = calculate_similarity_score(
            {"volume_mm3": current_part["volume"], "complexity_score": current_part["complexity"]},
            {"volume_mm3": calibrated_part["volume_mm3"], "complexity_score": calibrated_part["complexity_score"]}
        )
        
        # Use exponential weighting to give more importance to similar parts
        weight = math.exp(similarity * 2)  # exp(2) ≈ 7.4 for perfect similarity
        total_weight += weight
        weighted_time += weight * calibrated_part[time_type]
    
    if total_weight > 0:
        return weighted_time / total_weight
    return None

def calculate_machining_time(volume_mm3, complexity_score, material):
    """Calculate estimated machining time based on calibration data and complexity."""
    params = MATERIAL_PARAMS[material]
    calibration_data = params.get("calibration_data", [])
    
    current_part = {
        "volume": volume_mm3,
        "complexity": complexity_score
    }
    
    # Try to estimate times from calibration data
    setup_time = estimate_time_from_calibration(current_part, calibration_data, "setup_time")
    programming_time = estimate_time_from_calibration(current_part, calibration_data, "programming_time")
    machining_time = estimate_time_from_calibration(current_part, calibration_data, "machining_time")
    
    if machining_time is None:
        # Fallback to default calculation if no calibration data
        base_machining_time = params["target_machining_time"]
        complexity_multiplier = 1 + ((complexity_score / 100) * params["complexity_factor"] - 0.5)
        machining_time = base_machining_time * complexity_multiplier
    
    if setup_time is None:
        setup_time = params["setup_time"]
    
    if programming_time is None:
        programming_time = params["programming_time"]
    
    # Calculate confidence score based on number and similarity of calibration points
    confidence_score = 0
    if calibration_data:
        best_similarity = max(
            calculate_similarity_score(
                {"volume_mm3": current_part["volume"], "complexity_score": current_part["complexity"]},
                {"volume_mm3": cp["volume_mm3"], "complexity_score": cp["complexity_score"]}
            )
            for cp in calibration_data
        )
        confidence_score = min(100, (len(calibration_data) * 20 * best_similarity))
    
    total_time = setup_time + programming_time + machining_time
    
    return {
        "total_time": round(total_time, 2),
        "setup_time": round(setup_time, 2),
        "programming_time": round(programming_time, 2),
        "machining_time": round(machining_time, 2),
        "confidence_score": round(confidence_score, 2),
        "calibration_points_used": len(calibration_data)
    }

def calculate_face_area(vertices):
    """Calculate area of a face using triangulation."""
    if len(vertices) < 3:
        return 0
        
    # Use first vertex as reference point
    ref = vertices[0]
    total_area = 0
    
    # Triangulate the face and sum areas
    for i in range(1, len(vertices) - 1):
        v1 = np.array(vertices[i]) - np.array(ref)
        v2 = np.array(vertices[i + 1]) - np.array(ref)
        # Calculate area of triangle using cross product
        area = np.linalg.norm(np.cross(v1, v2)) / 2
        total_area += area
        
    return total_area

def extract_faces_and_vertices(file_path):
    """Extract faces and their vertices from STEP file."""
    faces = []
    vertices_dict = {}
    edges_dict = {}
    current_face = None
    
    try:
        with open(file_path, 'r') as f:
            content = f.readlines()
            
        for line in content:
            # Extract CARTESIAN_POINT definitions
            if 'CARTESIAN_POINT' in line:
                match = re.search(r"#(\d+)=CARTESIAN_POINT\([^(]*\(([-+]?\d*\.?\d+),([-+]?\d*\.?\d+),([-+]?\d*\.?\d+)\)", line)
                if match:
                    point_id = match.group(1)
                    x = float(match.group(2))
                    y = float(match.group(3))
                    z = float(match.group(4))
                    vertices_dict[point_id] = [x, y, z]
            
            # Extract EDGE_CURVE definitions (connects vertices)
            elif 'EDGE_CURVE' in line:
                edge_match = re.search(r"#(\d+)=EDGE_CURVE\([^,]*,[^,]*,#(\d+),#(\d+)", line)
                if edge_match:
                    edge_id = edge_match.group(1)
                    start_vertex = edge_match.group(2)
                    end_vertex = edge_match.group(3)
                    edges_dict[edge_id] = (start_vertex, end_vertex)
            
            # Extract ORIENTED_EDGE definitions (defines face boundaries)
            elif 'ORIENTED_EDGE' in line:
                edge_ref = re.search(r"#(\d+)", line)
                if edge_ref and current_face is not None:
                    edge_id = edge_ref.group(1)
                    if edge_id in edges_dict:
                        start_vertex, end_vertex = edges_dict[edge_id]
                        if start_vertex in vertices_dict:
                            current_face['vertices'].append(vertices_dict[start_vertex])
                        if end_vertex in vertices_dict:
                            current_face['vertices'].append(vertices_dict[end_vertex])
            
            # Start new face
            elif 'ADVANCED_FACE' in line:
                if current_face and current_face['vertices']:
                    # Remove duplicate vertices and ensure face is properly closed
                    unique_vertices = []
                    seen = set()
                    for v in current_face['vertices']:
                        v_tuple = tuple(v)
                        if v_tuple not in seen:
                            unique_vertices.append(v)
                            seen.add(v_tuple)
                    if len(unique_vertices) >= 3:
                        current_face['vertices'] = unique_vertices
                        faces.append(current_face)
                current_face = {'vertices': [], 'normal': None}
        
        # Add last face if valid
        if current_face and len(current_face['vertices']) >= 3:
            faces.append(current_face)
            
        logger.info(f"Extracted {len(faces)} faces with vertices")
        
    except Exception as e:
        logger.error(f"Error extracting geometry: {str(e)}")
        return []
        
    return faces

def calculate_face_normal(vertices):
    """Calculate normal vector for a face using Newell's method."""
    if len(vertices) < 3:
        return None
        
    normal = np.array([0.0, 0.0, 0.0])
    for i in range(len(vertices)):
        j = (i + 1) % len(vertices)
        v1 = np.array(vertices[i])
        v2 = np.array(vertices[j])
        normal[0] += (v1[1] - v2[1]) * (v1[2] + v2[2])
        normal[1] += (v1[2] - v2[2]) * (v1[0] + v2[0])
        normal[2] += (v1[0] - v2[0]) * (v1[1] + v2[1])
    
    length = np.linalg.norm(normal)
    if length > 0:
        return normal / length
    return None

def calculate_part_volume(file_path):
    """Calculate volume of the part using face areas and normals."""
    faces = extract_faces_and_vertices(file_path)
    if not faces:
        logger.error("No valid faces found in STEP file")
        return None
        
    total_volume = 0
    reference_point = np.array([0.0, 0.0, 0.0])
    
    for face in faces:
        if len(face['vertices']) >= 3:
            # Calculate face normal using Newell's method
            normal = calculate_face_normal(face['vertices'])
            if normal is None:
                continue
                
            # Calculate face area using triangulation
            area = calculate_face_area(face['vertices'])
            
            # Calculate face centroid
            centroid = np.mean(face['vertices'], axis=0)
            
            # Calculate signed volume contribution using divergence theorem
            volume_contribution = np.dot(centroid - reference_point, normal) * area / 3.0
            total_volume += volume_contribution
    
    volume = abs(total_volume)  # Take absolute value for positive volume
    logger.info(f"Calculated part volume: {volume} mm³")
    return volume if volume > 0 else None

def analyze_step_file(file_path, material):
    """Analyze a STEP file and return geometric properties."""
    try:
        cartesian_points = extract_cartesian_points(file_path)
        
        if not cartesian_points:
            logger.error("No cartesian points found in STEP file")
            raise ValueError("No geometric data found in STEP file")
        
        # Calculate bounding box
        bounding_box = calculate_bounding_box(cartesian_points)
        
        # Count entities
        with open(file_path, 'r') as f:
            content = f.read()
            
        entity_counts = {
            "CARTESIAN_POINT": len(cartesian_points),
            "ADVANCED_FACE": len(re.findall(r"ADVANCED_FACE", content)),
            "VERTEX_POINT": len(re.findall(r"VERTEX_POINT", content)),
            "EDGE_CURVE": len(re.findall(r"EDGE_CURVE", content))
        }
        
        logger.info(f"Entity counts: {entity_counts}")
        
        # Calculate actual part volume with improved method
        part_volume = calculate_part_volume(file_path)
        if part_volume is None or part_volume <= 0:
            # Fallback to bounding box volume if actual volume calculation fails
            logger.warning("Failed to calculate actual volume or volume is zero, using bounding box approximation")
            part_volume = (
                bounding_box["dimensions"][0] * 
                bounding_box["dimensions"][1] * 
                bounding_box["dimensions"][2]
            )
            # Apply a typical solid part fill factor (e.g., 60% of bounding box)
            part_volume *= 0.6
        
        logger.info(f"Final part volume: {part_volume} mm³")
        
        # Calculate raw stock info using bounding box
        raw_stock = calculate_raw_stock(bounding_box, material)
        
        # Calculate material removal
        material_removal = calculate_material_removal(part_volume, raw_stock["volume_mm3"])
        
        # Calculate complexity score
        total_entities = sum(entity_counts.values())
        unique_entity_types = len([count for count in entity_counts.values() if count > 0])
        complexity_score = min(100, (total_entities / 100) + (unique_entity_types * 5))
        
        # Calculate machining time with improved calibration
        time_estimate = calculate_machining_time(part_volume, complexity_score, material)
        
        return {
            "basic_info": {
                "volume_mm3": part_volume,
                "weight_kg": (part_volume / 1000000) * MATERIAL_PARAMS[material]["density"],
                "bounding_box_mm": bounding_box
            },
            "raw_stock": raw_stock,
            "material_removal": material_removal,
            "complexity": {
                "face_count": entity_counts.get("ADVANCED_FACE", 0),
                "vertex_count": entity_counts.get("VERTEX_POINT", 0),
                "edge_count": entity_counts.get("EDGE_CURVE", 0),
                "total_entities": total_entities,
                "unique_entity_types": unique_entity_types
            },
            "machining_estimate": {
                "complexity_score": round(complexity_score, 2),
                "estimated_machine_time_minutes": time_estimate["total_time"],
                "setup_time_minutes": time_estimate["setup_time"],
                "programming_time_minutes": time_estimate["programming_time"],
                "machining_time_minutes": time_estimate["machining_time"],
                "confidence_score": time_estimate["confidence_score"],
                "calibration_points_used": time_estimate["calibration_points_used"],
                "complexity_level": get_complexity_level(complexity_score)
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing STEP file: {str(e)}", exc_info=True)
        raise ValueError(f"Error analyzing STEP file: {str(e)}")

@app.post("/calibrate")
async def calibrate_timing(
    file: UploadFile = File(...),
    material: str = Form(...),
    setup_time: float = Form(...),
    programming_time: float = Form(...),
    machining_time: float = Form(...)
):
    """Add calibration data for a specific part."""
    logger.info(f"Received calibration request for {file.filename} with material: {material}")
    
    if material not in MATERIAL_PARAMS:
        raise HTTPException(status_code=400, detail=f"Unsupported material: {material}")
    
    try:
        # Validate timing values
        if setup_time < 0 or programming_time < 0 or machining_time < 0:
            raise HTTPException(status_code=400, detail="Time values cannot be negative")
        
        # Analyze the file first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.step') as temp_file:
            try:
                content = await file.read()
                temp_file.write(content)
                temp_file.flush()
                
                analysis = analyze_step_file(temp_file.name, material)
                
                # Add calibration data
                calibration_entry = {
                    "filename": file.filename,
                    "volume_mm3": analysis["basic_info"]["volume_mm3"],
                    "complexity_score": analysis["machining_estimate"]["complexity_score"],
                    "setup_time": setup_time,
                    "programming_time": programming_time,
                    "machining_time": machining_time,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                # Initialize calibration_data if it doesn't exist
                if "calibration_data" not in MATERIAL_PARAMS[material]:
                    MATERIAL_PARAMS[material]["calibration_data"] = []
                
                MATERIAL_PARAMS[material]["calibration_data"].append(calibration_entry)
                MATERIAL_PARAMS[material]["setup_time"] = setup_time
                MATERIAL_PARAMS[material]["programming_time"] = programming_time
                
                # Save updated parameters
                save_material_params()
                
                logger.info(f"Calibration data added successfully for {file.filename}")
                return {"message": "Calibration data added successfully"}
                
            finally:
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    logger.error(f"Error cleaning up temporary file: {e}")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during calibration: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during calibration: {str(e)}")

@app.post("/analyze")
async def analyze_step_file_endpoint(
    file: UploadFile = File(...),
    material: str = Form(...),
):
    """
    Analyze a STEP file and return geometric properties.
    """
    logger.info(f"Received file: {file.filename} with material: {material}")
    
    if material not in MATERIAL_PARAMS:
        raise HTTPException(status_code=400, detail=f"Unsupported material: {material}")
    
    if not file.filename.lower().endswith(('.step', '.stp')):
        raise HTTPException(status_code=400, detail="Only STEP files are allowed")
    
    # Create temporary file to store uploaded STEP file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.step') as temp_file:
        try:
            # Write uploaded file to temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            logger.info(f"File saved temporarily at: {temp_file.name}")
            
            # Analyze the STEP file
            try:
                analysis = analyze_step_file(temp_file.name, material)
                logger.info("Analysis completed successfully")
                return analysis
            except ValueError as e:
                logger.error(f"Error analyzing file: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file.name)
                logger.info("Temporary file cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {str(e)}")

@app.get("/")
async def read_root():
    return {"message": "Simple STEP File Analyzer API"}

def extract_cartesian_points(step_file_path):
    """Extract all cartesian points from STEP file data."""
    points = []
    logger.debug(f"Starting to process STEP file")
    
    try:
        # Read the STEP file content directly
        with open(step_file_path, 'r') as f:
            content = f.read()
        
        # Find all CARTESIAN_POINT entries
        # Pattern matches: #NUMBER=CARTESIAN_POINT('name',(X,Y,Z));
        pattern = r"#\d+=CARTESIAN_POINT\([^)]+,\(([-+]?\d*\.?\d+,[-+]?\d*\.?\d+,[-+]?\d*\.?\d+)\)"
        matches = re.finditer(pattern, content)
        
        for match in matches:
            try:
                # Extract the coordinates
                coords_str = match.group(1)
                logger.debug(f"Found coordinates: {coords_str}")
                
                # Split the coordinates and convert to float
                coords = [float(x.strip()) for x in coords_str.split(',')]
                if len(coords) >= 3:
                    points.append(coords[:3])
                    logger.debug(f"Successfully extracted point: {coords[:3]}")
            except Exception as e:
                logger.warning(f"Error processing coordinates {coords_str}: {e}")
                continue
        
        logger.info(f"Extracted {len(points)} points from STEP file")
        if points:
            logger.debug(f"First few points: {points[:5]}")
        
        if not points:
            # Try alternative pattern for different STEP file format
            pattern = r"CARTESIAN_POINT\s*\(\s*'[^']*'\s*,\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)\s*\)"
            matches = re.finditer(pattern, content)
            
            for match in matches:
                try:
                    coords = [float(match.group(1)), float(match.group(2)), float(match.group(3))]
                    points.append(coords)
                    logger.debug(f"Successfully extracted point (alternative format): {coords}")
                except Exception as e:
                    logger.warning(f"Error processing coordinates from alternative format: {e}")
                    continue
            
            logger.info(f"Extracted {len(points)} points using alternative format")
    
    except Exception as e:
        logger.error(f"Error processing STEP file: {e}")
        raise ValueError(f"Error processing STEP file: {e}")
    
    return points

def calculate_bounding_box(cartesian_points):
    """Calculate bounding box from a list of cartesian points."""
    if not cartesian_points:
        return None
    
    points = np.array(cartesian_points)
    min_corner = np.min(points, axis=0)
    max_corner = np.max(points, axis=0)
    dimensions = max_corner - min_corner
    
    logger.debug(f"Bounding box calculated: min={min_corner}, max={max_corner}, dimensions={dimensions}")
    return {
        "dimensions": dimensions.tolist(),
        "min_corner": min_corner.tolist(),
        "max_corner": max_corner.tolist()
    }

def get_complexity_level(score: float) -> str:
    """Convert numerical score to descriptive complexity level."""
    if score < 20:
        return "Simple"
    elif score < 40:
        return "Moderate"
    elif score < 60:
        return "Complex"
    elif score < 80:
        return "Very Complex"
    else:
        return "Extremely Complex"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 