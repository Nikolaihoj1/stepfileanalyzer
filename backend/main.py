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
import cadquery as cq
from cadquery.occ_impl.shapes import Vertex, Face
from cadquery.occ_impl.geom import BoundBox
import logging
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
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
    margin = 5.0  # Fixed 5mm margin
    dimensions = bounding_box["dimensions"]
    
    # Calculate raw dimensions with margin
    raw_dimensions = [d + (2 * margin) for d in dimensions]
    
    # Calculate raw volume
    raw_volume = raw_dimensions[0] * raw_dimensions[1] * raw_dimensions[2]
    
    # Convert density from g/cm³ to kg/mm³
    # 1 g/cm³ = 1 / (100^3) kg/mm³ = 1e-6 kg/mm³
    density_kg_per_mm3 = MATERIAL_PARAMS[material]["density"] * 1e-6
    weight_kg = raw_volume * density_kg_per_mm3
    
    return {
        "dimensions": raw_dimensions,
        "volume_mm3": raw_volume,
        "weight_kg": weight_kg
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
            
        logger.info(f"Starting STEP file parsing, found {len(content)} lines")
        
        # Debug counters
        point_count = 0
        edge_count = 0
        face_count = 0
        
        # More flexible regex patterns
        cartesian_point_patterns = [
            r"#(\d+)\s*=\s*CARTESIAN_POINT\s*\([^(]*\(([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)",
            r"CARTESIAN_POINT\s*\([^(]*\(([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)"
        ]
        
        edge_patterns = [
            r"#(\d+)\s*=\s*EDGE_CURVE\s*\([^,]*,\s*[^,]*,\s*#(\d+)\s*,\s*#(\d+)",
            r"EDGE_CURVE\s*\([^,]*,\s*[^,]*,\s*#(\d+)\s*,\s*#(\d+)"
        ]
        
        for line_num, line in enumerate(content, 1):
            try:
                # Try to extract CARTESIAN_POINT using multiple patterns
                point_found = False
                for pattern in cartesian_point_patterns:
                    match = re.search(pattern, line)
                    if match:
                        if len(match.groups()) == 4:  # Pattern with ID
                            point_id = match.group(1)
                            x, y, z = map(float, match.groups()[1:])
                        else:  # Pattern without ID
                            point_id = str(len(vertices_dict) + 1)
                            x, y, z = map(float, match.groups())
                        vertices_dict[point_id] = [x, y, z]
                        point_count += 1
                        point_found = True
                        if point_count % 100 == 0:
                            logger.debug(f"Processed {point_count} points")
                        break
                
                if point_found:
                    continue
                
                # Try to extract EDGE_CURVE using multiple patterns
                edge_found = False
                for pattern in edge_patterns:
                    match = re.search(pattern, line)
                    if match:
                        if len(match.groups()) == 3:  # Pattern with ID
                            edge_id = match.group(1)
                            start_vertex = match.group(2)
                            end_vertex = match.group(3)
                        else:  # Pattern without ID
                            edge_id = str(len(edges_dict) + 1)
                            start_vertex = match.group(1)
                            end_vertex = match.group(2)
                        edges_dict[edge_id] = (start_vertex, end_vertex)
                        edge_count += 1
                        edge_found = True
                        break
                
                if edge_found:
                    continue
                
                # Handle ADVANCED_FACE
                if 'ADVANCED_FACE' in line or 'FACE_SURFACE' in line:
                    # Process previous face if it exists
                    if current_face and current_face['vertices']:
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
                            face_count += 1
                            logger.debug(f"Added face {face_count} with {len(unique_vertices)} vertices")
                    
                    # Start new face
                    current_face = {'vertices': [], 'normal': None}
                    continue
                
                # Extract ORIENTED_EDGE references
                if 'ORIENTED_EDGE' in line:
                    edge_refs = re.findall(r"#(\d+)", line)
                    for edge_id in edge_refs:
                        if edge_id in edges_dict:
                            start_vertex, end_vertex = edges_dict[edge_id]
                            if start_vertex in vertices_dict and current_face is not None:
                                current_face['vertices'].append(vertices_dict[start_vertex])
                            if end_vertex in vertices_dict and current_face is not None:
                                current_face['vertices'].append(vertices_dict[end_vertex])
                
            except Exception as e:
                logger.warning(f"Error processing line {line_num}: {str(e)}\nLine content: {line.strip()}")
                continue
        
        # Process the last face
        if current_face and len(current_face['vertices']) >= 3:
            faces.append(current_face)
            face_count += 1
        
        logger.info("STEP file parsing complete:")
        logger.info(f"- Found {point_count} points")
        logger.info(f"- Found {edge_count} edges")
        logger.info(f"- Found {face_count} faces")
        logger.info(f"- Vertices dictionary size: {len(vertices_dict)}")
        logger.info(f"- Edges dictionary size: {len(edges_dict)}")
        logger.info(f"- Final faces count: {len(faces)}")
        
        if not faces:
            # Try to create a simple face from all vertices if no faces were found
            if len(vertices_dict) >= 3:
                logger.warning("No faces found, attempting to create a single face from all vertices")
                all_vertices = list(vertices_dict.values())
                faces.append({'vertices': all_vertices, 'normal': None})
                logger.info(f"Created single face with {len(all_vertices)} vertices")
            else:
                logger.error("No valid faces found in STEP file. This could be due to:")
                logger.error("1. Invalid file format")
                logger.error("2. No geometric data in the file")
                logger.error("3. Unsupported geometry representation")
                raise ValueError("No valid faces found in STEP file")
            
        return faces
        
    except Exception as e:
        logger.error(f"Error extracting geometry: {str(e)}")
        raise ValueError(f"Error extracting geometry: {str(e)}")

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
        
    # First, collect all vertices to calculate the bounding box
    all_vertices = []
    for face in faces:
        all_vertices.extend(face['vertices'])
    
    if not all_vertices:
        logger.error("No vertices found in STEP file")
        return None
        
    # Convert to numpy array for calculations
    vertices_array = np.array(all_vertices)
    
    # Calculate bounding box using the same method as calculate_bounding_box
    min_coords = np.min(vertices_array, axis=0)
    max_coords = np.max(vertices_array, axis=0)
    dimensions = np.abs(max_coords - min_coords)
    sorted_indices = np.argsort(dimensions)[::-1]
    
    # Transform vertices to align with sorted dimensions
    vertices_array = vertices_array[:, sorted_indices]
    center = np.mean(vertices_array, axis=0)
    
    total_volume = 0
    valid_face_count = 0
    
    for face in faces:
        if len(face['vertices']) >= 3:
            # Transform face vertices to match sorted dimensions
            vertices = np.array(face['vertices'])[:, sorted_indices]
            
            # Triangulate the face
            for i in range(1, len(vertices) - 1):
                # Create a triangle from vertices[0], vertices[i], vertices[i+1]
                v0 = vertices[0] - center
                v1 = vertices[i] - center
                v2 = vertices[i + 1] - center
                
                # Calculate signed volume contribution using the divergence theorem
                # Volume = (1/6) * |dot(v0, cross(v1, v2))|
                volume_contribution = abs(np.dot(v0, np.cross(v1, v2))) / 6.0
                total_volume += volume_contribution
                valid_face_count += 1
                
                logger.debug(f"Triangle {valid_face_count}: Volume contribution={volume_contribution:.2f}")
    
    if valid_face_count == 0:
        logger.error("No valid faces found for volume calculation")
        return None
        
    logger.info(f"Calculated part volume from {valid_face_count} triangles: {total_volume:.2f} mm³")
    return total_volume if total_volume > 0 else None

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
        volume_calculation_method = "exact"
        
        if part_volume is None or part_volume <= 0:
            # Fallback to bounding box volume if actual volume calculation fails
            logger.warning("Failed to calculate exact volume, falling back to bounding box approximation")
            bounding_box_volume = (
                bounding_box["dimensions"][0] * 
                bounding_box["dimensions"][1] * 
                bounding_box["dimensions"][2]
            )
            # Apply a typical solid part fill factor based on complexity
            face_count = entity_counts.get("ADVANCED_FACE", 0)
            if face_count > 0:
                # More faces typically means more complex geometry and less filled volume
                fill_factor = max(0.2, min(0.8, 1.0 - (face_count / 1000)))
            else:
                fill_factor = 0.6  # Default fill factor
            
            part_volume = bounding_box_volume * fill_factor
            volume_calculation_method = "bounding_box"
            logger.info(f"Estimated part volume using bounding box with {fill_factor:.2f} fill factor: {part_volume:.2f} mm³")
        
        # Calculate raw stock info using bounding box
        raw_stock = calculate_raw_stock(bounding_box, material)
        
        # Ensure raw stock volume is always larger than part volume
        if raw_stock["volume_mm3"] <= part_volume:
            # Add additional margin to ensure raw stock is larger
            scale_factor = (part_volume / raw_stock["volume_mm3"]) * 1.1  # Add 10% extra
            raw_stock["dimensions"] = [d * scale_factor**(1/3) for d in raw_stock["dimensions"]]
            raw_stock["volume_mm3"] = raw_stock["dimensions"][0] * raw_stock["dimensions"][1] * raw_stock["dimensions"][2]
            
            # Convert density from g/cm³ to kg/mm³
            density_kg_per_mm3 = MATERIAL_PARAMS[material]["density"] * 1e-6
            raw_stock["weight_kg"] = raw_stock["volume_mm3"] * density_kg_per_mm3
            
            logger.info(f"Adjusted raw stock dimensions to ensure larger than part volume. Scale factor: {scale_factor:.2f}")
        
        # Calculate material removal
        material_removal = calculate_material_removal(part_volume, raw_stock["volume_mm3"])
        
        # Calculate part weight using density
        # Convert density from g/cm³ to kg/mm³
        density_kg_per_mm3 = MATERIAL_PARAMS[material]["density"] * 1e-6
        part_weight_kg = part_volume * density_kg_per_mm3
        
        # Calculate complexity score
        total_entities = sum(entity_counts.values())
        unique_entity_types = len([count for count in entity_counts.values() if count > 0])
        complexity_score = min(100, (total_entities / 100) + (unique_entity_types * 5))
        
        # Calculate machining time with improved calibration
        time_estimate = calculate_machining_time(part_volume, complexity_score, material)
        
        return {
            "basic_info": {
                "volume_mm3": part_volume,
                "volume_calculation_method": volume_calculation_method,
                "weight_kg": part_weight_kg,
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
async def analyze_step_file(file: UploadFile = File(...), material: str = Form(...)):
    """
    Analyze a STEP file and return its geometric properties.
    """
    try:
        if material not in MATERIAL_PARAMS:
            raise HTTPException(status_code=400, detail=f"Unsupported material: {material}")
            
        # Create a temporary file to save the uploaded content
        temp_file = Path("temp_step_file.step")
        try:
            # Save uploaded file
            with open(temp_file, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Load the STEP file using CadQuery
            model = cq.importers.importStep(str(temp_file))
            shape = model.val()
            
            # Get basic measurements using CadQuery
            results = analyze_step_with_cadquery(str(temp_file))
            
            # Count entities using CadQuery's high-level API
            entity_counts = {
                "face_count": len(list(shape.faces())),
                "vertex_count": len(list(shape.vertices())),
                "edge_count": len(list(shape.edges()))
            }
            entity_counts["total_entities"] = sum(entity_counts.values())
            
            # Calculate part weight using density
            density_kg_per_mm3 = MATERIAL_PARAMS[material]["density"] * 1e-6  # Convert g/cm³ to kg/mm³
            part_weight_kg = results["volume"] * density_kg_per_mm3
            
            # Calculate raw stock dimensions (add margin)
            margin = MATERIAL_PARAMS[material]["stock_margin"]
            raw_dimensions = [d + (2 * margin) for d in results["dimensions"]]
            raw_volume = raw_dimensions[0] * raw_dimensions[1] * raw_dimensions[2]
            raw_weight = raw_volume * density_kg_per_mm3
            
            # Calculate material removal
            material_removal = {
                "removed_volume_mm3": raw_volume - results["volume"],
                "removal_percentage": ((raw_volume - results["volume"]) / raw_volume) * 100
            }
            
            # Calculate complexity score based on multiple factors
            # 1. Surface area to volume ratio (normalized)
            sa_to_vol_ratio = results["surface_area"] / results["volume"]
            sa_to_vol_score = min(50, (sa_to_vol_ratio * 1000))  # Scale and cap at 50
            
            # 2. Entity count score (normalized)
            max_expected_entities = 10000  # Adjust based on typical part complexity
            entity_score = min(50, (entity_counts["total_entities"] / max_expected_entities) * 50)
            
            # Combine scores
            complexity_score = sa_to_vol_score + entity_score
            
            # Calculate machining time estimate
            time_estimate = calculate_machining_time(results["volume"], complexity_score, material)
            
            # Format the response
            response = {
                "basic_info": {
                    "volume_mm3": results["volume"],
                    "weight_kg": part_weight_kg,
                    "bounding_box_mm": {
                        "dimensions": results["dimensions"],
                        "min_corner": results["bounding_box"]["min"],
                        "max_corner": results["bounding_box"]["max"]
                    }
                },
                "raw_stock": {
                    "dimensions": raw_dimensions,
                    "volume_mm3": raw_volume,
                    "weight_kg": raw_weight
                },
                "material_removal": material_removal,
                "complexity": {
                    "surface_area_mm2": results["surface_area"],
                    "face_count": entity_counts["face_count"],
                    "vertex_count": entity_counts["vertex_count"],
                    "edge_count": entity_counts["edge_count"],
                    "total_entities": entity_counts["total_entities"],
                    "surface_area_to_volume_ratio": sa_to_vol_ratio
                },
                "machining_estimate": {
                    "complexity_score": round(complexity_score, 2),
                    "complexity_level": get_complexity_level(complexity_score),
                    "estimated_machine_time_minutes": time_estimate["total_time"],
                    "setup_time_minutes": time_estimate["setup_time"],
                    "programming_time_minutes": time_estimate["programming_time"],
                    "machining_time_minutes": time_estimate["machining_time"],
                    "confidence_score": time_estimate["confidence_score"],
                    "calibration_points_used": time_estimate["calibration_points_used"]
                }
            }
            
            return response
            
        finally:
            # Clean up the temporary file
            if temp_file.exists():
                temp_file.unlink()
                
    except Exception as e:
        logger.error(f"Error processing STEP file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
    
    # Convert points to numpy array for efficient computation
    points = np.array(cartesian_points)
    
    # Find the principal axes using PCA to align with natural part orientation
    mean = np.mean(points, axis=0)
    centered = points - mean
    
    # Calculate min and max along each axis
    min_corner = np.min(points, axis=0)
    max_corner = np.max(points, axis=0)
    dimensions = np.abs(max_corner - min_corner)  # Use absolute value to ensure positive dimensions
    
    # Sort dimensions to match expected order (length, width, height)
    sorted_indices = np.argsort(dimensions)[::-1]  # Sort in descending order
    dimensions = dimensions[sorted_indices]
    min_corner = min_corner[sorted_indices]
    max_corner = max_corner[sorted_indices]
    
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

@app.post("/geometry")
async def get_geometry(file: UploadFile = File(...)):
    """Extract geometry data from STEP file for visualization."""
    try:
        # Save uploaded file temporarily
        temp_path = "temp_step_file.step"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        try:
            # Load the STEP file using CadQuery
            logger.info("Loading STEP file...")
            model = cq.importers.importStep(temp_path)
            shape = model.val()
            
            if shape is None:
                raise ValueError("Could not load shape from STEP file")
            
            # Initialize lists for geometry data
            vertices = []
            faces = []
            vertex_map = {}
            current_vertex_index = 0
            
            # First collect all vertices from the shape
            logger.info("Processing vertices from shape...")
            shape_vertices = list(shape.vertices())
            logger.info(f"Found {len(shape_vertices)} vertices in shape")
            
            if not shape_vertices:
                # Try to get vertices from edges if no vertices found directly
                logger.info("No vertices found directly, trying to extract from edges...")
                for edge in shape.edges():
                    start_vertex = edge.startPoint()
                    end_vertex = edge.endPoint()
                    vertices.extend([list(start_vertex), list(end_vertex)])
                logger.info(f"Found {len(vertices)} vertices from edges")
            else:
                # Process vertices from shape
                for vertex in shape_vertices:
                    pos = vertex.toTuple()
                    vertices.append(list(pos))
            
            # Process faces
            logger.info("Processing faces...")
            shape_faces = list(shape.faces())
            logger.info(f"Found {len(shape_faces)} faces in shape")
            
            for face_idx, face in enumerate(shape_faces):
                try:
                    # Get triangulation of the face with smaller tolerance
                    tess = face.tessellate(tolerance=0.01)
                    if not tess or len(tess) != 2:
                        logger.warning(f"Invalid tessellation result for face {face_idx}")
                        continue
                    
                    face_vertices, triangles = tess
                    logger.debug(f"Face {face_idx}: {len(face_vertices)} vertices, {len(triangles)} triangle indices")
                    
                    if not face_vertices or not triangles:
                        logger.warning(f"Empty tessellation data for face {face_idx}")
                        continue
                    
                    # Add vertices from this face
                    face_vertex_indices = []
                    for vertex in face_vertices:
                        vertex_tuple = tuple(vertex)
                        if vertex_tuple not in vertex_map:
                            vertex_map[vertex_tuple] = len(vertices)
                            vertices.append(list(vertex))
                            face_vertex_indices.append(len(vertices) - 1)
                        else:
                            face_vertex_indices.append(vertex_map[vertex_tuple])
                    
                    # Add triangles using the correct vertex indices
                    for i in range(0, len(triangles), 3):
                        if i + 2 < len(triangles):
                            try:
                                v1_idx = face_vertex_indices[triangles[i]]
                                v2_idx = face_vertex_indices[triangles[i + 1]]
                                v3_idx = face_vertex_indices[triangles[i + 2]]
                                if v1_idx != v2_idx and v2_idx != v3_idx and v3_idx != v1_idx:
                                    faces.append([v1_idx, v2_idx, v3_idx])
                            except IndexError as e:
                                logger.warning(f"Invalid triangle indices at face {face_idx}, triangle {i//3}: {str(e)}")
                                continue
                
                except Exception as e:
                    logger.warning(f"Error processing face {face_idx}: {str(e)}")
                    continue
            
            logger.info(f"Final geometry: {len(vertices)} vertices and {len(faces)} triangles")
            
            if not vertices:
                raise ValueError("No vertices could be extracted from the file")
            
            if not faces:
                logger.warning("No faces extracted, attempting to create triangulation from vertices...")
                # If we have vertices but no faces, try to create a simple triangulation
                if len(vertices) >= 3:
                    for i in range(0, len(vertices) - 2, 3):
                        faces.append([i, i+1, i+2])
                    logger.info(f"Created {len(faces)} triangles from vertices")
            
            # Get bounding box using BoundingBox() method
            bbox = shape.BoundingBox()
            if not bbox:
                # Calculate bounding box manually if the method fails
                logger.warning("BoundingBox() failed, calculating manually...")
                vertices_array = np.array(vertices)
                bbox_min = vertices_array.min(axis=0)
                bbox_max = vertices_array.max(axis=0)
            else:
                bbox_min = [bbox.xmin, bbox.ymin, bbox.zmin]
                bbox_max = [bbox.xmax, bbox.ymax, bbox.zmax]
            
            dimensions = [
                bbox_max[0] - bbox_min[0],
                bbox_max[1] - bbox_min[1],
                bbox_max[2] - bbox_min[2]
            ]
            
            # Count geometric entities
            entity_counts = {
                "vertices": len(vertices),
                "faces": len(faces),
                "edges": len(list(shape.edges()))
            }
            
            logger.info(f"Final counts: {entity_counts}")
            
            if not vertices or not faces:
                raise ValueError("No valid geometry data extracted from the file")
            
            return {
                "vertices": vertices,
                "faces": faces,
                "metadata": {
                    "original_bounds": {
                        "min": bbox_min,
                        "max": bbox_max,
                        "dimensions": dimensions
                    },
                    "entity_counts": entity_counts
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing geometry: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing geometry: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"Error handling file upload: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error handling file upload: {str(e)}")

def analyze_step_with_cadquery(file_path):
    """
    Analyze a STEP file using CadQuery to extract accurate geometric information.
    
    Parameters:
    -----------
    file_path : str
        Path to the STEP file
        
    Returns:
    --------
    dict
        Dictionary containing:
        - dimensions: [length, width, height] in mm
        - volume: in mm³
        - surface_area: in mm²
        - center_of_mass: [x, y, z] coordinates
    """
    try:
        # Load the STEP file
        result = cq.importers.importStep(file_path)
        
        # Create a workplane from the imported result
        part = cq.Workplane("XY").add(result)
        
        # Get the bounding box using CadQuery's built-in methods
        bb = part.objects[0].BoundingBox()
        
        # Calculate dimensions
        dimensions = [
            bb.xmax - bb.xmin,  # length
            bb.ymax - bb.ymin,  # width
            bb.zmax - bb.zmin   # height
        ]
        
        # Get the solid for volume and area calculations
        solid = part.val()
        
        # Calculate properties
        volume = solid.Volume()
        surface_area = solid.Area()
        
        # Get center of mass
        com = solid.Center()
        center_of_mass = [com.x, com.y, com.z]
        
        return {
            "dimensions": dimensions,
            "volume": volume,
            "surface_area": surface_area,
            "center_of_mass": center_of_mass,
            "bounding_box": {
                "min": [bb.xmin, bb.ymin, bb.zmin],
                "max": [bb.xmax, bb.ymax, bb.zmax]
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing STEP file with CadQuery: {str(e)}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 