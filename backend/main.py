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
    title="HMT - Fast-Quote",
    description="Advanced STEP File Analysis and Machining Time Estimation",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://192.168.0.12:3000",  # Local network access
        "http://127.0.0.1:3000",
        "http://94.145.236.170:3000",     # External IP frontend access
        "http://94.145.236.170:8000",     # External IP backend access
        "https://94.145.236.170:3000",    # HTTPS frontend
        "https://94.145.236.170:8000"     # HTTPS backend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]  # Add this to expose all headers
)

# Material properties
MATERIAL_PARAMS = {
    "aluminum_6082": {
        "name": "Aluminum 6082",
        "density": 2.7,  # g/cm³
        "base_removal_rate": 25000,  # cubic mm per minute
        "complexity_factor": 0.5,    # multiplier for complex geometries
        "setup_time": 60,          # minutes
        "programming_time": 60,    # minutes
        "tool_change_time": 5,      # minutes per tool change
        "finishing_factor": 0.3,    # multiplier for finishing operations
        "target_machining_time": 60,  # target machining time in minutes
        "stock_margin": 5,  # mm to add to each dimension for raw stock
        "calibration_data": {},  # Dictionary to store calibration data by filename
        "machinability_rating": 1.0  # Base reference for machinability
    },
    "steel_s355": {
        "name": "Steel S355",
        "density": 7.85,  # g/cm³
        "base_removal_rate": 8000,   # Lower removal rate for steel
        "complexity_factor": 0.7,    # Higher complexity factor due to material hardness
        "setup_time": 90,           # Longer setup time for steel
        "programming_time": 75,     # More complex programming for steel
        "tool_change_time": 8,      # More frequent tool changes
        "finishing_factor": 0.4,    # More finishing time needed
        "target_machining_time": 120,  # Longer base machining time
        "stock_margin": 6,  # Slightly larger margin for steel
        "calibration_data": {},
        "machinability_rating": 0.4  # Harder to machine than aluminum
    },
    "steel_30crni": {
        "name": "Steel 30CrNi",
        "density": 7.85,  # g/cm³
        "base_removal_rate": 6000,   # Lower removal rate for alloy steel
        "complexity_factor": 0.8,    # Higher complexity due to material properties
        "setup_time": 100,          # More setup time needed
        "programming_time": 90,     # More complex programming
        "tool_change_time": 10,     # More frequent tool changes
        "finishing_factor": 0.5,    # More finishing time needed
        "target_machining_time": 150,  # Longer machining time
        "stock_margin": 6,  # Similar margin to other steels
        "calibration_data": {},
        "machinability_rating": 0.3  # Harder to machine than S355
    },
    "stainless_316l": {
        "name": "Stainless Steel 316L",
        "density": 8.0,  # g/cm³
        "base_removal_rate": 4000,   # Much lower removal rate for stainless
        "complexity_factor": 0.9,    # Highest complexity factor
        "setup_time": 120,          # Longest setup time
        "programming_time": 100,    # Most complex programming
        "tool_change_time": 12,     # Most frequent tool changes
        "finishing_factor": 0.6,    # Most finishing time needed
        "target_machining_time": 180,  # Longest machining time
        "stock_margin": 7,  # Larger margin for stainless
        "calibration_data": {},
        "machinability_rating": 0.2  # Hardest to machine
    }
}

# Ensure data directory exists
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)

# Initialize material_params.json if it doesn't exist
material_params_file = os.path.join(data_dir, 'material_params.json')
if not os.path.exists(material_params_file):
    logger.info("Creating initial material_params.json")
    with open(material_params_file, 'w') as f:
        json.dump(MATERIAL_PARAMS, f, indent=2)
    logger.info("Initial material_params.json created")

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
                loaded_params = json.load(f)
                # Ensure calibration_data is a dictionary for each material
                for material in loaded_params:
                    if "calibration_data" not in loaded_params[material]:
                        loaded_params[material]["calibration_data"] = {}
                    elif isinstance(loaded_params[material]["calibration_data"], list):
                        # Convert old list format to dictionary format
                        calibration_dict = {}
                        for entry in loaded_params[material]["calibration_data"]:
                            if "filename" in entry:
                                calibration_dict[entry["filename"]] = entry
                        loaded_params[material]["calibration_data"] = calibration_dict
                
                global MATERIAL_PARAMS
                MATERIAL_PARAMS = loaded_params
            logger.info("Material parameters loaded successfully")
    except Exception as e:
        logger.error(f"Error loading material parameters: {e}")

# Load saved parameters on startup
load_material_params()

# Add this near the top where other globals are defined
ANALYSIS_HISTORY = {}

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
    """Calculate similarity score between two parts based on volume, complexity, and material removal."""
    # Volume ratio (0-1)
    volume_ratio = min(part1["volume_mm3"], part2["volume_mm3"]) / max(part1["volume_mm3"], part2["volume_mm3"])
    
    # Complexity similarity (0-1)
    complexity_diff = abs(part1["complexity_score"] - part2["complexity_score"]) / 100
    complexity_similarity = 1 - complexity_diff
    
    # Material removal similarity (0-1)
    removal_ratio1 = part1["material_removal"]["removal_percentage"] / 100
    removal_ratio2 = part2["material_removal"]["removal_percentage"] / 100
    removal_similarity = 1 - abs(removal_ratio1 - removal_ratio2)
    
    # Weight the factors (adjust weights as needed)
    weights = {
        "volume": 0.4,
        "complexity": 0.3,
        "removal": 0.3
    }
    
    similarity = (
        volume_ratio * weights["volume"] +
        complexity_similarity * weights["complexity"] +
        removal_similarity * weights["removal"]
    )
    
    return similarity

def calculate_machining_time(volume_mm3, complexity_score, material, material_removal):
    """Calculate estimated machining time based on calibration data and complexity."""
    params = MATERIAL_PARAMS[material]
    calibration_data = params.get("calibration_data", {})
    
    current_part = {
        "volume_mm3": volume_mm3,
        "complexity_score": complexity_score,
        "material_removal": material_removal
    }
    
    # Find similar parts
    similar_parts = []
    for part_name, calibrated_part in calibration_data.items():
        similarity = calculate_similarity_score(current_part, calibrated_part)
        if similarity > 0.7:  # Only use parts with >70% similarity
            similar_parts.append({
                "filename": part_name,
                "similarity": similarity,
                "setup_time": calibrated_part["setup_time"],
                "programming_time": calibrated_part["programming_time"],
                "machining_time": calibrated_part["machining_time"]
            })
    
    # Sort by similarity
    similar_parts.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Calculate base times
    if similar_parts:
        # Calculate weighted average times
        total_weight = sum(part["similarity"] for part in similar_parts)
        setup_time = sum(part["similarity"] * part["setup_time"] for part in similar_parts) / total_weight
        programming_time = sum(part["similarity"] * part["programming_time"] for part in similar_parts) / total_weight
        machining_time = sum(part["similarity"] * part["machining_time"] for part in similar_parts) / total_weight
        
        # Calculate confidence score based on similarities
        best_similarity = similar_parts[0]["similarity"]
        confidence_score = min(100, best_similarity * 100)
    else:
        # Fallback to basic calculation if no similar parts found
        base_time = params["target_machining_time"]
        complexity_multiplier = 1 + ((complexity_score / 100) * params["complexity_factor"] - 0.5)
        removal_multiplier = 1 + (material_removal["removal_percentage"] / 200)
        
        setup_time = params["setup_time"]
        programming_time = params["programming_time"]
        machining_time = base_time * complexity_multiplier * removal_multiplier
        confidence_score = 0
    
    # Calculate batch quantities with efficiency improvements
    batch_quantities = [1, 5, 10, 20, 50]
    batch_estimates = {}
    
    for quantity in batch_quantities:
        # Setup time is constant per batch
        batch_setup = setup_time
        
        # Programming time is constant per batch
        batch_programming = programming_time
        
        # Machining time improves with quantity due to optimizations and learning curve
        # Using a learning curve factor: efficiency improves as quantity increases
        learning_factor = 1 - (math.log(quantity, 50) * 0.15)  # Max 15% improvement at 50 pieces
        batch_machining_per_part = machining_time * learning_factor
        total_machining = batch_machining_per_part * quantity
        
        # Calculate totals
        total_time = batch_setup + batch_programming + total_machining
        time_per_part = total_time / quantity
        
        batch_estimates[str(quantity)] = {
            "total_time": round(total_time, 2),
            "time_per_part": round(time_per_part, 2),
            "setup_time": round(batch_setup, 2),
            "programming_time": round(batch_programming, 2),
            "machining_time_per_part": round(batch_machining_per_part, 2),
            "total_machining_time": round(total_machining, 2)
        }
    
    return {
        "total_time": round(setup_time + programming_time + machining_time, 2),
        "setup_time": round(setup_time, 2),
        "programming_time": round(programming_time, 2),
        "machining_time": round(machining_time, 2),
        "confidence_score": round(confidence_score, 2),
        "calibration_points_used": len(similar_parts),
        "similar_parts": similar_parts[:3],  # Return top 3 similar parts
        "batch_estimates": batch_estimates
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
    try:
        # Load the STEP file using CadQuery
        model = cq.importers.importStep(file_path)
        shape = model.val()
        
        if shape is None:
            raise ValueError("Could not load shape from STEP file")
        
        # Get tessellated representation of the entire shape
        logger.info("Tessellating shape...")
        tess = shape.tessellate(tolerance=0.01)
        vertices = [list(v) for v in tess[0]]  # Convert vertices to lists
        faces = [list(f) for f in tess[1]]     # Convert faces to lists
        
        logger.info(f"Tessellation complete: {len(vertices)} vertices, {len(faces)} faces")
        
        return {
            "vertices": vertices,
            "faces": faces,
            "normal": None  # Not needed for visualization
        }
        
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

async def analyze_step_file(file: UploadFile, material: str):
    """
    Analyze a STEP file and return its geometric properties.
    """
    try:
        if material not in MATERIAL_PARAMS:
            raise HTTPException(status_code=400, detail=f"Unsupported material: {material}")
            
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.step') as temp_file:
            try:
                # Save uploaded file
                content = await file.read()
                temp_file.write(content)
                temp_file.flush()
                
                # Reset file position for future reads
                await file.seek(0)
                
                # Load the STEP file using CadQuery
                model = cq.importers.importStep(str(temp_file.name))
                shape = model.val()
                
                # Get basic measurements using CadQuery
                results = analyze_step_with_cadquery(str(temp_file.name))
                
                # Get features data from results
                features_data = results.get("features_data", {"estimated_axes": 3})
                
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
                time_estimate = calculate_machining_time(
                    volume_mm3=results["volume"],
                    complexity_score=complexity_score,
                    material=material,
                    material_removal=material_removal
                )
                
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
                        "required_axes": features_data["estimated_axes"],
                        "estimated_machine_time_minutes": time_estimate["total_time"],
                        "setup_time_minutes": time_estimate["setup_time"],
                        "programming_time_minutes": time_estimate["programming_time"],
                        "machining_time_minutes": time_estimate["machining_time"],
                        "confidence_score": time_estimate["confidence_score"],
                        "calibration_points_used": time_estimate["calibration_points_used"],
                        "similar_parts": time_estimate.get("similar_parts", []),
                        "batch_estimates": time_estimate.get("batch_estimates", {})
                    }
                }
                
                return response
                
            finally:
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    logger.error(f"Error cleaning up temporary file: {e}")
                    
    except Exception as e:
        logger.error(f"Error processing STEP file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
        analysis = await analyze_step_file_endpoint(file=file, material=material)
        
        # Add calibration data
        calibration_entry = {
            "filename": file.filename,
            "volume_mm3": analysis["basic_info"]["volume_mm3"],
            "complexity_score": analysis["machining_estimate"]["complexity_score"],
            "setup_time": setup_time,
            "programming_time": programming_time,
            "machining_time": machining_time,
            "material_removal": analysis["material_removal"],
            "timestamp": datetime.datetime.now().isoformat(),
            "is_calibrated": True
        }
        
        # Initialize calibration_data if it doesn't exist
        if "calibration_data" not in MATERIAL_PARAMS[material]:
            MATERIAL_PARAMS[material]["calibration_data"] = {}
        
        # Store calibration by filename, overwriting any previous calibration for this file
        MATERIAL_PARAMS[material]["calibration_data"][file.filename] = calibration_entry
        
        # Save updated parameters
        save_material_params()
        
        logger.info(f"Calibration data added successfully for {file.filename}")
        return {"message": "Calibration data added successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during calibration: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during calibration: {str(e)}")

@app.get("/history")
async def get_analysis_history(material: str):
    """Get history of all analyzed parts, including both calibrated and non-calibrated parts."""
    if material not in MATERIAL_PARAMS:
        raise HTTPException(status_code=400, detail=f"Unsupported material: {material}")
        
    try:
        # Get calibration data (these are the calibrated parts)
        calibration_data = MATERIAL_PARAMS[material].get("calibration_data", {})
        
        # Combine calibrated and non-calibrated parts
        history = []
        
        # Add calibrated parts
        for filename, data in calibration_data.items():
            entry = data.copy()
            entry["filename"] = filename
            entry["is_calibrated"] = True
            entry["calibration_timestamp"] = entry.get("timestamp")
            
            # Calculate batch estimates for calibrated parts
            total_time = entry["setup_time"] + entry["programming_time"] + entry["machining_time"]
            batch_estimates = {}
            batch_quantities = [1, 5, 10, 20, 50]
            
            for quantity in batch_quantities:
                # Setup time is constant per batch
                batch_setup = entry["setup_time"]
                
                # Programming time is constant per batch
                batch_programming = entry["programming_time"]
                
                # Machining time improves with quantity due to optimizations and learning curve
                learning_factor = 1 - (math.log(quantity, 50) * 0.15)  # Max 15% improvement at 50 pieces
                batch_machining_per_part = entry["machining_time"] * learning_factor
                total_machining = batch_machining_per_part * quantity
                
                # Calculate totals
                batch_total_time = batch_setup + batch_programming + total_machining
                time_per_part = batch_total_time / quantity
                
                batch_estimates[str(quantity)] = {
                    "total_time": round(batch_total_time, 2),
                    "time_per_part": round(time_per_part, 2),
                    "setup_time": round(batch_setup, 2),
                    "programming_time": round(batch_programming, 2),
                    "machining_time_per_part": round(batch_machining_per_part, 2),
                    "total_machining_time": round(total_machining, 2)
                }
            
            # Add batch estimates to the entry
            entry["machining_estimate"] = {
                "batch_estimates": batch_estimates,
                "total_time": total_time,
                "setup_time_minutes": entry["setup_time"],
                "programming_time_minutes": entry["programming_time"],
                "machining_time_minutes": entry["machining_time"]
            }
            
            history.append(entry)
            
        # Add analyzed but non-calibrated parts
        if material in ANALYSIS_HISTORY:
            for filename, data in ANALYSIS_HISTORY[material].items():
                if filename not in calibration_data:  # Only add if not already calibrated
                    entry = data.copy()
                    entry["filename"] = filename
                    entry["is_calibrated"] = False
                    history.append(entry)
        
        # Sort by timestamp, newest first
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return {
            "history": history,
            "total_entries": len(history),
            "calibrated_count": len(calibration_data),
            "analyzed_count": len(history) - len(calibration_data)
        }
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")

@app.post("/analyze")
async def analyze_step_file_endpoint(file: UploadFile = File(...), material: str = Form(...)):
    """Analyze a STEP file and return its geometric properties."""
    try:
        # Validate material
        if material not in MATERIAL_PARAMS:
            logger.error(f"Invalid material specified: {material}")
            raise HTTPException(status_code=400, detail=f"Unsupported material: {material}")
            
        # Create a temporary file to save the uploaded content
        temp_file = Path("temp_step_file.step")
        try:
            # Save uploaded file
            content = await file.read()
            with open(temp_file, "wb") as buffer:
                buffer.write(content)
            await file.seek(0)  # Reset file position for future reads
            
            # Load and validate the STEP file
            try:
                model = cq.importers.importStep(str(temp_file))
                shape = model.val()
                
                if shape is None:
                    raise ValueError("Could not load shape from STEP file")
                    
                # Get basic measurements using CadQuery
                logger.info(f"Analyzing STEP file: {file.filename}")
                results = analyze_step_with_cadquery(str(temp_file))
                
                # Get features data from results
                features_data = results.get("features_data", {"estimated_axes": 3})
                
                # Count entities
                entity_counts = {
                    "face_count": len(list(shape.faces())),
                    "vertex_count": len(list(shape.vertices())),
                    "edge_count": len(list(shape.edges()))
                }
                entity_counts["total_entities"] = sum(entity_counts.values())
                
                # Calculate part weight
                density_kg_per_mm3 = MATERIAL_PARAMS[material]["density"] * 1e-6
                part_weight_kg = results["volume"] * density_kg_per_mm3
                
                # Calculate raw stock dimensions
                margin = MATERIAL_PARAMS[material]["stock_margin"]
                raw_dimensions = [d + (2 * margin) for d in results["dimensions"]]
                raw_volume = raw_dimensions[0] * raw_dimensions[1] * raw_dimensions[2]
                raw_weight = raw_volume * density_kg_per_mm3
                
                # Calculate material removal
                material_removal = {
                    "removed_volume_mm3": raw_volume - results["volume"],
                    "removal_percentage": ((raw_volume - results["volume"]) / raw_volume) * 100
                }
                
                # Calculate complexity score
                sa_to_vol_ratio = results["surface_area"] / results["volume"]
                sa_to_vol_score = min(50, (sa_to_vol_ratio * 1000))
                
                max_expected_entities = 10000
                entity_score = min(50, (entity_counts["total_entities"] / max_expected_entities) * 50)
                
                complexity_score = sa_to_vol_score + entity_score
                
                # Calculate machining time
                time_estimate = calculate_machining_time(
                    volume_mm3=results["volume"],
                    complexity_score=complexity_score,
                    material=material,
                    material_removal=material_removal
                )
                
                # Format response
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
                        "required_axes": features_data["estimated_axes"],
                        "setup_time_minutes": time_estimate["setup_time"],
                        "programming_time_minutes": time_estimate["programming_time"],
                        "machining_time_minutes": time_estimate["machining_time"],
                        "estimated_machine_time_minutes": time_estimate["total_time"],
                        "confidence_score": time_estimate["confidence_score"],
                        "calibration_points_used": time_estimate["calibration_points_used"],
                        "similar_parts": time_estimate.get("similar_parts", []),
                        "batch_estimates": time_estimate.get("batch_estimates", {})
                    }
                }
                
                # Before returning the response, save to history
                analysis_entry = {
                    "filename": file.filename,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "volume_mm3": response["basic_info"]["volume_mm3"],
                    "complexity_score": response["machining_estimate"]["complexity_score"],
                    "material_removal": response["material_removal"],
                    "machining_estimate": response["machining_estimate"],
                    "is_calibrated": False
                }
                
                # Initialize material history if it doesn't exist
                if material not in ANALYSIS_HISTORY:
                    ANALYSIS_HISTORY[material] = {}
                
                # Save to history
                ANALYSIS_HISTORY[material][file.filename] = analysis_entry
                
                return response
                
            except Exception as e:
                logger.error(f"Error processing STEP file: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid STEP file: {str(e)}")
                
        finally:
            # Clean up temporary file
            if temp_file.exists():
                temp_file.unlink()
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

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
            content = await file.read()
            f.write(content)
            await file.seek(0)  # Reset file position
        
        try:
            # Extract geometry using the improved function
            geometry_data = extract_faces_and_vertices(temp_path)
            
            # Load shape for metadata
            model = cq.importers.importStep(temp_path)
            shape = model.val()
            
            # Get bounding box
            bbox = shape.BoundingBox()
            bbox_min = [bbox.xmin, bbox.ymin, bbox.zmin]
            bbox_max = [bbox.xmax, bbox.ymax, bbox.zmax]
            dimensions = [
                bbox_max[0] - bbox_min[0],
                bbox_max[1] - bbox_min[1],
                bbox_max[2] - bbox_min[2]
            ]
            
            # Count geometric entities
            entity_counts = {
                "vertices": len(geometry_data["vertices"]),
                "faces": len(geometry_data["faces"]),
                "edges": len(list(shape.edges()))
            }
            
            logger.info(f"Final counts: {entity_counts}")
            
            return {
                "vertices": geometry_data["vertices"],
                "faces": geometry_data["faces"],
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

def determine_required_axes(shape, faces):
    """
    Determine if 3-axis or 5-axis machining is required.
    Finds the main face (largest) and then checks side faces for holes/pockets.
    Uses 3-axis if no side holes/pockets, 5-axis if there are any.
    
    Parameters:
    -----------
    shape : cadquery.Shape
        The CadQuery shape object
    faces : list
        List of faces with vertices and normals
        
    Returns:
    --------
    dict
        Dictionary containing:
        - estimated_axes: int (3 or 5)
        - reasons: list of reasons for the axis determination
    """
    reasons = []
    
    try:
        # 1. Find the main face (largest face)
        largest_face = None
        largest_area = 0
        largest_normal = None
        
        for face in shape.faces():
            area = face.Area()
            if area > largest_area:
                largest_area = area
                largest_face = face
                largest_normal = face.normalAt()
        
        if not largest_face or not largest_normal:
            return {
                "estimated_axes": 3,
                "reasons": ["No faces found, defaulting to 3-axis"]
            }
        
        # 2. Find side faces (faces perpendicular to main face)
        side_faces = []
        for face in shape.faces():
            if face == largest_face:
                continue
                
            normal = face.normalAt()
            # Calculate dot product to check if perpendicular (should be close to 0)
            dot_product = abs(normal.x * largest_normal.x + 
                            normal.y * largest_normal.y + 
                            normal.z * largest_normal.z)
            
            if dot_product < 0.1:  # Faces are nearly perpendicular
                side_faces.append(face)
        
        # 3. Check for holes/pockets in side faces
        feature_count = 0
        for face in side_faces:
            wires = list(face.innerWires())
            feature_count += len(wires)
            if feature_count > 0:  # Stop counting once we find any features
                break
        
        # 4. Determine required axes
        if feature_count > 0:
            required_axes = 5
            reasons.append(f"5-axis required: Found {feature_count} holes/pockets in side faces")
        else:
            required_axes = 3
            reasons.append("3-axis suitable: No holes/pockets found in side faces")
        
        # Add analysis details
        reasons.append(f"Analysis details: Checked {len(side_faces)} side faces")
        
        return {
            "estimated_axes": required_axes,
            "reasons": reasons
        }
        
    except Exception as e:
        logger.warning(f"Error determining required axes: {str(e)}")
        return {
            "estimated_axes": 3,
            "reasons": ["Default to 3-axis due to analysis error"]
        }

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
        - features_data: dict with machining features analysis
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
        
        # Extract faces for feature analysis
        faces = extract_faces_and_vertices(file_path)
        
        # Determine required machining axes
        features_data = determine_required_axes(solid, faces)
        
        return {
            "dimensions": dimensions,
            "volume": volume,
            "surface_area": surface_area,
            "center_of_mass": center_of_mass,
            "bounding_box": {
                "min": [bb.xmin, bb.ymin, bb.zmin],
                "max": [bb.xmax, bb.ymax, bb.zmax]
            },
            "features_data": features_data
        }
        
    except Exception as e:
        logger.error(f"Error analyzing STEP file with CadQuery: {str(e)}")
        raise

@app.get("/materials")
async def get_materials():
    """Get list of all available materials and their calibration counts."""
    try:
        materials_info = {}
        for material in MATERIAL_PARAMS:
            calibration_count = len(MATERIAL_PARAMS[material].get("calibration_data", {}))
            materials_info[material] = {
                "name": MATERIAL_PARAMS[material]["name"],
                "calibration_count": calibration_count,
                "density": MATERIAL_PARAMS[material]["density"],
                "machinability_rating": MATERIAL_PARAMS[material]["machinability_rating"]
            }
        return materials_info
    except Exception as e:
        logger.error(f"Error retrieving materials info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving materials info: {str(e)}")

@app.get("/calibrations/{material}")
async def get_calibrations_for_material(material: str):
    """Get all calibration data for a specific material."""
    if material not in MATERIAL_PARAMS:
        raise HTTPException(status_code=400, detail=f"Unsupported material: {material}")
    
    try:
        calibrations = MATERIAL_PARAMS[material].get("calibration_data", {})
        # Convert to list and sort by timestamp
        calibration_list = []
        for filename, data in calibrations.items():
            entry = data.copy()
            entry["filename"] = filename
            calibration_list.append(entry)
        
        # Sort by timestamp descending (newest first)
        calibration_list.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return {
            "material_name": MATERIAL_PARAMS[material]["name"],
            "calibration_count": len(calibration_list),
            "calibrations": calibration_list
        }
    except Exception as e:
        logger.error(f"Error retrieving calibrations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving calibrations: {str(e)}")

@app.delete("/calibrations/{material}")
async def reset_calibrations(material: str):
    """Reset all calibration data for a specific material."""
    if material not in MATERIAL_PARAMS:
        raise HTTPException(status_code=400, detail=f"Unsupported material: {material}")
    
    try:
        # Reset calibration data
        MATERIAL_PARAMS[material]["calibration_data"] = {}
        # Save to file
        save_material_params()
        return {"message": f"Calibration data reset for {MATERIAL_PARAMS[material]['name']}"}
    except Exception as e:
        logger.error(f"Error resetting calibrations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resetting calibrations: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 