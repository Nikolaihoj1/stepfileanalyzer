import trimesh
import numpy as np
from typing import Dict, Any

def analyze_geometry(step_file_path: str) -> Dict[str, Any]:
    """
    Analyze geometric properties of a STEP file using trimesh.
    
    Args:
        step_file_path (str): Path to the STEP file
        
    Returns:
        dict: Dictionary containing geometric properties
    """
    try:
        # Import the STEP file
        mesh = trimesh.load(step_file_path)
        
        # Get the bounding box
        bounds = mesh.bounds
        extents = mesh.extents
        
        # Calculate dimensions
        dimensions = {
            "length": float(extents[0]),
            "width": float(extents[1]),
            "height": float(extents[2])
        }
        
        # Calculate volume and surface area
        volume = float(mesh.volume)
        surface_area = float(mesh.area)
        
        # Get center of mass
        center_of_mass = mesh.center_mass.tolist()
        
        # Count faces and edges
        face_count = len(mesh.faces)
        edge_count = len(mesh.edges)
        
        return {
            "volume": volume,
            "surface_area": surface_area,
            "dimensions": dimensions,
            "bounding_box": {
                "min": bounds[0].tolist(),
                "max": bounds[1].tolist()
            },
            "center_of_mass": center_of_mass,
            "face_count": face_count,
            "edge_count": edge_count
        }
        
    except Exception as e:
        raise Exception(f"Error analyzing STEP file: {str(e)}") 