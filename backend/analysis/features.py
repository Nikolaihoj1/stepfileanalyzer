import trimesh
import numpy as np
from typing import Dict, Any, List
from scipy.spatial import distance

def analyze_features(step_file_path: str) -> Dict[str, Any]:
    """
    Analyze manufacturing features in a STEP file using trimesh.
    
    Args:
        step_file_path (str): Path to the STEP file
        
    Returns:
        dict: Dictionary containing feature analysis
    """
    try:
        # Import the STEP file
        mesh = trimesh.load(step_file_path)
        
        # Initialize feature lists
        holes = []
        pockets = []
        small_radii = []
        
        # Analyze mesh features
        # Find circular features by analyzing face normals and edge loops
        for face_id, face in enumerate(mesh.faces):
            vertices = mesh.vertices[mesh.faces[face_id]]
            normal = mesh.face_normals[face_id]
            
            # Calculate approximate radius if the face is circular
            if len(vertices) > 2:
                center = vertices.mean(axis=0)
                radii = distance.cdist([center], vertices)[0]
                avg_radius = float(np.mean(radii))
                
                if np.std(radii) < 0.1:  # If radii are similar, might be circular
                    if avg_radius < 1.0:  # Small radius threshold (1mm)
                        small_radii.append(avg_radius)
                    else:
                        holes.append({
                            "radius": avg_radius,
                            "axis": normal.tolist(),
                            "location": center.tolist()
                        })
            
            # Detect potential pockets by analyzing planar faces
            # Simple detection: faces with significant area and consistent normal
            face_area = trimesh.triangles.area([vertices])[0]
            if face_area > 10.0:  # Arbitrary threshold for pocket detection
                pockets.append({
                    "normal": normal.tolist(),
                    "location": center.tolist(),
                    "area": float(face_area)
                })
        
        # Estimate required axes
        axis_count = estimate_required_axes(holes, pockets)
        
        return {
            "holes": {
                "count": len(holes),
                "details": holes
            },
            "pockets": {
                "count": len(pockets),
                "details": pockets
            },
            "small_radii": {
                "count": len(small_radii),
                "min_radius": min(small_radii) if small_radii else None,
                "details": small_radii
            },
            "estimated_axes": axis_count
        }
        
    except Exception as e:
        raise Exception(f"Error analyzing features: {str(e)}")

def estimate_required_axes(holes: List[Dict[str, Any]], pockets: List[Dict[str, Any]]) -> int:
    """
    Estimate the number of axes required for machining.
    
    Args:
        holes (list): List of hole features
        pockets (list): List of pocket features
        
    Returns:
        int: Estimated number of required axes (3, 4, or 5)
    """
    unique_directions = set()
    
    # Analyze hole directions
    for hole in holes:
        direction = tuple(hole["axis"])
        unique_directions.add(direction)
    
    # Analyze pocket normals
    for pocket in pockets:
        direction = tuple(pocket["normal"])
        unique_directions.add(direction)
    
    # Decision logic for required axes
    if len(unique_directions) <= 1:
        return 3  # Only vertical features
    elif len(unique_directions) <= 2:
        return 4  # Features in two directions
    else:
        return 5  # Features in multiple directions 