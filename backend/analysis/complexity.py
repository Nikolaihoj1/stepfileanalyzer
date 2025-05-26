import numpy as np

def calculate_complexity(geometry_data: dict, features_data: dict) -> dict:
    """
    Calculate the overall machining complexity of the part.
    
    Args:
        geometry_data (dict): Geometric properties from geometry analysis
        features_data (dict): Feature information from feature analysis
        
    Returns:
        dict: Complexity metrics and scores
    """
    # Initialize complexity factors
    volume_score = calculate_volume_complexity(geometry_data)
    feature_score = calculate_feature_complexity(features_data)
    axis_score = calculate_axis_complexity(features_data["estimated_axes"])
    precision_score = calculate_precision_complexity(features_data)
    
    # Calculate overall complexity score (0-100)
    overall_score = np.mean([
        volume_score,
        feature_score,
        axis_score,
        precision_score
    ])
    
    # Estimate machining time factors
    time_factors = estimate_machining_time_factors(
        geometry_data,
        features_data,
        overall_score
    )
    
    return {
        "overall_score": round(overall_score, 2),
        "component_scores": {
            "volume_complexity": round(volume_score, 2),
            "feature_complexity": round(feature_score, 2),
            "axis_complexity": round(axis_score, 2),
            "precision_complexity": round(precision_score, 2)
        },
        "machining_factors": time_factors,
        "complexity_level": get_complexity_level(overall_score)
    }

def calculate_volume_complexity(geometry_data: dict) -> float:
    """Calculate complexity based on volume and surface area ratio."""
    volume = geometry_data["volume"]
    surface_area = geometry_data["surface_area"]
    
    # Calculate surface area to volume ratio (normalized)
    if volume > 0:
        ratio = (surface_area ** (2/3)) / (volume ** (1/3))
        # Normalize to 0-100 scale (empirical values)
        return min(100, max(0, (ratio - 4.8) * 20))
    return 50  # Default value if volume is 0

def calculate_feature_complexity(features_data: dict) -> float:
    """Calculate complexity based on feature count and types."""
    hole_count = features_data["holes"]["count"]
    pocket_count = features_data["pockets"]["count"]
    small_radii_count = features_data["small_radii"]["count"]
    
    # Weight different features
    feature_score = (
        hole_count * 5 +
        pocket_count * 10 +
        small_radii_count * 15
    )
    
    # Normalize to 0-100 scale
    return min(100, feature_score)

def calculate_axis_complexity(axis_count: int) -> float:
    """Calculate complexity based on required axes."""
    axis_scores = {
        3: 20,    # 3-axis machining
        4: 60,    # 4-axis machining
        5: 100    # 5-axis machining
    }
    return axis_scores.get(axis_count, 50)

def calculate_precision_complexity(features_data: dict) -> float:
    """Calculate complexity based on required precision."""
    small_radii = features_data["small_radii"]
    min_radius = small_radii.get("min_radius", float('inf'))
    
    if min_radius == float('inf'):
        return 0
    
    # Score based on minimum radius (smaller radius = higher complexity)
    return min(100, max(0, (1 / min_radius) * 20))

def estimate_machining_time_factors(geometry_data: dict,
                                  features_data: dict,
                                  complexity_score: float) -> dict:
    """Estimate factors affecting machining time."""
    # Basic time estimation factors
    setup_time = 15  # Base setup time in minutes
    
    # Adjust setup time based on axes
    if features_data["estimated_axes"] == 4:
        setup_time *= 1.5
    elif features_data["estimated_axes"] == 5:
        setup_time *= 2
    
    # Estimate material removal time (simplified)
    volume = geometry_data["volume"]
    removal_rate = 1000  # mm³/min (basic assumption)
    
    # Adjust removal rate based on complexity
    removal_rate *= (1 - (complexity_score / 200))  # Higher complexity = slower removal
    
    machining_time = volume / removal_rate if removal_rate > 0 else 0
    
    return {
        "estimated_setup_time_minutes": round(setup_time, 1),
        "estimated_machining_time_minutes": round(machining_time, 1),
        "total_estimated_time_minutes": round(setup_time + machining_time, 1)
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