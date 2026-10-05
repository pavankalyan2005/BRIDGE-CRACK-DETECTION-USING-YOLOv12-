"""
Structural Health Evaluation & AASHTO Condition State Engine
Calculates Structural Health Index (SHI), AASHTO Element-Level rating, and repair protocols.
"""

from typing import Dict, Any, List
from core.config import CONDITION_STATES

def evaluate_structural_condition(
    metrology: Dict[str, Any],
    detections: List[Dict[str, Any]],
    image_area_mm2: float
) -> Dict[str, Any]:
    """
    Computes overall Structural Health Index (0-100), AASHTO CS rating, and engineering action plan.
    """
    base_health = 100.0
    penalties = {}

    # 1. Crack Width Penalty (Max 40 points)
    max_w = metrology.get("max_width_mm", 0.0)
    if max_w > 0:
        width_penalty = min(40.0, (max_w / 2.5) * 35.0 + (metrology.get("avg_width_mm", 0.0) / 1.5) * 5.0)
        penalties["crack_width_penalty"] = round(width_penalty, 1)
    else:
        width_penalty = 0.0

    # 2. Defect Density Penalty (Max 25 points)
    surface_area = metrology.get("surface_area_mm2", 0.0)
    if image_area_mm2 > 0 and surface_area > 0:
        density_ratio = surface_area / image_area_mm2
        density_penalty = min(25.0, density_ratio * 400.0)
        penalties["defect_density_penalty"] = round(density_penalty, 1)
    else:
        density_penalty = 0.0

    # 3. Defect Classification Severity (Max 35 points from YOLO)
    yolo_penalty = 0.0
    for d in detections:
        weight = d.get("severity_weight", 1.0)
        score = d.get("score", 0.5)
        yolo_penalty += weight * score * 4.5
    yolo_penalty = min(35.0, yolo_penalty)
    penalties["multi_defect_penalty"] = round(yolo_penalty, 1)

    # Calculate final SHI
    total_penalty = width_penalty + density_penalty + yolo_penalty
    shi_score = max(5.0, min(100.0, base_health - total_penalty))
    shi_score = round(shi_score, 1)

    # Determine AASHTO / FHWA Condition State
    if shi_score >= 85.0:
        cs_key = "CS1"
        urgency = "Routine Monitoring (12 - 24 Months)"
    elif shi_score >= 70.0:
        cs_key = "CS2"
        urgency = "Scheduled Maintenance (3 - 6 Months)"
    elif shi_score >= 50.0:
        cs_key = "CS3"
        urgency = "Priority Engineering Intervention (30 Days)"
    else:
        cs_key = "CS4"
        urgency = "IMMEDIATE STRUCTURAL SAFETY ACTION (< 48 Hours)"

    cs_info = CONDITION_STATES[cs_key]

    # Generate tailored engineering recommendations
    recommendations = []
    
    # Check defect types
    class_names = [d.get("class_name", "") for d in detections]
    
    if max_w > 1.5 or "Spalling / Cavity" in class_names:
        recommendations.append("Apply high-modulus polymer modified structural repair mortar and epoxy injection.")
    elif max_w > 0.3:
        recommendations.append("Perform low-viscosity pressure epoxy resin injection to prevent water ingress and carbonation.")
    elif max_w > 0:
        recommendations.append("Apply breathable silane/siloxane surface water-repellent coating.")

    if any("Rebar" in name for name in class_names):
        recommendations.append("Execute concrete chipping, grit blast corroded rebar to Sa 2.5, apply zinc-rich sacrificial primer.")

    if "Alligator / Fatigue Crack" in class_names or metrology.get("branch_count", 0) > 3:
        recommendations.append("Install Carbon-Fiber Reinforced Polymer (CFRP) laminate wraps for flexural/shear retrofitting.")

    if "Expansion Joint Damage" in class_names:
        recommendations.append("Clean joint channel, replace elastomeric seal gland, and inspect bearing seat level.")

    if not recommendations:
        recommendations.append("Structure in sound state. Continue regular drone and non-destructive visual monitoring.")

    return {
        "structural_health_index": shi_score,
        "condition_state": cs_key,
        "rating_label": cs_info["rating"],
        "rating_color": cs_info["color"],
        "description": cs_info["desc"],
        "urgency": urgency,
        "penalties": penalties,
        "recommendations": recommendations,
        "load_bearing_risk": "Low" if shi_score >= 70 else ("Moderate" if shi_score >= 50 else "High / Critical")
    }
