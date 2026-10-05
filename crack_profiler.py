"""
Civil Engineering Metrology & Crack Profiling Engine
Performs Skeletonization, Euclidean Distance Transform (EDT) width measurement,
branching analysis, and orientation estimation.
"""

from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from core.config import DEFAULT_MM_PER_PIXEL, CRACK_WIDTH_THRESHOLDS

def analyze_crack_geometry(
    binary_mask: np.ndarray,
    mm_per_pixel: float = DEFAULT_MM_PER_PIXEL
) -> Dict[str, Any]:
    """
    Analyzes binary crack mask and computes metric structural parameters.
    """
    if binary_mask is None or np.count_nonzero(binary_mask) == 0:
        return {
            "has_cracks": False,
            "max_width_mm": 0.0,
            "avg_width_mm": 0.0,
            "min_width_mm": 0.0,
            "p95_width_mm": 0.0,
            "total_length_mm": 0.0,
            "surface_area_mm2": 0.0,
            "branch_count": 0,
            "junction_count": 0,
            "orientation_deg": 0.0,
            "orientation_type": "None",
            "width_distribution": [],
            "severity_category": "None",
            "skeleton_points_count": 0
        }

    # Ensure binary uint8 0/1
    mask_bool = binary_mask > 0
    
    # 1. Morphological Skeletonization (Medial Axis)
    skeleton = skeletonize(mask_bool)
    skeleton_uint8 = skeleton.astype(np.uint8) * 255

    # 2. Euclidean Distance Transform (EDT)
    edt_map = distance_transform_edt(mask_bool)
    
    # Extract widths along skeleton points
    # Local crack width is 2 * EDT value
    skeleton_coords = np.where(skeleton)
    skeleton_edt_values = edt_map[skeleton_coords]

    if len(skeleton_edt_values) > 0:
        widths_px = skeleton_edt_values * 2.0
        widths_mm = widths_px * mm_per_pixel

        max_width_mm = float(np.max(widths_mm))
        avg_width_mm = float(np.mean(widths_mm))
        min_width_mm = float(np.min(widths_mm))
        p95_width_mm = float(np.percentile(widths_mm, 95))
    else:
        max_width_mm = avg_width_mm = min_width_mm = p95_width_mm = 0.0
        widths_mm = np.array([0.0])

    # 3. Total Length Calculation (Geodesic pixel graph distance)
    # Count orthogonal vs diagonal steps in skeleton
    skel_indices = list(zip(skeleton_coords[0], skeleton_coords[1]))
    skel_set = set(skel_indices)
    
    total_length_px = 0.0
    for y, x in skel_indices:
        # Check 8 neighbors (only right/down to avoid double counting)
        for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            ny, nx = y + dy, x + dx
            if (ny, nx) in skel_set:
                if dy != 0 and dx != 0:
                    total_length_px += 1.4142  # diagonal
                else:
                    total_length_px += 1.0     # orthogonal

    total_length_mm = float(total_length_px * mm_per_pixel)

    # 4. Surface Area
    total_crack_pixels = int(np.count_nonzero(mask_bool))
    surface_area_mm2 = float(total_crack_pixels * (mm_per_pixel ** 2))

    # 5. Junction & Branch Detection (3x3 neighbor convolution)
    kernel_neighbors = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton_uint8 // 255, -1, kernel_neighbors)
    
    # Skeleton points with >= 3 neighbors are branching junctions
    junctions = (neighbor_count >= 3) & skeleton
    junction_count = int(np.count_nonzero(junctions))
    
    # End points (1 neighbor)
    end_points = (neighbor_count == 1) & skeleton
    branch_count = max(1, int(np.count_nonzero(end_points) // 2 + junction_count))

    # 6. Orientation & Angle (Principal Axis via Moments)
    moments = cv2.moments(binary_mask)
    if moments["mu20"] + moments["mu02"] > 0:
        # Calculate angle of least inertia / major crack propagation axis
        angle_rad = 0.5 * np.arctan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"])
        orientation_deg = float(np.degrees(angle_rad))
    else:
        orientation_deg = 0.0

    # Classify orientation
    abs_angle = abs(orientation_deg)
    if abs_angle <= 25 or abs_angle >= 155:
        orientation_type = "Longitudinal (Parallel to Traffic / Span)"
    elif 65 <= abs_angle <= 115:
        orientation_type = "Transverse (Perpendicular to Span / Flexure)"
    else:
        orientation_type = "Diagonal (Shear Stress / Torsion)"

    # 7. Severity Category based on Max Crack Width (Civil Engineering standard)
    if max_width_mm < CRACK_WIDTH_THRESHOLDS["hairline"]:
        severity_category = "Hairline (Insignificant - Monitor)"
    elif max_width_mm < CRACK_WIDTH_THRESHOLDS["moderate"]:
        severity_category = "Moderate (Water Seepage / Carbonation Risk)"
    elif max_width_mm < CRACK_WIDTH_THRESHOLDS["severe"]:
        severity_category = "Severe (Rebar Corrosion & Spalling Hazard)"
    else:
        severity_category = "Critical (Structural Load Re-evaluation Required)"

    # 8. Width distribution histogram for interactive chart
    hist, bin_edges = np.histogram(widths_mm, bins=8, range=(0, max(max_width_mm * 1.1, 0.5)))
    width_distribution = [
        {"bin": f"{round(bin_edges[i], 2)}-{round(bin_edges[i+1], 2)} mm", "count": int(hist[i])}
        for i in range(len(hist))
    ]

    return {
        "has_cracks": True,
        "max_width_mm": round(max_width_mm, 2),
        "avg_width_mm": round(avg_width_mm, 2),
        "min_width_mm": round(min_width_mm, 2),
        "p95_width_mm": round(p95_width_mm, 2),
        "total_length_mm": round(total_length_mm, 1),
        "surface_area_mm2": round(surface_area_mm2, 1),
        "branch_count": branch_count,
        "junction_count": junction_count,
        "orientation_deg": round(orientation_deg, 1),
        "orientation_type": orientation_type,
        "severity_category": severity_category,
        "width_distribution": width_distribution,
        "skeleton_points_count": len(skeleton_edt_values)
    }

def measure_caliper_width_at_point(
    binary_mask: np.ndarray,
    point_xy: Tuple[int, int],
    mm_per_pixel: float = DEFAULT_MM_PER_PIXEL
) -> Dict[str, Any]:
    """
    Measures exact local normal width when user clicks on a crack point in the UI.
    """
    x, y = int(point_xy[0]), int(point_xy[1])
    h, w = binary_mask.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return {"valid": False, "error": "Point out of image bounds"}

    # Compute EDT
    mask_bool = binary_mask > 0
    if not mask_bool[y, x]:
        # Search in 5px neighborhood
        y_min, y_max = max(0, y - 5), min(h, y + 6)
        x_min, x_max = max(0, x - 5), min(w, x + 6)
        patch = mask_bool[y_min:y_max, x_min:x_max]
        if not np.any(patch):
            return {"valid": False, "error": "No crack detected at clicked location"}
        py, px = np.where(patch)
        y, x = y_min + py[0], x_min + px[0]

    edt_map = distance_transform_edt(mask_bool)
    local_radius_px = edt_map[y, x]
    local_width_px = local_radius_px * 2.0
    local_width_mm = local_width_px * mm_per_pixel

    return {
        "valid": True,
        "point": [x, y],
        "width_px": round(float(local_width_px), 2),
        "width_mm": round(float(local_width_mm), 2),
        "scale_mm_per_pixel": mm_per_pixel
    }
