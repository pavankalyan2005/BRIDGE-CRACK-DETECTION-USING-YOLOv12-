import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
SAMPLE_DIR = STATIC_DIR / "samples"
UPLOADS_DIR = BASE_DIR / "uploads"
REPORTS_DIR = STATIC_DIR / "reports"
WEIGHTS_DIR = BASE_DIR / "weights"

# Ensure runtime directories exist
for directory in [STATIC_DIR, TEMPLATES_DIR, SAMPLE_DIR, UPLOADS_DIR, REPORTS_DIR, WEIGHTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Defect Classification Config for YOLOv12
DEFECT_CLASSES = {
    0: {"name": "Longitudinal Crack", "color": "#FF3366", "severity_weight": 1.2, "code": "LC-01"},
    1: {"name": "Transverse Crack", "color": "#FF9900", "severity_weight": 1.5, "code": "TC-02"},
    2: {"name": "Alligator / Fatigue Crack", "color": "#FF0000", "severity_weight": 2.0, "code": "AC-03"},
    3: {"name": "Spalling / Cavity", "color": "#9933FF", "severity_weight": 2.2, "code": "SP-04"},
    4: {"name": "Exposed Rebar & Corrosion", "color": "#CC0033", "severity_weight": 2.5, "code": "RC-05"},
    5: {"name": "Efflorescence / Leaching", "color": "#00CCCC", "severity_weight": 0.8, "code": "EF-06"},
    6: {"name": "Expansion Joint Damage", "color": "#FFCC00", "severity_weight": 1.8, "code": "JD-07"},
}

# Metrology Calibration Defaults
DEFAULT_MM_PER_PIXEL = 0.25  # 1 pixel = 0.25 mm at standard drone standoff distance (1.5m)
CRACK_WIDTH_THRESHOLDS = {
    "hairline": 0.3,    # < 0.3mm: Hairline / Low Risk
    "moderate": 1.0,    # 0.3 - 1.0mm: Moderate Risk (Water ingress danger)
    "severe": 2.5,      # 1.0 - 2.5mm: Severe Structural Concern
    "critical": 2.5     # > 2.5mm: Critical Failure Risk
}

# AASHTO / FHWA Element-Level Condition States
CONDITION_STATES = {
    "CS1": {"rating": "Good", "color": "#00E676", "desc": "Minimal or no defects. Normal periodic maintenance."},
    "CS2": {"rating": "Fair", "color": "#FFD600", "desc": "Minor to moderate cracking/spalling. Scheduled preventative repair."},
    "CS3": {"rating": "Poor", "color": "#FF6D00", "desc": "Significant defects affecting structural integrity. Engineering repair required."},
    "CS4": {"rating": "Severe", "color": "#D50000", "desc": "Critical structural hazard. Immediate intervention / load restriction needed."}
}

# Inference Hyperparameters
YOLO_CONF_THRESHOLD = 0.35
YOLO_IOU_THRESHOLD = 0.45
UNET_MASK_THRESHOLD = 0.40
INPUT_IMAGE_SIZE = 512
