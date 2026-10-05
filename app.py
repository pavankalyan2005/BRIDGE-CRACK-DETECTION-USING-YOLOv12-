"""
BridgeVisionNet: AI-Based Bridge Crack Detection and Structural Health Monitoring
FastAPI Application Entrypoint
"""

import os
import io
import time
import uuid
import base64
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.config import (
    STATIC_DIR, TEMPLATES_DIR, SAMPLE_DIR, UPLOADS_DIR, REPORTS_DIR,
    DEFECT_CLASSES, DEFAULT_MM_PER_PIXEL, YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD, UNET_MASK_THRESHOLD
)
from core.logger import logger
from ai_engine.pipeline import BridgeVisionPipeline
from metrology.crack_profiler import measure_caliper_width_at_point
from telemetry.iot_simulator import telemetry_simulator
from reporting.pdf_generator import generate_inspection_pdf
from reporting.csv_exporter import export_inspection_csv

app = FastAPI(
    title="BridgeVisionNet",
    description="AI-Based Bridge Crack Detection and Structural Health Monitoring Using YOLOv12 and U-Net++",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialize AI Pipeline
pipeline = BridgeVisionPipeline()

# In-memory inspection cache for interactive tools (caliper, reports)
INSPECTION_CACHE = {}

def encode_image_base64(img_bgr: np.ndarray, quality: int = 90) -> str:
    """Encodes BGR numpy image to JPEG base64 string."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', img_bgr, encode_param)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

def create_visual_overlays(
    img_rgb: np.ndarray,
    detections: List[dict],
    binary_mask: np.ndarray,
    mm_per_pixel: float
) -> dict:
    """
    Generates distinct visual layers for the interactive UI layer switcher:
    - raw
    - yolo_boxes
    - unet_mask
    - skeleton_layer
    - width_heatmap
    - composite_augmented
    """
    h, w = img_rgb.shape[:2]
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # 1. YOLO Layer (Bounding Boxes and Badges)
    yolo_bgr = img_bgr.copy()
    for d in detections:
        box = d["box"]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        color_hex = d.get("color", "#FF0000").lstrip("#")
        # Hex to BGR
        r, g, b = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
        color_bgr = (b, g, r)

        # Draw rounded rectangle / box
        cv2.rectangle(yolo_bgr, (x1, y1), (x2, y2), color_bgr, 2)
        
        # Label badge
        label = f"{d.get('class_name', 'Defect')} ({int(d.get('score', 0.0)*100)}%)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(yolo_bgr, (x1, max(0, y1 - 22)), (x1 + tw + 10, y1), color_bgr, -1)
        cv2.putText(yolo_bgr, label, (x1 + 5, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. U-Net++ Segmentation Mask (Glowing Cyan / Red overlay)
    unet_bgr = img_bgr.copy()
    if binary_mask is not None and np.count_nonzero(binary_mask) > 0:
        mask_overlay = np.zeros_like(img_bgr)
        # Paint crack pixels bright cyan-blue (BGR: 255, 200, 0)
        mask_overlay[binary_mask > 0] = [255, 200, 0]
        # Blend
        unet_bgr = cv2.addWeighted(unet_bgr, 0.65, mask_overlay, 0.65, 0)
        # Crack contour highlight
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(unet_bgr, contours, -1, (255, 255, 0), 1)

    # 3. Skeleton Centerline Layer (Yellow Medial Axis)
    from skimage.morphology import skeletonize
    skel_bgr = img_bgr.copy()
    if binary_mask is not None and np.count_nonzero(binary_mask) > 0:
        skel = skeletonize(binary_mask > 0)
        skel_bgr[skel] = [0, 255, 255] # Neon Yellow
        # Dilate slightly for visibility
        skel_dilated = cv2.dilate(skel.astype(np.uint8)*255, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
        skel_bgr[skel_dilated > 0] = [0, 240, 255]

    # 4. Crack Width Heatmap (Color scale: Green -> Yellow -> Orange -> Red)
    from scipy.ndimage import distance_transform_edt
    heatmap_bgr = img_bgr.copy()
    if binary_mask is not None and np.count_nonzero(binary_mask) > 0:
        edt = distance_transform_edt(binary_mask > 0)
        widths_mm = (edt * 2.0) * mm_per_pixel
        # Normalize widths (0.0 to 2.5 mm -> 0 to 255)
        norm_widths = np.clip(widths_mm / 2.5 * 255.0, 0, 255).astype(np.uint8)
        color_map = cv2.applyColorMap(norm_widths, cv2.COLORMAP_JET)
        
        # Apply color map only on crack pixels
        mask_3c = cv2.merge([binary_mask, binary_mask, binary_mask]) > 0
        heatmap_overlay = img_bgr.copy()
        heatmap_overlay[mask_3c] = color_map[mask_3c]
        heatmap_bgr = cv2.addWeighted(img_bgr, 0.4, heatmap_overlay, 0.6, 0)

    # 5. Composite Augmented View (YOLO boxes + U-Net++ Heatmap overlay + Medial Skeleton)
    composite_bgr = heatmap_bgr.copy()
    for d in detections:
        box = d["box"]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        color_hex = d.get("color", "#FF0000").lstrip("#")
        r, g, b = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
        color_bgr = (b, g, r)
        cv2.rectangle(composite_bgr, (x1, y1), (x2, y2), color_bgr, 2)
        label = f"{d.get('class_name', 'Defect')}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(composite_bgr, (x1, max(0, y1 - 20)), (x1 + tw + 8, y1), color_bgr, -1)
        cv2.putText(composite_bgr, label, (x1 + 4, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return {
        "raw": encode_image_base64(img_bgr),
        "yolo": encode_image_base64(yolo_bgr),
        "unet": encode_image_base64(unet_bgr),
        "skeleton": encode_image_base64(skel_bgr),
        "heatmap": encode_image_base64(heatmap_bgr),
        "composite": encode_image_base64(composite_bgr),
    }

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the main BridgeVisionNet UI Dashboard."""
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard template not found.")
    with open(index_file, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/samples")
async def list_sample_images():
    """Returns the list of preset bridge inspection images."""
    samples = [
        {
            "id": "sample_1",
            "name": "Deck Span Longitudinal Crack",
            "category": "Bridge Deck",
            "filename": "sample_1_deck_longitudinal_crack.jpg",
            "component": "Bridge Deck Segment D-04",
            "gps": "37.8199° N, 122.4783° W",
            "description": "Longitudinal flexural crack running along traffic direction with micro-branching."
        },
        {
            "id": "sample_2",
            "name": "Pier Shear & Transverse Crack",
            "category": "Substructure Pier",
            "filename": "sample_2_pier_transverse_crack.jpg",
            "component": "Pier #3 Shear Column",
            "gps": "37.8205° N, 122.4791° W",
            "description": "Transverse shear fracture across load-bearing pier with active efflorescence calcite leaching."
        },
        {
            "id": "sample_3",
            "name": "Girder Spalling & Corroded Rebar",
            "category": "Superstructure Girder",
            "filename": "sample_3_girder_spall_rebar.jpg",
            "component": "Prestressed Box Girder G-02",
            "gps": "37.8211° N, 122.4798° W",
            "description": "Severe concrete spalling crater with exposed corroded reinforcement steel and delamination."
        },
        {
            "id": "sample_4",
            "name": "Fatigue Alligator Map Cracking",
            "category": "Asphalt / Deck",
            "filename": "sample_4_alligator_fatigue_cracking.jpg",
            "component": "Approach Slab North",
            "gps": "37.8190° N, 122.4775° W",
            "description": "Interconnected multi-directional alligator fatigue network caused by heavy axle repetitions."
        },
        {
            "id": "sample_5",
            "name": "Expansion Joint Breakdown & Spall",
            "category": "Expansion Joint",
            "filename": "sample_5_expansion_joint_damage.jpg",
            "component": "Abutment Expansion Joint EJ-1",
            "gps": "37.8220° N, 122.4810° W",
            "description": "Damaged elastomeric seal with crushed concrete nosing edge and water percolation."
        },
        {
            "id": "sample_6",
            "name": "Healthy Benchmark Concrete Deck",
            "category": "Baseline Benchmark",
            "filename": "sample_6_healthy_benchmark_deck.jpg",
            "component": "New Span Cured Slab C-01",
            "gps": "37.8185° N, 122.4768° W",
            "description": "Unblemished baseline concrete surface meeting AASHTO CS1 Condition State standards."
        }
    ]
    return {"samples": samples}

@app.post("/api/inspect")
async def run_inspection(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    mm_per_pixel: float = Form(DEFAULT_MM_PER_PIXEL),
    yolo_conf: float = Form(YOLO_CONF_THRESHOLD),
    yolo_iou: float = Form(YOLO_IOU_THRESHOLD),
    unet_thresh: float = Form(UNET_MASK_THRESHOLD),
    component_name: str = Form("Bridge Span Section A-1"),
    gps_coords: str = Form("37.8199° N, 122.4783° W")
):
    """
    Executes end-to-end dual AI inference + metrology on an uploaded file or preset sample.
    """
    try:
        if file is not None and file.filename:
            image_bytes = await file.read()
            img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_rgb = np.array(img_pil)
        elif sample_id:
            sample_file_map = {
                "sample_1": "sample_1_deck_longitudinal_crack.jpg",
                "sample_2": "sample_2_pier_transverse_crack.jpg",
                "sample_3": "sample_3_girder_spall_rebar.jpg",
                "sample_4": "sample_4_alligator_fatigue_cracking.jpg",
                "sample_5": "sample_5_expansion_joint_damage.jpg",
                "sample_6": "sample_6_healthy_benchmark_deck.jpg"
            }
            sample_filename = sample_file_map.get(sample_id, "sample_1_deck_longitudinal_crack.jpg")
            sample_path = SAMPLE_DIR / sample_filename
            if not sample_path.exists():
                raise HTTPException(status_code=404, detail=f"Sample file {sample_filename} not found.")
            img_bgr = cv2.imread(str(sample_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            # Default to sample 1
            sample_path = SAMPLE_DIR / "sample_1_deck_longitudinal_crack.jpg"
            img_bgr = cv2.imread(str(sample_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Run Dual Pipeline
        inspection_result = pipeline.inspect(
            image_input=img_rgb,
            mm_per_pixel=mm_per_pixel,
            yolo_conf=yolo_conf,
            yolo_iou=yolo_iou,
            unet_thresh=unet_thresh,
            component_name=component_name,
            gps_coords=gps_coords
        )

        # Extract binary mask for visual layer generation
        orig_h, orig_w = img_rgb.shape[:2]
        _, binary_mask = pipeline.unet_engine.segment(img_rgb, threshold=unet_thresh)

        # Create multi-layer visual overlays
        layer_overlays = create_visual_overlays(
            img_rgb=img_rgb,
            detections=inspection_result["detections"],
            binary_mask=binary_mask,
            mm_per_pixel=mm_per_pixel
        )

        # Cache inspection session for interactive tools & reports
        inspection_id = str(uuid.uuid4())[:8]
        INSPECTION_CACHE[inspection_id] = {
            "result": inspection_result,
            "binary_mask": binary_mask,
            "mm_per_pixel": mm_per_pixel,
            "img_rgb": img_rgb,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Correlate live IoT sensor telemetry with defect severity
        shi_val = inspection_result["health_assessment"]["structural_health_index"]
        severity_ratio = (100.0 - shi_val) / 100.0
        telemetry_data = telemetry_simulator.generate_telemetry_frame(crack_severity_ratio=severity_ratio)

        return {
            "inspection_id": inspection_id,
            "metrics": inspection_result["inference_metrics"],
            "detections": inspection_result["detections"],
            "metrology": inspection_result["metrology"],
            "health_assessment": inspection_result["health_assessment"],
            "dimensions": inspection_result["dimensions"],
            "layers": layer_overlays,
            "telemetry": telemetry_data,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        logger.error(f"Inspection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inspection failed: {str(e)}")

class CaliperRequest(BaseModel):
    inspection_id: str
    point_x: int
    point_y: int

@app.post("/api/metrology/caliper")
async def measure_caliper(req: CaliperRequest):
    """
    Interactive Digital Caliper: Measures normal crack width at user-clicked pixel coordinate.
    """
    cached = INSPECTION_CACHE.get(req.inspection_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Inspection session expired or not found.")

    caliper_result = measure_caliper_width_at_point(
        binary_mask=cached["binary_mask"],
        point_xy=(req.point_x, req.point_y),
        mm_per_pixel=cached["mm_per_pixel"]
    )
    return caliper_result

@app.get("/api/telemetry/live")
async def get_live_telemetry(severity: float = 0.2):
    """
    Returns real-time IoT bridge sensor telemetry and FFT modal frequency spectrum.
    """
    return telemetry_simulator.generate_telemetry_frame(crack_severity_ratio=severity)

class ReportRequest(BaseModel):
    inspection_id: str
    bridge_name: Optional[str] = "Golden Gate North Viaduct (Pier 4)"
    structure_id: Optional[str] = "BR-CIV-2026-904"

@app.post("/api/report/pdf")
async def generate_pdf(req: ReportRequest):
    """
    Builds printable PDF Structural Health Inspection Certificate.
    """
    cached = INSPECTION_CACHE.get(req.inspection_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Inspection session not found.")

    pdf_path = generate_inspection_pdf(
        inspection_id=req.inspection_id,
        inspection_data=cached["result"],
        bridge_name=req.bridge_name,
        structure_id=req.structure_id
    )

    return FileResponse(
        path=pdf_path,
        filename=f"Bridge_Inspection_Report_{req.inspection_id}.pdf",
        media_type="application/pdf"
    )

@app.post("/api/report/csv")
async def generate_csv(req: ReportRequest):
    """
    Exports CSV measurement table.
    """
    cached = INSPECTION_CACHE.get(req.inspection_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Inspection session not found.")

    csv_path = export_inspection_csv(
        inspection_id=req.inspection_id,
        data=cached["result"]
    )

    return FileResponse(
        path=csv_path,
        filename=f"Inspection_{req.inspection_id}_data.csv",
        media_type="text/csv"
    )

@app.get("/api/system/status")
async def system_status():
    """Returns AI model parameters, execution device, and status."""
    return {
        "status": "ONLINE",
        "device": pipeline.device_name,
        "cuda_available": False,
        "models": {
            "unet_plus_plus": {
                "name": "Nested U-Net++ (Dense Skip & Deep Supervision)",
                "parameters": "9.16M",
                "input_resolution": "512x512",
                "channels": 3,
                "attention": "Squeeze-and-Excitation (SE)"
            },
            "yolo_v12": {
                "name": "YOLOv12 Multi-Scale Defect Detector",
                "parameters": "14.28M",
                "classes": len(DEFECT_CLASSES),
                "attention": "Area-Attention Backbone"
            }
        },
        "defect_classes": DEFECT_CLASSES
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
