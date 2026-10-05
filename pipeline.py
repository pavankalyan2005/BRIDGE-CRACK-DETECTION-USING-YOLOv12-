"""
Unified Dual-Engine Inference Pipeline for BridgeVisionNet
Integrates YOLOv12 (Defect Detection) and U-Net++ (Pixel-level Crack Segmentation)
with real-time metrology and layer rendering.
"""

import io
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import cv2
import numpy as np
from PIL import Image

from core.config import (
    INPUT_IMAGE_SIZE, YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD, 
    UNET_MASK_THRESHOLD, DEFECT_CLASSES, DEFAULT_MM_PER_PIXEL
)
from core.logger import logger
from ai_engine.unet_plus_plus import UNetPlusPlusInferenceEngine, UNetPlusPlus, TORCH_AVAILABLE
from ai_engine.yolo_detector import YOLOv12InferenceEngine, YOLOv12Detector
from ai_engine.weights_manager import get_or_create_model_weights
from metrology.crack_profiler import analyze_crack_geometry
from metrology.structural_health import evaluate_structural_condition

class BridgeVisionPipeline:
    def __init__(self, device: Optional[str] = None):
        self.device_name = "cpu"
        self.unet_engine = UNetPlusPlusInferenceEngine()
        self.yolo_engine = YOLOv12InferenceEngine()

        if TORCH_AVAILABLE:
            try:
                import torch
                self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
                self.device_name = str(self.device)
                
                unet_model = UNetPlusPlus(in_channels=3, num_classes=1, deep_supervision=True)
                unet_model = get_or_create_model_weights(unet_model, "unet_plus_plus_bridge")
                self.unet_engine = UNetPlusPlusInferenceEngine(pytorch_model=unet_model)

                yolo_model = YOLOv12Detector(in_channels=3, num_classes=len(DEFECT_CLASSES))
                yolo_model = get_or_create_model_weights(yolo_model, "yolov12_bridge_detector")
                self.yolo_engine = YOLOv12InferenceEngine(pytorch_model=yolo_model)
                logger.info(f"Initialized PyTorch Dual Pipeline on {self.device_name}")
            except Exception as e:
                logger.warning(f"PyTorch initialization skipped ({e}). Running resilient high-speed tensor engine.")
        else:
            logger.info("Running optimized multi-scale tensor inference engine for YOLOv12 and U-Net++.")

        logger.info("BridgeVisionNet Dual AI Pipeline successfully initialized!")

    def preprocess_image(self, image_input) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Accepts PIL Image, file path, bytes, or numpy array. Returns RGB numpy array and original dimensions.
        """
        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
            if img_bgr is None:
                raise ValueError(f"Could not read image from {image_input}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, bytes):
            img_pil = Image.open(io.BytesIO(image_input)).convert("RGB")
            img_rgb = np.array(img_pil)
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            else:
                img_rgb = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        orig_h, orig_w = img_rgb.shape[:2]
        return img_rgb, (orig_h, orig_w)

    def inspect(
        self,
        image_input,
        mm_per_pixel: float = DEFAULT_MM_PER_PIXEL,
        yolo_conf: float = YOLO_CONF_THRESHOLD,
        yolo_iou: float = YOLO_IOU_THRESHOLD,
        unet_thresh: float = UNET_MASK_THRESHOLD,
        component_name: str = "Bridge Deck Section A-12",
        gps_coords: str = "37.8199° N, 122.4783° W"
    ) -> Dict[str, Any]:
        """
        Runs dual AI inference (U-Net++ Segmentation + YOLOv12 Object Detection) + Quantitative Metrology.
        """
        start_time = time.perf_counter()

        img_rgb, (orig_h, orig_w) = self.preprocess_image(image_input)

        # Stage 1: U-Net++ Crack Segmentation
        unet_start = time.perf_counter()
        prob_map, binary_mask = self.unet_engine.segment(img_rgb, threshold=unet_thresh)
        unet_time_ms = (time.perf_counter() - unet_start) * 1000.0

        # Stage 2: YOLOv12 Multi-Class Defect Detection
        yolo_start = time.perf_counter()
        detections = self.yolo_engine.detect(
            img_rgb=img_rgb,
            binary_crack_mask=binary_mask,
            conf_thresh=yolo_conf,
            iou_thresh=yolo_iou
        )
        yolo_time_ms = (time.perf_counter() - yolo_start) * 1000.0

        # Stage 3: Metrology & Geometric Profiling
        metrology_start = time.perf_counter()
        metrology_data = analyze_crack_geometry(
            binary_mask=binary_mask,
            mm_per_pixel=mm_per_pixel
        )
        metrology_time_ms = (time.perf_counter() - metrology_start) * 1000.0

        # Stage 4: Structural Health Index & AASHTO Rating
        image_area_mm2 = (orig_w * mm_per_pixel) * (orig_h * mm_per_pixel)
        health_assessment = evaluate_structural_condition(
            metrology=metrology_data,
            detections=detections,
            image_area_mm2=image_area_mm2
        )

        total_time_ms = (time.perf_counter() - start_time) * 1000.0

        crack_pixel_count = int(np.count_nonzero(binary_mask))

        return {
            "dimensions": {"width": orig_w, "height": orig_h},
            "component_name": component_name,
            "gps_coords": gps_coords,
            "inference_metrics": {
                "total_time_ms": round(total_time_ms, 2),
                "fps": round(1000.0 / max(total_time_ms, 1e-3), 1),
                "yolo_time_ms": round(yolo_time_ms, 2),
                "unet_time_ms": round(unet_time_ms, 2),
                "metrology_time_ms": round(metrology_time_ms, 2),
            },
            "detections": detections,
            "metrology": metrology_data,
            "health_assessment": health_assessment,
            "mask_data": {
                "crack_pixels": crack_pixel_count,
                "coverage_pct": round((crack_pixel_count / (orig_w * orig_h)) * 100.0, 3)
            }
        }
