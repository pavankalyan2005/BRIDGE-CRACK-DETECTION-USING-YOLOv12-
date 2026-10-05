"""
Weights Manager for BridgeVisionNet
Manages neural weights when PyTorch is available, with safe fallback handling.
"""

from pathlib import Path
from core.config import WEIGHTS_DIR
from core.logger import logger

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except (ImportError, OSError, Exception):
    torch = None
    nn = None
    TORCH_AVAILABLE = False

def init_weights_with_priors(model, model_type: str = "unet"):
    if not TORCH_AVAILABLE or model is None:
        return
    try:
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    except Exception as e:
        logger.warning(f"Weight initialization warning: {e}")

def get_or_create_model_weights(model, model_name: str):
    if not TORCH_AVAILABLE or model is None:
        return model
    weights_path = WEIGHTS_DIR / f"{model_name}.pth"
    if weights_path.exists():
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
            logger.info(f"Loaded neural weights from {weights_path}")
            return model
        except Exception as e:
            logger.warning(f"Could not load weights: {e}")

    init_weights_with_priors(model)
    try:
        torch.save(model.state_dict(), weights_path)
    except Exception:
        pass

    return model
