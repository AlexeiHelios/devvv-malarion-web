"""
MALARION — Binary Validator inference.
bv_predict() mirrors the notebook's function exactly:
  - Resize crop to 224×224
  - ImageNet normalisation
  - Sigmoid threshold = BV_THRESH (0.60)
"""
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from config import BV_THRESH, BV_RESIZE, BV_MEAN, BV_STD

# ── Inference transform (MUST match BV training) ──────────────────────
INFER_TRANSFORM = T.Compose([
    T.Resize(BV_RESIZE),
    T.ToTensor(),
    T.Normalize(mean=BV_MEAN, std=BV_STD),
])


def bv_predict(bv_model,
               crop_bgr: np.ndarray,
               device:   torch.device,
               threshold: float = BV_THRESH) -> tuple[bool, float]:
    """
    Run BinaryValidator on a single BGR crop.

    Args:
        bv_model:  BinaryValidator (already on device, eval mode)
        crop_bgr:  np.ndarray  H×W×3 BGR
        device:    torch.device
        threshold: sigmoid decision boundary (default BV_THRESH=0.60)

    Returns:
        (is_parasite: bool, probability: float)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False, 0.0

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor   = INFER_TRANSFORM(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(bv_model(tensor)).item()

    return prob >= threshold, prob
