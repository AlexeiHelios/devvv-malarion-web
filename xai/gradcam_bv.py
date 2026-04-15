"""
MALARION — Grad-CAM for BinaryValidator (ResNet18 layer4).
Copied verbatim from notebook Part A of Grad-CAM XAI Cell 2.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from inference.bv_infer import INFER_TRANSFORM


class GradCAM_BV:
    """
    Grad-CAM on BinaryValidator ResNet18 — targets layer4.

    Usage:
        gcam = GradCAM_BV(bv_model, device)
        cam, prob = gcam.generate(crop_bgr)
        overlay   = gcam.overlay(crop_bgr, cam)
    """

    def __init__(self, model, device):
        self.model  = model
        self.device = device
        self._feat  = None
        self._grad  = None
        self._hook_handles = []

    def _register(self):
        self._remove()
        layer = self.model.backbone.layer4

        def fwd_hook(m, inp, out):
            self._feat = out.detach()

        def bwd_hook(m, gin, gout):
            self._grad = gout[0].detach()

        self._hook_handles.append(layer.register_forward_hook(fwd_hook))
        self._hook_handles.append(layer.register_full_backward_hook(bwd_hook))

    def _remove(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def generate(self, crop_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Returns (cam_normalised [H,W] float32 0-1, bv_probability float).
        Exact Grad-CAM computation from notebook.
        """
        self._register()
        self.model.eval()

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor   = INFER_TRANSFORM(Image.fromarray(crop_rgb)).unsqueeze(0).to(self.device)
        tensor.requires_grad_(False)

        # Forward with gradient tracking on input clone
        tensor_in = tensor.clone().requires_grad_(True)
        logit     = self.model(tensor_in)
        prob      = torch.sigmoid(logit).item()

        # Backward w.r.t. positive parasite class
        self.model.zero_grad()
        logit.backward()

        # Grad-CAM computation (verbatim from notebook)
        grads   = self._grad                                  # [1, C, H, W]
        feats   = self._feat                                  # [1, C, H, W]
        weights = grads.mean(dim=(2, 3), keepdim=True)       # [1, C, 1, 1]
        cam     = (weights * feats).sum(dim=1).squeeze(0)    # [H, W]
        cam     = F.relu(cam)
        cam     = cam.cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        self._remove()
        return cam, prob

    def overlay(self, crop_bgr: np.ndarray,
                cam: np.ndarray,
                alpha: float = 0.45) -> np.ndarray:
        """Returns BGR overlay image (JET colourmap at alpha=0.45)."""
        H, W        = crop_bgr.shape[:2]
        cam_resized = cv2.resize(cam, (W, H))
        heatmap     = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(crop_bgr, 1 - alpha, heatmap, alpha, 0)
