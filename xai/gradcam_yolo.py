"""
MALARION — Grad-CAM for YOLO+CBAM (last neck layer model.model[-2]).
Copied verbatim from notebook Part B of Grad-CAM XAI Cell 2.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as Tv
from PIL import Image as PILImage


# Transform applied to full image before YOLO neck forward pass
_NECK_TRANSFORM = Tv.Compose([
    Tv.ToTensor(),
    Tv.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class GradCAM_YOLO:
    """
    Grad-CAM on YOLO+CBAM — hooks model.model[-2] (last neck layer).

    Uses channel-mean activation map (no backward required) for stability
    with Ultralytics' multi-output head. Falls back to stored _feat if
    the layer traversal fails.

    Usage:
        gcam = GradCAM_YOLO(yolo_model, device)
        cam_crop, cam_full = gcam.generate(img_bgr, box_xyxy)
        overlay            = gcam.overlay_full(img_bgr, cam_full, boxes=...)
    """

    def __init__(self, yolo_model, device):
        self.model  = yolo_model
        self.device = device
        self._feat  = None
        self._grad  = None
        self._handles = []

    def _target_layer(self):
        """model[-2] = last neck layer before Detect/Segment head."""
        return self.model.model.model[-2]

    def _register(self):
        self._remove()
        layer = self._target_layer()

        def fwd(m, inp, out):
            if isinstance(out, torch.Tensor) and out.dim() == 4:
                self._feat = out

        def bwd(m, gin, gout):
            if gout[0] is not None and gout[0].dim() == 4:
                self._grad = gout[0].detach()

        self._handles.append(layer.register_forward_hook(fwd))
        self._handles.append(layer.register_full_backward_hook(bwd))

    def _remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def generate(self, img_bgr: np.ndarray,
                 box_xyxy: list) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            img_bgr:  full BGR image
            box_xyxy: [x1, y1, x2, y2] pixel coords of the detection

        Returns:
            (cam_crop [H_box, W_box] 0-1,
             cam_full [H_img, W_img] 0-1)
        """
        self._register()
        H, W = img_bgr.shape[:2]

        # ─────────────────────────────────────────────────────────────
        # CROP REGION: Add 20% padding around the bounding box
        # ─────────────────────────────────────────────────────────────
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        box_w = x2 - x1
        box_h = y2 - y1
        pad_x = int(box_w * 0.2)
        pad_y = int(box_h * 0.2)
        
        # Crop with padding, clipped to image bounds
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(W, x2 + pad_x)
        crop_y2 = min(H, y2 + pad_y)
        
        crop_h = crop_y2 - crop_y1
        crop_w = crop_x2 - crop_x1
        
        # Crop the image
        cropped = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

        # ─────────────────────────────────────────────────────────────
        # GRADCAM on cropped region only
        # ─────────────────────────────────────────────────────────────
        img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        tensor  = _NECK_TRANSFORM(
            PILImage.fromarray(cv2.resize(img_rgb, (640, 640)))
        ).unsqueeze(0).to(self.device)

        self._feat = None
        self._grad = None

        self.model.model.model.eval()
        tensor.requires_grad_(True)

        try:
            # Traverse backbone+neck layers up to target, capture features
            feat_out = None
            x = tensor
            for i, layer in enumerate(self.model.model.model[:-1]):
                x = layer(x)
                if layer is self._target_layer():
                    feat_out = x
                    break

            if feat_out is None or not isinstance(feat_out, torch.Tensor):
                raise ValueError("Could not capture neck features")

            # Channel-mean activation (GradCAM++ style, no backward needed)
            cam = feat_out.detach().mean(dim=1).squeeze(0)
            cam = F.relu(cam).cpu().numpy()

        except Exception:
            # Fallback to stored forward hook capture
            if self._feat is not None:
                cam = self._feat.detach().mean(dim=1).squeeze(0)
                cam = F.relu(cam).cpu().numpy()
            else:
                cam = np.ones((20, 20))

        self._remove()

        if cam.max() > 0:
            cam = cam / cam.max()

        # ─────────────────────────────────────────────────────────────
        # MAP CAM BACK TO ORIGINAL AND FULL IMAGE
        # ─────────────────────────────────────────────────────────────
        # Resize CAM to cropped region size
        cam_crop = cv2.resize(cam.astype(np.float32), (crop_w, crop_h))
        
        # Create full-image CAM with zeros outside crop region
        cam_full = np.zeros((H, W), dtype=np.float32)
        cam_full[crop_y1:crop_y2, crop_x1:crop_x2] = cam_crop

        return cam_crop, cam_full

    def overlay_full(self, img_bgr: np.ndarray,
                     cam_full: np.ndarray,
                     boxes: list | None = None,
                     alpha: float = 0.40) -> np.ndarray:
        """
        Overlay CAM on full image with optional detection boxes.

        boxes: list of (box_xyxy, color_bgr, label_str)
        """
        heatmap = cv2.applyColorMap(
            (cam_full * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)

        if boxes:
            for (bx, color, label) in boxes:
                x1, y1, x2, y2 = [int(v) for v in bx]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(overlay, label, (x1, max(y1 - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return overlay
