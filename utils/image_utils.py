"""
MALARION — Image utility functions.
"""
import base64
import cv2
import numpy as np

from config import CLASS_NAMES, NC, SPECIES_COLORS_BGR


def encode_image_b64(img_bgr: np.ndarray, quality: int = 90) -> str | None:
    """Encode a BGR image as JPEG base64 string."""
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def draw_detections(img_bgr: np.ndarray,
                    detections: list[dict],
                    draw_filtered: bool = True) -> np.ndarray:
    """
    Draw bounding boxes on a copy of img_bgr.
      - BV-kept   → species colour box
      - BV-filtered → red box
    Each box is labelled with class_name + yolo_conf.
    """
    out = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box_xyxy"]]
        kept   = det["bv_kept"]
        species = det.get("species", "falciparum")

        if kept:
            color = SPECIES_COLORS_BGR.get(species, (0, 200, 0))
        else:
            if not draw_filtered:
                continue
            color = (0, 0, 220)

        label = f"{det['class_name'][:12]} {det['yolo_conf']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    return out


def overlay_heatmap(img_bgr: np.ndarray,
                    cam: np.ndarray,
                    alpha: float = 0.40) -> np.ndarray:
    """Apply JET colourmap overlay on a BGR image."""
    H, W       = img_bgr.shape[:2]
    cam_scaled = cv2.resize(cam.astype(np.float32), (W, H))
    heatmap    = cv2.applyColorMap(
        (cam_scaled * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)


def resize_for_display(img_bgr: np.ndarray,
                       max_dim: int = 960) -> np.ndarray:
    """Proportionally resize so the longest side ≤ max_dim."""
    H, W = img_bgr.shape[:2]
    scale = min(max_dim / max(H, W), 1.0)
    if scale == 1.0:
        return img_bgr
    nW, nH = int(W * scale), int(H * scale)
    return cv2.resize(img_bgr, (nW, nH), interpolation=cv2.INTER_AREA)
