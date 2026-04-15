"""
MALARION — YOLO inference + crop extraction.
Exactly mirrors the notebook's Cell 3 (slide-level detection loop).
"""
import cv2
import numpy as np
import torch

from config import CONF_THRESH, IOU_THRESH, IMG_SIZE, CROP_PAD, CLASS_NAMES, NC


def extract_crop(img_bgr: np.ndarray,
                 x1: int, y1: int, x2: int, y2: int) -> np.ndarray | None:
    """
    Crop a detection box from the image with CROP_PAD padding.
    Returns None if the resulting crop is degenerate (zero area).
    Mirrors notebook's extract_crop() exactly.
    """
    H, W = img_bgr.shape[:2]
    x1 = max(0, x1 - CROP_PAD)
    y1 = max(0, y1 - CROP_PAD)
    x2 = min(W, x2 + CROP_PAD)
    y2 = min(H, y2 + CROP_PAD)
    if x2 <= x1 or y2 <= y1:
        return None
    return img_bgr[y1:y2, x1:x2].copy()


def run_yolo(yolo_model,
             img_bgr: np.ndarray,
             conf_thresh: float = CONF_THRESH,
             iou_thresh:  float = IOU_THRESH,
             img_size:    int   = IMG_SIZE) -> dict:
    """
    Run YOLO on a single BGR image.

    Returns:
        {
          "det_boxes_xyxy":  list[list[float]],   # pixel coords [x1,y1,x2,y2]
          "det_boxes_xyxyn": list[list[float]],   # normalised  [x1,y1,x2,y2]
          "det_cls":         list[int],
          "det_conf":        list[float],
          "crops":           list[np.ndarray|None],
          "img_hw":          (H, W),
        }
    """
    H, W = img_bgr.shape[:2]

    results = yolo_model.predict(
        source=img_bgr,
        imgsz=img_size,
        conf=conf_thresh,
        iou=iou_thresh,
        verbose=False,
    )
    res = results[0]

    det_boxes_xyxy  = []
    det_boxes_xyxyn = []
    det_cls         = []
    det_conf        = []
    crops           = []

    if res.boxes is not None and len(res.boxes):
        boxes_px  = res.boxes.xyxy.cpu().numpy()
        boxes_n   = res.boxes.xyxyn.cpu().numpy()
        cls_ids   = res.boxes.cls.cpu().numpy().astype(int)
        confs     = res.boxes.conf.cpu().numpy()

        for i in range(len(boxes_px)):
            x1, y1, x2, y2 = boxes_px[i]
            det_boxes_xyxy.append([float(x1), float(y1), float(x2), float(y2)])
            det_boxes_xyxyn.append([float(v) for v in boxes_n[i]])
            det_cls.append(int(cls_ids[i]))
            det_conf.append(float(confs[i]))

            crop = extract_crop(img_bgr, int(x1), int(y1), int(x2), int(y2))
            crops.append(crop)

    return {
        "det_boxes_xyxy":  det_boxes_xyxy,
        "det_boxes_xyxyn": det_boxes_xyxyn,
        "det_cls":         det_cls,
        "det_conf":        det_conf,
        "crops":           crops,
        "img_hw":          (H, W),
    }
