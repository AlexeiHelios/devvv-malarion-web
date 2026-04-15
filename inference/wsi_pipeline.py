"""
MALARION — WSI (Whole Slide Image) simulation pipeline.

Accepts a large image (real WSI or synthetic stitched slide),
cuts it into 960×960 tiles with configurable overlap, runs the
existing YOLO→BV pipeline on each tile, then aggregates all
tile-level results into a single slide-level report.

The YOLO and BV models are UNCHANGED — they receive exactly the
same 960×960 BGR input they were trained on.
"""
import cv2
import numpy as np
import logging
from collections import defaultdict

from config import (
    CLASS_NAMES, NC, SLIDE_THRESHOLDS,
    CONF_THRESH, IOU_THRESH, CROP_PAD,
)
from inference.yolo_infer import run_yolo, extract_crop
from inference.bv_infer import bv_predict

log = logging.getLogger(__name__)

# ── WSI tiling constants ──────────────────────────────────────────────
WSI_TILE_SIZE = 960      # must match YOLO training resolution
WSI_OVERLAP   = 64       # px overlap between adjacent tiles


def tile_image(img_bgr: np.ndarray) -> list[dict]:
    """
    Cut a large BGR image into overlapping 960×960 tiles.

    Returns list of tile dicts:
        { "tile": np.ndarray, "x": int, "y": int,
          "row": int, "col": int }
    """
    H, W = img_bgr.shape[:2]
    step  = WSI_TILE_SIZE - WSI_OVERLAP
    tiles = []

    for row_i, y in enumerate(range(0, H, step)):
        for col_i, x in enumerate(range(0, W, step)):
            x2 = min(x + WSI_TILE_SIZE, W)
            y2 = min(y + WSI_TILE_SIZE, H)
            tile = img_bgr[y:y2, x:x2]

            # Pad edge tiles to full 960×960
            if tile.shape[0] < WSI_TILE_SIZE or tile.shape[1] < WSI_TILE_SIZE:
                padded = np.zeros(
                    (WSI_TILE_SIZE, WSI_TILE_SIZE, 3), dtype=np.uint8)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded

            tiles.append({
                "tile": tile,
                "x":    x,
                "y":    y,
                "row":  row_i,
                "col":  col_i,
            })

    return tiles


def run_wsi_pipeline(img_bgr:     np.ndarray,
                     yolo_model,
                     bv_model,
                     device,
                     uses_bv:     bool,
                     conf_thresh: float = CONF_THRESH,
                     iou_thresh:  float = IOU_THRESH,
                     bv_thresh:   float = 0.60) -> dict:
    """
    Full WSI pipeline on a large BGR image.

    Steps:
        1. Tile the image into 960×960 patches
        2. Run YOLO on each tile
        3. Run BV on each detection crop (if uses_bv)
        4. Remap box coords back to full-image space
        5. Aggregate all tiles → slide-level report

    Returns:
        {
          "tile_records":   list[dict],   # per-tile results
          "slide_report":   dict,         # aggregated slide stats
          "all_detections": list[dict],   # all dets in slide space
          "wsi_hw":         (H, W),
          "tile_grid":      (n_rows, n_cols),
          "total_tiles":    int,
        }

    Raises:
        ValueError: If image is smaller than 1920px on longest side.
    """
    H, W  = img_bgr.shape[:2]

    # ── Defensive size guard — reject undersized images ──────────────
    MIN_WSI_SIZE = 1920
    if max(H, W) < MIN_WSI_SIZE:
        msg = (
            f"Image too small for WSI mode ({W}×{H}px). "
            f"WSI mode requires at least {MIN_WSI_SIZE}px on the longest side."
        )
        log.warning(msg)
        raise ValueError(msg)
    tiles = tile_image(img_bgr)

    n_cols = max(t["col"] for t in tiles) + 1
    n_rows = max(t["row"] for t in tiles) + 1

    log.info(f"WSI {W}×{H}px → {len(tiles)} tiles "
             f"({n_rows} rows × {n_cols} cols), "
             f"step={WSI_TILE_SIZE - WSI_OVERLAP}px")

    tile_records  = []
    all_dets      = []

    for tile_info in tiles:
        tile     = tile_info["tile"]
        x_off    = tile_info["x"]
        y_off    = tile_info["y"]
        row_i    = tile_info["row"]
        col_i    = tile_info["col"]
        tile_idx = row_i * n_cols + col_i

        # ── YOLO on this tile ─────────────────────────────────────────
        yolo_result = run_yolo(
            yolo_model, tile,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            img_size=WSI_TILE_SIZE,
        )

        raw_count       = len(yolo_result["det_boxes_xyxy"])
        validated_count = 0
        raw_per_class   = [0] * NC
        val_per_class   = [0] * NC
        kept_cls        = []
        tile_dets       = []

        boxes   = yolo_result["det_boxes_xyxy"]
        cls_ids = yolo_result["det_cls"]
        confs   = yolo_result["det_conf"]
        crops   = yolo_result["crops"]

        for i in range(raw_count):
            cls_id = cls_ids[i]
            if 0 <= cls_id < NC:
                raw_per_class[cls_id] += 1

            # BV filtering
            crop = crops[i]
            if uses_bv and bv_model is not None:
                is_par, bv_conf = bv_predict(
                    bv_model, crop, device, bv_thresh)
            else:
                is_par, bv_conf = True, 1.0

            # Remap tile-space box → full WSI space
            bx1, by1, bx2, by2 = boxes[i]
            slide_box = [
                float(bx1 + x_off), float(by1 + y_off),
                float(bx2 + x_off), float(by2 + y_off),
            ]

            det = {
                "tile_idx":       tile_idx,
                "row":            row_i,
                "col":            col_i,
                "box_xyxy_tile":  [float(bx1), float(by1),
                                   float(bx2), float(by2)],
                "box_xyxy_slide": slide_box,
                "class_id":       cls_id,
                "class_name":     CLASS_NAMES[cls_id] if 0 <= cls_id < NC
                                  else "unknown",
                "species":        CLASS_NAMES[cls_id].split("_")[0]
                                  if 0 <= cls_id < NC else "unknown",
                "stage":          CLASS_NAMES[cls_id].split("_")[1]
                                  if 0 <= cls_id < NC and "_" in CLASS_NAMES[cls_id]
                                  else "?",
                "yolo_conf":      round(float(confs[i]), 4),
                "bv_conf":        round(bv_conf, 4),
                "bv_kept":        is_par,
            }
            tile_dets.append(det)
            all_dets.append(det)

            if is_par:
                validated_count += 1
                if 0 <= cls_id < NC:
                    val_per_class[cls_id] += 1
                kept_cls.append(cls_id)

        tile_verdict = "infected" if validated_count >= 1 else "healthy"

        tile_records.append({
            "tile_idx":        tile_idx,
            "row":             row_i,
            "col":             col_i,
            "x_offset":        x_off,
            "y_offset":        y_off,
            "raw_count":       raw_count,
            "validated_count": validated_count,
            "raw_per_class":   raw_per_class,
            "val_per_class":   val_per_class,
            "kept_cls":        kept_cls,
            "tile_verdict":    tile_verdict,
        })

        log.debug(f"  Tile [{row_i},{col_i}] "
                  f"raw={raw_count} kept={validated_count} → {tile_verdict}")

    # ── Aggregate all tiles ───────────────────────────────────────────
    slide_report = _aggregate(tile_records, all_dets, H, W,
                              n_rows, n_cols)

    return {
        "tile_records":   tile_records,
        "slide_report":   slide_report,
        "all_detections": all_dets,
        "wsi_hw":         (H, W),
        "tile_grid":      (n_rows, n_cols),
        "total_tiles":    len(tiles),
    }


def _aggregate(tile_records: list,
               all_dets:     list,
               H: int, W: int,
               n_rows: int, n_cols: int) -> dict:
    """Merge per-tile counts into slide-level report."""
    total_raw       = sum(t["raw_count"]       for t in tile_records)
    total_validated = sum(t["validated_count"] for t in tile_records)
    agg_raw_pc      = [0] * NC
    agg_val_pc      = [0] * NC
    species_summary = defaultdict(int)
    stage_summary   = defaultdict(int)

    for tile in tile_records:
        for i in range(NC):
            agg_raw_pc[i] += tile["raw_per_class"][i]
            agg_val_pc[i] += tile["val_per_class"][i]

    for det in all_dets:
        if det["bv_kept"]:
            species_summary[det["species"]] += 1
            stage_summary[det["stage"]]     += 1

    threshold_predictions = {
        f"thresh_{T}": ("infected" if total_validated >= T else "healthy")
        for T in SLIDE_THRESHOLDS
    }

    infected_tiles = [t for t in tile_records if t["validated_count"] >= 1]

    return {
        "slide_verdict":             threshold_predictions["thresh_1"],
        "raw_count":                 total_raw,
        "validated_count":           total_validated,
        "raw_count_per_class":       agg_raw_pc,
        "validated_count_per_class": agg_val_pc,
        "species_summary":           dict(species_summary),
        "stage_summary":             dict(stage_summary),
        "threshold_predictions":     threshold_predictions,
        "is_false_negative":         total_raw == 0,
        "class_names":               CLASS_NAMES,
        # WSI-specific extras
        "total_tiles":               len(tile_records),
        "infected_tiles":            len(infected_tiles),
        "healthy_tiles":             len(tile_records) - len(infected_tiles),
        "tile_grid":                 (n_rows, n_cols),
        "wsi_dimensions":            (W, H),
    }


def build_wsi_heatmap(img_bgr:      np.ndarray,
                      all_dets:     list,
                      tile_records: list,
                      n_rows:       int,
                      n_cols:       int,
                      scale:        float = 0.25) -> dict:
    """
    Build two visualisations for the WSI result panel:

    1. density_overlay  — full-slide thumbnail with JET heatmap
                          of validated detection density
    2. verdict_grid_img — colour grid showing per-tile verdict
                          (red=infected, green=healthy)

    Returns { "density_overlay": np.ndarray BGR,
              "verdict_grid":    np.ndarray BGR }
    """
    H, W = img_bgr.shape[:2]
    out_w = max(1, int(W * scale))
    out_h = max(1, int(H * scale))

    # ── Density map ───────────────────────────────────────────────────
    density = np.zeros((H, W), dtype=np.float32)
    for det in all_dets:
        if not det["bv_kept"]:
            continue
        x1, y1, x2, y2 = det["box_xyxy_slide"]
        cx = int(np.clip((x1 + x2) / 2, 0, W - 1))
        cy = int(np.clip((y1 + y2) / 2, 0, H - 1))
        density[cy, cx] += 1.0

    density = cv2.GaussianBlur(density, (61, 61), 0)
    if density.max() > 0:
        density /= density.max()

    thumb        = cv2.resize(img_bgr, (out_w, out_h))
    density_sm   = cv2.resize(density, (out_w, out_h))
    heatmap      = cv2.applyColorMap(
        (density_sm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay      = cv2.addWeighted(thumb, 0.55, heatmap, 0.45, 0)

    # Draw tile grid lines
    step_w = int(WSI_TILE_SIZE * scale)
    step_h = int(WSI_TILE_SIZE * scale)
    for gx in range(0, out_w, step_w):
        cv2.line(overlay, (gx, 0), (gx, out_h), (50, 50, 50), 1)
    for gy in range(0, out_h, step_h):
        cv2.line(overlay, (0, gy), (out_w, gy), (50, 50, 50), 1)

    # ── Verdict grid ──────────────────────────────────────────────────
    CELL = 60
    grid_img = np.zeros((n_rows * CELL, n_cols * CELL, 3), dtype=np.uint8)
    for tile in tile_records:
        r, c = tile["row"], tile["col"]
        if r >= n_rows or c >= n_cols:
            continue
        color = (40, 180, 80) if tile["tile_verdict"] == "healthy" \
                else (50, 50, 210)
        y1g = r * CELL;  y2g = y1g + CELL
        x1g = c * CELL;  x2g = x1g + CELL
        cv2.rectangle(grid_img, (x1g, y1g), (x2g - 1, y2g - 1), color, -1)
        cv2.rectangle(grid_img, (x1g, y1g), (x2g - 1, y2g - 1), (20, 20, 20), 1)
        label = str(tile["validated_count"])
        cv2.putText(grid_img, label,
                    (x1g + CELL // 2 - 6, y1g + CELL // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return {
        "density_overlay": overlay,
        "verdict_grid":    grid_img,
    }
