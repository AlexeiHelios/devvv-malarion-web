"""
MALARION — Slide-level analysis.
Mirrors notebook Cell 4 (BV filtering + class-aware counting)
and the threshold sweep from the slide-level analysis section.
"""
from config import CLASS_NAMES, NC, SLIDE_THRESHOLDS
from inference.bv_infer import bv_predict


def run_slide_analysis(yolo_result: dict,
                       bv_model,
                       device,
                       bv_thresh: float,
                       uses_bv:   bool) -> dict:
    """
    Apply BV filtering (if uses_bv) and compute slide-level statistics.

    Args:
        yolo_result: output of run_yolo()
        bv_model:    BinaryValidator | None
        device:      torch.device
        bv_thresh:   BV sigmoid threshold
        uses_bv:     whether to apply BV filtering

    Returns:
        slide_record dict with all counts, per-class breakdowns,
        detection-level kept_flags, and slide verdict.
    """
    crops    = yolo_result["det_boxes_xyxy"]
    det_cls  = yolo_result["det_cls"]
    det_conf = yolo_result["det_conf"]
    raw_crops = yolo_result["crops"]

    raw_count                 = len(raw_crops)
    raw_count_per_class       = [0] * NC
    validated_count_per_class = [0] * NC
    validated_count           = 0

    kept_cls   = []
    kept_conf  = []
    kept_flags = []      # True = BV kept (or no BV used), False = BV filtered

    for i, crop in enumerate(raw_crops):
        cls_id = int(det_cls[i]) if i < len(det_cls) else 0

        # Raw class tally (mirrors notebook exactly)
        if 0 <= cls_id < NC:
            raw_count_per_class[cls_id] += 1

        # BV decision
        if uses_bv and bv_model is not None:
            is_par, bv_conf = bv_predict(bv_model, crop, device, bv_thresh)
        else:
            # No BV — all detections are kept
            is_par, bv_conf = True, 1.0

        kept_flags.append(is_par)

        if is_par:
            validated_count += 1
            if 0 <= cls_id < NC:
                validated_count_per_class[cls_id] += 1
            kept_cls.append(cls_id)
            kept_conf.append(bv_conf)

    # ── Species / stage summaries ─────────────────────────────────────
    species_summary = {}
    stage_summary   = {}
    for cls_id in kept_cls:
        name    = CLASS_NAMES[cls_id] if 0 <= cls_id < NC else "unknown"
        species = name.split("_")[0]
        stage   = name.split("_")[1] if "_" in name else "?"
        species_summary[species] = species_summary.get(species, 0) + 1
        stage_summary[stage]     = stage_summary.get(stage, 0) + 1

    # ── Slide verdict at multiple thresholds ──────────────────────────
    threshold_predictions = {
        f"thresh_{T}": ("infected" if validated_count >= T else "healthy")
        for T in SLIDE_THRESHOLDS
    }

    # ── False-negative flag ───────────────────────────────────────────
    is_false_negative = (raw_count == 0)   # YOLO found nothing at all

    return {
        "raw_count":                  raw_count,
        "validated_count":            validated_count,
        "raw_count_per_class":        raw_count_per_class,
        "validated_count_per_class":  validated_count_per_class,
        "kept_cls":                   kept_cls,
        "kept_conf":                  kept_conf,
        "kept_flags":                 kept_flags,
        "species_summary":            species_summary,
        "stage_summary":              stage_summary,
        "threshold_predictions":      threshold_predictions,
        "slide_verdict":              threshold_predictions["thresh_1"],
        "is_false_negative":          is_false_negative,
    }


def build_detection_list(yolo_result: dict,
                         slide_record: dict) -> list[dict]:
    """
    Build per-detection list for the API response and Gemini prompt.
    Assigns XAI category (TP/FP/FILT/HARD) based on BV confidence
    and YOLO confidence heuristics (GT-free version for inference).
    """
    boxes   = yolo_result["det_boxes_xyxy"]
    cls_ids = yolo_result["det_cls"]
    confs   = yolo_result["det_conf"]
    flags   = slide_record["kept_flags"]

    detections = []
    bv_conf_iter = iter(slide_record["kept_conf"])

    for i in range(len(boxes)):
        cls_id   = int(cls_ids[i]) if i < len(cls_ids) else 0
        cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < NC else "unknown"
        yolo_c   = float(confs[i]) if i < len(confs) else 0.0
        bv_kept  = flags[i] if i < len(flags) else False
        bv_c     = next(bv_conf_iter) if bv_kept else 0.0

        # XAI category (heuristic without GT labels)
        if not bv_kept:
            cat = "FILT"
        elif yolo_c <= 0.35 or (bv_kept and 0.45 <= bv_c <= 0.70):
            cat = "HARD"
        else:
            cat = "TP"   # default for kept detections (FP needs GT labels)

        detections.append({
            "index":     i,
            "box_xyxy":  [float(v) for v in boxes[i]],
            "class_id":  cls_id,
            "class_name": cls_name,
            "species":   cls_name.split("_")[0],
            "stage":     cls_name.split("_")[1] if "_" in cls_name else "?",
            "yolo_conf": round(yolo_c, 4),
            "bv_conf":   round(bv_c, 4),
            "bv_kept":   bv_kept,
            "xai_category": cat,
        })

    return detections
