"""
MALARION — /api/predict  (POST)

Full pipeline per request:
  1. Validate + decode uploaded image
  2. Load pipeline (model_id 1-5) from singleton registry
  3. Run YOLO inference → crops
  4. Run BV filtering (if model uses BV)
  5. Compute slide-level stats + per-class breakdown
  6. Generate annotated image
  7. Run GradCAM (YOLO neck + BV layer4)
  8. Return JSON response with all images base64-encoded

/api/explain  (POST)
  Async-friendly endpoint that only runs Gemini XAI narrative.
  Call this after /api/predict returns (fire-and-forget pattern).
"""
import io
import logging
import traceback

import cv2
import numpy as np
from flask import Blueprint, request, jsonify

from config import (
    CLASS_NAMES, NC,
    CONF_THRESH, IOU_THRESH, IMG_SIZE, BV_THRESH,
    PIPELINE_REGISTRY,
)
from models.loader import load_pipeline, get_device
from inference.yolo_infer import run_yolo
from inference.slide_analysis import run_slide_analysis, build_detection_list
from xai.gradcam_bv import GradCAM_BV
from xai.gradcam_yolo import GradCAM_YOLO
from xai.gemini_xai import (
    build_annotated_image,
    build_prompt,
    gemini_explain,
)
from utils.image_utils import encode_image_b64, draw_detections, overlay_heatmap
from inference.wsi_pipeline import run_wsi_pipeline, build_wsi_heatmap

predict_bp = Blueprint("predict", __name__)
log = logging.getLogger(__name__)

# Images larger than this on either dimension are treated as WSI
WSI_MIN_DIMENSION = 1920


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _decode_upload(file_storage) -> np.ndarray:
    """Read a werkzeug FileStorage into a BGR ndarray."""
    buf = np.frombuffer(file_storage.read(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — unsupported format or corrupt file.")
    return img


def _parse_int(value, default: int, name: str, lo: int, hi: int) -> int:
    try:
        v = int(value)
        if not (lo <= v <= hi):
            raise ValueError
        return v
    except (TypeError, ValueError):
        log.warning(f"Invalid {name}={value!r}, using default={default}")
        return default


def _parse_float(value, default: float, name: str, lo: float, hi: float) -> float:
    try:
        v = float(value)
        if not (lo <= v <= hi):
            raise ValueError
        return v
    except (TypeError, ValueError):
        log.warning(f"Invalid {name}={value!r}, using default={default}")
        return default


# ─────────────────────────────────────────────────────────────────────
# POST /api/predict
# ─────────────────────────────────────────────────────────────────────

@predict_bp.route("/api/predict", methods=["POST"])
def predict():
    """
    Expected multipart/form-data fields:
      image       : image file (jpg / png / tiff)
      model_id    : int 1-5  (default 5)
      conf_thresh : float    (default 0.25)
      iou_thresh  : float    (default 0.45)
      bv_thresh   : float    (default 0.60)

    Returns JSON with annotated image, heatmaps, slide report.
    """
    # ── 1. Parse request ─────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image file in request (field: 'image')"}), 400

    model_id    = _parse_int(request.form.get("model_id",    5), 5, "model_id",    1, 5)
    conf_thresh = _parse_float(request.form.get("conf_thresh", CONF_THRESH), CONF_THRESH, "conf_thresh", 0.01, 1.0)
    iou_thresh  = _parse_float(request.form.get("iou_thresh",  IOU_THRESH),  IOU_THRESH,  "iou_thresh",  0.01, 1.0)
    bv_thresh   = _parse_float(request.form.get("bv_thresh",   BV_THRESH),   BV_THRESH,   "bv_thresh",   0.01, 1.0)

    try:
        img_bgr = _decode_upload(request.files["image"])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # ── 2. Load pipeline ──────────────────────────────────────────────
    try:
        pipeline = load_pipeline(model_id)
    except FileNotFoundError as e:
        return jsonify({
            "error": str(e),
            "hint":  "Place the required .pt / .pth files in the weights/ directory.",
        }), 503
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    yolo_model = pipeline["yolo"]
    bv_model   = pipeline["bv"]
    uses_bv    = pipeline["uses_bv"]
    uses_cbam  = pipeline["uses_cbam"]
    device     = get_device()

    # ── 3. YOLO inference ─────────────────────────────────────────────
    try:
        yolo_result = run_yolo(
            yolo_model, img_bgr,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            img_size=IMG_SIZE,
        )
    except Exception as e:
        log.error(f"YOLO inference failed: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"YOLO inference failed: {e}"}), 500

    # ── 4. BV filtering + slide analysis ─────────────────────────────
    try:
        slide_record = run_slide_analysis(
            yolo_result, bv_model, device,
            bv_thresh=bv_thresh,
            uses_bv=uses_bv,
        )
        detections = build_detection_list(yolo_result, slide_record)
    except Exception as e:
        log.error(f"Slide analysis failed: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Slide analysis failed: {e}"}), 500

    # ── 5. Build annotated image ──────────────────────────────────────
    img_annotated = build_annotated_image(img_bgr, yolo_result, slide_record)

    # ── 6. GradCAM — YOLO neck ───────────────────────────────────────
    yolo_gradcam_b64 = None
    try:
        gcam_yolo = GradCAM_YOLO(yolo_model, device)
        if detections:
            # Use first kept detection box, or first detection if none kept
            ref_det = next((d for d in detections if d["bv_kept"]), detections[0])
            _, cam_full = gcam_yolo.generate(img_bgr, ref_det["box_xyxy"])
            yolo_overlay = gcam_yolo.overlay_full(img_annotated, cam_full)
        else:
            # No detections — still generate full-image CAM for reference
            dummy_box = [0, 0, img_bgr.shape[1], img_bgr.shape[0]]
            _, cam_full = gcam_yolo.generate(img_bgr, dummy_box)
            yolo_overlay = gcam_yolo.overlay_full(img_bgr, cam_full)
        yolo_gradcam_b64 = encode_image_b64(yolo_overlay)
    except Exception as e:
        log.warning(f"YOLO GradCAM failed (non-fatal): {e}")

    # ── 7. GradCAM — BV (per kept detection) ─────────────────────────
    bv_gradcam_panels = []
    if uses_bv and bv_model is not None:
        gcam_bv = GradCAM_BV(bv_model, device)
        crops   = yolo_result["crops"]
        for det in detections:
            idx  = det["index"]
            crop = crops[idx] if idx < len(crops) else None
            if crop is None or not det["bv_kept"]:
                continue
            try:
                cam, bv_prob = gcam_bv.generate(crop)
                overlay      = gcam_bv.overlay(crop, cam)
                bv_gradcam_panels.append({
                    "detection_index": idx,
                    "class_name":      det["class_name"],
                    "bv_conf":         round(bv_prob, 4),
                    "crop_original":   encode_image_b64(crop),
                    "bv_gradcam":      encode_image_b64(overlay),
                })
            except Exception as e:
                log.warning(f"BV GradCAM failed for det {idx}: {e}")

    # ── 8. Assemble response ──────────────────────────────────────────
    response = {
        "status":        "ok",
        "model_id":      model_id,
        "pipeline_name": pipeline["name"],
        "description":   pipeline["description"],
        "uses_bv":       uses_bv,
        "uses_cbam":     uses_cbam,

        # YOLO inference results (for XAI report generation)
        "yolo_result": {
            "det_boxes_xyxy": [d.tolist() if hasattr(d, 'tolist') else d for d in yolo_result.get("det_boxes_xyxy", [])],
            "det_cls":        [int(c) if hasattr(c, '__int__') else c for c in yolo_result.get("det_cls", [])],
            "det_conf":       [float(c) if hasattr(c, '__float__') else c for c in yolo_result.get("det_conf", [])],
            "crops":          [],
        },

        # Slide-level report
        "slide_report": {
            "slide_verdict":              slide_record["slide_verdict"],
            "raw_count":                  slide_record["raw_count"],
            "validated_count":            slide_record["validated_count"],
            "raw_count_per_class":        slide_record["raw_count_per_class"],
            "validated_count_per_class":  slide_record["validated_count_per_class"],
            "species_summary":            slide_record["species_summary"],
            "stage_summary":              slide_record["stage_summary"],
            "threshold_predictions":      slide_record["threshold_predictions"],
            "is_false_negative":          slide_record["is_false_negative"],
            "class_names":                CLASS_NAMES,
        },

        # Per-detection list
        "detections": detections,

        # Images (base64 JPEG)
        "images": {
            "annotated":       encode_image_b64(img_annotated),
            "yolo_gradcam":    yolo_gradcam_b64,
            "bv_gradcam_panels": bv_gradcam_panels,
        },

        # Gemini status placeholder — call /api/explain separately
        "gemini_xai": {
            "status":   "pending",
            "raw_text": None,
            "sections": None,
        },
    }

    return jsonify(response), 200


# ─────────────────────────────────────────────────────────────────────
# POST /api/explain
# Accepts same fields as /api/predict PLUS the yolo/slide JSON already
# computed, so we don't re-run inference.
# ─────────────────────────────────────────────────────────────────────

@predict_bp.route("/api/explain", methods=["POST"])
def explain():
    """
    Generate Gemini XAI narrative for a slide.

    Expected JSON body (output of /api/predict can be piped in):
      {
        "model_id":     int,
        "slide_report": { ... },
        "detections":   [ ... ],
        "image_b64":    "<base64 annotated JPEG>"   (optional)
      }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    model_id    = int(data.get("model_id", 5))
    slide_rep   = data.get("slide_report", {})
    detections  = data.get("detections", [])
    image_b64   = data.get("image_b64")
    pipeline_name = PIPELINE_REGISTRY.get(model_id, {}).get("name", f"Model {model_id}")

    # Reconstruct minimal yolo_result and slide_record for prompt builder
    yolo_result = {
        "det_boxes_xyxy": [d["box_xyxy"] for d in detections],
        "det_cls":        [d["class_id"] for d in detections],
        "det_conf":       [d["yolo_conf"] for d in detections],
        "crops":          [],
    }
    slide_record = {
        "raw_count":       slide_rep.get("raw_count", 0),
        "validated_count": slide_rep.get("validated_count", 0),
        "kept_cls":        [d["class_id"] for d in detections if d.get("bv_kept")],
        "kept_conf":       [d["bv_conf"]  for d in detections if d.get("bv_kept")],
        "kept_flags":      [d.get("bv_kept", False) for d in detections],
        "slide_verdict":   slide_rep.get("slide_verdict", "unknown"),
        "is_false_negative": slide_rep.get("is_false_negative", False),
        "species_summary": slide_rep.get("species_summary", {}),
        "stage_summary":   slide_rep.get("stage_summary", {}),
    }

    # Decode annotated image (or create blank placeholder)
    if image_b64:
        import base64
        buf = np.frombuffer(base64.b64decode(image_b64), dtype=np.uint8)
        img_ann = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img_ann is None:
            img_ann = np.zeros((960, 960, 3), dtype=np.uint8)
    else:
        img_ann = np.zeros((960, 960, 3), dtype=np.uint8)

    try:
        parts  = build_prompt(yolo_result, slide_record, img_ann, pipeline_name)
        result = gemini_explain(parts)
    except Exception as e:
        log.error(f"Gemini explain failed: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Gemini explain failed: {e}"}), 500

    return jsonify({"gemini_xai": result}), 200


# ─────────────────────────────────────────────────────────────────────
# POST /api/predict_wsi
# Handles large / stitched slide images via tile-based inference.
# The YOLO and BV models are UNCHANGED — each tile is 960×960.
# ─────────────────────────────────────────────────────────────────────

@predict_bp.route("/api/predict_wsi", methods=["POST"])
def predict_wsi():
    """
    WSI (Whole Slide Image) endpoint.
    Accepts the same fields as /api/predict.
    Automatically tiles the image and aggregates results.

    Extra response fields vs /api/predict:
      slide_report.total_tiles
      slide_report.infected_tiles
      slide_report.tile_grid
      slide_report.wsi_dimensions
      images.wsi_density_overlay   — heatmap thumbnail
      images.wsi_verdict_grid      — per-tile verdict grid
      tile_records                 — per-tile raw results
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file (field: 'image')"}), 400

    model_id    = _parse_int(request.form.get("model_id",    5), 5, "model_id",    1, 5)
    conf_thresh = _parse_float(request.form.get("conf_thresh", CONF_THRESH), CONF_THRESH, "conf_thresh", 0.01, 1.0)
    iou_thresh  = _parse_float(request.form.get("iou_thresh",  IOU_THRESH),  IOU_THRESH,  "iou_thresh",  0.01, 1.0)
    bv_thresh   = _parse_float(request.form.get("bv_thresh",   BV_THRESH),   BV_THRESH,   "bv_thresh",   0.01, 1.0)

    try:
        img_bgr = _decode_upload(request.files["image"])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    H, W = img_bgr.shape[:2]
    log.info(f"WSI request: {W}×{H}px  model={model_id}")

    # ── Size guard — images smaller than 2×TILE_SIZE are not suitable for WSI ──
    # A standard FOV image (960×960) uploaded in WSI mode would be split into
    # tiles that are mostly black padding, causing missed detections.
    # Minimum requirement: at least 1920px on the longest side.
    MIN_WSI_SIZE = 1920
    if max(H, W) < MIN_WSI_SIZE:
        return jsonify({
            "error": (
                f"Image too small for WSI mode ({W}×{H}px). "
                f"WSI mode requires at least {MIN_WSI_SIZE}px on the longest side. "
                f"Please use Standard FOV mode for single microscopy images."
            ),
            "hint": "Switch the toggle to 'Standard FOV' and re-upload.",
            "image_size": f"{W}×{H}",
            "required_min": MIN_WSI_SIZE,
        }), 400

    # ── Load pipeline ─────────────────────────────────────────────────
    try:
        pipeline = load_pipeline(model_id)
    except FileNotFoundError as e:
        return jsonify({"error": str(e), "hint": "Place weight files in weights/"}), 503
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    yolo_model = pipeline["yolo"]
    bv_model   = pipeline["bv"]
    uses_bv    = pipeline["uses_bv"]
    uses_cbam  = pipeline["uses_cbam"]
    device     = get_device()

    # ── Run WSI pipeline ──────────────────────────────────────────────
    try:
        wsi_result = run_wsi_pipeline(
            img_bgr, yolo_model, bv_model, device,
            uses_bv=uses_bv,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            bv_thresh=bv_thresh,
        )
    except ValueError as e:
        # Validation error (e.g. undersized image) — return 400
        log.warning(f"WSI validation failed: {e}")
        return jsonify({
            "error": str(e),
            "hint": "Switch the toggle to 'Standard FOV' and re-upload.",
        }), 400
    except Exception as e:
        log.error(f"WSI pipeline failed: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"WSI inference failed: {e}"}), 500

    tile_records   = wsi_result["tile_records"]
    slide_report   = wsi_result["slide_report"]
    all_dets       = wsi_result["all_detections"]
    n_rows, n_cols = wsi_result["tile_grid"]

    # ── Build heatmap visualisations ──────────────────────────────────
    wsi_density_b64 = None
    wsi_grid_b64    = None
    try:
        viz = build_wsi_heatmap(
            img_bgr, all_dets, tile_records, n_rows, n_cols, scale=0.25)
        wsi_density_b64 = encode_image_b64(viz["density_overlay"])
        wsi_grid_b64    = encode_image_b64(viz["verdict_grid"])
    except Exception as e:
        log.warning(f"WSI heatmap failed (non-fatal): {e}")

    # ── Serialise tile records (drop numpy arrays) ────────────────────
    tile_records_json = [
        {k: v for k, v in t.items()}
        for t in tile_records
    ]

    return jsonify({
        "status":        "ok",
        "mode":          "wsi",
        "model_id":      model_id,
        "pipeline_name": pipeline["name"],
        "uses_bv":       uses_bv,
        "uses_cbam":     uses_cbam,
        "slide_report":  slide_report,
        "detections":    all_dets,
        "tile_records":  tile_records_json,
        "images": {
            "wsi_density_overlay": wsi_density_b64,
            "wsi_verdict_grid":    wsi_grid_b64,
        },
        "gemini_xai": {"status": "pending"},
    }), 200


# ─────────────────────────────────────────────────────────────────────
# POST /api/generate_xai_report
# Generate detailed XAI report on demand (optional feature)
# ─────────────────────────────────────────────────────────────────────

@predict_bp.route("/api/generate_xai_report", methods=["POST"])
def generate_xai_report():
    """
    Generate a detailed Gemini XAI report for the given analysis result.
    Expects the prediction data in JSON.
    
    Request body:
    {
        "yolo_result": {...},
        "slide_report": {...},  (or slide_record)
        "detections": [...],
        "pipeline_name": "...",
        "image_b64": "..." (optional, for context)
    }
    
    Returns:
    {
        "status": "ok" | "error",
        "gemini_xai": { ... detailed report ... },
        "report_text": "Plain text version for download"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body"}), 400
            
        yolo_result = data.get("yolo_result", {})
        slide_report = data.get("slide_report", data.get("slide_record", {}))  # Support both names
        detections = data.get("detections", [])
        pipeline_name = data.get("pipeline_name", "Unknown")
        image_b64 = data.get("image_b64")
        
        # Reconstruct slide_record from slide_report and detections
        # (frontend sends response data, we need internal structure)
        slide_record = {
            "raw_count":       slide_report.get("raw_count", 0),
            "validated_count": slide_report.get("validated_count", 0),
            "kept_cls":        [d.get("class_id", 0) for d in detections if d.get("bv_kept")],
            "kept_conf":       [d.get("bv_conf", 0.0) for d in detections if d.get("bv_kept")],
            "kept_flags":      [d.get("bv_kept", False) for d in detections],
            "slide_verdict":   slide_report.get("slide_verdict", "unknown"),
            "is_false_negative": slide_report.get("is_false_negative", False),
            "species_summary": slide_report.get("species_summary", {}),
            "stage_summary":   slide_report.get("stage_summary", {}),
        }
        
        # Reconstruct image if available
        if image_b64:
            import base64
            try:
                buf = np.frombuffer(base64.b64decode(image_b64), dtype=np.uint8)
                img_ann = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img_ann is None:
                    img_ann = np.zeros((960, 960, 3), dtype=np.uint8)
            except Exception:
                img_ann = np.zeros((960, 960, 3), dtype=np.uint8)
        else:
            img_ann = np.zeros((960, 960, 3), dtype=np.uint8)
        
        # Build prompt and get Gemini explanation
        parts = build_prompt(yolo_result, slide_record, img_ann, pipeline_name)
        result = gemini_explain(parts)
        
        if not result:
            return jsonify({"error": "Gemini report generation failed"}), 500
        
        # Format the report text for download
        report_text = _format_report_for_download(
            slide_record, yolo_result, result, pipeline_name
        )
        
        return jsonify({
            "status": "ok",
            "gemini_xai": result,
            "report_text": report_text
        }), 200
        
    except Exception as e:
        log.error(f"XAI report generation failed: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Report generation failed: {e}"}), 500


def _format_report_for_download(slide_record: dict, yolo_result: dict, 
                                gemini_result: dict, pipeline_name: str) -> str:
    """
    Format the report as downloadable text with metadata and detailed analysis.
    """
    lines = [
        "=" * 80,
        "MALARION — Malaria Parasite Detection & Explainable AI Report",
        "=" * 80,
        "",
        f"Pipeline: {pipeline_name}",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "─" * 80,
        "ANALYSIS SUMMARY",
        "─" * 80,
        "",
        f"Raw detections: {slide_record.get('raw_count', 0)}",
        f"BV-validated: {slide_record.get('validated_count', 0)}",
        f"Slide verdict: {slide_record.get('slide_verdict', 'unknown')}",
        "",
    ]
    
    # Add species and stage breakdown
    species_summary = slide_record.get('species_summary', {})
    stage_summary = slide_record.get('stage_summary', {})
    
    if species_summary:
        lines.append("Species Breakdown:")
        for species, count in species_summary.items():
            lines.append(f"  {species}: {count}")
        lines.append("")
    
    if stage_summary:
        lines.append("Life Stage Breakdown:")
        for stage, count in stage_summary.items():
            lines.append(f"  {stage}: {count}")
        lines.append("")
    
    lines.extend([
        "─" * 80,
        "GEMINI XAI — CLINICAL ANALYSIS",
        "─" * 80,
        "",
    ])
    
    # Add Gemini sections
    if isinstance(gemini_result, dict):
        for section_key in ["1. SLIDE ASSESSMENT", "2. DETECTION QUALITY", 
                           "3. BV FILTER EFFECT", "4. CLINICAL VERDICT"]:
            if section_key in gemini_result:
                lines.append(section_key)
                lines.append(gemini_result[section_key])
                lines.append("")
    else:
        lines.append(str(gemini_result))
    
    lines.extend([
        "─" * 80,
        "End of Report",
        "=" * 80,
    ])
    
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# GET /api/download_report
# Download the generated report as plain text
# ─────────────────────────────────────────────────────────────────────

@predict_bp.route("/api/download_report", methods=["POST"])
def download_report():
    """
    Download the XAI report as a .txt file.
    
    Request body:
    {
        "report_text": "...",
        "filename": "malarion_report.txt" (optional)
    }
    """
    from flask import send_file, make_response
    
    try:
        data = request.get_json()
        if not data or "report_text" not in data:
            return jsonify({"error": "No report_text provided"}), 400
        
        report_text = data["report_text"]
        filename = data.get("filename", f"malarion_report_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        
        # Create text file in memory
        output = io.BytesIO()
        output.write(report_text.encode('utf-8'))
        output.seek(0)
        
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        response.headers["Content-Type"] = "text/plain; charset=utf-8"
        
        return response, 200
        
    except Exception as e:
        log.error(f"Report download failed: {e}")
        return jsonify({"error": str(e)}), 500
