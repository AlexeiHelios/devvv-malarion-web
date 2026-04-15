"""
MALARION — /api/health  and  /api/models
"""
import torch
from flask import Blueprint, jsonify

from config import PIPELINE_REGISTRY
from models.loader import missing_weights, get_device

health_bp = Blueprint("health", __name__)


@health_bp.route("/api/health", methods=["GET"])
def health():
    """Basic liveness probe."""
    return jsonify({
        "status": "ok",
        "device": str(get_device()),
        "cuda_available": torch.cuda.is_available(),
    }), 200


@health_bp.route("/api/models", methods=["GET"])
def models_status():
    """
    Returns status of all 5 pipeline model files —
    which weight files are present and which are missing.
    """
    absent = missing_weights()
    absent_set = {(m["model_id"], m["component"]) for m in absent}

    pipelines = []
    for mid, cfg in PIPELINE_REGISTRY.items():
        yolo_ok = (mid, "yolo") not in absent_set
        bv_ok   = (mid, "bv")   not in absent_set if cfg["uses_bv"] else None
        ready   = yolo_ok and (bv_ok is not False)
        pipelines.append({
            "model_id":    mid,
            "name":        cfg["name"],
            "description": cfg["description"],
            "uses_bv":     cfg["uses_bv"],
            "uses_cbam":   cfg["uses_cbam"],
            "yolo_weight": cfg["yolo_path"].name,
            "bv_weight":   cfg["bv_path"].name if cfg["bv_path"] else None,
            "yolo_ready":  yolo_ok,
            "bv_ready":    bv_ok,
            "ready":       ready,
        })

    return jsonify({
        "pipelines":      pipelines,
        "missing_files":  absent,
        "all_ready":      len(absent) == 0,
    }), 200
