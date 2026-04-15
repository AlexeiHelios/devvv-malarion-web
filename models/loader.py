"""
MALARION — Model loader / registry.

Models are loaded ONCE at application startup and stored in a module-level
dict. Each pipeline variant (1-5) gets its own entry. Thread-safe for
read-only inference (no in-place ops on shared model state).
"""
import threading
import logging
from pathlib import Path

import torch
from ultralytics import YOLO as UltralyticsYOLO

from config import PIPELINE_REGISTRY, BV_THRESH
from models.architectures import BinaryValidator

log = logging.getLogger(__name__)

# ── Global registry ───────────────────────────────────────────────────
_registry: dict = {}          # { model_id: { "yolo": ..., "bv": ... } }
_lock = threading.Lock()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_yolo(path: Path):
    """Load an Ultralytics YOLO model and move to device."""
    if not path.exists():
        raise FileNotFoundError(
            f"YOLO weight file not found: {path}\n"
            f"Place the .pt file in the weights/ directory."
        )
    model = UltralyticsYOLO(str(path))
    model.to(DEVICE)
    log.info(f"  YOLO loaded: {path.name}  (device={DEVICE})")
    return model


def _load_bv(path: Path) -> BinaryValidator:
    """Load a BinaryValidator checkpoint."""
    if not path.exists():
        raise FileNotFoundError(
            f"BV weight file not found: {path}\n"
            f"Place the .pth file in the weights/ directory."
        )
    model = BinaryValidator().to(DEVICE)
    ckpt  = torch.load(str(path), map_location=DEVICE)
    state = ckpt.get("state_dict", ckpt)   # handle wrapped checkpoints
    model.load_state_dict(state)
    model.eval()

    if "val_f1" in ckpt:
        log.info(f"  BV loaded:   {path.name}  val_f1={ckpt['val_f1']:.4f}")
    else:
        log.info(f"  BV loaded:   {path.name}")
    return model


def load_pipeline(model_id: int) -> dict:
    """
    Return the cached pipeline dict for model_id, loading it on first call.

    Returns:
        {
          "yolo":     UltralyticsYOLO instance,
          "bv":       BinaryValidator | None,
          "uses_bv":  bool,
          "uses_cbam": bool,
          "name":     str,
        }
    """
    global _registry

    if model_id not in PIPELINE_REGISTRY:
        raise ValueError(f"Unknown model_id={model_id}. Valid: 1–5.")

    with _lock:
        if model_id not in _registry:
            cfg  = PIPELINE_REGISTRY[model_id]
            log.info(f"Loading pipeline {model_id}: {cfg['name']}")

            yolo = _load_yolo(cfg["yolo_path"])
            bv   = _load_bv(cfg["bv_path"]) if cfg["uses_bv"] else None

            _registry[model_id] = {
                "yolo":      yolo,
                "bv":        bv,
                "uses_bv":   cfg["uses_bv"],
                "uses_cbam": cfg["uses_cbam"],
                "name":      cfg["name"],
                "description": cfg["description"],
            }
            log.info(f"Pipeline {model_id} ready.")

    return _registry[model_id]


def preload_all():
    """Attempt to preload all pipelines at startup (optional)."""
    for mid in PIPELINE_REGISTRY:
        try:
            load_pipeline(mid)
        except FileNotFoundError as e:
            log.warning(f"Pipeline {mid} skipped at preload: {e}")


def get_device() -> torch.device:
    return DEVICE


def missing_weights() -> list[dict]:
    """Return info about weight files that are absent from disk."""
    missing = []
    for mid, cfg in PIPELINE_REGISTRY.items():
        for label, path in [("yolo", cfg["yolo_path"]),
                             ("bv",   cfg["bv_path"])]:
            if path is not None and not Path(path).exists():
                missing.append({
                    "model_id":  mid,
                    "pipeline":  cfg["name"],
                    "component": label,
                    "filename":  Path(path).name,
                    "full_path": str(path),
                })
    return missing
