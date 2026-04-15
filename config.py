"""
MALARION — Central configuration.
All thresholds, paths and constants extracted verbatim from the notebook.
"""
import os
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"

# ── Model weight files ────────────────────────────────────────────────
YOLO_BASELINE_PT = WEIGHTS_DIR / "best_malarion_v1.pt"
BV_BASELINE_PTH  = WEIGHTS_DIR / "bv_resnet18_best.pth"
BV_HN_PTH        = WEIGHTS_DIR / "bv_resnet18_hn_best.pth"
YOLO_CBAM_PT     = WEIGHTS_DIR / "c3_yolo_cbam_best.pt"

# ── Detection hyperparameters ─────────────────────────────────────────
CONF_THRESH = 0.25
IOU_THRESH  = 0.45
IMG_SIZE    = 960
CROP_PAD    = 4

# ── BV hyperparameters (from BV-9B recall-safe sweep) ────────────────
BV_THRESH = 0.60         # from BV-9B recall-safe sweep (recall >= 0.94)
IOU_MATCH   = 0.45

# ── Slide-level thresholds ────────────────────────────────────────────
SLIDE_THRESHOLDS = [1, 2, 3]

# ── BV transform (MUST match training) ───────────────────────────────
BV_RESIZE = (224, 224)
BV_MEAN   = [0.485, 0.456, 0.406]
BV_STD    = [0.229, 0.224, 0.225]

# ── 16-class taxonomy (4 species × 4 life stages) ────────────────────
CLASS_NAMES = [
    "falciparum_R", "falciparum_T", "falciparum_S", "falciparum_G",
    "vivax_R",      "vivax_T",      "vivax_S",      "vivax_G",
    "ovale_R",      "ovale_T",      "ovale_S",      "ovale_G",
    "malariae_R",   "malariae_T",   "malariae_S",   "malariae_G",
]
NC = len(CLASS_NAMES)

# ── Species colour palette (BGR for OpenCV) ───────────────────────────
SPECIES_COLORS_BGR = {
    "falciparum": (70,  57,  230),
    "vivax":      (143, 157, 42),
    "ovale":      (106, 196, 233),
    "malariae":   (147, 76,  106),
}

# ── XAI interpretations (verbatim from notebook) ─────────────────────
INTERPRETATIONS = {
    "TP":   ["CAM concentrated on parasite cytoplasm/nucleus — correct localization.",
             "CBAM channels align with ring-stage morphology; tight focus on cell body.",
             "Activation peaks over chromatin dot — strong discriminative signal.",
             "Spatial attention isolates parasite from red blood cell background.",
             "High-confidence TP: CAM energy fully within GT bounding box."],
    "FP":   ["CAM highlights stain artifact or overlapping cell — not a parasite.",
             "Diffuse activation over background debris; no concentrated parasite signal.",
             "Focus on RBC boundary rather than intracellular parasite structure.",
             "Activation spread across multiple cells — ambiguous morphology confuses model.",
             "High YOLO conf but CAM shows no parasite-specific feature highlighted."],
    "FILT": ["BV correctly rejects: CAM energy on background texture, not parasite.",
             "Layer4 activation diffuse — ResNet finds no discriminative parasite feature.",
             "Low BV confidence: crop contains staining artifact, not parasite chromatin.",
             "BV layer4 shows suppressed activation — no nucleus/cytoplasm pattern found.",
             "Filtered correctly: CAM energy on image edge/noise, not intracellular region."],
    "HARD": ["Near-threshold BV score: partial ring structure present but incomplete.",
             "Low YOLO confidence: small parasite size causes weak feature response.",
             "Ambiguous case: overlapping cells cause split CAM activation.",
             "Hard negative: stain granule mimics ring stage — BV barely rejects.",
             "Low conf detection: CAM energy split between parasite and background equally."],
}

# ── Gemini ────────────────────────────────────────────────────────────
GEMINI_MODEL   = "models/gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_EXPECTED_SECTIONS = [
    "1. SLIDE ASSESSMENT",
    "2. DETECTION QUALITY",
    "3. BV FILTER EFFECT",
    "4. CLINICAL VERDICT",
]

# ── Pipeline registry ─────────────────────────────────────────────────
PIPELINE_REGISTRY = {
    1: {"name": "YOLO Baseline",
        "description": "YOLOv8 baseline — detection only, no validator",
        "uses_bv": False, "uses_cbam": False,
        "yolo_path": YOLO_BASELINE_PT, "bv_path": None},
    2: {"name": "YOLO Baseline + BV (no hard-negative mining)",
        "description": "YOLOv8 baseline + ResNet18 BV without hard-negative mining",
        "uses_bv": True, "uses_cbam": False,
        "yolo_path": YOLO_BASELINE_PT, "bv_path": BV_BASELINE_PTH},
    3: {"name": "YOLO Baseline + BV (with hard-negative mining)",
        "description": "YOLOv8 baseline + ResNet18 BV retrained with hard-negative mining",
        "uses_bv": True, "uses_cbam": False,
        "yolo_path": YOLO_BASELINE_PT, "bv_path": BV_HN_PTH},
    4: {"name": "YOLO + CBAM Attention",
        "description": "YOLOv8 with CBAM channel+spatial attention at neck — no BV",
        "uses_bv": False, "uses_cbam": True,
        "yolo_path": YOLO_CBAM_PT, "bv_path": None},
    5: {"name": "YOLO CBAM + BV (hard-negative mining)",
        "description": "Full pipeline: CBAM-YOLO + hard-negative-trained BV validator",
        "uses_bv": True, "uses_cbam": True,
        "yolo_path": YOLO_CBAM_PT, "bv_path": BV_HN_PTH},
}
