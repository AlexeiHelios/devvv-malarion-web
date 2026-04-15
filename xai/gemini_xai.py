"""
MALARION — Gemini Flash XAI narrative generator.
Mirrors the notebook's Section 4-5 (build_prompt + gemini_explain) exactly.
"""
import base64
import time
import logging

import cv2
import numpy as np

from google import genai
from google.genai import types

from config import (
    GEMINI_MODEL, GEMINI_API_KEY,
    GEMINI_EXPECTED_SECTIONS,
    CLASS_NAMES, NC, BV_THRESH,
)

log = logging.getLogger(__name__)

# ── System context (verbatim from notebook) ───────────────────────────
SYSTEM_CONTEXT_WITH_BV = """You are MALARION-XAI, a clinical AI assistant specialising
in malaria parasite detection and Explainable AI for deep-learning pipelines.

The MALARION system uses a two-stage pipeline on thin blood-smear images:
  Stage 1 — YOLOv8m-seg + CBAM: detects and segments parasite instances
             across 16 classes (4 species x 4 life stages).
             Species: falciparum, vivax, ovale, malariae
             Stages:  R=Ring, T=Trophozoite, S=Schizont, G=Gametocyte
  Stage 2 — Binary Validator (BV) ResNet18: filters false detections.

In the annotated image:
  GREEN boxes = detections accepted by BV (confirmed parasite)
  RED boxes   = detections rejected by BV (filtered as non-parasite)
  No boxes    = YOLO found nothing (possible false negative)""".strip()

SYSTEM_CONTEXT_NO_BV = """You are MALARION-XAI, a clinical AI assistant specialising
in malaria parasite detection and Explainable AI for deep-learning pipelines.

The MALARION system detects parasites on thin blood-smear images using YOLOv8 models
across 16 classes (4 species x 4 life stages):
  Species: falciparum, vivax, ovale, malariae
  Stages:  R=Ring, T=Trophozoite, S=Schizont, G=Gametocyte

In the annotated image:
  GREEN boxes = YOLO detections (parasite instances)
  No boxes    = YOLO found nothing (possible false negative)""".strip()

FORMAT_RULES = """
OUTPUT FORMAT — Enhanced detailed analysis (follow exactly):
- Begin with "1. SLIDE ASSESSMENT" — no preamble
- Use plain text only (no markdown formatting)
- Write EXACTLY 4 sections in this form:
    1. SLIDE ASSESSMENT
    2. DETECTION QUALITY
    3. DETECTION EVALUATION
    4. CLINICAL VERDICT
- Write 5-8 detailed sentences per section, explaining:
  * Specific findings and their clinical significance
  * Numerical data (counts, percentages, confidence scores)
  * Parasites identified by species and life stage
  * Quality assessment of detections (true positive rate)
  * Impact of validation/filtering on detections (if applicable)
  * Confidence levels and any ambiguities
  * Recommended next steps for clinical decision-making
- No bullet points; use flowing narrative prose
- Include interpretations of visual patterns and AI confidence
- Do NOT add sections beyond 4; do not add RECOMMENDATIONS or other sections""".strip()

_FALLBACK = (
    "1. SLIDE ASSESSMENT\n[Gemini API unavailable after retries]\n\n"
    "2. DETECTION QUALITY\n[N/A]\n\n"
    "3. DETECTION EVALUATION\n[N/A]\n\n"
    "4. CLINICAL VERDICT\n[N/A]"
)


def _init_gemini():
    if not GEMINI_API_KEY:
        log.warning("GEMINI_API_KEY not set — XAI narrative disabled.")
        return None
    client = genai.Client(api_key=GEMINI_API_KEY)
    log.info(f"Gemini client ready: {GEMINI_MODEL}")
    return client


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = _init_gemini()
    return _client


def _strip_preamble(text: str) -> str:
    """Remove any text before the first expected section header."""
    for section in GEMINI_EXPECTED_SECTIONS:
        idx = text.find(section)
        if idx != -1:
            return text[idx:].strip()
    return text.strip()


def _validate_format(text: str) -> tuple[bool, list]:
    missing = [s for s in GEMINI_EXPECTED_SECTIONS if s not in text]
    return len(missing) == 0, missing


def build_annotated_image(img_bgr: np.ndarray,
                          yolo_result: dict,
                          slide_record: dict) -> np.ndarray:
    """
    Draw green/red boxes on a copy of img_bgr.
    Green = BV-kept, Red = BV-filtered.
    No text overlays.
    """
    img_ann   = img_bgr.copy()
    boxes     = yolo_result["det_boxes_xyxy"]
    det_cls   = yolo_result["det_cls"]
    det_conf  = yolo_result["det_conf"]
    kept_flags = slide_record["kept_flags"]

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        kept     = kept_flags[i] if i < len(kept_flags) else False
        color    = (0, 200, 0) if kept else (0, 0, 220)
        cv2.rectangle(img_ann,
                      (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    return img_ann


def build_prompt(yolo_result: dict,
                 slide_record: dict,
                 img_ann: np.ndarray,
                 pipeline_name: str = "",
                 uses_bv: bool = True) -> list:
    """
    Build the Gemini parts list [prompt_text, image_dict].
    Mirrors notebook's build_prompt() exactly.
    """
    raw_count       = slide_record["raw_count"]
    validated_count = slide_record["validated_count"]
    pred_slide      = slide_record["slide_verdict"]
    is_fn           = slide_record["is_false_negative"]
    kept_cls        = slide_record["kept_cls"]

    det_cls  = yolo_result["det_cls"]
    det_conf = yolo_result["det_conf"]
    flags    = slide_record["kept_flags"]

    # Per-detection lines
    det_lines = []
    for i in range(len(yolo_result["det_boxes_xyxy"])):
        cls_id   = int(det_cls[i]) if i < len(det_cls) else 0
        cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < NC else "unknown"
        conf     = float(det_conf[i]) if i < len(det_conf) else 0.0
        kept     = flags[i] if i < len(flags) else False
        
        if uses_bv:
            status   = "BV-KEPT (green box)" if kept else "BV-FILTERED (red box)"
        else:
            status   = "YOLO detection" if kept else "[not shown]"
        
        det_lines.append(
            f"    [{i}] {cls_name:<18}  yolo_conf={conf:.3f}  {status}")

    # Species / stage summary of kept detections
    species_kept, stage_kept = {}, {}
    for cls_id in kept_cls:
        name    = CLASS_NAMES[cls_id] if 0 <= cls_id < NC else "unknown"
        species = name.split("_")[0]
        stage   = name.split("_")[1] if "_" in name else "?"
        species_kept[species] = species_kept.get(species, 0) + 1
        stage_kept[stage]     = stage_kept.get(stage, 0) + 1

    species_str = ", ".join(f"{v}x {k}" for k, v in species_kept.items()) or "none"
    stage_str   = ", ".join(f"{v}x {k}" for k, v in stage_kept.items())   or "none"

    fn_warning = ""
    if is_fn:
        fn_warning = """
CRITICAL — FALSE NEGATIVE CASE:
The ground-truth label is INFECTED but YOLO detected ZERO parasites.
No boxes appear in the image. In section 2 (DETECTION QUALITY), focus
on what early-stage or low-density infection features YOLO likely missed
and why (small ring size, low contrast, sparse parasitaemia).
In section 3 (DETECTION EVALUATION), note any validation system performance.
In section 4 (CLINICAL VERDICT), flag this as a missed diagnosis requiring
manual microscopist review.\n"""

    # Conditionally format section 3 based on whether BV is used
    if uses_bv:
        section3_desc = f"— {raw_count} raw detections reduced to {validated_count} kept by Binary Validator.\n   {'Validator had no crops to evaluate — YOLO found nothing.' if is_fn else 'Describe filtering appropriateness, types of detections removed vs retained, and BV confidence scores.'}"
        bv_note = ""
    else:
        section3_desc = f"— all {validated_count} YOLO detections reported without additional validation.\n   {'No detections to evaluate.' if is_fn else 'Comment on detection confidence distribution and any borderline cases needing verification.'}"
        bv_note = "\nIMPORTANT: This model does NOT use a Binary Validator (BV). Do NOT mention BV or validation filtering in your analysis."
    
    # Summary line for detection info (conditional BV info)
    if uses_bv:
        det_summary = f"Raw YOLO detections      : {raw_count}\nBV-validated (kept)      : {validated_count}\nBV threshold             : {BV_THRESH}"
    else:
        det_summary = f"Total YOLO detections    : {validated_count}"

    prompt_text = f"""{SYSTEM_CONTEXT_WITH_BV if uses_bv else SYSTEM_CONTEXT_NO_BV}

--- PIPELINE: {pipeline_name} ---
Model prediction         : {pred_slide.upper()}
{det_summary}
Species detected (kept)  : {species_str}
Life stages (kept)       : {stage_str}{bv_note}

Per-detection breakdown:
{chr(10).join(det_lines) if det_lines else "    (no detections — blank slide)"}
{fn_warning}
1. SLIDE ASSESSMENT — describe the blood smear appearance and spatial
   distribution of detections (or explain blank appearance if no boxes).
   Comment on RBC morphology, staining quality, and parasite density.

2. DETECTION QUALITY — comment on the detected species and life stages.
   Are the classes consistent with expected morphology?
   Note missed regions, misclassifications, or unexpected findings.
   {"Focus on what was MISSED and why — this is a false negative case." if is_fn else ""}

3. DETECTION EVALUATION {section3_desc}

4. CLINICAL VERDICT — is the slide-level prediction ({pred_slide.upper()}) reliable?
   {"IMPORTANT: flag this as a false negative requiring urgent manual review." if is_fn else
    "What should a microscopist verify or double-check?"}

{FORMAT_RULES}""".strip()

    # Encode annotated image as JPEG base64
    ok, buf = cv2.imencode(".jpg", img_ann, [cv2.IMWRITE_JPEG_QUALITY, 90])
    parts = [prompt_text]
    if ok:
        parts.append({
            "mime_type": "image/jpeg",
            "data":      base64.b64encode(buf.tobytes()).decode("utf-8"),
        })

    return parts


def gemini_explain(parts: list,
                   retries: int = 3,
                   delay: float = 5.0) -> dict:
    """
    Call Gemini Flash and return parsed 4-section explanation.

    Returns:
        {
          "status":  "ok" | "unavailable" | "no_key",
          "raw_text": str,
          "sections": { "slide_assessment": str, ... }
        }
    """
    client = _get_client()
    if client is None:
        return {
            "status":   "no_key",
            "raw_text": _FALLBACK,
            "sections": _parse_sections(_FALLBACK),
        }

    # Build content list for new SDK
    # parts[0] = prompt text string, parts[1] = image dict (optional)
    prompt_text = parts[0]
    contents = [prompt_text]
    if len(parts) > 1 and isinstance(parts[1], dict):
        img_dict = parts[1]
        contents.append(
            types.Part.from_bytes(
                data=__import__("base64").b64decode(img_dict["data"]),
                mime_type=img_dict["mime_type"],
            )
        )

    # Strip "models/" prefix — new SDK uses bare model name
    model_name = GEMINI_MODEL.replace("models/", "")

    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
            )
            raw  = response.text
            text = _strip_preamble(raw)
            ok, missing = _validate_format(text)
            if not ok:
                log.warning(f"Gemini format warning — missing: {missing}")
            return {
                "status":   "ok",
                "raw_text": text,
                "sections": _parse_sections(text),
            }
        except Exception as e:
            log.warning(f"Gemini error attempt {attempt}/{retries}: {e}")
            if attempt < retries:
                time.sleep(delay)

    return {
        "status":   "unavailable",
        "raw_text": _FALLBACK,
        "sections": _parse_sections(_FALLBACK),
    }


def _parse_sections(text: str) -> dict:
    """Split the 4-section text into a dict keyed by section slug."""
    section_map = {
        "1. SLIDE ASSESSMENT":  "slide_assessment",
        "2. DETECTION QUALITY": "detection_quality",
        "3. DETECTION EVALUATION":  "detection_evaluation",
        "3. BV FILTER EFFECT":  "detection_evaluation",  # Legacy mapping for backwards compatibility
        "4. CLINICAL VERDICT":  "clinical_verdict",
        "5. RECOMMENDATIONS": "recommendations",  # Optional extra section, not used in UI
    }
    result  = {v: "" for v in set(section_map.values())}
    current = None

    for line in text.splitlines():
        stripped = line.strip()
        matched  = False
        for header, slug in section_map.items():
            if stripped.startswith(header):
                current = slug
                result[current] = stripped[len(header):].strip()
                matched = True
                break
        if not matched and current:
            result[current] += ("\n" + line) if result[current] else line

    return {k: v.strip() for k, v in result.items()}
