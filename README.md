# 🦟 MALARION — Flask Production App

Malaria parasite detection system converted from the research notebook into a
production-ready Flask application. Full pipeline preserved: YOLO inference →
BV filtering → Grad-CAM XAI → Gemini Flash clinical narrative.

---

## Weight files required

Place these files inside `weights/` before starting the server.

| Filename | Source notebook path | Used by |
|---|---|---|
| `best_malarion_v1.pt` | `models/best_malarion_v1.pt` | Pipelines 1, 2, 3 |
| `bv_resnet18_best.pth` | `models/bv_resnet18_best.pth` | Pipeline 2 |
| `bv_resnet18_hn_best.pth` | `models/bv_resnet18_hn_best.pth` | Pipelines 3, 5 |
| `c3_yolo_cbam_best.pt` | `models/c3_yolo_cbam/weights/best.pt` | Pipelines 4, 5 |

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set Gemini API key (optional — XAI narrative disabled if absent)
export GEMINI_API_KEY="your_key_here"

# 4. Place weight files in weights/

# 5. Run
python app.py
```

Open http://localhost:5000

---

## API reference

### `POST /api/predict`

**Form fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `image` | file | — | Blood smear slide (.jpg .png .tiff) |
| `model_id` | int 1–5 | 5 | Pipeline variant |
| `conf_thresh` | float | 0.25 | YOLO confidence threshold |
| `iou_thresh` | float | 0.45 | YOLO NMS IoU threshold |
| `bv_thresh` | float | 0.60 | BV sigmoid threshold |

**Response:**
```json
{
  "status": "ok",
  "model_id": 5,
  "pipeline_name": "YOLO CBAM + BV (hard-negative mining)",
  "slide_report": {
    "slide_verdict": "infected",
    "raw_count": 12,
    "validated_count": 8,
    "raw_count_per_class": [...],
    "validated_count_per_class": [...],
    "species_summary": {"falciparum": 5, "vivax": 3},
    "stage_summary": {"R": 4, "T": 3, "S": 1},
    "threshold_predictions": {
      "thresh_1": "infected",
      "thresh_2": "infected",
      "thresh_3": "infected"
    },
    "is_false_negative": false,
    "class_names": [...]
  },
  "detections": [
    {
      "index": 0,
      "box_xyxy": [120, 85, 210, 175],
      "class_id": 1,
      "class_name": "falciparum_T",
      "species": "falciparum",
      "stage": "T",
      "yolo_conf": 0.74,
      "bv_conf": 0.82,
      "bv_kept": true,
      "xai_category": "TP"
    }
  ],
  "images": {
    "annotated": "<base64 JPEG>",
    "yolo_gradcam": "<base64 JPEG>",
    "bv_gradcam_panels": [
      {
        "detection_index": 0,
        "class_name": "falciparum_T",
        "bv_conf": 0.82,
        "crop_original": "<base64>",
        "bv_gradcam": "<base64>"
      }
    ]
  },
  "gemini_xai": {"status": "pending"}
}
```

### `POST /api/explain`

Call after `/api/predict`. Accepts the predict response body as JSON.
Returns the Gemini 4-section clinical narrative.

```json
{
  "gemini_xai": {
    "status": "ok",
    "raw_text": "1. SLIDE ASSESSMENT\n...",
    "sections": {
      "slide_assessment": "...",
      "detection_quality": "...",
      "bv_filter_effect": "...",
      "clinical_verdict": "..."
    }
  }
}
```

### `GET /api/health`

Liveness probe. Returns device info.

### `GET /api/models`

Returns readiness status for all 5 pipelines, listing missing weight files.

---

## Project structure

```
malarion/
├── app.py                     Flask application factory
├── config.py                  All constants, thresholds, class names, pipeline registry
├── requirements.txt
├── weights/                   ← place .pt / .pth files here
│   ├── best_malarion_v1.pt
│   ├── bv_resnet18_best.pth
│   ├── bv_resnet18_hn_best.pth
│   └── c3_yolo_cbam_best.pt
├── models/
│   ├── architectures.py       ChannelAttention, SpatialAttention, CBAM, BinaryValidator
│   └── loader.py              Singleton registry — loads once at startup
├── inference/
│   ├── yolo_infer.py          run_yolo() + extract_crop()
│   ├── bv_infer.py            bv_predict() with exact ImageNet transform
│   └── slide_analysis.py      BV filtering loop, per-class counts, slide verdict
├── xai/
│   ├── gradcam_bv.py          GradCAM_BV  (ResNet18 layer4, standard Grad-CAM)
│   ├── gradcam_yolo.py        GradCAM_YOLO (neck layer, channel-mean activation)
│   └── gemini_xai.py          Gemini Flash prompt builder + 4-section parser
├── utils/
│   └── image_utils.py         encode_image_b64, draw_detections, overlay_heatmap
├── routes/
│   ├── predict.py             POST /api/predict  POST /api/explain
│   └── health.py              GET /api/health    GET /api/models
├── templates/
│   └── index.html             Single-page UI
└── static/
    ├── css/style.css
    └── js/app.js
```

---

## 16 detection classes

4 species × 4 life stages (R=Ring, T=Trophozoite, S=Schizont, G=Gametocyte):

| Species | Classes |
|---|---|
| Falciparum | falciparum_R, falciparum_T, falciparum_S, falciparum_G |
| Vivax | vivax_R, vivax_T, vivax_S, vivax_G |
| Ovale | ovale_R, ovale_T, ovale_S, ovale_G |
| Malariae | malariae_R, malariae_T, malariae_S, malariae_G |

---

## Production deployment

```bash
# Single worker required — GPU models are not thread-safe
gunicorn -w 1 -b 0.0.0.0:5000 --timeout 120 app:app
```
