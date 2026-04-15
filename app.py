"""
MALARION — Flask application entry point.

Run:
    python app.py                        # development
    gunicorn -w 1 -b 0.0.0.0:5000 app:app   # production (single worker — GPU shared state)
"""
import logging
import sys
from pathlib import Path

from flask import Flask, render_template

# ── Ensure project root is on sys.path ───────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from routes.predict import predict_bp
from routes.health  import health_bp
from models.loader  import missing_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("malarion")


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # ── Blueprints ────────────────────────────────────────────────────
    app.register_blueprint(predict_bp)
    app.register_blueprint(health_bp)

    # ── Index route ───────────────────────────────────────────────────
    @app.route("/")
    def index():
        return render_template("index.html")

    # ── Startup weight check ──────────────────────────────────────────
    with app.app_context():
        absent = missing_weights()
        if absent:
            log.warning("=" * 60)
            log.warning("  MISSING WEIGHT FILES — some pipelines will be unavailable")
            for m in absent:
                log.warning(
                    f"  Pipeline {m['model_id']} ({m['pipeline']}) "
                    f"→ missing {m['component'].upper()}: {m['filename']}"
                )
            log.warning("  Place .pt / .pth files in:  weights/")
            log.warning("=" * 60)
        else:
            log.info("All weight files present — all 5 pipelines ready.")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
