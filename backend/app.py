"""
DermaAI — Flask Backend
========================
Endpoints:
  POST /api/analyze          → fast prediction (heatmap async via job_id)
  GET  /api/heatmap/<job_id> → poll for GradCAM heatmap
  GET  /api/dermatologists   → nearby dermatologist data
  POST /api/report           → generate + download PDF report
  POST /api/chat             → chatbot (Gemini → NVIDIA → Ollama fallback)
  GET  /                     → serve frontend index.html
"""

import os

# Disable torch dynamo/compiler — prevents torchvision import errors on some PyTorch versions
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_BACKEND"] = "eager"

import uuid
import threading
import requests
from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
from dotenv import load_dotenv

from model.model_loader import load_model, predict, CLASSES
from utils.gradcam import GradCAM, overlay_heatmap, image_to_b64
from utils.predictor import preprocess_image
from utils.report_gen import generate_report

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

ALLOWED_EXTENSIONS    = {"png", "jpg", "jpeg", "bmp", "webp"}
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB

# ──────────────────────────────────────────────────────────────────────────────
# Load model once at startup
# ──────────────────────────────────────────────────────────────────────────────
print("⏳  Loading DenseNet121 skin-cancer model …")
model  = load_model()
device = next(model.parameters()).device
gradcam_engine = GradCAM(model)
print("✅  DenseNet121 model ready.")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ──────────────────────────────────────────────────────────────────────────────
# Frontend serving
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ──────────────────────────────────────────────────────────────────────────────
# Async GradCAM  (background thread + poll endpoint)
# ──────────────────────────────────────────────────────────────────────────────
_heatmap_jobs: dict = {}   # job_id → { status, heatmap_b64 }

def _run_gradcam(job_id: str, tensor, pil_img, class_idx):
    try:
        tensor_grad = tensor.clone().requires_grad_(True)
        heatmap     = gradcam_engine.generate(tensor_grad, class_idx=class_idx)
        heatmap_b64 = overlay_heatmap(heatmap, pil_img)
        _heatmap_jobs[job_id] = {"status": "done", "heatmap_b64": heatmap_b64}
    except Exception as e:
        _heatmap_jobs[job_id] = {"status": "error", "error": str(e)}


@app.route("/api/heatmap/<job_id>", methods=["GET"])
def get_heatmap(job_id):
    """Frontend polls this until status == 'done'."""
    job = _heatmap_jobs.get(job_id)
    if not job:
        return jsonify({"status": "pending"}), 202
    return jsonify(job)


# ──────────────────────────────────────────────────────────────────────────────
# API: Analyze image  (fast — prediction only, heatmap is async)
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use PNG, JPG, JPEG, BMP, or WEBP."}), 400

    try:
        file_bytes = file.read()

        # 1. Preprocess
        pil_img, tensor = preprocess_image(file_bytes)
        tensor = tensor.to(device)

        # 2. Fast prediction — no backward pass, sub-second
        result = predict(pil_img, model, device)

        # 3. Encode original image
        original_b64 = image_to_b64(pil_img)

        # 4. Kick off GradCAM in background (non-blocking)
        job_id    = str(uuid.uuid4())
        class_idx = CLASSES.index(result["prediction"]) if not result["is_uncertain"] else None
        _heatmap_jobs[job_id] = {"status": "pending"}
        threading.Thread(
            target=_run_gradcam,
            args=(job_id, tensor.detach(), pil_img, class_idx),
            daemon=True,
        ).start()

        # 5. Return prediction immediately
        return jsonify({
            "risk_level"     : result["risk_level"],
            "confidence"     : result["confidence"],
            "prediction"     : result["prediction"],
            "all_probs"      : result["all_probs"],
            "urgency"        : result["urgency"],
            "message"        : result["message"],
            "advice"         : result["advice"],
            "is_uncertain"   : result["is_uncertain"],
            "original_b64"   : original_b64,
            "heatmap_b64"    : None,      # arrives via polling
            "heatmap_job_id" : job_id,
        })

    except Exception as e:
        app.logger.exception("Error during analysis")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


# ──────────────────────────────────────────────────────────────────────────────
# API: Nearby dermatologists
# ──────────────────────────────────────────────────────────────────────────────
MOCK_DERMATOLOGISTS = [
    {
        "name": "Dr. Priya Nair — Skin & Laser Clinic",
        "address": "12, MG Road, Bangalore, KA 560001",
        "phone": "+91 98765 43210",
        "rating": 4.8, "distance": "0.8 km",
        "lat": 12.9756, "lng": 77.6011,
        "open": True, "specialty": "Dermatology & Cosmetology",
    },
    {
        "name": "Apollo Skin Care Centre",
        "address": "Apollo Hospital, Koramangala, Bangalore",
        "phone": "+91 80 2658 0000",
        "rating": 4.6, "distance": "1.4 km",
        "lat": 12.9352, "lng": 77.6245,
        "open": True, "specialty": "Dermato-oncology",
    },
    {
        "name": "Dr. Arun Sharma — Dermatology & Surgery",
        "address": "45, Richmond Rd, Bangalore, KA 560025",
        "phone": "+91 94830 21045",
        "rating": 4.5, "distance": "2.1 km",
        "lat": 12.9612, "lng": 77.5955,
        "open": False, "specialty": "Skin Cancer & Biopsy",
    },
    {
        "name": "Manipal Dermatology Dept.",
        "address": "Manipal Hospital, Old Airport Rd, Bangalore",
        "phone": "+91 80 2502 4444",
        "rating": 4.7, "distance": "3.5 km",
        "lat": 12.9592, "lng": 77.6476,
        "open": True, "specialty": "Clinical Dermatology",
    },
    {
        "name": "SkinEssentials by Dr. Mehta",
        "address": "23, Brigade Rd, Bangalore, KA 560001",
        "phone": "+91 98400 77712",
        "rating": 4.4, "distance": "4.2 km",
        "lat": 12.9732, "lng": 77.6073,
        "open": True, "specialty": "Skin Screening & Mole Mapping",
    },
]


@app.route("/api/dermatologists", methods=["GET"])
def dermatologists():
    return jsonify({"results": MOCK_DERMATOLOGISTS})


# ──────────────────────────────────────────────────────────────────────────────
# API: Generate PDF report
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/api/report", methods=["POST"])
def report():
    data = request.get_json(force=True)
    # heatmap_b64 may be None if user downloads before heatmap arrives
    heatmap_b64  = data.get("heatmap_b64") or ""
    original_b64 = data.get("original_b64", "")

    for field in ["risk_level", "confidence", "original_b64"]:
        if not data.get(field):
            return jsonify({"error": f"Missing field: {field}"}), 400

    try:
        pdf_bytes = generate_report(
            risk_level=data["risk_level"],
            confidence=float(data["confidence"]),
            heatmap_b64=heatmap_b64,
            original_b64=original_b64,
        )

        response = make_response(pdf_bytes)
        response.headers["Content-Type"] = "application/pdf"
        response.headers["Content-Disposition"] = "attachment; filename=DermaAI_Report.pdf"
        return response

    except Exception as e:
        app.logger.exception("Error generating report")
        return jsonify({"error": f"Report generation failed: {str(e)}"}), 500


# ──────────────────────────────────────────────────────────────────────────────
# API: Chatbot  (Gemini → NVIDIA NIM → Ollama fallback)
# ──────────────────────────────────────────────────────────────────────────────
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")


@app.route("/api/chat", methods=["POST"])
def chat():
    data         = request.get_json(force=True)
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    configs = [
        {
            "name": "Gemini",
            "url": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
            "headers": {"Content-Type": "application/json"},
            "payload": {"contents": [{"parts": [{"text": user_message}]}]},
        },
        {
            "name": "NVIDIA NIM (Kimi)",
            "url": "https://integrate.api.nvidia.com/v1/chat/completions",
            "headers": {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"},
            "payload": {
                "model": "moonshotai/kimi-k2.5",
                "messages": [{"role": "user", "content": user_message}],
                "max_tokens": 16384, "temperature": 1.0, "top_p": 1.0,
            },
        },
        {
            "name": "Ollama",
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}", "Content-Type": "application/json"},
            "payload": {
                "model": "deepseek/deepseek-r1:free",
                "messages": [{"role": "user", "content": user_message}],
                "max_tokens": 16384, "temperature": 1.0,
            },
        },
    ]

    for api in configs:
        app.logger.info(f"Attempting chat with: {api['name']}")
        try:
            resp = requests.post(api["url"], headers=api["headers"], json=api["payload"], timeout=30)
            if resp.status_code == 200:
                rj = resp.json()
                try:
                    if api["name"] == "Gemini":
                        reply = rj["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        reply = rj["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    reply = "Sorry, received an unexpected response format."
                return jsonify({"reply": reply, "provider": api["name"]})
            app.logger.warning(f"{api['name']} failed: {resp.status_code}")
        except requests.exceptions.RequestException as e:
            app.logger.error(f"Connection error with {api['name']}: {e}")

    return jsonify({"error": "All AI providers failed. Please try again later."}), 500


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
