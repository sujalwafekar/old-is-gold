"""
DermaAI — Flask Backend
========================
Endpoints:
  POST /api/analyze          → image upload → risk + confidence + heatmap
  GET  /api/dermatologists   → mock nearby dermatologist data
  POST /api/report           → generate + download PDF report
  GET  /                     → serve frontend index.html
  GET  /<path>               → serve any frontend static file
"""

import os
import base64
from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import torch

from model.model_loader import load_model, logits_to_risk
from utils.gradcam import GradCAM, overlay_heatmap, image_to_b64
from utils.predictor import preprocess_image
from utils.report_gen import generate_report

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024   # 10 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ──────────────────────────────────────────────────────────────────────────────
# Load model once at startup
# ──────────────────────────────────────────────────────────────────────────────
print("⏳  Loading EfficientNet-B0 …")
model = load_model()
gradcam = GradCAM(model)
print("✅  Model ready.")


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
# API: Analyze image
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

        # 2. Inference (no-grad) to get logits for risk mapping
        with torch.no_grad():
            logits = model(tensor)

        risk_dict = logits_to_risk(logits)
        risk_level = risk_dict["risk"]
        confidence = risk_dict["display_confidence"]

        # 3. Grad-CAM (requires grad)
        tensor_grad = tensor.clone().requires_grad_(True)
        heatmap = gradcam.generate(tensor_grad)

        # 4. Overlay
        heatmap_b64 = overlay_heatmap(heatmap, pil_img)

        # 5. Encode original
        original_b64 = image_to_b64(pil_img)

        return jsonify({
            "risk_level": risk_level,
            "confidence": confidence,
            "true_label": risk_dict["true_label"],
            "prediction": risk_dict["prediction"],
            "urgency": risk_dict["urgency"],
            "message": risk_dict["message"],
            "advice": risk_dict["advice"],
            "heatmap_b64": heatmap_b64,
            "original_b64": original_b64,
        })

    except Exception as e:
        app.logger.exception("Error during analysis")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


# ──────────────────────────────────────────────────────────────────────────────
# API: Nearby dermatologists (mock data)
# ──────────────────────────────────────────────────────────────────────────────
MOCK_DERMATOLOGISTS = [
    {
        "name": "Dr. Priya Nair — Skin & Laser Clinic",
        "address": "12, MG Road, Bangalore, KA 560001",
        "phone": "+91 98765 43210",
        "rating": 4.8,
        "distance": "0.8 km",
        "lat": 12.9756,
        "lng": 77.6011,
        "open": True,
        "specialty": "Dermatology & Cosmetology",
    },
    {
        "name": "Apollo Skin Care Centre",
        "address": "Apollo Hospital, Koramangala, Bangalore",
        "phone": "+91 80 2658 0000",
        "rating": 4.6,
        "distance": "1.4 km",
        "lat": 12.9352,
        "lng": 77.6245,
        "open": True,
        "specialty": "Dermato-oncology",
    },
    {
        "name": "Dr. Arun Sharma — Dermatology & Surgery",
        "address": "45, Richmond Rd, Bangalore, KA 560025",
        "phone": "+91 94830 21045",
        "rating": 4.5,
        "distance": "2.1 km",
        "lat": 12.9612,
        "lng": 77.5955,
        "open": False,
        "specialty": "Skin Cancer & Biopsy",
    },
    {
        "name": "Manipal Dermatology Dept.",
        "address": "Manipal Hospital, Old Airport Rd, Bangalore",
        "phone": "+91 80 2502 4444",
        "rating": 4.7,
        "distance": "3.5 km",
        "lat": 12.9592,
        "lng": 77.6476,
        "open": True,
        "specialty": "Clinical Dermatology",
    },
    {
        "name": "SkinEssentials by Dr. Mehta",
        "address": "23, Brigade Rd, Bangalore, KA 560001",
        "phone": "+91 98400 77712",
        "rating": 4.4,
        "distance": "4.2 km",
        "lat": 12.9732,
        "lng": 77.6073,
        "open": True,
        "specialty": "Skin Screening & Mole Mapping",
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
    required = ["risk_level", "confidence", "heatmap_b64", "original_b64"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    try:
        pdf_bytes = generate_report(
            risk_level=data["risk_level"],
            confidence=float(data["confidence"]),
            heatmap_b64=data["heatmap_b64"],
            original_b64=data["original_b64"],
        )

        response = make_response(pdf_bytes)
        response.headers["Content-Type"] = "application/pdf"
        response.headers["Content-Disposition"] = "attachment; filename=DermaAI_Report.pdf"
        return response

    except Exception as e:
        app.logger.exception("Error generating report")
        return jsonify({"error": f"Report generation failed: {str(e)}"}), 500


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
