# DermaAI 🧬

**AI-powered skin risk detection with Grad-CAM explainability**

---

## Quick Start

```bash
# 1. Install dependencies (Python 3.10+ required)
cd backend
python -m pip install -r requirements.txt

# 2. Run the server
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## Project Structure

```
skin cancer/
├── backend/
│   ├── app.py              # Flask server (main entry point)
│   ├── requirements.txt    # Python dependencies
│   ├── model/
│   │   └── model_loader.py # EfficientNet-B0 + risk mapping
│   └── utils/
│       ├── gradcam.py      # Grad-CAM heatmap generator
│       ├── predictor.py    # Image preprocessing + inference
│       └── report_gen.py   # PDF report generator (fpdf2)
└── frontend/
    ├── index.html          # Landing page
    ├── analyze.html        # Analysis + results page
    ├── css/
    │   ├── index.css
    │   └── analyze.css
    └── js/
        ├── upload.js       # Drag-and-drop upload
        ├── analyze.js      # API calls + result rendering
        └── map.js          # Leaflet.js dermatologist map
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze` | Upload image → risk + heatmap |
| GET | `/api/dermatologists` | Mock nearby dermatologist data |
| POST | `/api/report` | Generate downloadable PDF |

---

## Features

- 🔬 **EfficientNet-B0** deep learning model (ImageNet pretrained)
- 🗺️ **Grad-CAM** visual explainability heatmap
- 📊 **3-tier risk classification**: Low / Medium / High
- 📍 **Nearby dermatologist finder** (Leaflet.js map)
- 📄 **Downloadable PDF report** with analysis summary
- ⚡ **Real-time** — analysis completes in under 3 seconds

> **Disclaimer:** DermaAI is for screening purposes only and does not constitute medical advice.
