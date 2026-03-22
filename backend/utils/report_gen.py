"""
PDF report generator for DermaAI.
Uses fpdf2 to produce a single-page clinical summary.
"""
from fpdf import FPDF, XPos, YPos
from datetime import datetime
import base64
import io
import tempfile
import os


RISK_COLOR = {
    "Low":    (34, 197, 94),    # green
    "Medium": (234, 179,  8),   # amber
    "High":   (239, 68,  68),   # red
}

DISCLAIMER = (
    "DISCLAIMER: DermaAI is an AI-assisted screening tool for informational "
    "purposes only. It is NOT a substitute for professional medical diagnosis. "
    "Always consult a certified dermatologist for clinical evaluation and "
    "treatment guidance."
)


class DermaReport(FPDF):
    def header(self):
        self.set_fill_color(15, 20, 40)
        self.rect(0, 0, 210, 30, "F")
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(100, 220, 255)
        self.set_y(8)
        self.cell(0, 12, "DermaAI  ·  Skin Risk Analysis Report", align="C")
        self.set_text_color(150, 160, 180)
        self.set_font("Helvetica", "", 9)
        self.set_y(20)
        self.cell(0, 6, f"Generated: {datetime.now().strftime('%B %d, %Y  %H:%M')}", align="C")
        self.ln(16)

    def footer(self):
        self.set_y(-20)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(120, 130, 150)
        self.multi_cell(0, 4, DISCLAIMER, align="C")


def generate_report(
    risk_level: str,
    confidence: float,
    heatmap_b64: str,
    original_b64: str,
) -> bytes:
    """
    Generate a PDF report and return raw bytes.
    """
    pdf = DermaReport(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.add_page()

    # ── Risk badge ────────────────────────────────────────────────────────
    r, g, b = RISK_COLOR.get(risk_level, (100, 100, 100))
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 26)
    pdf.cell(0, 18, f"Risk Level:  {risk_level.upper()}", align="C", fill=True)
    pdf.ln(6)

    # ── Confidence ────────────────────────────────────────────────────────
    pdf.set_text_color(30, 30, 60)
    pdf.set_font("Helvetica", "", 13)
    conf_pct = round(confidence * 100, 1)
    pdf.cell(0, 8, f"Model Confidence:  {conf_pct}%", align="C")
    pdf.ln(10)

    # ── Separating line ───────────────────────────────────────────────────
    pdf.set_draw_color(200, 210, 230)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)

    # ── Images: original + heatmap side by side ───────────────────────────
    def b64_to_temp_png(b64_str: str) -> str:
        data = base64.b64decode(b64_str)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(data)
        tmp.close()
        return tmp.name

    orig_path = b64_to_temp_png(original_b64)
    heat_path = b64_to_temp_png(heatmap_b64)

    img_y = pdf.get_y()
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(60, 70, 90)

    # Labels
    pdf.set_x(15)
    pdf.cell(88, 6, "Original Image", align="C")
    pdf.set_x(107)
    pdf.cell(88, 6, "Grad-CAM Heatmap", align="C")
    pdf.ln(7)

    img_y = pdf.get_y()
    pdf.image(orig_path, x=15, y=img_y, w=88, h=72)
    pdf.image(heat_path, x=107, y=img_y, w=88, h=72)
    pdf.set_y(img_y + 76)

    os.unlink(orig_path)
    os.unlink(heat_path)

    # ── Gradient bar ──────────────────────────────────────────────────────
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(60, 70, 90)
    pdf.cell(0, 6, "Risk Confidence Scale", align="C")
    pdf.ln(7)

    bar_x = 30
    bar_w = 150
    bar_h = 8
    bar_y = pdf.get_y()

    # Draw gradient bar segments
    segments = 100
    for i in range(segments):
        ratio = i / segments
        rv = int(34 + (239 - 34) * ratio)
        gv = int(197 + (68 - 197) * ratio)
        bv = int(94 + (68 - 94) * ratio)
        pdf.set_fill_color(rv, gv, bv)
        pdf.rect(bar_x + (bar_w * i / segments), bar_y, bar_w / segments + 0.2, bar_h, "F")

    # Confidence marker
    marker_x = bar_x + bar_w * confidence
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(marker_x - 1, bar_y - 2, 2, bar_h + 4, "F")
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(30, 30, 60)
    pdf.set_xy(marker_x - 8, bar_y + bar_h + 2)
    pdf.cell(16, 5, f"{conf_pct}%", align="C")

    # Labels
    pdf.set_xy(bar_x, bar_y + bar_h + 2)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(34, 197, 94)
    pdf.cell(25, 5, "Low Risk")
    pdf.set_text_color(239, 68, 68)
    pdf.set_x(bar_x + bar_w - 22)
    pdf.cell(25, 5, "High Risk", align="R")
    pdf.ln(16)

    # ── Analysis notes ────────────────────────────────────────────────────
    pdf.set_draw_color(200, 210, 230)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 20, 40)
    pdf.cell(0, 7, "Analysis Summary", align="L")
    pdf.ln(7)

    notes = {
        "Low": (
            "The analyzed skin region shows minimal irregular patterns. "
            "No significant risk indicators were detected. Continue regular "
            "self-monitoring and annual dermatologist check-ups."
        ),
        "Medium": (
            "Moderate risk patterns were detected in the highlighted region. "
            "Some irregular pigmentation or texture may be present. A professional "
            "dermatologist consultation within 2–4 weeks is recommended."
        ),
        "High": (
            "Significant risk indicators detected in the highlighted region. "
            "Irregular borders, color variation, or asymmetry may be present. "
            "Please consult a dermatologist as soon as possible for a clinical evaluation."
        ),
    }
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 60, 80)
    pdf.multi_cell(0, 6, notes.get(risk_level, ""), align="L")

    # Return bytes
    return bytes(pdf.output())
