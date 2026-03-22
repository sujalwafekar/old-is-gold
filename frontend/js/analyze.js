/**
 * DermaAI — Analysis orchestrator
 * Sends image to backend, renders results.
 */

const API_BASE = 'http://127.0.0.1:5000';

const aBtn = document.getElementById('analyze-btn');
const aLabel = document.getElementById('analyze-label');
const analyzeSpinner = document.getElementById('analyze-spinner');
const errorBox = document.getElementById('error-box');
const errorMsg = document.getElementById('error-msg');
const uploadSection = document.getElementById('upload-section');
const resultsSection = document.getElementById('results-section');

// Result elements
const riskBannerInner = document.getElementById('risk-banner').querySelector('.risk-banner-inner') || null;
const riskLevelText = document.getElementById('risk-level-text');
const riskIcon = document.getElementById('risk-icon');
const riskAction = document.getElementById('risk-action');
const confPct = document.getElementById('conf-pct');
const confFill = document.getElementById('conf-fill');
const origImg = document.getElementById('orig-img');
const heatImg = document.getElementById('heat-img');
const downloadBtn = document.getElementById('download-btn');
const dlLabel = document.getElementById('dl-label');
const dlSpinner = document.getElementById('dl-spinner');

// Stored result for PDF generation
let currentResult = null;

/* ── Helpers ─────────────────────────────────────────────── */
function showError(msg) {
    errorBox.classList.remove('hidden');
    errorMsg.textContent = msg;
}
function hideError() { errorBox.classList.add('hidden'); }

const RISK_META = {
    Low: {
        icon: '🟢',
        cls: 'low',
        action: 'Low risk detected. Continue regular self-monitoring and annual dermatologist check-ups.'
    },
    Medium: {
        icon: '🟡',
        cls: 'medium',
        action: 'Moderate risk patterns found. A dermatologist consultation within 2–4 weeks is recommended.'
    },
    High: {
        icon: '🔴',
        cls: 'high',
        action: 'Significant risk indicators detected. Please consult a dermatologist as soon as possible.'
    },
};

function renderBanner(risk, confidence) {
    const meta = RISK_META[risk] || RISK_META['Low'];
    const banner = document.getElementById('risk-banner');
    const inner = banner.querySelector('.risk-banner-inner') ||
        (() => { const d = document.createElement('div'); d.className = 'risk-banner-inner'; banner.appendChild(d); return d; })();

    // Build inner content
    inner.innerHTML = `
    <div class="risk-label-wrap">
      <span class="risk-icon" id="risk-icon">${meta.icon}</span>
      <div>
        <div class="risk-super">Risk Classification</div>
        <div class="risk-level-text" id="risk-level-text">${risk}</div>
      </div>
    </div>
    <div class="confidence-wrap">
      <div class="conf-label">
        <span>Model Confidence</span>
        <span class="conf-pct" id="conf-pct">0%</span>
      </div>
      <div class="conf-track"><div class="conf-fill" id="conf-fill"></div></div>
    </div>
    <div class="risk-action-text" id="risk-action">${meta.action}</div>
  `;

    // Remove old risk classes
    inner.classList.remove('low', 'medium', 'high');
    inner.classList.add(meta.cls);

    // Animate confidence bar after render
    requestAnimationFrame(() => {
        const fill = document.getElementById('conf-fill');
        const pct = document.getElementById('conf-pct');
        if (fill) {
            setTimeout(() => { fill.style.width = `${Math.round(confidence * 100)}%`; }, 80);
        }
        if (pct) {
            animateCounter(pct, 0, Math.round(confidence * 100), 1200, v => v + '%');
        }
    });
}

function animateCounter(el, from, to, duration, fmt = v => v) {
    const start = performance.now();
    const step = (now) => {
        const p = Math.min((now - start) / duration, 1);
        const val = Math.round(from + (to - from) * easeOut(p));
        el.textContent = fmt(val);
        if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
}

function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

/* ── Analyze ─────────────────────────────────────────────── */
aBtn.addEventListener('click', async () => {
    const file = window.getSelectedFile();
    if (!file) return;

    hideError();
    aBtn.disabled = true;
    aLabel.textContent = 'Analyzing …';
    analyzeSpinner.classList.remove('hidden');

    const form = new FormData();
    form.append('image', file);

    try {
        const resp = await fetch(`${API_BASE}/api/analyze`, {
            method: 'POST',
            body: form,
        });

        const data = await resp.json();

        if (!resp.ok || data.error) {
            throw new Error(data.error || 'Unknown server error');
        }

        currentResult = data;

        // Render results
        renderBanner(data.risk_level, data.confidence);
        origImg.src = `data:image/png;base64,${data.original_b64}`;
        heatImg.src = `data:image/png;base64,${data.heatmap_b64}`;

        // Populate Detailed Readout
        document.getElementById('ro-true-label').textContent = data.true_label || 'N/A';
        document.getElementById('ro-prediction').textContent = data.prediction || 'Unknown';
        document.getElementById('ro-confidence').textContent = (data.confidence * 100).toFixed(2) + '%';
        document.getElementById('ro-urgency').textContent = data.urgency || 'None';
        document.getElementById('ro-message').textContent = data.message || 'No additional message.';
        document.getElementById('ro-advice').textContent = data.advice || 'Follow standard skin care guidelines.';

        // Show results, hide upload
        uploadSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });

    } catch (err) {
        showError(err.message || 'Analysis failed. Is the server running?');
    } finally {
        aBtn.disabled = false;
        aLabel.textContent = '✦  Analyze Image';
        analyzeSpinner.classList.add('hidden');
    }
});

/* ── Download PDF ────────────────────────────────────────── */
downloadBtn.addEventListener('click', async () => {
    if (!currentResult) return;

    downloadBtn.disabled = true;
    dlLabel.textContent = 'Generating PDF …';
    dlSpinner.classList.remove('hidden');

    try {
        const resp = await fetch(`${API_BASE}/api/report`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentResult),
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.error || 'PDF generation failed');
        }

        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'DermaAI_Report.pdf';
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);

    } catch (err) {
        alert('Report error: ' + err.message);
    } finally {
        downloadBtn.disabled = false;
        dlLabel.textContent = 'Download PDF Report';
        dlSpinner.classList.add('hidden');
    }
});
