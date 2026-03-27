/**
 * DermaAI — Clinical Analysis Logic
 */
(() => {

    const API_BASE = 'https://sujal1207-dermaai.hf.space';

    const uploadSection  = document.getElementById('upload-section');
    const resultsSection = document.getElementById('results-section');
    const analyzeBtn     = document.getElementById('analyze-btn');
    const analyzeLabel   = document.getElementById('analyze-label');
    const analyzeSpinner = document.getElementById('analyze-spinner');
    const errorBox       = document.getElementById('error-box');
    const errorMsg       = document.getElementById('error-msg');

    let currentResult = null;

    function showError(msg) {
        errorBox.classList.remove('hidden');
        errorMsg.textContent = msg;
    }

    function hideError() {
        errorBox.classList.add('hidden');
    }

    function generateReportId() {
        return `DAI-${Math.floor(Math.random() * 90000) + 10000}`;
    }

    // ── Poll for heatmap ──────────────────────────────────────────────────────
    function pollHeatmap(jobId, maxAttempts = 30) {
        const heatImg = document.getElementById('heat-img');
        const heatLabel = document.getElementById('heat-status');

        let attempts = 0;
        const interval = setInterval(async () => {
            attempts++;
            try {
                const resp = await fetch(`${API_BASE}/api/heatmap/${jobId}`);
                const data = await resp.json();

                if (data.status === 'done') {
                    clearInterval(interval);
                    heatImg.src = `data:image/png;base64,${data.heatmap_b64}`;
                    heatImg.style.opacity = '1';
                    if (heatLabel) heatLabel.textContent = 'ACTIVATION_MAP_01';
                    // Update stored result with heatmap for PDF download
                    if (currentResult) currentResult.heatmap_b64 = data.heatmap_b64;
                } else if (data.status === 'error' || attempts >= maxAttempts) {
                    clearInterval(interval);
                    if (heatLabel) heatLabel.textContent = 'Heatmap unavailable';
                }
            } catch {
                // network error — keep polling
            }
        }, 1500);
    }

    // ── Render results ────────────────────────────────────────────────────────
    function renderResults(data) {
        const tStart = performance.now();

        let theme = 'low';
        if (data.risk_level === 'Medium') theme = 'medium';
        if (data.risk_level === 'High')   theme = 'high';

        document.getElementById('report-id').textContent   = `Report #${generateReportId()}`;
        document.getElementById('report-date').textContent = new Date().toLocaleDateString('en-US', {
            year: 'numeric', month: 'short', day: 'numeric'
        });

        // Original image — show immediately
        document.getElementById('orig-img').src = `data:image/png;base64,${data.original_b64}`;

        // Heatmap — show shimmer while loading
        const heatImg = document.getElementById('heat-img');
        heatImg.src   = '';
        heatImg.style.opacity = '0.3';
        const heatLabel = document.getElementById('heat-status');
        if (heatLabel) heatLabel.textContent = 'Generating heatmap…';

        // Prediction card
        const predCard = document.getElementById('prediction-card');
        predCard.className = `card prediction-card ${theme}`;

        document.getElementById('risk-badge').textContent       = `${data.risk_level} Risk`.toUpperCase();
        document.getElementById('pred-class-main').textContent  = data.prediction;

        const confNum = (data.confidence * 100).toFixed(1);
        document.getElementById('conf-pct').textContent = `${confNum}%`;

        document.getElementById('detail-class').textContent   = data.prediction;
        document.getElementById('detail-urgency').textContent = data.urgency;
        document.getElementById('diag-message').textContent   = data.message;
        document.getElementById('diag-advice').textContent    = data.advice;

        // Reveal results
        uploadSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });

        requestAnimationFrame(() => {
            document.getElementById('conf-fill').style.width = `${confNum}%`;
        });

        const duration = ((performance.now() - tStart + 400) / 1000).toFixed(1);
        document.getElementById('detail-time').textContent = `${duration}s`;

        if (window.loadDermatologists) window.loadDermatologists();
    }

    // ── Start Analysis ────────────────────────────────────────────────────────
    analyzeBtn.addEventListener('click', async () => {
        const file = window.getSelectedFile ? window.getSelectedFile() : null;
        if (!file) return;

        hideError();
        analyzeBtn.disabled = true;
        analyzeLabel.textContent = 'Analyzing...';
        analyzeSpinner.classList.remove('hidden');

        const form = new FormData();
        form.append('image', file);

        try {
            const resp = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: form });
            const data = await resp.json();

            if (!resp.ok || data.error) throw new Error(data.error || 'Server error occurred');

            currentResult = data;
            renderResults(data);

            // Start polling for heatmap
            if (data.heatmap_job_id) pollHeatmap(data.heatmap_job_id);

        } catch (err) {
            showError(err.message || 'Analysis failed. Make sure the server is running.');
        } finally {
            analyzeBtn.disabled = false;
            analyzeLabel.textContent = 'Start Clinical Analysis';
            analyzeSpinner.classList.add('hidden');
        }
    });

    // ── Download PDF ──────────────────────────────────────────────────────────
    document.getElementById('download-btn').addEventListener('click', async () => {
        if (!currentResult) return;
        const btn     = document.getElementById('download-btn');
        const label   = document.getElementById('dl-label');
        const spinner = document.getElementById('dl-spinner');

        btn.disabled = true;
        label.textContent = 'Generating PDF...';
        spinner.classList.remove('hidden');

        try {
            const resp = await fetch(`${API_BASE}/api/report`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(currentResult),
            });

            if (!resp.ok) throw new Error('PDF generation failed');

            const blob = await resp.blob();
            const url  = URL.createObjectURL(blob);
            const a    = document.createElement('a');
            a.href     = url;
            a.download = 'DermaAI_Clinical_Report.pdf';
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
        } catch (err) {
            alert('Report error: ' + err.message);
        } finally {
            btn.disabled = false;
            label.textContent = '📥 Download PDF Report';
            spinner.classList.add('hidden');
        }
    });

    // ── New Analysis ──────────────────────────────────────────────────────────
    document.getElementById('new-analysis-btn').addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        document.getElementById('conf-fill').style.width = '0%';
        if (window.resetUpload) window.resetUpload();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

})();
