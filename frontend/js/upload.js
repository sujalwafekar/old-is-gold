/**
 * DermaAI — Upload Zone Logic
 * Handles drag-and-drop, file selection, and preview.
 */

const dropZone      = document.getElementById('drop-zone');
const fileInput     = document.getElementById('file-input');
const browseBtn     = document.getElementById('browse-btn');
const dropContent   = document.getElementById('drop-content');
const previewContent= document.getElementById('preview-content');
const previewImg    = document.getElementById('preview-img');
const previewName   = document.getElementById('preview-name');
const analyzeBtn    = document.getElementById('analyze-btn');
const analyzeLabel  = document.getElementById('analyze-label');
const changeBtn     = document.getElementById('change-btn');

let selectedFile = null;

/* ── Helpers ─────────────────────────────────────────────── */
function showPreview(file) {
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewName.textContent = file.name;
  dropContent.classList.add('hidden');
  previewContent.classList.remove('hidden');
  analyzeBtn.disabled = false;
  analyzeLabel.textContent = '✦  Analyze Image';
}

function resetUpload() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  previewContent.classList.add('hidden');
  dropContent.classList.remove('hidden');
  analyzeBtn.disabled = true;
  analyzeLabel.textContent = 'Select an image first';
}

function validateFile(file) {
  const allowed = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/bmp'];
  if (!allowed.includes(file.type)) {
    showError('Unsupported file type. Please use PNG, JPG, WEBP, or BMP.');
    return false;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError('File is too large. Maximum size is 10 MB.');
    return false;
  }
  return true;
}

/* ── Browse button ───────────────────────────────────────── */
browseBtn.addEventListener('click', () => fileInput.click());
changeBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file && validateFile(file)) showPreview(file);
});

/* ── Drag and drop ───────────────────────────────────────── */
dropZone.addEventListener('click', (e) => {
  if (e.target === dropZone || dropContent.contains(e.target)) {
    if (!previewContent.contains(e.target)) fileInput.click();
  }
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

['dragleave', 'dragend'].forEach(evt =>
  dropZone.addEventListener(evt, () => dropZone.classList.remove('drag-over'))
);

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && validateFile(file)) showPreview(file);
});

/* ── New analysis reset ──────────────────────────────────── */
document.getElementById('new-analysis-btn').addEventListener('click', () => {
  document.getElementById('results-section').classList.add('hidden');
  document.getElementById('upload-section').classList.remove('hidden');
  resetUpload();
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

/* Export for analyze.js */
window.getSelectedFile = () => selectedFile;
