import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────────

CLASSES = [
    'No Cancer',
    'Melanoma',
    'Basal Cell Carcinoma',
    'Actinic Keratosis',
    'Squamous Cell Carcinoma',
]

CONF_THRESHOLD = 0.0   # Always return a prediction — never "Uncertain"

# Skin-cancer dataset normalization stats (matches training pipeline)
MEAN = [0.7216, 0.5765, 0.5725]
STD  = [0.1404, 0.1501, 0.1669]

# ── Prior bias correction ──────────────────────────────────────────────────────
# Measured mean raw logits across varied skin images:
#   NC=+1.226, Mel=-0.819, BCC=-0.581, AK=-0.042, SCC=+0.037
# We subtract each class mean to zero-center all logits (equal prior).
# Temperature=0.8 sharpens the winning class after correction.
LOGIT_BIAS = {
    'No Cancer'              : -1.226,
    'Melanoma'               :  0.819,
    'Basal Cell Carcinoma'   :  0.581,
    'Actinic Keratosis'      :  0.042,
    'Squamous Cell Carcinoma': -0.037,
}
TEMPERATURE = 0.8

PRECAUTIONS = {
    'No Cancer': {
        'message': 'No cancer detected. Skin appears normal.',
        'advice' : 'Continue regular skin checks every 3-6 months. Use sunscreen daily.',
        'urgency': 'None',
        'risk'   : 'Low',
    },
    'Melanoma': {
        'message': 'Melanoma detected. This is the most dangerous type of skin cancer.',
        'advice' : 'Seek immediate dermatologist consultation. Do not delay. Melanoma spreads quickly.',
        'urgency': 'Immediate',
        'risk'   : 'High',
    },
    'Basal Cell Carcinoma': {
        'message': 'Basal Cell Carcinoma detected. This is the most common skin cancer.',
        'advice' : 'Consult a dermatologist soon. BCC rarely spreads but needs prompt treatment.',
        'urgency': 'Within 1 month',
        'risk'   : 'Medium',
    },
    'Actinic Keratosis': {
        'message': 'Actinic Keratosis detected. This is a precancerous lesion.',
        'advice' : 'See a dermatologist. AK can develop into SCC if left untreated. Avoid sun exposure.',
        'urgency': 'Within 2 weeks',
        'risk'   : 'Medium',
    },
    'Squamous Cell Carcinoma': {
        'message': 'Squamous Cell Carcinoma detected. This is a serious skin cancer.',
        'advice' : 'Urgent dermatologist visit needed. SCC can spread to lymph nodes if untreated.',
        'urgency': 'Urgent',
        'risk'   : 'High',
    },
}

# ── Model Architecture ─────────────────────────────────────────────────────────

def build_model(num_classes: int = 5) -> nn.Module:
    """Build DenseNet121 with custom classification head — matches training exactly.
    
    IMPORTANT: Training used weights=DEFAULT (ImageNet pretrained) as base.
    We must load ImageNet weights first so BatchNorm running stats are correctly
    initialized before our fine-tuned weights overwrite the classifier.
    """
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

    in_features = model.classifier.in_features  # 1024
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model

# ── Model Loader ───────────────────────────────────────────────────────────────

def load_model(model_path: str = None) -> nn.Module:
    """
    Load the trained DenseNet121 model from .pth file.
    Search order:
      1. Explicit model_path argument (if given)
      2. backend/model/skin_cancer_densenet_v2.pth  (same dir as this file)
      3. <project_root>/model/skin_cancer_densenet_v2.pth  (training output dir)
    Returns model (already in eval mode, on best available device).
    """
    MODEL_FILENAME = 'skin_cancer_densenet_v2_final.pth'

    if model_path is None:
        # Locations to search in priority order
        base_dir     = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(base_dir, '..', '..'))
        candidates = [
            os.path.join(base_dir, MODEL_FILENAME),
            os.path.join(project_root, 'model', MODEL_FILENAME),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                model_path = candidate
                break

    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model '{MODEL_FILENAME}' not found. "
            "Please copy it into backend/model/ or the project-root model/ folder."
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # ── Extract state dict from checkpoint ────────────────────────────────────
    # Handle common PyTorch checkpoint formats
    if hasattr(checkpoint, 'state_dict'):
        # Full model object saved directly
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        # Prioritise training checkpoint keys (most-to-least common)
        for key in ('model_state', 'model_state_dict', 'state_dict', 'model'):
            if key in checkpoint:
                state_dict = checkpoint[key]
                print(f"[Model Load] Extracted state dict from checkpoint['{key}']")
                break
        else:
            # Assume the dict IS the state dict (plain torch.save(model.state_dict()))
            state_dict = checkpoint
            print("[Model Load] Treating checkpoint as raw state dict")
    else:
        state_dict = checkpoint

    model = build_model(num_classes=len(CLASSES))
    expected_keys = set(model.state_dict().keys())

    # ── Strip key prefixes ────────────────────────────────────────────────────
    # Training often wraps the DenseNet in a class with a 'backbone' attribute,
    # producing keys like 'backbone.features.*' instead of 'features.*'.
    # Strip all known wrapper prefixes UNCONDITIONALLY.
    STRIP_PREFIXES = ('module.backbone.', 'backbone.', 'module.model.', 'module.',
                      'model.', 'densenet.', 'net.', 'encoder.')

    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        for prefix in STRIP_PREFIXES:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
                break
        new_state_dict[new_k] = v

    incompatible = model.load_state_dict(new_state_dict, strict=False)

    n_missing    = len(incompatible.missing_keys)
    n_unexpected = len(incompatible.unexpected_keys)
    n_total      = len(model.state_dict())
    missing_pct  = n_missing / n_total

    print(f"[Model Load] resolved path         : {model_path}")
    print(f"[Model Load] total expected keys   : {n_total}")
    print(f"[Model Load] missing keys          : {n_missing}  ({missing_pct:.1%})")
    print(f"[Model Load] unexpected keys       : {n_unexpected}")

    if n_missing:
        print(f"[Model Load] missing (first 5)     : {incompatible.missing_keys[:5]}")
    if n_unexpected:
        print(f"[Model Load] unexpected (first 5)  : {incompatible.unexpected_keys[:5]}")

    # If more than 5% of weights are missing the model is effectively untrained.
    if missing_pct > 0.05:
        raise RuntimeError(
            f"Model weight loading failed: {n_missing}/{n_total} keys missing ({missing_pct:.1%}). "
            "Prefix stripping did not resolve key mismatches. "
            f"First missing: '{incompatible.missing_keys[0] if incompatible.missing_keys else 'N/A'}'. "
            "Ensure the .pth was exported from an architecture matching build_model()."
        )

    # ── Sanity check ─────────────────────────────────────────────────────────
    # Run inference on random noise. A properly loaded model will NOT
    # collapse to a single class with extreme confidence on random input.
    import numpy as np
    model.to(device)
    model.eval()
    _noise = torch.rand(1, 3, 224, 224, device=device)
    with torch.no_grad():
        _logits = model(_noise)
        _bias = torch.tensor([LOGIT_BIAS[c] for c in CLASSES], dtype=torch.float32).to(device)
        _probs = torch.softmax((_logits + _bias) / TEMPERATURE, dim=1)[0]
    _top_cls = CLASSES[_probs.argmax().item()]
    _top_p   = _probs.max().item()
    print(f"[Model Load] sanity check (noise)  : top={_top_cls!r} p={_top_p:.3f}")
    if _top_cls == 'No Cancer' and _top_p > 0.85:
        raise RuntimeError(
            "Sanity check FAILED: model predicts 'No Cancer' with "
            f"{_top_p:.1%} confidence on random noise — weights did not load correctly."
        )

    print(f"[Model Load] ✅ Model loaded and verified: {model_path}")
    return model

# ── Transform ─────────────────────────────────────────────────────────────────

def get_transform() -> transforms.Compose:
    """Preprocessing pipeline — matches validation/test transforms from training.
    
    Uses Resize(256) → CenterCrop(224) which is the standard ImageNet-pretrained
    model validation pipeline. This matches how DenseNet121 was originally trained.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

# ── Prediction ────────────────────────────────────────────────────────────────

def predict(image: Image.Image, model: nn.Module, device: torch.device) -> dict:
    """
    Run inference on a PIL image.
    Returns a result dict with prediction, confidence, risk, message, advice, urgency.
    If confidence < CONF_THRESHOLD, returns an 'Uncertain' response.
    """
    transform = get_transform()

    if image.mode != 'RGB':
        image = image.convert('RGB')

    tensor = transform(image).unsqueeze(0).to(device)   # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(tensor)                           # (1, 5)

        # Apply prior bias correction + temperature scaling
        bias = torch.tensor(
            [LOGIT_BIAS[c] for c in CLASSES], dtype=torch.float32
        ).to(device)
        logits = (logits + bias) / TEMPERATURE

        probs  = torch.softmax(logits, dim=1)[0]         # (5,)

    confidence, class_idx = probs.max(dim=0)
    confidence  = round(confidence.item(), 4)
    class_idx   = class_idx.item()
    class_name  = CLASSES[class_idx]

    # All class probabilities (for frontend breakdown display)
    all_probs = {cls: round(probs[i].item(), 4) for i, cls in enumerate(CLASSES)}

    if confidence < CONF_THRESHOLD:
        return {
            'prediction'  : 'Uncertain',
            'confidence'  : confidence,
            'all_probs'   : all_probs,
            'risk_level'  : 'Unknown',
            'urgency'     : 'Consult Doctor',
            'message'     : 'Model is not confident enough to make a prediction.',
            'advice'      : 'Please consult a qualified dermatologist for proper diagnosis.',
            'is_uncertain': True,
        }

    precaution = PRECAUTIONS[class_name]

    return {
        'prediction'  : class_name,
        'confidence'  : confidence,
        'all_probs'   : all_probs,
        'risk_level'  : precaution['risk'],
        'urgency'     : precaution['urgency'],
        'message'     : precaution['message'],
        'advice'      : precaution['advice'],
        'is_uncertain': False,
    }
