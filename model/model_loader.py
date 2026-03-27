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

CONF_THRESHOLD = 0.70

MEAN = [0.7216, 0.5765, 0.5725]
STD  = [0.1404, 0.1501, 0.1669]

PRECAUTIONS = {
    'No Cancer': {
        'message': 'No cancer detected. Skin appears normal.',
        'advice' : 'Continue regular skin checks every 3-6 months. Use sunscreen daily.',
        'urgency': 'None',
    },
    'Melanoma': {
        'message': 'Melanoma detected. This is the most dangerous type of skin cancer.',
        'advice' : 'Seek immediate dermatologist consultation. Do not delay. Melanoma spreads quickly.',
        'urgency': 'Immediate',
    },
    'Basal Cell Carcinoma': {
        'message': 'Basal Cell Carcinoma detected. This is the most common skin cancer.',
        'advice' : 'Consult a dermatologist soon. BCC rarely spreads but needs prompt treatment.',
        'urgency': 'Within 1 month',
    },
    'Actinic Keratosis': {
        'message': 'Actinic Keratosis detected. This is a precancerous lesion.',
        'advice' : 'See a dermatologist. AK can develop into SCC if left untreated. Avoid sun exposure.',
        'urgency': 'Within 2 weeks',
    },
    'Squamous Cell Carcinoma': {
        'message': 'Squamous Cell Carcinoma detected. This is a serious skin cancer.',
        'advice' : 'Urgent dermatologist visit needed. SCC can spread to lymph nodes if untreated.',
        'urgency': 'Urgent',
    },
}

# ── Model Architecture ─────────────────────────────────────────────────────────

def build_model(num_classes: int = 5) -> nn.Module:
    """Build DenseNet121 with custom classification head — same as training."""
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

    # Replace classifier with the same head used during training
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

def load_model(model_path: str = None) -> tuple[nn.Module, torch.device]:
    """
    Load the trained DenseNet121 model from .pth file.
    Returns (model, device).
    """
    if model_path is None:
        # Default: model file sits next to this script inside model/
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'skin_cancer_densenet_v2.pth')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device)

    model = build_model(num_classes=len(CLASSES))
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()

    return model, device

# ── Transform ─────────────────────────────────────────────────────────────────

def get_transform() -> transforms.Compose:
    """Preprocessing pipeline — matches validation/test transforms from training."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

# ── Prediction ────────────────────────────────────────────────────────────────

def predict(image: Image.Image, model: nn.Module, device: torch.device) -> dict:
    """
    Run inference on a PIL image.
    Returns a result dict with prediction, confidence, message, advice, urgency.
    If confidence < CONF_THRESHOLD, returns a 'Consult a Doctor' response.
    """
    transform = get_transform()

    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(tensor)                          # (1, 5)
        probs  = torch.softmax(logits, dim=1)[0]        # (5,)

    confidence, class_idx = probs.max(dim=0)
    confidence  = round(confidence.item(), 4)
    class_idx   = class_idx.item()
    class_name  = CLASSES[class_idx]

    # All class probabilities (for debugging / frontend display)
    all_probs = {cls: round(probs[i].item(), 4) for i, cls in enumerate(CLASSES)}

    # Confidence threshold check
    if confidence < CONF_THRESHOLD:
        return {
            'prediction'  : 'Uncertain',
            'confidence'  : confidence,
            'all_probs'   : all_probs,
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
        'urgency'     : precaution['urgency'],
        'message'     : precaution['message'],
        'advice'      : precaution['advice'],
        'is_uncertain': False,
    }
