import torch
from torchvision import transforms
from PIL import Image
import io


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def preprocess_image(file_bytes: bytes) -> tuple:
    """
    Loads image bytes → PIL Image + normalized tensor.
    Returns (pil_image, tensor)
    """
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = TRANSFORM(pil_img).unsqueeze(0)  # (1, 3, 224, 224)
    return pil_img, tensor


def run_inference(model, tensor: torch.Tensor):
    """
    Runs model forward pass and returns (logits, class_idx).
    Does NOT call backward — use GradCAM separately.
    """
    with torch.no_grad():
        logits = model(tensor)
    class_idx = logits.argmax(dim=1).item()
    return logits, class_idx
