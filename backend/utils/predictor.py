import torch
from PIL import Image
import io

from model.model_loader import get_transform


def preprocess_image(file_bytes: bytes) -> tuple:
    """
    Loads image bytes → PIL Image + normalized tensor (skin-cancer stats).
    Returns (pil_image, tensor)
    """
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    transform = get_transform()
    tensor = transform(pil_img).unsqueeze(0)   # (1, 3, 224, 224)
    return pil_img, tensor
