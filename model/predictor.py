import torch
from torchvision import transforms
from PIL import Image
import io


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.7216, 0.5765, 0.5725],
        std=[0.1404, 0.1501, 0.1669]
    ),
])


def preprocess_image(file_bytes: bytes) -> tuple:
    """
    Loads image bytes → PIL Image + normalized tensor.
    Returns (pil_image, tensor)
    """
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = TRANSFORM(pil_img).unsqueeze(0)  # (1, 3, 224, 224)
    return pil_img, tensor
