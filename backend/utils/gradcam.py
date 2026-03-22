import numpy as np
import torch
import cv2
import base64
from PIL import Image
import io


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for EfficientNet-B0.
    Hooks into the last convolutional block to produce a heatmap.
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # EfficientNet-B0: last conv layer is in features[-1][0]
        target_layer = self.model.features[-1][0]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Returns: base64-encoded PNG of heatmap blended over original image.
        input_tensor: (1, 3, 224, 224) normalized tensor
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backprop for the target class
        score = output[0, class_idx]
        score.backward()

        # Pool gradients across spatial dims
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])  # (C,)
        activations = self.activations[0]                  # (C, H, W)

        # Weight activation maps
        for i in range(activations.shape[0]):
            activations[i] *= pooled_grads[i]

        heatmap = activations.mean(dim=0).numpy()          # (H, W)
        heatmap = np.maximum(heatmap, 0)

        # Normalize
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap


def overlay_heatmap(heatmap: np.ndarray, original_pil: Image.Image) -> str:
    """
    Blends Grad-CAM heatmap over original image.
    Returns base64-encoded PNG string.
    """
    orig_np = np.array(original_pil.resize((224, 224)).convert("RGB"))

    # Resize heatmap to image dimensions
    heatmap_resized = cv2.resize(heatmap, (224, 224))

    # Apply JET colormap
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend
    blended = cv2.addWeighted(orig_np, 0.55, heatmap_color, 0.45, 0)

    # Draw a semi-transparent annotation circle at heat centroid
    yx = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
    cy, cx = int(yx[0]), int(yx[1])
    cv2.circle(blended, (cx, cy), 18, (255, 255, 255), 2)

    # Encode to base64 PNG
    pil_blended = Image.fromarray(blended)
    buf = io.BytesIO()
    pil_blended.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64


def image_to_b64(pil_img: Image.Image) -> str:
    """Encode a PIL image to base64 PNG string."""
    pil_img = pil_img.resize((224, 224)).convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
