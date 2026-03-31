from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .model import choose_device, prepare_image


def generate_gradcam(
    image_path: str | Path,
    model: nn.Module,
    device: str | torch.device | None = None,
    target_class: int | None = None,
) -> np.ndarray:
    """Generate a Grad-CAM heatmap for the given image and model.

    Returns an RGB numpy array (H, W, 3) with the heatmap overlaid on the
    original image, suitable for saving directly as a PNG.
    """
    resolved_device = device if isinstance(device, torch.device) else choose_device(str(device or "auto"))
    inputs = prepare_image(image_path).unsqueeze(0).to(resolved_device)

    # Identify the last conv layer — works for both model layouts
    if hasattr(model, "backbone"):
        target_layer = model.backbone.layer4[-1]
    else:
        target_layer = model.layer4[-1]

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.eval()
        logits = model(inputs)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        model.zero_grad()
        logits[0, target_class].backward()

        # Pool gradients across spatial dims -> channel weights
        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations[0]).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().detach().numpy()

        # Normalize to 0-1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize to original image size and overlay
        with Image.open(image_path) as orig:
            orig_rgb = orig.convert("RGB")
            w, h = orig_rgb.size

        cam_resized = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        ).astype(np.float32) / 255.0

        # Apply colormap (jet-style: blue -> green -> red)
        heatmap = np.zeros((h, w, 3), dtype=np.float32)
        heatmap[..., 0] = np.clip(cam_resized * 2.0, 0, 1)
        heatmap[..., 1] = np.clip(2.0 * cam_resized - 0.5, 0, 1)
        heatmap[..., 2] = np.clip(1.0 - cam_resized * 2.0, 0, 1)

        orig_array = np.array(orig_rgb).astype(np.float32) / 255.0
        blended = 0.55 * orig_array + 0.45 * heatmap
        blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

        return blended
    finally:
        fwd_handle.remove()
        bwd_handle.remove()


def save_gradcam(
    image_path: str | Path,
    model: nn.Module,
    target_path: Path,
    device: str | torch.device | None = None,
    target_class: int | None = None,
) -> Path:
    """Generate and save a Grad-CAM heatmap overlay to target_path."""
    heatmap = generate_gradcam(image_path, model, device=device, target_class=target_class)
    Image.fromarray(heatmap).save(target_path)
    return target_path
