"""GSD estimation inference utilities for Gradio application.

Provides functions to load and run the RegressionTreeCNN model for
Ground Sampling Distance estimation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torchvision.transforms import Normalize

if TYPE_CHECKING:
    from app.models import RegressionTreeCNN as RegressionTreeCNNType

try:
    from app.config import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        get_gsd_weights_path,
    )
    from app.models import RegressionTreeCNN
except ImportError:
    from config import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        get_gsd_weights_path,
    )
    from models import RegressionTreeCNN


# =============================================================================
# Inference utilities
# =============================================================================


def gsd_cm_to_m(gsd_cm: float) -> float:
    """Convert GSD from cm/pixel to m/pixel."""
    return gsd_cm / 100.0


def load_gsd_model(
    device: str | torch.device = "cpu",
) -> "RegressionTreeCNNType":
    """
    Load GSD estimation model from checkpoint.

    Downloads weights from Hugging Face Hub.

    Args:
        device: Device to load the model on.

    Returns:
        RegressionTreeCNN model in eval mode.
    """
    # Download weights from HF Hub
    checkpoint_path = get_gsd_weights_path()

    # Create model without pretrained weights (will load from checkpoint)
    model = RegressionTreeCNN(
        backbone="resnet101",
        num_exponent_classes=5,
        vector_dim=16,
        gsd_min=15.0,
        gsd_max=480.0,
        pretrained=False,
    )

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Checkpoint may contain 'model_state_dict', 'model' or be just state_dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def estimate_gsd(
    model: "RegressionTreeCNNType",
    image: torch.Tensor,
    device: str | torch.device = "cpu",
) -> float:
    """
    Estimate GSD (Ground Sampling Distance) for an image.

    Args:
        model: Loaded RegressionTreeCNN model.
        image: RGB image tensor [C, H, W] or [B, C, H, W], values in [0, 1].
        device: Device for inference.

    Returns:
        Estimated GSD in meters/pixel. Returns 0.3 (default) if estimation fails.
    """
    import math

    # Ensure batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)

    image = image.to(device)

    with torch.inference_mode():
        output = model(image)
        gsd_cm = output["gsd"].mean().item()

    # Validate model output
    if math.isnan(gsd_cm) or math.isinf(gsd_cm) or gsd_cm <= 0:
        return 0.3  # Return default value

    return gsd_cm_to_m(gsd_cm)


def get_gsd_transform() -> Normalize:
    """Get ImageNet normalization transform for GSD model."""
    return Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
