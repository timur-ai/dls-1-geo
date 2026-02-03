"""Configuration constants for the building segmentation application."""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# =============================================================================
# Hugging Face Hub configuration
# =============================================================================

HF_SEGMENTATION_REPO: str = "MindForgeTim/building-segmentation"
HF_SEGMENTATION_FILE: str = "seg_final_model.pth"

HF_GSD_REPO: str = "MindForgeTim/dls-gsd-model"
HF_GSD_FILE: str = "gsd_model_resnet101.pth"


def download_weights(repo_id: str, filename: str) -> Path:
    """
    Download model weights from Hugging Face Hub.

    Uses standard HF cache directory (~/.cache/huggingface/hub) which is
    cross-platform and handles caching automatically.

    Args:
        repo_id: Hugging Face Hub repository ID (e.g., "user/repo").
        filename: Name of the file in the HF repository.

    Returns:
        Path to the downloaded weights file in HF cache.

    Raises:
        Exception: If download fails (network error, file not found, etc.).
    """
    logger.info("Loading weights from HF Hub: %s/%s", repo_id, filename)

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )
        logger.info("Weights loaded from: %s", downloaded_path)
        return Path(downloaded_path)
    except Exception:
        logger.exception("Failed to download weights from %s/%s", repo_id, filename)
        raise


def get_segmentation_weights_path() -> Path:
    """Get path to segmentation model weights, downloading if needed."""
    return download_weights(HF_SEGMENTATION_REPO, HF_SEGMENTATION_FILE)


def get_gsd_weights_path() -> Path:
    """Get path to GSD model weights, downloading if needed."""
    return download_weights(HF_GSD_REPO, HF_GSD_FILE)


# =============================================================================
# Tiling parameters
# =============================================================================

TILE_SIZE: int = 512  # Tile size for inference
TILE_OVERLAP: int = 64  # Overlap between tiles for seamless stitching

# =============================================================================
# ImageNet normalization (for pretrained models)
# =============================================================================

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# =============================================================================
# Inference parameters
# =============================================================================

# Ground Sampling Distance in meters per pixel (Inria dataset standard)
DEFAULT_GSD_M: float = 0.3

# Binarization threshold for segmentation (sigmoid output > threshold)
SEGMENTATION_THRESHOLD: float = 0.5

# Minimum component area in pixels (noise filtering)
MIN_BUILDING_AREA_PX: int = 100

# =============================================================================
# Visualization
# =============================================================================

# Overlay visualization settings
OVERLAY_COLOR: tuple[int, int, int] = (255, 0, 0)  # Red (RGB)
OVERLAY_ALPHA: float = 0.5  # 50% transparency

# =============================================================================
# Supported formats
# =============================================================================

# Supported image formats
SUPPORTED_IMAGE_FORMATS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tiff", ".tif")

# Mask values
MASK_BUILDING_VALUE: int = 255  # white = building
MASK_BACKGROUND_VALUE: int = 0  # black = background
