"""Inference utilities for building segmentation application."""

from __future__ import annotations

import threading
from typing import Literal

import cv2
import numpy as np
import torch

try:
    from app.config import (
        DEFAULT_GSD_M,
        MIN_BUILDING_AREA_PX,
        SEGMENTATION_THRESHOLD,
        TILE_OVERLAP,
        TILE_SIZE,
        get_segmentation_weights_path,
    )
    from app.models import UNet
    from app.preprocessing import (
        create_tiles,
        image_to_tensor,
        merge_tiles,
        prepare_batch_for_model,
        resize_image,
    )
except ImportError:
    from config import (
        DEFAULT_GSD_M,
        MIN_BUILDING_AREA_PX,
        SEGMENTATION_THRESHOLD,
        TILE_OVERLAP,
        TILE_SIZE,
        get_segmentation_weights_path,
    )
    from models import UNet
    from preprocessing import (
        create_tiles,
        image_to_tensor,
        merge_tiles,
        prepare_batch_for_model,
        resize_image,
    )

# =============================================================================
# Global model cache (singleton pattern for lazy loading)
# =============================================================================

_seg_model: torch.nn.Module | None = None
_device: torch.device | None = None
_model_lock = threading.Lock()


# =============================================================================
# Device utilities
# =============================================================================


def get_device() -> torch.device:
    """
    Get the best available device (CUDA if available, else CPU).

    Returns:
        torch.device for model inference.
    """
    global _device
    if _device is None:
        with _model_lock:
            # Double-check after acquiring lock
            if _device is None:
                _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


# =============================================================================
# Model loading
# =============================================================================


def load_segmentation_model(
    device: torch.device | None = None,
) -> torch.nn.Module:
    """
    Load UNet segmentation model from checkpoint.

    Downloads weights from Hugging Face Hub if not available locally.

    Args:
        device: Device to load model on. If None, uses get_device().

    Returns:
        UNet model in eval mode.
    """
    global _seg_model

    if _seg_model is not None:
        return _seg_model

    with _model_lock:
        # Double-check after acquiring lock
        if _seg_model is not None:
            return _seg_model

        if device is None:
            device = get_device()

        # Download weights from HF Hub
        checkpoint_path = get_segmentation_weights_path()

        # Create model without pretrained weights (will load from checkpoint)
        model = UNet(num_classes=1, pretrained_encoder=False)

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Checkpoint may contain 'model_state_dict' or be just state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        _seg_model = model
        return model


def get_segmentation_model() -> torch.nn.Module:
    """Get cached segmentation model, loading if necessary."""
    global _seg_model
    if _seg_model is None:
        _seg_model = load_segmentation_model()
    return _seg_model


# =============================================================================
# Single tile inference
# =============================================================================


def _predict_single_tile(
    model: torch.nn.Module,
    tile: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Run inference on a single tile.

    Args:
        model: Loaded segmentation model.
        tile: RGB tile array (H, W, 3).
        device: Device for inference.

    Returns:
        Probability map (H, W) as float32 in [0, 1].
    """
    # Convert to tensor (1, 3, H, W), normalized
    tensor = image_to_tensor(tile, normalize=True).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.sigmoid(logits)

    # Back to numpy (H, W)
    prob_map = probs[0, 0].cpu().numpy()
    return prob_map


def _predict_batch(
    model: torch.nn.Module,
    tiles: list[np.ndarray],
    device: torch.device,
    batch_size: int = 8,
) -> list[np.ndarray]:
    """
    Run inference on multiple tiles in batches.

    Args:
        model: Loaded segmentation model.
        tiles: List of RGB tile arrays.
        device: Device for inference.
        batch_size: Number of tiles per batch.

    Returns:
        List of probability maps (H, W) as float32.
    """
    if not tiles:
        return []

    prob_maps = []

    for i in range(0, len(tiles), batch_size):
        batch_tiles = tiles[i : i + batch_size]
        tensor = prepare_batch_for_model(batch_tiles, normalize=True).to(device)

        with torch.inference_mode():
            logits = model(tensor)
            probs = torch.sigmoid(logits)

        # Convert to numpy
        for j in range(probs.shape[0]):
            prob_map = probs[j, 0].cpu().numpy()
            prob_maps.append(prob_map)

    return prob_maps


# =============================================================================
# Main prediction function
# =============================================================================


def predict_mask(
    image: np.ndarray,
    mode: Literal["tiling", "resize"] = "tiling",
    tile_size: int = TILE_SIZE,
    overlap: int = TILE_OVERLAP,
    threshold: float = SEGMENTATION_THRESHOLD,
    batch_size: int = 8,
) -> np.ndarray:
    """
    Predict building segmentation mask from input image.

    Args:
        image: RGB image array of shape (H, W, 3).
        mode: Processing mode - 'tiling' for accuracy, 'resize' for speed.
        tile_size: Tile size for tiling mode.
        overlap: Overlap between tiles.
        threshold: Binarization threshold.
        batch_size: Batch size for tile inference.

    Returns:
        Binary mask of shape (H, W) where 255 = building, 0 = background.
    """
    h, w = image.shape[:2]
    device = get_device()
    model = get_segmentation_model()

    if mode == "resize":
        # Resize mode: simple and fast
        resized = resize_image(image, (tile_size, tile_size))
        prob_map = _predict_single_tile(model, resized, device)
        # Resize probability map back to original size
        prob_map_full = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)

    else:
        # Tiling mode: accurate, for large images
        # Handle small images
        if h <= tile_size and w <= tile_size:
            # Image smaller than tile — pad to tile_size
            padded = np.zeros((tile_size, tile_size, 3), dtype=image.dtype)
            padded[:h, :w] = image
            prob_map = _predict_single_tile(model, padded, device)
            prob_map_full = prob_map[:h, :w]
        else:
            # Split into tiles
            tiles, positions = create_tiles(image, tile_size, overlap)

            # Batch inference
            prob_maps = _predict_batch(model, tiles, device, batch_size)

            # Merge tiles
            prob_map_full = merge_tiles(
                prob_maps, positions, (h, w), tile_size, overlap
            )

    # Binarize
    binary_mask = (prob_map_full > threshold).astype(np.uint8) * 255

    return binary_mask


# =============================================================================
# Area calculation
# =============================================================================


def calculate_area(
    mask: np.ndarray,
    gsd_m: float = DEFAULT_GSD_M,
) -> float:
    """
    Calculate total building area in square meters.

    Args:
        mask: Binary mask array where 255 = building, 0 = background.
        gsd_m: Ground Sampling Distance in meters per pixel.

    Returns:
        Total building area in m².
    """
    # Count building pixels
    building_pixels = np.count_nonzero(mask)

    # Area = pixel count × (GSD)²
    area_m2 = building_pixels * (gsd_m ** 2)

    return area_m2


# =============================================================================
# Building counting
# =============================================================================


def count_buildings(
    mask: np.ndarray,
    min_area_px: int = MIN_BUILDING_AREA_PX,
) -> int:
    """
    Count number of separate buildings using connected components.

    Args:
        mask: Binary mask array where 255 = building.
        min_area_px: Minimum component area in pixels to count as building.

    Returns:
        Number of detected buildings.
    """
    # Binarize mask (in case it's not strictly 0/255)
    binary = (mask > 127).astype(np.uint8)

    # Connected components with statistics
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    # Filter by minimum area (label 0 is background)
    count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area_px:
            count += 1

    return count
