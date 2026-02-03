"""Postprocessing utilities for visualization and statistics."""

from __future__ import annotations

import numpy as np

try:
    from app.config import OVERLAY_ALPHA, OVERLAY_COLOR
except ImportError:
    from config import OVERLAY_ALPHA, OVERLAY_COLOR


def create_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Create visualization with mask overlaid on the original image.

    Args:
        image: RGB image array of shape (H, W, 3).
        mask: Binary mask array of shape (H, W) where 255 = building.

    Returns:
        RGB image with semi-transparent overlay on building areas.

    Raises:
        ValueError: If image and mask dimensions do not match or mask is not 2D.
    """
    # Проверяем что маска 2D
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")

    # Проверяем совпадение размеров
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image shape {image.shape[:2]} does not match mask shape {mask.shape[:2]}"
        )

    # Создаём копию изображения как float для корректной арифметики
    overlay = image.astype(np.float32)

    # Создаём маску для наложения (где mask > 0)
    building_mask = mask > 0

    # Накладываем цвет на области зданий (работаем с float для избежания overflow)
    color = np.array(OVERLAY_COLOR, dtype=np.float32)
    overlay[building_mask] = (
        (1 - OVERLAY_ALPHA) * overlay[building_mask] + OVERLAY_ALPHA * color
    )

    # Clip и конвертируем обратно в uint8
    return np.clip(overlay, 0, 255).astype(np.uint8)


def calculate_coverage_percent(mask: np.ndarray) -> float:
    """
    Calculate building coverage as percentage of total area.

    Args:
        mask: Binary mask array where 255 = building.

    Returns:
        Coverage percentage (0-100).
    """
    total_pixels = mask.size
    building_pixels = np.count_nonzero(mask)
    return (building_pixels / total_pixels) * 100.0

