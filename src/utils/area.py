"""Building area calculation utilities."""
from __future__ import annotations

import numpy as np
import torch


def calculate_area_m2(
    mask: np.ndarray | torch.Tensor,
    gsd: float = 0.3,
) -> float:
    """
    Calculate building area in square meters from a binary mask.

    Args:
        mask: Binary segmentation mask (H, W) with building pixels = 1 or 255.
        gsd: Ground Sampling Distance in meters per pixel.
             Default 0.3 m/pixel for Inria dataset.

    Returns:
        Total building area in square meters.
    """
    # Конвертация torch tensor в numpy если нужно
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Подсчёт пикселей зданий (значения > 0)
    pixel_count = np.sum(mask > 0)

    # Площадь одного пикселя в м²
    pixel_area_m2 = gsd ** 2

    return float(pixel_count * pixel_area_m2)


def calculate_area_from_boxes(
    boxes: np.ndarray | torch.Tensor,
    gsd: float = 0.3,
) -> float:
    """
    Calculate total building area from bounding boxes.

    Args:
        boxes: Bounding boxes in format (N, 4) where each row is
               [x_min, y_min, x_max, y_max] in pixels.
        gsd: Ground Sampling Distance in meters per pixel.

    Returns:
        Total estimated building area in square meters.

    Note:
        This is an approximation. Actual building area is typically
        less than bounding box area due to irregular shapes.
    """
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()

    if len(boxes) == 0:
        return 0.0

    # Вычисление площади каждого bbox в пикселях
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    # Фильтрация degenerate boxes (отрицательная или нулевая площадь)
    valid_mask = (widths > 0) & (heights > 0)
    if not valid_mask.any():
        return 0.0

    pixel_areas = widths[valid_mask] * heights[valid_mask]

    # Общая площадь в м²
    pixel_area_m2 = gsd ** 2
    total_area = float(np.sum(pixel_areas) * pixel_area_m2)

    return total_area
