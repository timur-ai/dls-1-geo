"""Utilities for converting segmentation masks to bounding boxes."""
from __future__ import annotations

from typing import overload

import cv2
import numpy as np
import torch


def mask_to_bboxes(
    mask: np.ndarray,
    min_area: int = 100,
) -> np.ndarray:
    """
    Convert binary segmentation mask to bounding boxes using connected components.

    Args:
        mask: Binary mask (H, W) with building pixels = 255 or 1.
        min_area: Minimum component area in pixels to filter noise.

    Returns:
        Array of bounding boxes (N, 4) in format [x_min, y_min, x_max, y_max].
    """
    # Бинаризация маски
    if mask.max() > 1:
        binary_mask = (mask > 127).astype(np.uint8)
    else:
        binary_mask = (mask > 0).astype(np.uint8)

    # Connected components analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    boxes = []

    # Пропускаем фон (label 0)
    for label_idx in range(1, num_labels):
        # Статистики компоненты: x, y, width, height, area
        x = stats[label_idx, cv2.CC_STAT_LEFT]
        y = stats[label_idx, cv2.CC_STAT_TOP]
        w = stats[label_idx, cv2.CC_STAT_WIDTH]
        h = stats[label_idx, cv2.CC_STAT_HEIGHT]
        area = stats[label_idx, cv2.CC_STAT_AREA]

        # Фильтрация мелких компонент
        if area < min_area:
            continue

        # Формат: [x_min, y_min, x_max, y_max]
        boxes.append([x, y, x + w, y + h])

    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    return np.array(boxes, dtype=np.float32)


def filter_boxes_by_size(
    boxes: np.ndarray,
    min_size: int = 10,
    max_size: int = 2000,
) -> np.ndarray:
    """
    Filter bounding boxes by size.

    Args:
        boxes: Array of boxes (N, 4).
        min_size: Minimum side length.
        max_size: Maximum side length.

    Returns:
        Filtered boxes array.
    """
    if len(boxes) == 0:
        return boxes

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    # Фильтруем по размеру
    valid = (
        (widths >= min_size)
        & (heights >= min_size)
        & (widths <= max_size)
        & (heights <= max_size)
    )

    return boxes[valid]


def clip_boxes_to_image(
    boxes: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Clip bounding boxes to image boundaries.

    Args:
        boxes: Array of boxes (N, 4) in [x_min, y_min, x_max, y_max] format.
        height: Image height.
        width: Image width.

    Returns:
        Clipped boxes array.
    """
    if len(boxes) == 0:
        return boxes

    boxes = boxes.copy()
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height)

    return boxes


@overload
def filter_degenerate_boxes(
    boxes: np.ndarray,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]: ...


@overload
def filter_degenerate_boxes(
    boxes: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]: ...


def filter_degenerate_boxes(
    boxes: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor | None = None,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor | None]:
    """
    Filter out degenerate boxes with zero or negative width/height.

    Supports both numpy arrays and torch tensors.

    Args:
        boxes: Array/Tensor of boxes (N, 4) in [x_min, y_min, x_max, y_max] format.
        labels: Optional array/tensor of labels (N,) corresponding to boxes.

    Returns:
        Tuple of (filtered_boxes, filtered_labels). If labels was None, returns None.
        Output type matches input type (numpy or torch).
    """
    if len(boxes) == 0:
        return boxes, labels

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    valid_mask = (widths > 0) & (heights > 0)

    filtered_boxes = boxes[valid_mask]
    filtered_labels = labels[valid_mask] if labels is not None else None

    return filtered_boxes, filtered_labels
