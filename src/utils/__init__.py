"""Utility functions for training and evaluation."""
from __future__ import annotations

from src.utils.area import calculate_area_from_boxes, calculate_area_m2
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.metrics import (
    PixelMetrics,
    calculate_box_iou,
    calculate_iou,
    calculate_map,
    calculate_pixel_metrics,
)
from src.utils.seed import seed_everything
from src.utils.visualization import (
    plot_training_curves,
    visualize_detection,
    visualize_segmentation,
)

__all__ = [
    "seed_everything",
    "calculate_iou",
    "calculate_pixel_metrics",
    "calculate_box_iou",
    "calculate_map",
    "PixelMetrics",
    "calculate_area_m2",
    "calculate_area_from_boxes",
    "save_checkpoint",
    "load_checkpoint",
    "visualize_segmentation",
    "visualize_detection",
    "plot_training_curves",
]
