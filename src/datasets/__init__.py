"""Dataset classes for aerial image processing."""
from __future__ import annotations

from src.datasets.bbox_utils import (
    clip_boxes_to_image,
    filter_boxes_by_size,
    filter_degenerate_boxes,
    mask_to_bboxes,
)
from src.datasets.inria_dataset import InriaSegmentationDataset
from src.datasets.inria_detection_dataset import InriaDetectionDataset, detection_collate_fn
from src.datasets.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_detection_train_transforms,
    get_detection_val_transforms,
    get_train_transforms,
    get_val_transforms,
)

__all__ = [
    "InriaSegmentationDataset",
    "InriaDetectionDataset",
    "detection_collate_fn",
    "get_train_transforms",
    "get_val_transforms",
    "get_detection_train_transforms",
    "get_detection_val_transforms",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "mask_to_bboxes",
    "filter_boxes_by_size",
    "filter_degenerate_boxes",
    "clip_boxes_to_image",
]
