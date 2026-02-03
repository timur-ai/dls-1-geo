"""Faster R-CNN with FPN for building detection."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def create_faster_rcnn(
    num_classes: int = 2,  # background + building
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
    min_size: int = 800,
    max_size: int = 1333,
    box_score_thresh: float = 0.05,
    box_nms_thresh: float = 0.5,
    box_detections_per_img: int = 100,
) -> FasterRCNN:
    """
    Create Faster R-CNN model with ResNet-50 FPN backbone.

    Args:
        num_classes: Number of classes including background.
        pretrained_backbone: Use ImageNet pretrained backbone.
        trainable_backbone_layers: Number of trainable layers in backbone (0-5).
        min_size: Minimum image size for transform.
        max_size: Maximum image size for transform.
        box_score_thresh: Score threshold for predictions.
        box_nms_thresh: NMS IoU threshold.
        box_detections_per_img: Maximum detections per image.

    Returns:
        Faster R-CNN model instance.
    """
    # Создание backbone с FPN
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights="IMAGENET1K_V1" if pretrained_backbone else None,
        trainable_layers=trainable_backbone_layers,
    )

    # Создание модели Faster R-CNN
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        box_detections_per_img=box_detections_per_img,
    )

    return model


class BuildingDetector(nn.Module):
    """
    Wrapper around Faster R-CNN for building detection.

    Provides convenient interface for training and inference.
    """

    def __init__(
        self,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: int = 3,
        min_size: int = 800,
        max_size: int = 1333,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
    ) -> None:
        """
        Initialize building detector.

        Args:
            pretrained_backbone: Use ImageNet pretrained backbone.
            trainable_backbone_layers: Number of trainable layers.
            min_size: Minimum image size.
            max_size: Maximum image size.
            box_score_thresh: Score threshold for inference.
            box_nms_thresh: NMS threshold.
        """
        super().__init__()

        self.model = create_faster_rcnn(
            num_classes=2,  # background + building
            pretrained_backbone=pretrained_backbone,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
        )

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            images: List of image tensors (C, H, W).
            targets: List of target dicts with 'boxes' and 'labels' (training only).

        Returns:
            In training mode: dict with losses.
            In eval mode: list of prediction dicts with 'boxes', 'labels', 'scores'.
        """
        return self.model(images, targets)

    @torch.inference_mode()
    def predict(
        self,
        images: list[torch.Tensor],
        score_thresh: float = 0.5,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Run inference with score thresholding.

        Args:
            images: List of image tensors.
            score_thresh: Minimum score for predictions.

        Returns:
            List of prediction dicts.
        """
        self.eval()
        predictions = self.model(images)

        # Фильтрация по score
        filtered = []
        for pred in predictions:
            mask = pred["scores"] >= score_thresh
            filtered.append({
                "boxes": pred["boxes"][mask],
                "labels": pred["labels"][mask],
                "scores": pred["scores"][mask],
            })

        return filtered
