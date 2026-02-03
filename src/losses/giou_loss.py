"""Generalized IoU Loss for bounding box regression."""
from __future__ import annotations

import torch
import torch.nn as nn


def box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: (N, 4) boxes in [x1, y1, x2, y2] format.
        boxes2: (M, 4) boxes in [x1, y1, x2, y2] format.

    Returns:
        IoU matrix (N, M).
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    # Union
    union = area1[:, None] + area2 - inter

    return inter / (union + 1e-6)


def generalized_box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Generalized IoU between two sets of boxes.

    GIoU = IoU - (C - Union) / C

    where C is the smallest enclosing box.

    Args:
        boxes1: (N, 4) boxes in [x1, y1, x2, y2] format.
        boxes2: (N, 4) boxes in [x1, y1, x2, y2] format.

    Returns:
        GIoU values (N,).
    """
    # Areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Intersection
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    # Union
    union = area1 + area2 - inter

    # IoU
    iou = inter / (union + 1e-6)

    # Enclosing box
    enclose_lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    enclose_rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]

    # GIoU
    giou = iou - (enclose_area - union) / (enclose_area + 1e-6)

    return giou


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss for bounding box regression.

    Loss = 1 - GIoU
    """

    def __init__(self, reduction: str = "mean") -> None:
        """
        Initialize GIoU Loss.

        Args:
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GIoU Loss.

        Args:
            pred_boxes: Predicted boxes (N, 4).
            target_boxes: Target boxes (N, 4).

        Returns:
            Loss value.
        """
        giou = generalized_box_iou(pred_boxes, target_boxes)
        loss = 1 - giou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DIoULoss(nn.Module):
    """
    Distance IoU Loss for better convergence.

    DIoU = IoU - d²/c²

    where d is center distance and c is diagonal of enclosing box.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DIoU Loss."""
        # Centers
        pred_center = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
        target_center = (target_boxes[:, :2] + target_boxes[:, 2:]) / 2

        # Center distance squared
        d2 = ((pred_center - target_center) ** 2).sum(dim=1)

        # Enclosing box diagonal squared
        enclose_lt = torch.min(pred_boxes[:, :2], target_boxes[:, :2])
        enclose_rb = torch.max(pred_boxes[:, 2:], target_boxes[:, 2:])
        c2 = ((enclose_rb - enclose_lt) ** 2).sum(dim=1)

        # IoU
        area1 = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        area2 = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])

        lt = torch.max(pred_boxes[:, :2], target_boxes[:, :2])
        rb = torch.min(pred_boxes[:, 2:], target_boxes[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        union = area1 + area2 - inter
        iou = inter / (union + 1e-6)

        # DIoU
        diou = iou - d2 / (c2 + 1e-6)
        loss = 1 - diou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
