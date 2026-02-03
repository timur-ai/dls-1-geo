"""Focal Loss implementation for object detection."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for classification in object detection.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Helps with class imbalance by down-weighting easy examples.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """
        Initialize Focal Loss.

        Args:
            alpha: Balancing factor for positive/negative examples.
            gamma: Focusing parameter (higher = more focus on hard examples).
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Focal Loss.

        Args:
            inputs: Predicted logits (N,) or (N, C) for multiclass.
            targets: Ground truth labels (N,).

        Returns:
            Loss value.
        """
        # Binary case
        if inputs.dim() == 1 or inputs.size(1) == 1:
            inputs = inputs.view(-1)
            targets = targets.view(-1).float()

            bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            p = torch.sigmoid(inputs)
            p_t = p * targets + (1 - p) * (1 - targets)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = (1 - p_t) ** self.gamma

            loss = alpha_t * focal_weight * bce
        else:
            # Multiclass case
            ce_loss = F.cross_entropy(inputs, targets, reduction="none")
            p = F.softmax(inputs, dim=1)
            p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - p_t) ** self.gamma

            loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SigmoidFocalLoss(nn.Module):
    """
    Sigmoid Focal Loss for dense prediction tasks.

    Used in RetinaNet-style detectors.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        """
        Initialize Sigmoid Focal Loss.

        Args:
            alpha: Weighting factor.
            gamma: Focusing parameter.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Sigmoid Focal Loss.

        Args:
            inputs: Logits (N, C).
            targets: Binary targets (N, C) as one-hot or soft labels.

        Returns:
            Scalar loss.
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        loss = alpha_t * focal_weight * ce_loss

        return loss.mean()
