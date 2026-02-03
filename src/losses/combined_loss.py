"""Combined loss functions for segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.dice_loss import DiceLoss


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice Loss.

    Loss = α * BCE + β * DiceLoss

    This combination helps with:
    - BCE: pixel-wise accuracy, good for imbalanced classes
    - Dice: overlap/area accuracy, good for segmentation quality
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
    ) -> None:
        """
        Initialize combined loss.

        Args:
            bce_weight: Weight for BCE loss (α).
            dice_weight: Weight for Dice loss (β).
            smooth: Smoothing factor for Dice.
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            pred: Predicted logits (B, 1, H, W).
            target: Ground truth mask (B, 1, H, W) or (B, H, W).

        Returns:
            Scalar loss value.
        """
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Проверка совместимости размеров
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
            )

        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalDiceLoss(nn.Module):
    """
    Combined Focal Loss and Dice Loss.

    Focal Loss helps with class imbalance by down-weighting easy examples.
    """

    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        gamma: float = 2.0,
        alpha: float = 0.25,
        smooth: float = 1e-6,
    ) -> None:
        """
        Initialize combined loss.

        Args:
            focal_weight: Weight for Focal loss.
            dice_weight: Weight for Dice loss.
            gamma: Focal loss focusing parameter.
            alpha: Focal loss balancing parameter.
            smooth: Smoothing factor for Dice.
        """
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.gamma = gamma
        self.alpha = alpha
        self.dice = DiceLoss(smooth=smooth)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined Focal + Dice loss.

        Args:
            pred: Predicted logits (B, 1, H, W).
            target: Ground truth mask (B, 1, H, W) or (B, H, W).

        Returns:
            Scalar loss value.
        """
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Проверка совместимости размеров
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
            )

        # Focal Loss
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = (alpha_t * focal_weight * bce).mean()

        # Dice Loss
        dice_loss = self.dice(pred, target)

        return self.focal_weight * focal_loss + self.dice_weight * dice_loss
