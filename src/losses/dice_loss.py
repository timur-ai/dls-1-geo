"""Dice Loss implementation for segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        """
        Initialize Dice Loss.

        Args:
            smooth: Smoothing factor to avoid division by zero.
        """
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Dice Loss.

        Args:
            pred: Predicted logits (B, 1, H, W) - will be passed through sigmoid.
            target: Ground truth mask (B, 1, H, W) or (B, H, W), values 0 or 1.

        Returns:
            Scalar loss value.
        """
        # Применяем sigmoid к предсказаниям
        pred = torch.sigmoid(pred)

        # Убеждаемся что target имеет правильную форму
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Проверка совместимости размеров
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
            )

        # Per-batch Dice для корректного усреднения (аналогично SoftDiceLoss)
        batch_size = pred.size(0)
        pred_flat = pred.view(batch_size, -1)
        target_flat = target.view(batch_size, -1)

        # Dice coefficient per sample
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice.mean()


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss that works with probabilities directly.

    Useful when combined with other losses that expect logits.
    """

    def __init__(self, smooth: float = 1e-6, apply_sigmoid: bool = True) -> None:
        """
        Initialize Soft Dice Loss.

        Args:
            smooth: Smoothing factor.
            apply_sigmoid: Whether to apply sigmoid to predictions.
        """
        super().__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Soft Dice Loss.

        Args:
            pred: Predictions (B, 1, H, W).
            target: Ground truth (B, 1, H, W) or (B, H, W).

        Returns:
            Scalar loss value.
        """
        if self.apply_sigmoid:
            pred = torch.sigmoid(pred)

        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Проверка совместимости размеров
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
            )

        # Per-batch Dice для стабильности
        batch_size = pred.size(0)
        pred_flat = pred.view(batch_size, -1)
        target_flat = target.view(batch_size, -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice.mean()
