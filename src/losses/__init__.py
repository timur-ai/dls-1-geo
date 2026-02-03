"""Loss functions for training."""
from __future__ import annotations

from src.losses.combined_loss import BCEDiceLoss, FocalDiceLoss
from src.losses.dice_loss import DiceLoss, SoftDiceLoss
from src.losses.focal_loss import FocalLoss, SigmoidFocalLoss
from src.losses.giou_loss import DIoULoss, GIoULoss

__all__ = [
    "DiceLoss",
    "SoftDiceLoss",
    "BCEDiceLoss",
    "FocalDiceLoss",
    "FocalLoss",
    "SigmoidFocalLoss",
    "GIoULoss",
    "DIoULoss",
]
