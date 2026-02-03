"""Model checkpointing utilities."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)


def _get_model_state_dict(model: nn.Module) -> dict[str, Any]:
    """
    Extract state_dict from model, handling torch.compile() wrapped models.

    torch.compile() wraps the model in OptimizedModule, which adds '_orig_mod.'
    prefix to all state_dict keys. This function unwraps it for consistent saving.
    """
    # Проверяем, является ли модель скомпилированной (OptimizedModule)
    if hasattr(model, "_orig_mod"):
        # Модель скомпилирована — получаем state_dict из оригинальной модели
        return model._orig_mod.state_dict()
    return model.state_dict()


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    epoch: int,
    global_step: int,
    best_metric: float,
    path: str | Path,
    scaler: GradScaler | None = None,
    **extra_state: Any,
) -> None:
    """
    Save training checkpoint with all necessary state.

    Args:
        model: PyTorch model.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler (optional).
        epoch: Current epoch number.
        global_step: Global training step.
        best_metric: Best metric value achieved.
        path: Path to save checkpoint.
        scaler: GradScaler for mixed precision (optional).
        **extra_state: Additional state to save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = _get_model_state_dict(model)

    checkpoint = {
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        **extra_state,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    # Атомарное сохранение через временный файл
    tmp_path = path.with_suffix(".tmp")
    torch.save(checkpoint, tmp_path)
    # На Windows rename() падает если target существует — удаляем сначала
    if path.exists():
        path.unlink()
    tmp_path.rename(path)

    logger.info(f"Checkpoint saved to {path}")


def _load_model_state_dict(
    model: nn.Module,
    state_dict: dict[str, Any],
    strict: bool = True,
) -> None:
    """
    Load state_dict into model, handling torch.compile() wrapped models.

    If model is compiled (OptimizedModule), loads into the original module.
    """
    # Проверяем, является ли модель скомпилированной
    if hasattr(model, "_orig_mod"):
        # Модель скомпилирована — загружаем в оригинальную модель
        model._orig_mod.load_state_dict(state_dict, strict=strict)
    else:
        model.load_state_dict(state_dict, strict=strict)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    scaler: GradScaler | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    Load training checkpoint and restore state.

    Args:
        path: Path to checkpoint file.
        model: PyTorch model to load weights into.
        optimizer: Optimizer to restore state (optional).
        scheduler: Scheduler to restore state (optional).
        scaler: GradScaler to restore state (optional).
        device: Device to load checkpoint to.

    Returns:
        Dictionary with additional checkpoint state (epoch, global_step, etc.).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # weights_only=False необходим для загрузки optimizer/scheduler state_dict,
    # которые содержат не только тензоры. Файл загружается только из доверенного источника.
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    _load_model_state_dict(model, checkpoint["model_state_dict"])
    logger.info(f"Model weights loaded from {path}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Optimizer state restored")

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Scheduler state restored")

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        logger.info("GradScaler state restored")

    # Возвращаем все поля из checkpoint кроме state_dict'ов
    excluded_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
    }
    return {k: v for k, v in checkpoint.items() if k not in excluded_keys}
