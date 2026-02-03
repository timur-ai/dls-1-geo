"""Visualization utilities for predictions and ground truth."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_segmentation(
    image: np.ndarray | torch.Tensor,
    mask_gt: np.ndarray | torch.Tensor | None,
    mask_pred: np.ndarray | torch.Tensor | None,
    title: str = "",
    alpha: float = 0.5,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (15, 5),
) -> None:
    """
    Visualize image with ground truth and predicted segmentation masks.

    Args:
        image: Input image (H, W, 3) or (3, H, W), values in [0, 1] or [0, 255].
        mask_gt: Ground truth mask (H, W), optional.
        mask_pred: Predicted mask (H, W), optional.
        title: Figure title.
        alpha: Overlay transparency.
        save_path: Path to save figure (optional).
        figsize: Figure size.
    """
    # Конвертация tensors в numpy
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask_gt, torch.Tensor):
        mask_gt = mask_gt.detach().cpu().numpy()
    if isinstance(mask_pred, torch.Tensor):
        mask_pred = mask_pred.detach().cpu().numpy()

    # Если изображение в формате (C, H, W), транспонируем
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    # Нормализация в [0, 1] если нужно
    if image.max() > 1:
        image = image / 255.0

    # Определение количества subplot'ов
    n_plots = 1 + (mask_gt is not None) + (mask_pred is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    idx = 0

    # Исходное изображение
    axes[idx].imshow(image)
    axes[idx].set_title("Input Image")
    axes[idx].axis("off")
    idx += 1

    # Ground truth
    if mask_gt is not None:
        axes[idx].imshow(image)
        mask_overlay = np.zeros((*mask_gt.shape, 4))
        mask_overlay[mask_gt > 0] = [0, 1, 0, alpha]  # Зелёный
        axes[idx].imshow(mask_overlay)
        axes[idx].set_title("Ground Truth")
        axes[idx].axis("off")
        idx += 1

    # Prediction
    if mask_pred is not None:
        axes[idx].imshow(image)
        mask_overlay = np.zeros((*mask_pred.shape, 4))
        mask_overlay[mask_pred > 0.5] = [1, 0, 0, alpha]  # Красный
        axes[idx].imshow(mask_overlay)
        axes[idx].set_title("Prediction")
        axes[idx].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_detection(
    image: np.ndarray | torch.Tensor,
    boxes_gt: np.ndarray | None,
    boxes_pred: np.ndarray | None,
    scores_pred: np.ndarray | None = None,
    title: str = "",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (15, 5),
) -> None:
    """
    Visualize image with ground truth and predicted bounding boxes.

    Args:
        image: Input image (H, W, 3) or (3, H, W).
        boxes_gt: Ground truth boxes (N, 4) in format [x_min, y_min, x_max, y_max].
        boxes_pred: Predicted boxes (M, 4).
        scores_pred: Confidence scores for predicted boxes (M,).
        title: Figure title.
        save_path: Path to save figure.
        figsize: Figure size.
    """
    # Конвертация
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(boxes_gt, torch.Tensor):
        boxes_gt = boxes_gt.detach().cpu().numpy()
    if isinstance(boxes_pred, torch.Tensor):
        boxes_pred = boxes_pred.detach().cpu().numpy()
    if isinstance(scores_pred, torch.Tensor):
        scores_pred = scores_pred.detach().cpu().numpy()

    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    if image.max() > 1:
        image = image / 255.0

    n_plots = 1 + (boxes_gt is not None) + (boxes_pred is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    idx = 0

    # Исходное изображение
    axes[idx].imshow(image)
    axes[idx].set_title("Input Image")
    axes[idx].axis("off")
    idx += 1

    # Ground truth boxes
    if boxes_gt is not None:
        axes[idx].imshow(image)
        for box in boxes_gt:
            rect = plt.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                fill=False,
                edgecolor="green",
                linewidth=2,
            )
            axes[idx].add_patch(rect)
        axes[idx].set_title(f"Ground Truth ({len(boxes_gt)} boxes)")
        axes[idx].axis("off")
        idx += 1

    # Predicted boxes
    if boxes_pred is not None:
        axes[idx].imshow(image)
        for i, box in enumerate(boxes_pred):
            rect = plt.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
            axes[idx].add_patch(rect)
            if scores_pred is not None:
                axes[idx].text(
                    box[0],
                    box[1] - 5,
                    f"{scores_pred[i]:.2f}",
                    color="red",
                    fontsize=8,
                )
        axes[idx].set_title(f"Predictions ({len(boxes_pred)} boxes)")
        axes[idx].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    train_metrics: list[float] | None = None,
    val_metrics: list[float] | None = None,
    metric_name: str = "IoU",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 5),
    show: bool = False,
) -> None:
    """
    Plot training and validation curves.

    Args:
        train_losses: Training loss per epoch.
        val_losses: Validation loss per epoch (optional).
        train_metrics: Training metric per epoch (optional).
        val_metrics: Validation metric per epoch (optional).
        metric_name: Name of the metric.
        save_path: Path to save figure.
        figsize: Figure size.
        show: Whether to display the plot (useful in notebooks).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss plot
    axes[0].plot(train_losses, label="Train Loss", color="blue")
    if val_losses:
        axes[0].plot(val_losses, label="Val Loss", color="orange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Metric plot
    if train_metrics:
        axes[1].plot(train_metrics, label=f"Train {metric_name}", color="blue")
    if val_metrics:
        axes[1].plot(val_metrics, label=f"Val {metric_name}", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(metric_name)
    axes[1].set_title(f"Training {metric_name}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    elif not save_path:
        plt.show()
    else:
        plt.close()
