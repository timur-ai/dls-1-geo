"""Metrics calculation for segmentation and detection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass
class PixelMetrics:
    """Container for pixel-wise segmentation metrics."""

    iou: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def calculate_iou(
    pred: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Calculate Intersection over Union (IoU / Jaccard Index).

    Args:
        pred: Predicted mask, any shape with values in [0, 1] or binary.
        target: Ground truth mask, any shape, binary.
        threshold: Threshold to binarize predictions.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        IoU score in [0, 1].
    """
    # Конвертация в numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Flatten оба массива для корректного вычисления IoU
    # Это исключает проблемы с broadcasting при разных размерностях
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    # Бинаризация предсказаний
    pred_binary = (pred_flat > threshold).astype(np.float32)
    target_binary = (target_flat > 0).astype(np.float32)

    # IoU = intersection / union
    intersection = np.sum(pred_binary * target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return float(iou)


def calculate_pixel_metrics(
    pred: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    threshold: float = 0.5,
) -> PixelMetrics:
    """
    Calculate all pixel-wise segmentation metrics.

    Args:
        pred: Predicted mask, any shape, values in [0, 1] or binary.
        target: Ground truth mask, any shape, binary.
        threshold: Threshold to binarize predictions.

    Returns:
        PixelMetrics with iou, accuracy, precision, recall, f1.
    """
    # Конвертация в numpy и flatten
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Flatten для корректной работы с любыми размерностями
    pred_flat = (pred.flatten() > threshold).astype(np.int32)
    target_flat = (target.flatten() > 0).astype(np.int32)

    # Проверка совпадения длин после flatten
    if len(pred_flat) != len(target_flat):
        raise ValueError(
            f"Mismatch in flattened sizes: pred={len(pred_flat)}, target={len(target_flat)}. "
            f"Original shapes: pred={pred.shape}, target={target.shape}"
        )

    # Вычисление метрик
    iou = calculate_iou(pred, target, threshold)
    accuracy = accuracy_score(target_flat, pred_flat)
    precision = precision_score(target_flat, pred_flat, zero_division=0)
    recall = recall_score(target_flat, pred_flat, zero_division=0)
    f1 = f1_score(target_flat, pred_flat, zero_division=0)

    return PixelMetrics(
        iou=iou,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def calculate_box_iou(
    box1: np.ndarray,
    box2: np.ndarray,
) -> float:
    """
    Calculate IoU between two bounding boxes.

    Args:
        box1: First box [x_min, y_min, x_max, y_max].
        box2: Second box [x_min, y_min, x_max, y_max].

    Returns:
        IoU score in [0, 1].
    """
    # Координаты пересечения
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Площадь пересечения
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Площади боксов
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # IoU
    union = area1 + area2 - intersection
    if union == 0:
        return 0.0

    return intersection / union


def calculate_map(
    pred_boxes: list[np.ndarray],
    pred_scores: list[np.ndarray],
    gt_boxes: list[np.ndarray],
    iou_threshold: float = 0.5,
) -> float:
    """
    Calculate mean Average Precision (mAP) at given IoU threshold.

    Args:
        pred_boxes: List of predicted boxes per image, each (N, 4).
        pred_scores: List of confidence scores per image, each (N,).
        gt_boxes: List of ground truth boxes per image, each (M, 4).
        iou_threshold: IoU threshold for matching.

    Returns:
        mAP score in [0, 1].
    """
    # Валидация входных данных
    if len(pred_boxes) != len(pred_scores) or len(pred_boxes) != len(gt_boxes):
        raise ValueError(
            f"Input lists must have the same length: "
            f"pred_boxes={len(pred_boxes)}, pred_scores={len(pred_scores)}, gt_boxes={len(gt_boxes)}"
        )

    all_scores = []
    all_matches = []
    total_valid_gt = 0  # Счётчик валидных GT boxes после фильтрации

    for preds, scores, gts in zip(pred_boxes, pred_scores, gt_boxes):
        # Фильтрация degenerate boxes (нулевая ширина или высота)
        if len(preds) > 0:
            widths = preds[:, 2] - preds[:, 0]
            heights = preds[:, 3] - preds[:, 1]
            valid_pred_mask = (widths > 0) & (heights > 0)
            preds = preds[valid_pred_mask]
            scores = scores[valid_pred_mask]

        if len(gts) > 0:
            gt_widths = gts[:, 2] - gts[:, 0]
            gt_heights = gts[:, 3] - gts[:, 1]
            valid_gt_mask = (gt_widths > 0) & (gt_heights > 0)
            gts = gts[valid_gt_mask]

        # Подсчёт валидных GT boxes после фильтрации
        total_valid_gt += len(gts)

        if len(preds) == 0:
            continue

        # Сортировка по score
        order = np.argsort(-scores)
        preds = preds[order]
        scores = scores[order]

        gt_matched = np.zeros(len(gts), dtype=bool)

        for pred, score in zip(preds, scores):
            all_scores.append(score)
            matched = False

            for gt_idx, gt in enumerate(gts):
                if gt_matched[gt_idx]:
                    continue

                iou = calculate_box_iou(pred, gt)
                if iou >= iou_threshold:
                    gt_matched[gt_idx] = True
                    matched = True
                    break

            all_matches.append(1 if matched else 0)

    if len(all_scores) == 0:
        return 0.0

    # Сортировка по score для precision-recall curve
    order = np.argsort(-np.array(all_scores))
    all_matches = np.array(all_matches)[order]

    # Вычисление precision-recall с корректным total_gt
    tp_cumsum = np.cumsum(all_matches)
    fp_cumsum = np.cumsum(1 - all_matches)

    if total_valid_gt == 0:
        return 0.0

    recalls = tp_cumsum / total_valid_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # AP через интерполяцию (11-point)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recalls >= t
        if mask.any():
            ap += np.max(precisions[mask])
    ap /= 11

    return float(ap)
