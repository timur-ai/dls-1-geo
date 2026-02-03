"""Image preprocessing utilities for the building segmentation application."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T

try:
    from app.config import IMAGENET_MEAN, IMAGENET_STD, TILE_OVERLAP, TILE_SIZE
except ImportError:
    from config import IMAGENET_MEAN, IMAGENET_STD, TILE_OVERLAP, TILE_SIZE


# =============================================================================
# Image loading and validation
# =============================================================================


def load_image(source: str | Path | np.ndarray) -> np.ndarray:
    """
    Load an image from various input types.

    Args:
        source: File path (str or Path) or numpy array (RGB/RGBA format expected).

    Returns:
        RGB image as numpy array with shape (H, W, 3) and dtype uint8.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read as an image.
    """
    if isinstance(source, np.ndarray):
        # Уже numpy array - просто копируем
        image = source.copy()
    else:
        # Проверяем существование файла
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Image file not found: {source}")

        # Загружаем из файла (cv2 читает в BGR, IMREAD_UNCHANGED для альфа-канала)
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to read image: {source}")

        # Конвертируем BGR(A) -> RGB
        if image.ndim == 2:
            # Grayscale -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # BGRA -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            # BGR -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Обработка RGBA массивов (например, от Gradio)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    # Обработка grayscale массивов
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    return image.astype(np.uint8)


def validate_image(image: np.ndarray) -> bool:
    """
    Check if the image is a valid RGB image.

    Args:
        image: Input array to validate.

    Returns:
        True if valid RGB image, False otherwise.
    """
    # Проверяем базовые условия
    if not isinstance(image, np.ndarray):
        return False
    if image.ndim != 3:
        return False
    if image.shape[2] != 3:
        return False
    if image.size == 0:
        return False

    return True


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image for consistent processing.

    Currently performs basic normalization to uint8 range.
    Can be extended for model-specific preprocessing.

    Args:
        image: RGB image array.

    Returns:
        Normalized image as uint8 array.
    """
    # Приводим к uint8 если нужно
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            # Вероятно float [0, 1]
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    return image


# =============================================================================
# ImageNet normalization for pretrained models
# =============================================================================

# Кэшированный экземпляр нормализатора
_imagenet_normalizer: T.Normalize | None = None


def get_imagenet_normalize_transform() -> T.Normalize:
    """
    Get ImageNet normalization transform (cached singleton).

    Returns:
        torchvision.transforms.Normalize with ImageNet statistics.
    """
    global _imagenet_normalizer
    if _imagenet_normalizer is None:
        _imagenet_normalizer = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return _imagenet_normalizer


# =============================================================================
# Tiling utilities
# =============================================================================


def create_tiles(
    image: np.ndarray,
    tile_size: int = TILE_SIZE,
    overlap: int = TILE_OVERLAP,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """
    Split image into overlapping tiles.

    Args:
        image: RGB image array (H, W, 3).
        tile_size: Size of each tile (square).
        overlap: Overlap between adjacent tiles.

    Returns:
        Tuple of (tiles, positions):
            - tiles: List of tile arrays, each (tile_size, tile_size, 3)
            - positions: List of (y, x) top-left corner positions
    """
    h, w = image.shape[:2]
    stride = tile_size - overlap

    tiles = []
    positions = []

    # Генерируем позиции тайлов
    y_positions = list(range(0, h - tile_size + 1, stride))
    x_positions = list(range(0, w - tile_size + 1, stride))

    # Добавляем последний тайл если не покрыли край
    if not y_positions or y_positions[-1] + tile_size < h:
        y_positions.append(max(0, h - tile_size))
    if not x_positions or x_positions[-1] + tile_size < w:
        x_positions.append(max(0, w - tile_size))

    # Убираем дубликаты и сортируем
    y_positions = sorted(set(y_positions))
    x_positions = sorted(set(x_positions))

    for y in y_positions:
        for x in x_positions:
            tile = image[y : y + tile_size, x : x + tile_size]

            # Паддинг если тайл меньше нужного размера (край изображения)
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=image.dtype)
                padded[: tile.shape[0], : tile.shape[1]] = tile
                tile = padded

            tiles.append(tile)
            positions.append((y, x))

    return tiles, positions


def merge_tiles(
    tiles: list[np.ndarray],
    positions: list[tuple[int, int]],
    image_shape: tuple[int, int],
    tile_size: int = TILE_SIZE,
    overlap: int = TILE_OVERLAP,
) -> np.ndarray:
    """
    Merge tiles back into full image with weighted averaging in overlap zones.

    Args:
        tiles: List of tile arrays (tile_size, tile_size) — masks or probabilities.
        positions: List of (y, x) top-left corner positions.
        image_shape: Original image shape (H, W).
        tile_size: Size of each tile.
        overlap: Overlap between tiles.

    Returns:
        Merged image/mask array (H, W).
    """
    h, w = image_shape
    dtype = np.float32

    # Accumulator для взвешенного усреднения
    output = np.zeros((h, w), dtype=dtype)
    weights = np.zeros((h, w), dtype=dtype)

    # Создаём весовую маску для тайла (линейное затухание к краям в overlap зоне)
    tile_weight = _create_tile_weight(tile_size, overlap)

    for tile, (y, x) in zip(tiles, positions):
        # Определяем реальный размер (может быть меньше на краях)
        th = min(tile_size, h - y)
        tw = min(tile_size, w - x)

        tile_crop = tile[:th, :tw].astype(dtype)
        weight_crop = tile_weight[:th, :tw]

        output[y : y + th, x : x + tw] += tile_crop * weight_crop
        weights[y : y + th, x : x + tw] += weight_crop

    # Нормализация по весам
    weights = np.maximum(weights, 1e-8)
    output = output / weights

    return output


def _create_tile_weight(tile_size: int, overlap: int) -> np.ndarray:
    """
    Create weight mask for tile blending.

    Weights linearly decrease from 1.0 in center to lower values at edges
    within the overlap zone for smooth blending.

    Args:
        tile_size: Size of the tile.
        overlap: Overlap size.

    Returns:
        Weight mask array (tile_size, tile_size).
    """
    if overlap <= 0:
        return np.ones((tile_size, tile_size), dtype=np.float32)

    # Создаём 1D профиль веса
    weight_1d = np.ones(tile_size, dtype=np.float32)

    # Линейное затухание в overlap зонах
    ramp = np.linspace(0.1, 1.0, overlap)
    weight_1d[:overlap] = ramp
    weight_1d[-overlap:] = ramp[::-1]

    # 2D весовая маска как внешнее произведение
    weight_2d = np.outer(weight_1d, weight_1d)

    return weight_2d


# =============================================================================
# Resize utilities
# =============================================================================


def resize_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (512, 512),
) -> np.ndarray:
    """
    Resize image to target size.

    Args:
        image: RGB image array (H, W, 3).
        target_size: Target size (height, width).

    Returns:
        Resized image array.
    """
    return cv2.resize(
        image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR
    )


def resize_mask(
    mask: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    """
    Resize mask back to original size using nearest neighbor.

    Args:
        mask: Binary mask array (H, W).
        target_size: Target size (height, width).

    Returns:
        Resized mask array.
    """
    return cv2.resize(
        mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST
    )


# =============================================================================
# Tensor conversion for model inference
# =============================================================================


def prepare_batch_for_model(
    tiles: list[np.ndarray],
    normalize: bool = True,
) -> torch.Tensor:
    """
    Convert list of image tiles to normalized torch tensor batch.

    Args:
        tiles: List of RGB image arrays (H, W, 3), uint8 [0, 255].
        normalize: Whether to apply ImageNet normalization.

    Returns:
        Torch tensor (B, 3, H, W), float32, normalized if requested.
    """
    # Stack tiles: (B, H, W, 3)
    batch = np.stack(tiles, axis=0)

    # Convert to float [0, 1] and transpose to (B, C, H, W)
    batch = batch.astype(np.float32) / 255.0
    batch = np.transpose(batch, (0, 3, 1, 2))

    # To tensor
    tensor = torch.from_numpy(batch)

    # Apply ImageNet normalization
    if normalize:
        normalizer = get_imagenet_normalize_transform()
        tensor = normalizer(tensor)

    return tensor


def image_to_tensor(
    image: np.ndarray,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Convert single image to torch tensor.

    Args:
        image: RGB image array (H, W, 3), uint8 [0, 255].
        normalize: Whether to apply ImageNet normalization.

    Returns:
        Torch tensor (1, 3, H, W), float32.
    """
    return prepare_batch_for_model([image], normalize=normalize)
