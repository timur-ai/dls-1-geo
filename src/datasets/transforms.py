"""Data augmentation pipelines using albumentations."""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(
    crop_size: int = 512,
    p_flip: float = 0.5,
    p_rotate: float = 0.5,
    rotate_limit: int = 15,
) -> A.Compose:
    """
    Get training augmentation pipeline.

    Args:
        crop_size: Size of random crop.
        p_flip: Probability of horizontal/vertical flip.
        p_rotate: Probability of rotation.
        rotate_limit: Maximum rotation angle in degrees.

    Returns:
        Albumentations Compose transform.
    """
    return A.Compose([
        # Случайный кроп из большого изображения
        A.RandomCrop(height=crop_size, width=crop_size),
        # Геометрические аугментации
        A.HorizontalFlip(p=p_flip),
        A.VerticalFlip(p=p_flip),
        A.Rotate(limit=rotate_limit, p=p_rotate, border_mode=0),
        # Цветовые аугментации
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05,
            p=0.5,
        ),
        # Нормализация по ImageNet
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # Конвертация в PyTorch tensor
        ToTensorV2(),
    ])


def get_val_transforms(crop_size: int | None = None, use_random_crop: bool = False) -> A.Compose:
    """
    Get validation/inference augmentation pipeline.

    Only normalization, no augmentation for consistent evaluation.

    Args:
        crop_size: Optional crop size. If None, no cropping.
        use_random_crop: If True, uses RandomCrop (for multi-tile sampling).
                         If False (default), uses CenterCrop (for single deterministic tile).

    Returns:
        Albumentations Compose transform.
    """
    transforms_list = []

    if crop_size is not None:
        # RandomCrop для multi-tile sampling, CenterCrop для single tile
        if use_random_crop:
            transforms_list.append(A.RandomCrop(height=crop_size, width=crop_size))
        else:
            transforms_list.append(A.CenterCrop(height=crop_size, width=crop_size))

    transforms_list.extend([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    return A.Compose(transforms_list)


def get_detection_train_transforms(
    min_area: int = 100,
    min_visibility: float = 0.3,
) -> A.Compose:
    """
    Get training transforms for detection task.

    Assumes images are already cropped to the target size by the dataset.
    Only applies augmentations and normalization.

    Args:
        min_area: Minimum bounding box area in pixels after augmentation.
        min_visibility: Minimum visibility ratio for bounding boxes (0-1).

    Returns:
        Albumentations Compose transform with bbox support.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05,
                p=0.5,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",  # [x_min, y_min, x_max, y_max]
            label_fields=["labels"],
            min_visibility=min_visibility,
            min_area=min_area,
        ),
    )


def get_detection_val_transforms(
    min_area: int = 100,
    min_visibility: float = 0.3,
) -> A.Compose:
    """
    Get validation transforms for detection task.

    Assumes images are already cropped to the target size by the dataset.
    Only applies normalization.

    Args:
        min_area: Minimum bounding box area in pixels.
        min_visibility: Minimum visibility ratio for bounding boxes (0-1).

    Returns:
        Albumentations Compose transform with bbox support.
    """
    return A.Compose(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=min_visibility,
            min_area=min_area,
        ),
    )
