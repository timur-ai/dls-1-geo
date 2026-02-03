"""Detection Dataset for Inria Aerial Image Labeling Dataset."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import albumentations as A
import numpy as np
import rasterio
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.bbox_utils import clip_boxes_to_image, filter_boxes_by_size, mask_to_bboxes
from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)

# Города в training set
TRAIN_CITIES = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
TILES_PER_CITY = 36
VAL_TILES_PER_CITY = 5


class InriaDetectionDataset(Dataset):
    """
    Dataset for Inria Aerial Image Labeling (detection task).

    Converts segmentation masks to bounding boxes for building detection.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        transform: Callable | A.Compose | None = None,
        tile_size: int = 800,
        tiles_per_image: int = 5,
        min_box_area: int = 100,
        min_box_side: int = 10,
        seed: int = 42,
    ) -> None:
        """
        Initialize the detection dataset.

        Args:
            data_dir: Path to dataset root.
            split: One of 'train', 'val', or 'all'.
            transform: Albumentations transform with bbox support.
            tile_size: Size of tiles to crop.
            tiles_per_image: Number of tiles per image per epoch.
            min_box_area: Minimum bounding box area in pixels.
            min_box_side: Minimum bounding box side length in pixels.
            seed: Random seed for deterministic cropping (used in val split).
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.tile_size = tile_size
        self.tiles_per_image = tiles_per_image
        self.min_box_area = min_box_area
        self.min_box_side = min_box_side
        self.seed = seed

        self.images_dir = self.data_dir / "train" / "images"
        self.masks_dir = self.data_dir / "train" / "gt"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        self.samples = self._collect_samples()
        logger.info(f"Found {len(self.samples)} images for detection split '{split}'")

    def _collect_samples(self) -> list[tuple[Path, Path]]:
        """Collect image-mask pairs based on split."""
        samples = []

        for city in TRAIN_CITIES:
            for tile_idx in range(1, TILES_PER_CITY + 1):
                is_val_tile = tile_idx <= VAL_TILES_PER_CITY

                if self.split == "train" and is_val_tile:
                    continue
                if self.split == "val" and not is_val_tile:
                    continue

                image_name = f"{city}{tile_idx}.tif"
                image_path = self.images_dir / image_name
                mask_path = self.masks_dir / image_name

                if image_path.exists() and mask_path.exists():
                    samples.append((image_path, mask_path))

        return samples

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples) * self.tiles_per_image

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a single detection sample.

        Returns:
            Dictionary with:
            - 'image': Tensor (3, H, W)
            - 'boxes': Tensor (N, 4) in [x_min, y_min, x_max, y_max]
            - 'labels': Tensor (N,) all ones (single class)
        """
        image_idx = idx // self.tiles_per_image
        image_path, mask_path = self.samples[image_idx]

        # Загрузка изображения и маски
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        h, w = image.shape[:2]

        # RNG для случайного кропа:
        # - val: локальный RandomState с детерминированным seed для воспроизводимости
        # - train: глобальный np.random (seed устанавливается в worker_init_fn)
        if self.split == "val":
            rng = np.random.RandomState(self.seed + idx)
        else:
            rng = np.random

        # Кроп или padding по высоте
        if h > self.tile_size:
            y = rng.randint(0, h - self.tile_size + 1)
            image = image[y : y + self.tile_size, :, :]
            mask = mask[y : y + self.tile_size, :]
        elif h < self.tile_size:
            pad_h = self.tile_size - h
            image = np.pad(image, ((0, pad_h), (0, 0), (0, 0)), mode="constant")
            mask = np.pad(mask, ((0, pad_h), (0, 0)), mode="constant")

        # Кроп или padding по ширине
        if w > self.tile_size:
            x = rng.randint(0, w - self.tile_size + 1)
            image = image[:, x : x + self.tile_size, :]
            mask = mask[:, x : x + self.tile_size]
        elif w < self.tile_size:
            pad_w = self.tile_size - w
            image = np.pad(image, ((0, 0), (0, pad_w), (0, 0)), mode="constant")
            mask = np.pad(mask, ((0, 0), (0, pad_w)), mode="constant")

        # Конвертация маски в bounding boxes
        boxes = mask_to_bboxes(mask, min_area=self.min_box_area)
        boxes = filter_boxes_by_size(boxes, min_size=self.min_box_side, max_size=self.tile_size)
        boxes = clip_boxes_to_image(boxes, self.tile_size, self.tile_size)

        # Фильтрация вырожденных boxes (нулевая ширина или высота после clip)
        if len(boxes) > 0:
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            valid_mask = (widths > 0) & (heights > 0)
            boxes = boxes[valid_mask]

        # Labels: все 1 (один класс — building)
        labels = np.ones(len(boxes), dtype=np.int64)

        # Применение трансформаций (всегда для консистентной нормализации)
        if self.transform is not None:
            # Применяем transform всегда, даже для пустых boxes
            transformed = self.transform(
                image=image,
                bboxes=boxes.tolist() if len(boxes) > 0 else [],
                labels=labels.tolist() if len(labels) > 0 else [],
            )
            image = transformed["image"]

            # Обработка boxes после трансформации
            if len(transformed["bboxes"]) > 0:
                boxes = np.array(transformed["bboxes"], dtype=np.float32)
                labels = np.array(transformed["labels"], dtype=np.int64)

                # Фильтрация вырожденных boxes после трансформации
                # (Albumentations с min_visibility может оставить частичные boxes)
                widths = boxes[:, 2] - boxes[:, 0]
                heights = boxes[:, 3] - boxes[:, 1]
                valid_mask = (widths > 0) & (heights > 0)
                if not valid_mask.all():
                    boxes = boxes[valid_mask]
                    labels = labels[valid_mask]
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros(0, dtype=np.int64)
        else:
            # Fallback без трансформаций — применяем ImageNet нормализацию вручную
            image = image.astype(np.float32) / 255.0
            mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
            std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
            image = (image - mean) / std
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        # Конвертация в tensor (защита от кастомных transforms без ToTensorV2)
        if not isinstance(image, torch.Tensor):
            logger.warning("Image is not a tensor after transform — custom transform missing ToTensorV2?")
            if isinstance(image, np.ndarray):
                # Предполагаем формат (H, W, C) -> (C, H, W)
                if image.ndim == 3 and image.shape[2] in (1, 3):
                    image = torch.from_numpy(image.transpose(2, 0, 1)).float()
                else:
                    image = torch.from_numpy(image).float()

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
        }

    def _load_image(self, path: Path) -> np.ndarray:
        """Load image as RGB numpy array."""
        try:
            with rasterio.open(path) as src:
                image = src.read([1, 2, 3]).transpose(1, 2, 0)
        except Exception:
            image = np.array(Image.open(path).convert("RGB"))
        return image.astype(np.uint8)

    def _load_mask(self, path: Path) -> np.ndarray:
        """Load mask as grayscale numpy array."""
        try:
            with rasterio.open(path) as src:
                mask = src.read(1)
        except Exception:
            mask = np.array(Image.open(path).convert("L"))
        return mask.astype(np.uint8)

    @property
    def num_images(self) -> int:
        """Return number of unique images."""
        return len(self.samples)


def detection_collate_fn(
    batch: list[dict[str, Any]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """
    Custom collate function for detection DataLoader.

    Handles variable number of boxes per image.

    Args:
        batch: List of sample dictionaries.

    Returns:
        Tuple of (images_list, targets_list) as expected by Faster R-CNN.
    """
    images = []
    targets = []

    for sample in batch:
        images.append(sample["image"])
        targets.append({
            "boxes": sample["boxes"],
            "labels": sample["labels"],
        })

    return images, targets
