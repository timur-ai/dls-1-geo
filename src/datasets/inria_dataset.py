"""Custom Dataset for Inria Aerial Image Labeling Dataset."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable

import albumentations as A
import numpy as np
import rasterio
import rasterio.errors
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Города в training set
TRAIN_CITIES = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]

# Количество тайлов на город
TILES_PER_CITY = 36

# Первые N тайлов для валидации (как в оригинальной статье)
VAL_TILES_PER_CITY = 5


class InriaSegmentationDataset(Dataset):
    """
    Dataset for Inria Aerial Image Labeling (segmentation task).

    Each image is 5000x5000 pixels at 0.3m GSD resolution.
    Ground truth masks are binary (building=255, background=0).
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        transform: Callable | A.Compose | None = None,
        tile_size: int = 512,
        tiles_per_image: int = 10,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            data_dir: Path to dataset root (containing train/images and train/gt).
            split: One of 'train', 'val', or 'all'.
            transform: Albumentations transform pipeline.
            tile_size: Size of tiles to crop from images.
            tiles_per_image: Number of random tiles to sample per image per epoch.
                             For validation with CenterCrop, set to 1 to avoid duplicates.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.tile_size = tile_size
        
        # Проверка tiles_per_image на корректность
        if tiles_per_image < 1:
            raise ValueError(f"tiles_per_image must be >= 1, got {tiles_per_image}")
        self.tiles_per_image = tiles_per_image
        
        # Предупреждение о потенциальном дублировании при валидации
        if split == "val" and tiles_per_image > 1:
            logger.warning(
                f"Validation split with tiles_per_image={tiles_per_image} > 1. "
                "Ensure transforms use RandomCrop, not CenterCrop, to avoid duplicates."
            )

        # Пути к директориям
        self.images_dir = self.data_dir / "train" / "images"
        self.masks_dir = self.data_dir / "train" / "gt"

        # Проверка существования директорий
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        # Сбор файлов по split
        self.samples = self._collect_samples()
        logger.info(f"Found {len(self.samples)} images for split '{split}'")

    def _collect_samples(self) -> list[tuple[Path, Path]]:
        """Collect image-mask pairs based on split."""
        samples = []

        for city in TRAIN_CITIES:
            for tile_idx in range(1, TILES_PER_CITY + 1):
                # Определение принадлежности к train/val
                is_val_tile = tile_idx <= VAL_TILES_PER_CITY

                if self.split == "train" and is_val_tile:
                    continue
                if self.split == "val" and not is_val_tile:
                    continue

                # Формирование имени файла
                image_name = f"{city}{tile_idx}.tif"
                image_path = self.images_dir / image_name
                mask_path = self.masks_dir / image_name

                if image_path.exists() and mask_path.exists():
                    samples.append((image_path, mask_path))
                else:
                    logger.warning(f"Missing files for {city}{tile_idx}")

        return samples

    def __len__(self) -> int:
        """Return total number of samples (images * tiles_per_image)."""
        return len(self.samples) * self.tiles_per_image

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with 'image' and 'mask' tensors.
        """
        # Определение какое изображение и какой тайл
        image_idx = idx // self.tiles_per_image
        image_path, mask_path = self.samples[image_idx]

        # Загрузка изображения и маски
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        # Применение трансформаций
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            # Если нет трансформаций, конвертируем в tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float()

        # Конвертируем маску в тензор если это numpy array
        # (ToTensorV2 конвертирует только image, не mask)
        if not isinstance(mask, torch.Tensor):
            # Явно приводим к contiguous array перед конвертацией
            mask = torch.from_numpy(np.ascontiguousarray(mask)).float()

        # Бинаризация маски (0 или 1)
        mask = (mask > 0).float()

        return {"image": image, "mask": mask}

    def _load_image(self, path: Path) -> np.ndarray:
        """Load image as RGB numpy array."""
        try:
            # Попытка загрузки через rasterio (для GeoTIFF)
            with rasterio.open(path) as src:
                # Читаем как (C, H, W) и транспонируем в (H, W, C)
                image = src.read([1, 2, 3]).transpose(1, 2, 0)
        except rasterio.errors.RasterioIOError:
            # Fallback на PIL если rasterio не может открыть файл
            image = np.array(Image.open(path).convert("RGB"))

        return image.astype(np.uint8)

    def _load_mask(self, path: Path) -> np.ndarray:
        """Load mask as grayscale numpy array."""
        try:
            with rasterio.open(path) as src:
                mask = src.read(1)
        except rasterio.errors.RasterioIOError:
            # Fallback на PIL если rasterio не может открыть файл
            mask = np.array(Image.open(path).convert("L"))

        return mask.astype(np.uint8)

    def get_full_image(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get full resolution image and mask (for inference).

        Args:
            idx: Image index (not sample index).

        Returns:
            Tuple of (image, mask) as numpy arrays.
        """
        if idx >= len(self.samples):
            raise IndexError(f"Image index {idx} out of range")

        image_path, mask_path = self.samples[idx]
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        return image, mask

    @property
    def num_images(self) -> int:
        """Return number of unique images in dataset."""
        return len(self.samples)

    def get_image_path(self, idx: int) -> Path:
        """
        Get image file path by index.

        Args:
            idx: Image index (not sample index).

        Returns:
            Path to image file.
        """
        if idx >= len(self.samples):
            raise IndexError(f"Image index {idx} out of range")
        return self.samples[idx][0]

    def get_city_name(self, idx: int) -> str:
        """
        Get city name for image by index.

        Args:
            idx: Image index (not sample index).

        Returns:
            City name (e.g., 'austin', 'chicago', 'tyrol-w').
        """
        image_path = self.get_image_path(idx)
        # Извлекаем город через regex: всё до последней цифры в имени файла
        # austin1 -> austin, tyrol-w15 -> tyrol-w, vienna36 -> vienna
        match = re.match(r"^(.+?)\d+$", image_path.stem)
        if match:
            return match.group(1)
        return image_path.stem
