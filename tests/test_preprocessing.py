"""Tests for preprocessing module."""

from __future__ import annotations

import numpy as np
import torch

from app.preprocessing import (
    create_tiles,
    get_imagenet_normalize_transform,
    image_to_tensor,
    load_image,
    merge_tiles,
    normalize_image,
    prepare_batch_for_model,
    resize_image,
    validate_image,
)


class TestValidateImage:
    """Tests for validate_image function."""

    def test_valid_rgb_image(self) -> None:
        """Valid RGB image should return True."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        assert validate_image(image) is True

    def test_grayscale_image(self) -> None:
        """Grayscale image should return False."""
        image = np.zeros((100, 100), dtype=np.uint8)
        assert validate_image(image) is False

    def test_rgba_image(self) -> None:
        """RGBA image (4 channels) should return False."""
        image = np.zeros((100, 100, 4), dtype=np.uint8)
        assert validate_image(image) is False

    def test_empty_image(self) -> None:
        """Empty image should return False."""
        image = np.zeros((0, 0, 3), dtype=np.uint8)
        assert validate_image(image) is False

    def test_non_array(self) -> None:
        """Non-array input should return False."""
        assert validate_image("not an array") is False  # type: ignore[arg-type]
        assert validate_image(None) is False  # type: ignore[arg-type]


class TestLoadImage:
    """Tests for load_image function."""

    def test_load_from_array(self) -> None:
        """Loading from numpy array should return copy."""
        original = np.ones((50, 50, 3), dtype=np.uint8) * 128
        loaded = load_image(original)

        assert loaded.shape == original.shape
        assert loaded.dtype == np.uint8
        np.testing.assert_array_equal(loaded, original)

        # Проверяем что это копия, а не тот же объект
        loaded[0, 0, 0] = 255
        assert original[0, 0, 0] == 128


class TestNormalizeImage:
    """Tests for normalize_image function."""

    def test_uint8_passthrough(self) -> None:
        """uint8 image should pass through unchanged."""
        image = np.ones((10, 10, 3), dtype=np.uint8) * 100
        normalized = normalize_image(image)

        assert normalized.dtype == np.uint8
        np.testing.assert_array_equal(normalized, image)

    def test_float_0_1_conversion(self) -> None:
        """Float [0, 1] image should be converted to uint8."""
        image = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        normalized = normalize_image(image)

        assert normalized.dtype == np.uint8
        assert normalized[0, 0, 0] == 127  # 0.5 * 255 ≈ 127

    def test_int_conversion(self) -> None:
        """Integer array should be converted to uint8."""
        image = np.ones((10, 10, 3), dtype=np.int32) * 200
        normalized = normalize_image(image)

        assert normalized.dtype == np.uint8
        assert normalized[0, 0, 0] == 200


class TestCreateTiles:
    """Tests for create_tiles function."""

    def test_single_tile_for_small_image(self) -> None:
        """Image smaller than tile_size should return single tile."""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        tiles, positions = create_tiles(image, tile_size=512, overlap=64)

        assert len(tiles) == 1
        assert len(positions) == 1
        assert positions[0] == (0, 0)
        assert tiles[0].shape == (512, 512, 3)

    def test_correct_number_of_tiles(self) -> None:
        """Should create correct number of tiles with overlap."""
        # 1024x1024 image with 512 tile and 64 overlap → stride = 448
        # Positions: 0, 448, 512 (to cover edge) → 3 positions per dimension
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        tiles, positions = create_tiles(image, tile_size=512, overlap=64)

        # At least 4 tiles (2x2 grid minimum for 1024 image)
        assert len(tiles) >= 4
        assert len(positions) == len(tiles)

    def test_tiles_have_correct_shape(self) -> None:
        """All tiles should have the specified tile_size."""
        image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
        tiles, _ = create_tiles(image, tile_size=256, overlap=32)

        for tile in tiles:
            assert tile.shape == (256, 256, 3)

    def test_tiles_cover_entire_image(self) -> None:
        """Tiles should cover the entire image without gaps."""
        h, w = 500, 700
        image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        tiles, positions = create_tiles(image, tile_size=256, overlap=32)

        # Check that we have tiles covering edges
        max_y = max(pos[0] for pos in positions)
        max_x = max(pos[1] for pos in positions)

        assert max_y + 256 >= h
        assert max_x + 256 >= w


class TestMergeTiles:
    """Tests for merge_tiles function."""

    def test_merge_single_tile(self) -> None:
        """Merging single tile should return original values."""
        tile = np.ones((256, 256), dtype=np.float32) * 0.5
        tiles = [tile]
        positions = [(0, 0)]

        merged = merge_tiles(tiles, positions, (256, 256), tile_size=256, overlap=64)

        assert merged.shape == (256, 256)
        # Values should be approximately 0.5 (weighted average of single tile)
        assert np.allclose(merged, 0.5, atol=0.1)

    def test_merge_preserves_shape(self) -> None:
        """Merged output should have correct shape."""
        tiles = [np.ones((512, 512), dtype=np.float32) * 0.5] * 4
        positions = [(0, 0), (0, 448), (448, 0), (448, 448)]

        merged = merge_tiles(tiles, positions, (960, 960), tile_size=512, overlap=64)

        assert merged.shape == (960, 960)


class TestResizeImage:
    """Tests for resize_image function."""

    def test_resize_to_target_size(self) -> None:
        """Image should be resized to exact target size."""
        image = np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)
        resized = resize_image(image, target_size=(512, 512))

        assert resized.shape == (512, 512, 3)

    def test_resize_preserves_dtype(self) -> None:
        """Resizing should preserve uint8 dtype."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        resized = resize_image(image, target_size=(50, 50))

        assert resized.dtype == np.uint8


class TestPrepareBatchForModel:
    """Tests for prepare_batch_for_model function."""

    def test_output_shape(self) -> None:
        """Output should be (B, 3, H, W) tensor."""
        tiles = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(4)]
        batch = prepare_batch_for_model(tiles, normalize=False)

        assert batch.shape == (4, 3, 256, 256)

    def test_output_dtype(self) -> None:
        """Output should be float32 tensor."""
        tiles = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)]
        batch = prepare_batch_for_model(tiles, normalize=False)

        assert batch.dtype == torch.float32

    def test_values_normalized_to_0_1(self) -> None:
        """Without ImageNet norm, values should be in [0, 1]."""
        tiles = [np.ones((64, 64, 3), dtype=np.uint8) * 255]
        batch = prepare_batch_for_model(tiles, normalize=False)

        assert batch.min() >= 0.0
        assert batch.max() <= 1.0

    def test_imagenet_normalization(self) -> None:
        """With ImageNet norm, values can be outside [0, 1]."""
        tiles = [np.ones((64, 64, 3), dtype=np.uint8) * 128]
        batch = prepare_batch_for_model(tiles, normalize=True)

        # ImageNet normalization shifts mean, values won't be in [0, 1]
        assert batch.dtype == torch.float32


class TestImageToTensor:
    """Tests for image_to_tensor function."""

    def test_adds_batch_dimension(self) -> None:
        """Single image should get batch dimension."""
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        tensor = image_to_tensor(image, normalize=False)

        assert tensor.shape == (1, 3, 128, 128)


class TestGetImagenetNormalizeTransform:
    """Tests for get_imagenet_normalize_transform function."""

    def test_returns_normalize_transform(self) -> None:
        """Should return a Normalize transform."""
        transform = get_imagenet_normalize_transform()
        assert transform is not None

    def test_transform_works_on_tensor(self) -> None:
        """Transform should work on float tensor."""
        transform = get_imagenet_normalize_transform()
        tensor = torch.rand(3, 64, 64)
        normalized = transform(tensor)

        assert normalized.shape == tensor.shape
