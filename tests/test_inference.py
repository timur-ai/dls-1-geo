"""Tests for inference module."""

from __future__ import annotations

import numpy as np
import pytest

from app.inference import calculate_area, count_buildings


class TestCalculateArea:
    """Tests for calculate_area function."""

    def test_empty_mask_returns_zero(self) -> None:
        """Empty mask (no buildings) should return 0."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        area = calculate_area(mask, gsd_m=0.3)

        assert area == 0.0

    def test_full_mask_calculates_correctly(self) -> None:
        """Full white mask should calculate correct area."""
        # 100x100 pixels, all white (255)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        # GSD = 0.3 m/px → area = 10000 * 0.3^2 = 900 m²
        area = calculate_area(mask, gsd_m=0.3)

        assert area == pytest.approx(900.0)

    def test_partial_mask_calculates_correctly(self) -> None:
        """Partial mask should calculate correct area."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Fill 50x50 region with white
        mask[25:75, 25:75] = 255
        # 2500 pixels * 0.3^2 = 225 m²
        area = calculate_area(mask, gsd_m=0.3)

        assert area == pytest.approx(225.0)

    def test_different_gsd_values(self) -> None:
        """Different GSD values should give proportional areas."""
        mask = np.ones((100, 100), dtype=np.uint8) * 255

        area_03 = calculate_area(mask, gsd_m=0.3)
        area_06 = calculate_area(mask, gsd_m=0.6)

        # 0.6^2 / 0.3^2 = 4
        assert area_06 == pytest.approx(area_03 * 4)

    def test_returns_float(self) -> None:
        """Area should be a float."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        area = calculate_area(mask)

        assert isinstance(area, float)


class TestCountBuildings:
    """Tests for count_buildings function."""

    def test_empty_mask_returns_zero(self) -> None:
        """Empty mask should return 0 buildings."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        count = count_buildings(mask)

        assert count == 0

    def test_single_building(self) -> None:
        """Single connected component should return 1."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255  # 20x20 = 400 pixels (> min_area)
        count = count_buildings(mask, min_area_px=100)

        assert count == 1

    def test_multiple_buildings(self) -> None:
        """Multiple separated components should be counted correctly."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Three separate buildings
        mask[10:25, 10:25] = 255  # 225 pixels
        mask[10:25, 70:85] = 255  # 225 pixels
        mask[70:85, 40:55] = 255  # 225 pixels

        count = count_buildings(mask, min_area_px=100)

        assert count == 3

    def test_filters_small_components(self) -> None:
        """Small components below threshold should be filtered out."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Large building
        mask[10:30, 10:30] = 255  # 400 pixels
        # Small noise
        mask[80:85, 80:85] = 255  # 25 pixels

        count = count_buildings(mask, min_area_px=100)

        assert count == 1  # Only the large one

    def test_returns_int(self) -> None:
        """Count should be an integer."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        count = count_buildings(mask)

        assert isinstance(count, int)


class TestPredictMaskContract:
    """Contract tests for predict_mask function (without loading model)."""

    def test_output_shape_matches_input(self) -> None:
        """Output mask shape should match input image H×W."""
        # Импортируем здесь чтобы избежать загрузки модели при импорте модуля
        from app.inference import predict_mask

        # Используем маленькое изображение для быстрого теста
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        # Этот тест требует загрузки модели, пропускаем если модель недоступна
        try:
            mask = predict_mask(image, mode="resize")
            assert mask.shape == (64, 64)
        except FileNotFoundError:
            pytest.skip("Model checkpoint not available")

    def test_output_dtype_is_uint8(self) -> None:
        """Output mask should be uint8."""
        from app.inference import predict_mask

        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        try:
            mask = predict_mask(image, mode="resize")
            assert mask.dtype == np.uint8
        except FileNotFoundError:
            pytest.skip("Model checkpoint not available")

    def test_output_values_are_binary(self) -> None:
        """Output mask should contain only 0 or 255."""
        from app.inference import predict_mask

        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        try:
            mask = predict_mask(image, mode="resize")
            unique_values = np.unique(mask)
            assert all(v in [0, 255] for v in unique_values)
        except FileNotFoundError:
            pytest.skip("Model checkpoint not available")
