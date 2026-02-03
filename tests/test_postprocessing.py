"""Tests for postprocessing module."""

from __future__ import annotations

import numpy as np
import pytest

from app.postprocessing import (
    calculate_coverage_percent,
    create_overlay,
)


class TestCreateOverlay:
    """Tests for create_overlay function."""

    def test_returns_correct_shape(self) -> None:
        """Overlay should have same shape as input image."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        overlay = create_overlay(image, mask)

        assert overlay.shape == image.shape

    def test_no_change_with_empty_mask(self) -> None:
        """Empty mask should not modify the image."""
        image = np.ones((50, 50, 3), dtype=np.uint8) * 128
        mask = np.zeros((50, 50), dtype=np.uint8)
        overlay = create_overlay(image, mask)

        np.testing.assert_array_equal(overlay, image)

    def test_overlay_applied_on_mask_area(self) -> None:
        """Overlay color should be applied where mask > 0."""
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 5] = 255  # Один пиксель помечен как здание

        overlay = create_overlay(image, mask)

        # Пиксель должен иметь цвет overlay (красный с альфой)
        assert overlay[5, 5, 0] > 0  # Red channel
        # Остальные пиксели не изменились
        assert overlay[0, 0, 0] == 0

    def test_raises_on_shape_mismatch(self) -> None:
        """Should raise ValueError when image and mask shapes don't match."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)

        with pytest.raises(ValueError, match="does not match"):
            create_overlay(image, mask)


class TestCalculateCoveragePercent:
    """Tests for calculate_coverage_percent function."""

    def test_empty_mask_zero_coverage(self) -> None:
        """Empty mask should return 0% coverage."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        coverage = calculate_coverage_percent(mask)

        assert coverage == 0.0

    def test_full_mask_hundred_coverage(self) -> None:
        """Full mask should return 100% coverage."""
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        coverage = calculate_coverage_percent(mask)

        assert coverage == 100.0

    def test_partial_coverage(self) -> None:
        """Partial mask should return correct percentage."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:5, :] = 255  # 50% покрытия

        coverage = calculate_coverage_percent(mask)

        assert coverage == 50.0


