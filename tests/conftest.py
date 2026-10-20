"""Test fixtures and synthetic image generators."""

from pathlib import Path
from typing import Generator
import numpy as np
import pytest


@pytest.fixture
def synthetic_rgb_image() -> np.ndarray:
    """Generate a 100x120 synthetic 3-channel RGB image."""
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    # Add colored gradient / shapes
    img[:50, :60, 0] = 255  # Red quadrant
    img[:50, 60:, 1] = 255  # Green quadrant
    img[50:, :60, 2] = 255  # Blue quadrant
    img[50:, 60:, :] = 200  # Gray quadrant
    return img


@pytest.fixture
def synthetic_gray_image() -> np.ndarray:
    """Generate a 100x120 synthetic 2D grayscale image."""
    img = np.zeros((100, 120), dtype=np.uint8)
    img[25:75, 30:90] = 255  # White rectangle on black background
    return img


@pytest.fixture
def synthetic_rgba_image() -> np.ndarray:
    """Generate a 100x120 synthetic 4-channel RGBA image."""
    img = np.zeros((100, 120, 4), dtype=np.uint8)
    img[:, :, :3] = 180
    img[:, :, 3] = 255  # Full alpha
    return img


@pytest.fixture
def temp_image_path(tmp_path: Path) -> Generator[Path, None, None]:
    """Temporary image path fixture."""
    yield tmp_path / "test_image.png"
