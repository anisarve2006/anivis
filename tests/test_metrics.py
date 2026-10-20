"""Tests for image comparison and quality metrics."""

import numpy as np
import pytest

from visionkit.exceptions import ValidationError
import visionkit as vk


def test_mse_identical(synthetic_rgb_image):
    err = vk.mean_squared_error(synthetic_rgb_image, synthetic_rgb_image)
    assert err == 0.0


def test_mse_different(synthetic_rgb_image):
    distorted = synthetic_rgb_image.copy()
    distorted[0, 0, 0] = (int(distorted[0, 0, 0]) + 10) % 256
    err = vk.mean_squared_error(synthetic_rgb_image, distorted)
    assert err > 0.0


def test_mse_shape_mismatch(synthetic_rgb_image, synthetic_gray_image):
    with pytest.raises(ValidationError, match="shapes must match"):
        vk.mean_squared_error(synthetic_rgb_image, synthetic_gray_image)


def test_psnr_identical(synthetic_rgb_image):
    score = vk.psnr(synthetic_rgb_image, synthetic_rgb_image)
    assert score == float("inf")


def test_psnr_noisy(synthetic_rgb_image):
    noisy = np.clip(synthetic_rgb_image.astype(np.int16) + 5, 0, 255).astype(np.uint8)
    score = vk.psnr(synthetic_rgb_image, noisy)
    assert 20.0 < score < 60.0


def test_ssim_identical(synthetic_rgb_image):
    score = vk.ssim(synthetic_rgb_image, synthetic_rgb_image)
    assert pytest.approx(score, abs=1e-4) == 1.0


def test_ssim_distorted(synthetic_rgb_image):
    noisy = np.clip(synthetic_rgb_image.astype(np.int16) + 20, 0, 255).astype(np.uint8)
    score = vk.ssim(synthetic_rgb_image, noisy)
    assert 0.0 <= score < 1.0
