"""Tests for filtering operations."""

import numpy as np
import pytest

from visionkit.exceptions import InvalidKernelError
import visionkit as vk


def test_gaussian_blur(synthetic_rgb_image):
    blurred = vk.gaussian_blur(synthetic_rgb_image, kernel_size=(5, 5), sigma=1.2)
    assert blurred.shape == synthetic_rgb_image.shape
    assert blurred.dtype == synthetic_rgb_image.dtype


def test_gaussian_blur_single_int_kernel(synthetic_gray_image):
    blurred = vk.gaussian_blur(synthetic_gray_image, kernel_size=3)
    assert blurred.shape == synthetic_gray_image.shape


def test_gaussian_blur_even_kernel_raises():
    img = np.ones((20, 20), dtype=np.uint8)
    with pytest.raises(InvalidKernelError, match="must be odd"):
        vk.gaussian_blur(img, kernel_size=(4, 4))


def test_median_blur(synthetic_gray_image):
    blurred = vk.median_blur(synthetic_gray_image, kernel_size=3)
    assert blurred.shape == synthetic_gray_image.shape


def test_box_blur(synthetic_rgb_image):
    blurred = vk.box_blur(synthetic_rgb_image, kernel_size=(3, 3))
    assert blurred.shape == synthetic_rgb_image.shape


def test_bilateral_filter(synthetic_rgb_image):
    filtered = vk.bilateral_filter(synthetic_rgb_image, diameter=5, sigma_color=50, sigma_space=50)
    assert filtered.shape == synthetic_rgb_image.shape
