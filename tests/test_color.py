"""Tests for color conversions."""

import numpy as np
import pytest

from visionkit.exceptions import UnsupportedFormatError
import visionkit as vk


def test_grayscale_from_rgb(synthetic_rgb_image):
    gray = vk.grayscale(synthetic_rgb_image)
    assert gray.ndim == 2
    assert gray.shape == (synthetic_rgb_image.shape[0], synthetic_rgb_image.shape[1])


def test_grayscale_keep_dims(synthetic_rgb_image):
    gray_3d = vk.grayscale(synthetic_rgb_image, keep_dims=True)
    assert gray_3d.ndim == 3
    assert gray_3d.shape == (synthetic_rgb_image.shape[0], synthetic_rgb_image.shape[1], 1)


def test_grayscale_already_gray(synthetic_gray_image):
    gray = vk.grayscale(synthetic_gray_image)
    np.testing.assert_array_equal(gray, synthetic_gray_image)


def test_rgb_bgr_roundtrip(synthetic_rgb_image):
    bgr = vk.rgb_to_bgr(synthetic_rgb_image)
    rgb_back = vk.bgr_to_rgb(bgr)
    np.testing.assert_array_equal(rgb_back, synthetic_rgb_image)


def test_to_rgb_from_gray(synthetic_gray_image):
    rgb = vk.to_rgb(synthetic_gray_image)
    assert rgb.ndim == 3
    assert rgb.shape == (synthetic_gray_image.shape[0], synthetic_gray_image.shape[1], 3)
    np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 1])
    np.testing.assert_array_equal(rgb[:, :, 1], rgb[:, :, 2])


def test_to_rgb_from_rgba(synthetic_rgba_image):
    rgb = vk.to_rgb(synthetic_rgba_image)
    assert rgb.shape == (synthetic_rgba_image.shape[0], synthetic_rgba_image.shape[1], 3)


def test_to_rgb_already_rgb(synthetic_rgb_image):
    rgb = vk.to_rgb(synthetic_rgb_image)
    np.testing.assert_array_equal(rgb, synthetic_rgb_image)


def test_rgb_hsv_roundtrip(synthetic_rgb_image):
    hsv = vk.rgb_to_hsv(synthetic_rgb_image)
    rgb_back = vk.hsv_to_rgb(hsv)
    assert hsv.shape == synthetic_rgb_image.shape
    # Small rounding differences can occur in HSV color space math
    assert np.allclose(rgb_back, synthetic_rgb_image, atol=2)


def test_grayscale_from_rgba(synthetic_rgba_image):
    gray = vk.grayscale(synthetic_rgba_image, source_mode="rgba")
    assert gray.ndim == 2
    assert gray.shape == (synthetic_rgba_image.shape[0], synthetic_rgba_image.shape[1])


def test_grayscale_invalid_source_mode(synthetic_rgb_image):
    with pytest.raises(UnsupportedFormatError, match="Unsupported source_mode"):
        vk.grayscale(synthetic_rgb_image, source_mode="cmyk")  # type: ignore
