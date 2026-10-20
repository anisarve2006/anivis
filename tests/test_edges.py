"""Tests for edge detection algorithms."""

import numpy as np
import pytest

from visionkit.exceptions import UnsupportedFormatError, ValidationError
import visionkit as vk


def test_detect_edges_canny_on_gray(synthetic_gray_image):
    edges = vk.detect_edges(synthetic_gray_image, method="canny", low_threshold=50, high_threshold=150)
    assert edges.ndim == 2
    assert edges.shape == synthetic_gray_image.shape
    assert edges.dtype == np.uint8
    # There should be detected edges around the rectangle
    assert np.any(edges > 0)


def test_detect_edges_canny_on_rgb(synthetic_rgb_image):
    edges = vk.detect_edges(synthetic_rgb_image, method="canny")
    assert edges.ndim == 2
    assert edges.shape == (synthetic_rgb_image.shape[0], synthetic_rgb_image.shape[1])
    assert edges.dtype == np.uint8


def test_detect_edges_sobel(synthetic_gray_image):
    edges = vk.detect_edges(synthetic_gray_image, method="sobel")
    assert edges.ndim == 2
    assert edges.shape == synthetic_gray_image.shape


def test_detect_edges_laplacian(synthetic_gray_image):
    edges = vk.detect_edges(synthetic_gray_image, method="laplacian")
    assert edges.ndim == 2
    assert edges.shape == synthetic_gray_image.shape


def test_detect_edges_scharr(synthetic_gray_image):
    edges = vk.detect_edges(synthetic_gray_image, method="scharr")
    assert edges.ndim == 2
    assert edges.shape == synthetic_gray_image.shape


def test_detect_edges_invalid_thresholds(synthetic_gray_image):
    with pytest.raises(ValidationError, match="cannot exceed"):
        vk.detect_edges(synthetic_gray_image, low_threshold=200, high_threshold=100)


def test_detect_edges_invalid_method(synthetic_gray_image):
    with pytest.raises(UnsupportedFormatError, match="Unsupported edge detection method"):
        vk.detect_edges(synthetic_gray_image, method="magic_edges")  # type: ignore


def test_sobel_gradients(synthetic_gray_image):
    gx, gy, mag, ang = vk.sobel_gradients(synthetic_gray_image)
    assert gx.shape == synthetic_gray_image.shape
    assert gy.shape == synthetic_gray_image.shape
    assert mag.shape == synthetic_gray_image.shape
    assert ang.shape == synthetic_gray_image.shape
