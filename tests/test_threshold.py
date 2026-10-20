"""Tests for thresholding operations."""

import numpy as np
import pytest

from visionkit.exceptions import UnsupportedFormatError, ValidationError
import visionkit as vk


def test_threshold_binary(synthetic_gray_image):
    val, thresh_img = vk.threshold(synthetic_gray_image, thresh=127, maxval=255, method="binary")
    assert val == 127.0
    assert thresh_img.shape == synthetic_gray_image.shape
    unique_vals = set(np.unique(thresh_img))
    assert unique_vals.issubset({0, 255})


def test_threshold_otsu():
    # Create an image with bimodal pixel intensity distribution
    bimodal = np.zeros((100, 100), dtype=np.uint8)
    bimodal[:50, :] = 40
    bimodal[50:, :] = 200
    val, thresh_img = vk.threshold(bimodal, method="otsu")
    assert 40 <= val <= 200
    assert thresh_img.shape == bimodal.shape
    assert set(np.unique(thresh_img)).issubset({0, 255})


def test_threshold_invalid_method(synthetic_gray_image):
    with pytest.raises(UnsupportedFormatError, match="Unsupported threshold method"):
        vk.threshold(synthetic_gray_image, method="invalid_thresh")  # type: ignore


def test_adaptive_threshold(synthetic_gray_image):
    thresh_img = vk.adaptive_threshold(synthetic_gray_image, block_size=11, c=2.0)
    assert thresh_img.shape == synthetic_gray_image.shape
    assert thresh_img.dtype == np.uint8


def test_adaptive_threshold_invalid_block_size(synthetic_gray_image):
    with pytest.raises(ValidationError, match="must be an odd integer"):
        vk.adaptive_threshold(synthetic_gray_image, block_size=10)
