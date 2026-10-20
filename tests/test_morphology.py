"""Tests for morphology operations."""

import numpy as np
import pytest

from visionkit.exceptions import UnsupportedFormatError
import visionkit as vk


def test_structuring_element():
    rect = vk.get_structuring_element(shape="rect", kernel_size=(3, 3))
    assert rect.shape == (3, 3)
    np.testing.assert_array_equal(rect, np.ones((3, 3), dtype=np.uint8))


def test_structuring_element_invalid_shape():
    with pytest.raises(UnsupportedFormatError, match="Unsupported structuring element shape"):
        vk.get_structuring_element(shape="triangle")  # type: ignore


def test_erode_and_dilate(synthetic_gray_image):
    eroded = vk.erode(synthetic_gray_image, kernel_size=(3, 3))
    dilated = vk.dilate(synthetic_gray_image, kernel_size=(3, 3))

    assert eroded.shape == synthetic_gray_image.shape
    assert dilated.shape == synthetic_gray_image.shape
    # Erosion shrinks bright regions, dilation expands them
    assert np.sum(eroded) <= np.sum(synthetic_gray_image)
    assert np.sum(dilated) >= np.sum(synthetic_gray_image)


def test_morphology_open_and_close(synthetic_gray_image):
    opened = vk.morphology_open(synthetic_gray_image, kernel_size=(3, 3))
    closed = vk.morphology_close(synthetic_gray_image, kernel_size=(3, 3))

    assert opened.shape == synthetic_gray_image.shape
    assert closed.shape == synthetic_gray_image.shape


def test_morphology_gradient(synthetic_gray_image):
    grad = vk.morphology_gradient(synthetic_gray_image, kernel_size=(3, 3))
    assert grad.shape == synthetic_gray_image.shape


def test_top_hat_and_black_hat(synthetic_gray_image):
    th = vk.top_hat(synthetic_gray_image, kernel_size=(5, 5))
    bh = vk.black_hat(synthetic_gray_image, kernel_size=(5, 5))
    assert th.shape == synthetic_gray_image.shape
    assert bh.shape == synthetic_gray_image.shape
