"""Tests for transformation module."""

import numpy as np
import pytest

from visionkit.exceptions import ValidationError
import visionkit as vk


def test_resize_explicit_dimensions(synthetic_rgb_image):
    resized = vk.resize(synthetic_rgb_image, width=60, height=50)
    assert resized.shape == (50, 60, 3)


def test_resize_width_only_aspect_ratio(synthetic_rgb_image):
    # original is 100 high, 120 wide (ratio h/w = 100/120 = 5/6)
    resized = vk.resize(synthetic_rgb_image, width=240)
    assert resized.shape[1] == 240
    assert resized.shape[0] == 200


def test_resize_height_only_aspect_ratio(synthetic_rgb_image):
    resized = vk.resize(synthetic_rgb_image, height=50)
    assert resized.shape[0] == 50
    assert resized.shape[1] == 60


def test_resize_scale_factors(synthetic_rgb_image):
    scaled = vk.resize(synthetic_rgb_image, fx=0.5, fy=0.5)
    assert scaled.shape == (50, 60, 3)


def test_resize_missing_arguments(synthetic_rgb_image):
    with pytest.raises(ValidationError, match="Either"):
        vk.resize(synthetic_rgb_image)


def test_flip_horizontal(synthetic_gray_image):
    flipped = vk.flip(synthetic_gray_image, direction="horizontal")
    assert flipped.shape == synthetic_gray_image.shape
    np.testing.assert_array_equal(flipped[:, 0], synthetic_gray_image[:, -1])


def test_flip_vertical_and_both(synthetic_gray_image):
    v = vk.flip(synthetic_gray_image, direction="vertical")
    assert v.shape == synthetic_gray_image.shape
    b = vk.flip(synthetic_gray_image, direction="both")
    assert b.shape == synthetic_gray_image.shape


def test_crop(synthetic_rgb_image):
    cropped = vk.crop(synthetic_rgb_image, x=10, y=10, width=40, height=30)
    assert cropped.shape == (30, 40, 3)


def test_crop_out_of_bounds(synthetic_rgb_image):
    with pytest.raises(ValidationError, match="outside image bounds"):
        vk.crop(synthetic_rgb_image, x=1000, y=1000, width=50, height=50)
