"""Tests for validation module."""

import numpy as np
import pytest

from visionkit.exceptions import InvalidImageError, InvalidKernelError, ValidationError
from visionkit.validation import validate_image, validate_kernel_size, validate_numeric_range


def test_validate_image_valid(synthetic_rgb_image):
    result = validate_image(synthetic_rgb_image)
    assert result is synthetic_rgb_image


def test_validate_image_none():
    with pytest.raises(InvalidImageError, match="cannot be None"):
        validate_image(None)

    assert validate_image(None, allow_none=True) is None


def test_validate_image_not_ndarray():
    with pytest.raises(InvalidImageError, match="Expected 'image' to be a numpy.ndarray"):
        validate_image([1, 2, 3])


def test_validate_image_empty():
    with pytest.raises(InvalidImageError, match="empty"):
        validate_image(np.array([]))


def test_validate_image_invalid_dims():
    with pytest.raises(InvalidImageError, match="dimensions"):
        validate_image(np.ones((5,)))  # 1D array

    with pytest.raises(InvalidImageError, match="dimensions"):
        validate_image(np.ones((2, 2, 2, 2)))  # 4D array


def test_validate_image_invalid_channels():
    with pytest.raises(InvalidImageError, match="channels"):
        validate_image(np.ones((10, 10, 5), dtype=np.uint8))


def test_validate_kernel_size_int():
    assert validate_kernel_size(5) == (5, 5)


def test_validate_kernel_size_tuple():
    assert validate_kernel_size((3, 5)) == (3, 5)


def test_validate_kernel_size_even_rejected_by_default():
    with pytest.raises(InvalidKernelError, match="must be odd"):
        validate_kernel_size(4)


def test_validate_kernel_size_even_allowed():
    assert validate_kernel_size(4, allow_even=True) == (4, 4)


def test_validate_kernel_size_negative():
    with pytest.raises(InvalidKernelError, match="strictly positive"):
        validate_kernel_size(-3)


def test_validate_numeric_range():
    assert validate_numeric_range(5, "param", min_val=0, max_val=10) == 5

    with pytest.raises(ValidationError, match="greater than"):
        validate_numeric_range(-1, "param", min_val=0)

    with pytest.raises(ValidationError, match="less than"):
        validate_numeric_range(15, "param", max_val=10)
