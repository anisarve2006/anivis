"""Tests for image I/O operations."""

from pathlib import Path
import numpy as np
import pytest

from visionkit.exceptions import ImageReadError, UnsupportedFormatError
import visionkit as vk


def test_save_and_read_rgb_roundtrip(synthetic_rgb_image, tmp_path):
    out_path = tmp_path / "saved.png"
    saved_path = vk.save(synthetic_rgb_image, out_path)
    assert Path(saved_path).exists()

    loaded = vk.read(out_path, mode="rgb")
    assert loaded.shape == synthetic_rgb_image.shape
    # Lossless PNG should match exactly
    np.testing.assert_array_equal(loaded, synthetic_rgb_image)


def test_save_and_read_grayscale(synthetic_gray_image, tmp_path):
    out_path = tmp_path / "gray.png"
    vk.save(synthetic_gray_image, out_path)

    loaded = vk.read(out_path, mode="gray")
    assert loaded.ndim == 2
    np.testing.assert_array_equal(loaded, synthetic_gray_image)


def test_save_auto_creates_directories(synthetic_gray_image, tmp_path):
    nested_path = tmp_path / "nested" / "subfolder" / "image.png"
    vk.save(synthetic_gray_image, nested_path, create_dirs=True)
    assert nested_path.exists()


def test_read_missing_file():
    with pytest.raises(ImageReadError, match="Image file not found"):
        vk.read("non_existent_image_12345.png")


def test_read_invalid_mode(synthetic_rgb_image, tmp_path):
    img_path = tmp_path / "test.png"
    vk.save(synthetic_rgb_image, img_path)
    with pytest.raises(UnsupportedFormatError, match="Unsupported read mode"):
        vk.read(img_path, mode="invalid_mode")  # type: ignore


def test_save_invalid_image(tmp_path):
    with pytest.raises(Exception):
        vk.save("not_an_image", tmp_path / "out.png")  # type: ignore
