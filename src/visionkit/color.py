"""Color space conversions and adjustments."""

from typing import Literal
import cv2
import numpy as np

from visionkit.exceptions import UnsupportedFormatError
from visionkit.validation import validate_image


def grayscale(
    image: np.ndarray,
    keep_dims: bool = False,
    source_mode: Literal["rgb", "bgr", "rgba", "bgra"] = "rgb",
) -> np.ndarray:
    """Convert an image to grayscale.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D or 3D numpy array).
    keep_dims : bool, default=False
        If True, retains channel dimension as shape (H, W, 1). If False, returns (H, W).
    source_mode : {"rgb", "bgr", "rgba", "bgra"}, default="rgb"
        Color space of the input image.

    Returns
    -------
    np.ndarray
        Grayscale image.

    Raises
    ------
    InvalidImageError
        If the image is invalid.
    UnsupportedFormatError
        If the source_mode or channel count is unsupported.
    """
    validate_image(image, param_name="image")

    if image.ndim == 2:
        gray = image.copy()
    elif image.ndim == 3:
        channels = image.shape[2]
        src = source_mode.lower()
        if channels == 1:
            gray = image[:, :, 0].copy()
        elif channels == 3:
            if src == "rgb":
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif src == "bgr":
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                raise UnsupportedFormatError(
                    f"Unsupported source_mode '{source_mode}' for 3-channel image."
                )
        elif channels == 4:
            if src in ("rgb", "rgba"):
                gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            elif src in ("bgr", "bgra"):
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                raise UnsupportedFormatError(
                    f"Unsupported source_mode '{source_mode}' for 4-channel image."
                )
        else:
            raise UnsupportedFormatError(
                f"Cannot convert image with {channels} channels to grayscale."
            )
    else:
        raise UnsupportedFormatError(f"Unsupported image dimensions: {image.ndim}")

    if keep_dims:
        return np.expand_dims(gray, axis=-1)
    return gray


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to BGR."""
    validate_image(image, valid_channels=(3, 4))
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    return image


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB."""
    validate_image(image, valid_channels=(3, 4))
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return image


def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to HSV."""
    validate_image(image, valid_channels=(3,))
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def hsv_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an HSV image to RGB."""
    validate_image(image, valid_channels=(3,))
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Normalize any 2D or 3D image to a standard 3-channel RGB image."""
    validate_image(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image.copy()
