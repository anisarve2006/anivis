"""Image filtering, smoothing, and blur operations."""

from typing import Optional, Union
import cv2
import numpy as np

from anivis.types import KernelSize
from anivis.validation import validate_image, validate_kernel_size, validate_numeric_range


def gaussian_blur(
    image: np.ndarray,
    kernel_size: KernelSize = (5, 5),
    sigma: float = 1.0,
    sigma_y: Optional[float] = None,
) -> np.ndarray:
    """Apply Gaussian blur to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or multi-channel).
    kernel_size : int or tuple of (int, int), default=(5, 5)
        Gaussian kernel dimensions. Must consist of positive odd integers.
    sigma : float, default=1.0
        Gaussian kernel standard deviation along the X-axis.
    sigma_y : float, optional
        Gaussian kernel standard deviation along the Y-axis. Defaults to `sigma`.

    Returns
    -------
    np.ndarray
        Blurred image.

    Raises
    ------
    InvalidImageError
        If image is invalid.
    InvalidKernelError
        If kernel_size dimensions are non-positive or even.
    ValidationError
        If sigma values are negative.
    """
    validate_image(image, param_name="image")
    kx, ky = validate_kernel_size(kernel_size, param_name="kernel_size", allow_even=False)
    validate_numeric_range(sigma, "sigma", min_val=0.0)

    sig_y = sigma if sigma_y is None else sigma_y
    validate_numeric_range(sig_y, "sigma_y", min_val=0.0)

    return cv2.GaussianBlur(image, (kx, ky), sigmaX=float(sigma), sigmaY=float(sig_y))


def median_blur(
    image: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """Apply median filter to an image for salt-and-pepper noise reduction."""
    validate_image(image, param_name="image")
    kx, _ = validate_kernel_size(kernel_size, param_name="kernel_size", allow_even=False)
    return cv2.medianBlur(image, kx)


def box_blur(
    image: np.ndarray,
    kernel_size: KernelSize = (5, 5),
) -> np.ndarray:
    """Apply normalized box blur to an image."""
    validate_image(image, param_name="image")
    kx, ky = validate_kernel_size(kernel_size, param_name="kernel_size", allow_even=True)
    return cv2.blur(image, (kx, ky))


def bilateral_filter(
    image: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Apply edge-preserving bilateral filtering to an image."""
    validate_image(image, param_name="image")
    validate_numeric_range(diameter, "diameter", min_val=1)
    validate_numeric_range(sigma_color, "sigma_color", min_val=0.0)
    validate_numeric_range(sigma_space, "sigma_space", min_val=0.0)

    # OpenCV bilateralFilter requires 8-bit or floating-point single/three-channel image
    return cv2.bilateralFilter(
        image,
        d=int(diameter),
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space),
    )
