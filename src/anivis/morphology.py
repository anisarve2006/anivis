"""Morphological image processing operations."""

from typing import Literal, Optional
import cv2
import numpy as np

from anivis.exceptions import UnsupportedFormatError, ValidationError
from anivis.types import KernelSize
from anivis.validation import validate_image, validate_kernel_size, validate_numeric_range


SHAPE_MAP = {
    "rect": cv2.MORPH_RECT,
    "cross": cv2.MORPH_CROSS,
    "ellipse": cv2.MORPH_ELLIPSE,
}


def get_structuring_element(
    shape: Literal["rect", "cross", "ellipse"] = "rect",
    kernel_size: KernelSize = (3, 3),
) -> np.ndarray:
    """Create a morphological structuring element (kernel).

    Parameters
    ----------
    shape : {"rect", "cross", "ellipse"}, default="rect"
        Kernel shape.
    kernel_size : int or tuple of (int, int), default=(3, 3)
        Kernel dimensions.

    Returns
    -------
    np.ndarray
        Structuring element uint8 matrix.
    """
    kx, ky = validate_kernel_size(kernel_size, allow_even=True)
    shape_key = shape.lower()
    if shape_key not in SHAPE_MAP:
        raise UnsupportedFormatError(
            f"Unsupported structuring element shape '{shape}'. Choose from {list(SHAPE_MAP.keys())}."
        )
    return cv2.getStructuringElement(SHAPE_MAP[shape_key], (kx, ky))


def _resolve_kernel(
    kernel: Optional[np.ndarray],
    kernel_size: KernelSize,
) -> np.ndarray:
    """Helper to return provided kernel or generate default rectangular kernel."""
    if kernel is not None:
        if not isinstance(kernel, np.ndarray) or kernel.ndim != 2:
            raise ValidationError("Kernel must be a 2D numpy array.")
        return kernel
    return get_structuring_element("rect", kernel_size)


def erode(
    image: np.ndarray,
    kernel_size: KernelSize = (3, 3),
    iterations: int = 1,
    kernel: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Erode an image using a structuring element.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    kernel_size : int or tuple of (int, int), default=(3, 3)
        Kernel dimensions (used if `kernel` is None).
    iterations : int, default=1
        Number of erosion steps.
    kernel : np.ndarray, optional
        Custom structuring element.
    """
    validate_image(image)
    validate_numeric_range(iterations, "iterations", min_val=1)
    k = _resolve_kernel(kernel, kernel_size)
    return cv2.erode(image, k, iterations=iterations)


def dilate(
    image: np.ndarray,
    kernel_size: KernelSize = (3, 3),
    iterations: int = 1,
    kernel: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Dilate an image using a structuring element.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    kernel_size : int or tuple of (int, int), default=(3, 3)
        Kernel dimensions (used if `kernel` is None).
    iterations : int, default=1
        Number of dilation steps.
    kernel : np.ndarray, optional
        Custom structuring element.
    """
    validate_image(image)
    validate_numeric_range(iterations, "iterations", min_val=1)
    k = _resolve_kernel(kernel, kernel_size)
    return cv2.dilate(image, k, iterations=iterations)


def morphology_open(
    image: np.ndarray,
    kernel_size: KernelSize = (3, 3),
    iterations: int = 1,
    kernel: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Perform morphological opening (erosion followed by dilation).

    Useful for removing small noise and background specks.
    """
    validate_image(image)
    validate_numeric_range(iterations, "iterations", min_val=1)
    k = _resolve_kernel(kernel, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, k, iterations=iterations)


def morphology_close(
    image: np.ndarray,
    kernel_size: KernelSize = (3, 3),
    iterations: int = 1,
    kernel: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Perform morphological closing (dilation followed by erosion).

    Useful for closing small holes and gaps within foreground objects.
    """
    validate_image(image)
    validate_numeric_range(iterations, "iterations", min_val=1)
    k = _resolve_kernel(kernel, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, k, iterations=iterations)


def morphology_gradient(
    image: np.ndarray,
    kernel_size: KernelSize = (3, 3),
    kernel: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute morphological gradient (difference between dilation and erosion)."""
    validate_image(image)
    k = _resolve_kernel(kernel, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, k)


def top_hat(
    image: np.ndarray,
    kernel_size: KernelSize = (9, 9),
    kernel: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute Top Hat transform (difference between image and its opening)."""
    validate_image(image)
    k = _resolve_kernel(kernel, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, k)


def black_hat(
    image: np.ndarray,
    kernel_size: KernelSize = (9, 9),
    kernel: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute Black Hat transform (difference between closing and original image)."""
    validate_image(image)
    k = _resolve_kernel(kernel, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, k)
