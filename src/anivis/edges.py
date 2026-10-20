"""Edge detection algorithms and gradient operators."""

from typing import Tuple
import cv2
import numpy as np

from anivis.color import grayscale
from anivis.exceptions import UnsupportedFormatError, ValidationError
from anivis.types import EdgeMethod
from anivis.validation import validate_image, validate_numeric_range


def detect_edges(
    image: np.ndarray,
    method: EdgeMethod = "canny",
    low_threshold: float = 50.0,
    high_threshold: float = 150.0,
    aperture_size: int = 3,
    l2_gradient: bool = False,
) -> np.ndarray:
    """Detect edges in an image using specified algorithm.

    If a color image is provided, it is automatically converted to grayscale
    prior to edge calculation.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or RGB).
    method : {"canny", "sobel", "laplacian", "scharr"}, default="canny"
        Edge detection algorithm to apply.
    low_threshold : float, default=50.0
        Lower hysteresis threshold for Canny edge detector.
    high_threshold : float, default=150.0
        Upper hysteresis threshold for Canny edge detector.
    aperture_size : int, default=3
        Aperture size for Sobel / Laplacian operators (must be 1, 3, 5, or 7).
    l2_gradient : bool, default=False
        Whether to use L2 norm in Canny detector.

    Returns
    -------
    np.ndarray
        2D uint8 edge mask where edge pixels are 255 and background is 0.

    Raises
    ------
    InvalidImageError
        If image is invalid.
    ValidationError
        If thresholds or aperture sizes are invalid.
    UnsupportedFormatError
        If an unknown edge detection method is requested.
    """
    validate_image(image, param_name="image")

    # If image is multi-channel, convert to grayscale
    gray = grayscale(image) if image.ndim == 3 else image

    # Ensure grayscale is uint8
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    method_lower = method.lower()

    if method_lower == "canny":
        validate_numeric_range(low_threshold, "low_threshold", min_val=0.0)
        validate_numeric_range(high_threshold, "high_threshold", min_val=0.0)
        if low_threshold > high_threshold:
            raise ValidationError(
                f"low_threshold ({low_threshold}) cannot exceed high_threshold ({high_threshold})."
            )
        if aperture_size not in (3, 5, 7):
            raise ValidationError(
                f"aperture_size for Canny must be 3, 5, or 7, got {aperture_size}."
            )
        edges = cv2.Canny(
            gray,
            threshold1=float(low_threshold),
            threshold2=float(high_threshold),
            apertureSize=aperture_size,
            L2gradient=l2_gradient,
        )
        return edges

    if method_lower == "sobel":
        if aperture_size not in (1, 3, 5, 7):
            raise ValidationError(
                f"aperture_size for Sobel must be 1, 3, 5, or 7, got {aperture_size}."
            )
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=aperture_size)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=aperture_size)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        norm_mag = np.clip(magnitude / (magnitude.max() + 1e-8) * 255.0, 0, 255).astype(np.uint8)
        return norm_mag

    if method_lower == "laplacian":
        if aperture_size not in (1, 3, 5, 7):
            raise ValidationError(
                f"aperture_size for Laplacian must be 1, 3, 5, or 7, got {aperture_size}."
            )
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=aperture_size)
        norm_lap = np.clip(np.abs(lap) / (np.abs(lap).max() + 1e-8) * 255.0, 0, 255).astype(np.uint8)
        return norm_lap

    if method_lower == "scharr":
        grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        norm_mag = np.clip(magnitude / (magnitude.max() + 1e-8) * 255.0, 0, 255).astype(np.uint8)
        return norm_mag

    raise UnsupportedFormatError(
        f"Unsupported edge detection method '{method}'. Choose from ('canny', 'sobel', 'laplacian', 'scharr')."
    )


def sobel_gradients(
    image: np.ndarray,
    aperture_size: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute directional Sobel gradients, gradient magnitude, and orientation angle.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    aperture_size : int, default=3
        Kernel size (1, 3, 5, or 7).

    Returns
    -------
    grad_x : np.ndarray (float64)
        Horizontal gradient.
    grad_y : np.ndarray (float64)
        Vertical gradient.
    magnitude : np.ndarray (float64)
        Gradient magnitude.
    angles : np.ndarray (float64)
        Gradient direction in radians (-pi to pi).
    """
    validate_image(image)
    gray = grayscale(image) if image.ndim == 3 else image
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=aperture_size)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=aperture_size)
    magnitude = np.hypot(grad_x, grad_y)
    angles = np.arctan2(grad_y, grad_x)
    return grad_x, grad_y, magnitude, angles
