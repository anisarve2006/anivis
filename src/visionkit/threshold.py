"""Image thresholding and binarization methods."""

from typing import Literal, Tuple
import cv2
import numpy as np

from visionkit.color import grayscale
from visionkit.exceptions import UnsupportedFormatError, ValidationError
from visionkit.types import ThresholdMethod
from visionkit.validation import validate_image, validate_numeric_range


THRESHOLD_FLAG_MAP = {
    "binary": cv2.THRESH_BINARY,
    "binary_inv": cv2.THRESH_BINARY_INV,
    "trunc": cv2.THRESH_TRUNC,
    "tozero": cv2.THRESH_TOZERO,
    "tozero_inv": cv2.THRESH_TOZERO_INV,
    "otsu": cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    "triangle": cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE,
}


def threshold(
    image: np.ndarray,
    thresh: float = 127.0,
    maxval: float = 255.0,
    method: ThresholdMethod = "binary",
) -> Tuple[float, np.ndarray]:
    """Apply thresholding to produce a binary or partitioned image.

    Parameters
    ----------
    image : np.ndarray
        Input image. Converted to grayscale if color.
    thresh : float, default=127.0
        Threshold value. Ignored if method is 'otsu' or 'triangle'.
    maxval : float, default=255.0
        Maximum value to use with binary thresholding.
    method : {"binary", "binary_inv", "trunc", "tozero", "tozero_inv", "otsu", "triangle"}
        Thresholding algorithm.

    Returns
    -------
    actual_thresh : float
        The threshold value used (computed threshold for Otsu/Triangle).
    result : np.ndarray
        Thresholded uint8 image.
    """
    validate_image(image)
    validate_numeric_range(maxval, "maxval", min_val=0.0)

    gray = grayscale(image) if image.ndim == 3 else image
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    method_lower = method.lower()
    if method_lower not in THRESHOLD_FLAG_MAP:
        raise UnsupportedFormatError(
            f"Unsupported threshold method '{method}'. Choose from {list(THRESHOLD_FLAG_MAP.keys())}."
        )

    flag = THRESHOLD_FLAG_MAP[method_lower]
    initial_thresh = 0.0 if method_lower in ("otsu", "triangle") else float(thresh)
    actual_thresh, result = cv2.threshold(gray, initial_thresh, float(maxval), flag)
    return float(actual_thresh), result


def adaptive_threshold(
    image: np.ndarray,
    maxval: float = 255.0,
    method: Literal["mean", "gaussian"] = "gaussian",
    threshold_type: Literal["binary", "binary_inv"] = "binary",
    block_size: int = 11,
    c: float = 2.0,
) -> np.ndarray:
    """Apply adaptive thresholding using local neighborhood statistics.

    Parameters
    ----------
    image : np.ndarray
        Input image (converted to grayscale if color).
    maxval : float, default=255.0
        Non-zero value assigned to the pixels for which condition is satisfied.
    method : {"mean", "gaussian"}, default="gaussian"
        Adaptive method.
    threshold_type : {"binary", "binary_inv"}, default="binary"
        Threshold type.
    block_size : int, default=11
        Size of a pixel neighborhood used to calculate threshold (must be odd, > 1).
    c : float, default=2.0
        Constant subtracted from the mean or weighted mean.

    Returns
    -------
    np.ndarray
        Thresholded binary image.
    """
    validate_image(image)
    if block_size <= 1 or block_size % 2 == 0:
        raise ValidationError(f"block_size must be an odd integer greater than 1, got {block_size}.")

    gray = grayscale(image) if image.ndim == 3 else image
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    adapt_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method.lower() == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
    type_flag = cv2.THRESH_BINARY_INV if threshold_type.lower() == "binary_inv" else cv2.THRESH_BINARY

    return cv2.adaptiveThreshold(
        gray,
        maxValue=float(maxval),
        adaptiveMethod=adapt_method,
        thresholdType=type_flag,
        blockSize=block_size,
        C=float(c),
    )
