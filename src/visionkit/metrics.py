"""Image comparison and quality metrics (MSE, PSNR, SSIM)."""

from typing import Union
import cv2
import numpy as np

from visionkit.exceptions import ValidationError
from visionkit.validation import validate_image, validate_numeric_range


def mean_squared_error(
    image1: np.ndarray,
    image2: np.ndarray,
) -> float:
    """Compute the Mean Squared Error (MSE) between two images.

    Parameters
    ----------
    image1 : np.ndarray
        First image.
    image2 : np.ndarray
        Second image.

    Returns
    -------
    float
        Mean squared error value (>= 0.0).

    Raises
    ------
    ValidationError
        If images have different shapes.
    """
    validate_image(image1, param_name="image1")
    validate_image(image2, param_name="image2")

    if image1.shape != image2.shape:
        raise ValidationError(
            f"Image shapes must match for MSE calculation: {image1.shape} vs {image2.shape}."
        )

    # Fast SIMD path for uint8 or float32 arrays
    if image1.dtype == np.uint8 and image2.dtype == np.uint8:
        sq_diff = cv2.norm(image1, image2, cv2.NORM_L2SQR)
        return float(sq_diff / image1.size)

    diff = image1.astype(np.float32) - image2.astype(np.float32)
    return float(np.mean(diff * diff))


# Short alias
mse = mean_squared_error


def psnr(
    image1: np.ndarray,
    image2: np.ndarray,
    max_val: float = 255.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Parameters
    ----------
    image1 : np.ndarray
        First image.
    image2 : np.ndarray
        Second image.
    max_val : float, default=255.0
        Maximum possible pixel value (e.g. 255 for uint8, 1.0 for float).

    Returns
    -------
    float
        PSNR value in decibels. Returns float('inf') if images are identical.
    """
    validate_numeric_range(max_val, "max_val", min_val=0.0, inclusive_min=False)
    err = mean_squared_error(image1, image2)
    if err == 0.0:
        return float("inf")
    return float(20.0 * np.log10(max_val) - 10.0 * np.log10(err))


def ssim(
    image1: np.ndarray,
    image2: np.ndarray,
    k1: float = 0.01,
    k2: float = 0.03,
    win_size: int = 7,
    data_range: float = 255.0,
) -> float:
    """Compute the Structural Similarity Index (SSIM) between two images.

    Parameters
    ----------
    image1 : np.ndarray
        First image (grayscale or multi-channel).
    image2 : np.ndarray
        Second image.
    k1 : float, default=0.01
        SSIM stability constant.
    k2 : float, default=0.03
        SSIM stability constant.
    win_size : int, default=7
        Gaussian window size (must be odd positive integer).
    data_range : float, default=255.0
        Dynamic range of the pixel values.

    Returns
    -------
    float
        Mean SSIM index in range [-1.0, 1.0]. 1.0 indicates perfect structural identity.
    """
    validate_image(image1, param_name="image1")
    validate_image(image2, param_name="image2")

    if image1.shape != image2.shape:
        raise ValidationError(
            f"Image shapes must match for SSIM calculation: {image1.shape} vs {image2.shape}."
        )

    if win_size % 2 == 0 or win_size < 3:
        raise ValidationError(f"win_size must be an odd integer >= 3, got {win_size}.")

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # If multi-channel, compute per-channel and average
    if image1.ndim == 3 and image1.shape[2] > 1:
        channel_ssims = []
        for ch in range(image1.shape[2]):
            s = ssim(
                image1[:, :, ch],
                image2[:, :, ch],
                k1=k1,
                k2=k2,
                win_size=win_size,
                data_range=data_range,
            )
            channel_ssims.append(s)
        return float(np.mean(channel_ssims))

    # Single channel / 2D computation
    im1 = image1.astype(np.float64)
    im2 = image2.astype(np.float64)

    # Gaussian filter
    sigma = 1.5
    mu1 = cv2.GaussianBlur(im1, (win_size, win_size), sigma)
    mu2 = cv2.GaussianBlur(im2, (win_size, win_size), sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(im1 * im1, (win_size, win_size), sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(im2 * im2, (win_size, win_size), sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(im1 * im2, (win_size, win_size), sigma) - mu1_mu2

    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = num / (den + 1e-12)
    return float(np.mean(ssim_map))
