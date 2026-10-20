"""Validation utilities for images, kernels, and parameters."""

from typing import Iterable, Optional, Sequence, Tuple, Union
import numpy as np

from anivis.exceptions import InvalidImageError, InvalidKernelError, ValidationError
from anivis.types import KernelSize


DEFAULT_ALLOWED_DTYPES = (
    np.uint8,
    np.uint16,
    np.int32,
    np.float32,
    np.float64,
)


def validate_image(
    image: object,
    param_name: str = "image",
    allow_none: bool = False,
    min_dims: int = 2,
    max_dims: int = 3,
    valid_channels: Optional[Sequence[int]] = (1, 3, 4),
    allowed_dtypes: Optional[Sequence[object]] = DEFAULT_ALLOWED_DTYPES,
) -> np.ndarray:
    """Validate that the input is a valid image NumPy ndarray.

    Parameters
    ----------
    image : object
        Input array to validate.
    param_name : str, default="image"
        Name of the parameter for error reporting.
    allow_none : bool, default=False
        Whether None is an accepted value.
    min_dims : int, default=2
        Minimum number of array dimensions.
    max_dims : int, default=3
        Maximum number of array dimensions.
    valid_channels : sequence of int, optional
        Valid number of channels for 3D images. Defaults to (1, 3, 4).
    allowed_dtypes : sequence of dtypes, optional
        Allowed NumPy dtypes.

    Returns
    -------
    np.ndarray
        The validated numpy ndarray.

    Raises
    ------
    InvalidImageError
        If the image is not a numpy array, has invalid dimensions, or unsupported dtype.
    """
    if image is None:
        if allow_none:
            return None  # type: ignore[return-value]
        raise InvalidImageError(f"Parameter '{param_name}' cannot be None.")

    if not isinstance(image, np.ndarray):
        raise InvalidImageError(
            f"Expected '{param_name}' to be a numpy.ndarray, but received {type(image).__name__}."
        )

    if image.size == 0:
        raise InvalidImageError(f"'{param_name}' is empty (size is 0).")

    ndim = image.ndim
    if ndim < min_dims or ndim > max_dims:
        raise InvalidImageError(
            f"'{param_name}' must have {min_dims} to {max_dims} dimensions, but has {ndim} (shape: {image.shape})."
        )

    if ndim == 3 and valid_channels is not None:
        channels = image.shape[2]
        if channels not in valid_channels:
            raise InvalidImageError(
                f"'{param_name}' has {channels} channels, expected one of {tuple(valid_channels)} "
                f"(shape: {image.shape})."
            )

    if allowed_dtypes is not None:
        if not any(np.issubdtype(image.dtype, dt) for dt in allowed_dtypes):
            allowed_names = [getattr(dt, "__name__", str(dt)) for dt in allowed_dtypes]
            raise InvalidImageError(
                f"'{param_name}' has unsupported dtype '{image.dtype}'. "
                f"Supported dtypes: {allowed_names}."
            )

    return image


def validate_kernel_size(
    kernel_size: KernelSize,
    param_name: str = "kernel_size",
    allow_even: bool = False,
) -> Tuple[int, int]:
    """Validate and normalize kernel size to a (width, height) tuple of positive odd integers.

    Parameters
    ----------
    kernel_size : int or tuple of (int, int)
        Size of the kernel.
    param_name : str, default="kernel_size"
        Parameter name for error reporting.
    allow_even : bool, default=False
        Whether even kernel sizes are allowed.

    Returns
    -------
    tuple of (int, int)
        Normalized kernel dimensions (width, height).

    Raises
    ------
    InvalidKernelError
        If kernel_size is invalid.
    """
    if isinstance(kernel_size, int):
        kx, ky = kernel_size, kernel_size
    elif isinstance(kernel_size, (tuple, list)):
        if len(kernel_size) != 2:
            raise InvalidKernelError(
                f"'{param_name}' must be an integer or a tuple of two integers, got {kernel_size}."
            )
        kx, ky = kernel_size
        if not (isinstance(kx, int) and isinstance(ky, int)):
            raise InvalidKernelError(
                f"'{param_name}' dimensions must be integers, got ({type(kx).__name__}, {type(ky).__name__})."
            )
    else:
        raise InvalidKernelError(
            f"'{param_name}' must be an integer or tuple of two integers, got {type(kernel_size).__name__}."
        )

    if kx <= 0 or ky <= 0:
        raise InvalidKernelError(
            f"'{param_name}' dimensions must be strictly positive, got ({kx}, {ky})."
        )

    if not allow_even:
        if kx % 2 == 0 or ky % 2 == 0:
            raise InvalidKernelError(
                f"'{param_name}' dimensions must be odd numbers, got ({kx}, {ky})."
            )

    return (kx, ky)


def validate_numeric_range(
    value: Union[int, float],
    param_name: str,
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    inclusive_min: bool = True,
    inclusive_max: bool = True,
) -> Union[int, float]:
    """Validate that a numeric value lies within an expected range."""
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"Expected '{param_name}' to be a number, but received {type(value).__name__}."
        )

    if min_val is not None:
        if inclusive_min and value < min_val:
            raise ValidationError(
                f"'{param_name}' ({value}) must be greater than or equal to {min_val}."
            )
        if not inclusive_min and value <= min_val:
            raise ValidationError(
                f"'{param_name}' ({value}) must be strictly greater than {min_val}."
            )

    if max_val is not None:
        if inclusive_max and value > max_val:
            raise ValidationError(
                f"'{param_name}' ({value}) must be less than or equal to {max_val}."
            )
        if not inclusive_max and value >= max_val:
            raise ValidationError(
                f"'{param_name}' ({value}) must be strictly less than {max_val}."
            )

    return value
