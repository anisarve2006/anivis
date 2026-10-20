"""Geometric image transformations and resizing."""

from typing import Optional, Union
import cv2
import numpy as np

from visionkit.exceptions import UnsupportedFormatError, ValidationError
from visionkit.types import InterpolationMode
from visionkit.validation import validate_image, validate_numeric_range


INTERPOLATION_MAP = {
    "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos4": cv2.INTER_LANCZOS4,
}


def resize(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    interpolation: InterpolationMode = "linear",
) -> np.ndarray:
    """Resize an image to target dimensions or scale factors.

    If only width is supplied, height is automatically computed to preserve the
    original aspect ratio. Similarly, if only height is supplied, width is computed.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    width : int, optional
        Target image width in pixels.
    height : int, optional
        Target image height in pixels.
    fx : float, optional
        Scale factor along horizontal axis.
    fy : float, optional
        Scale factor along vertical axis.
    interpolation : {"linear", "nearest", "cubic", "area", "lanczos4"}, default="linear"
        Interpolation method to use.

    Returns
    -------
    np.ndarray
        Resized image array.

    Raises
    ------
    ValidationError
        If sizing parameters or scales are invalid.
    UnsupportedFormatError
        If an unknown interpolation mode is requested.
    """
    validate_image(image, param_name="image")

    orig_h, orig_w = image.shape[:2]

    interp_key = interpolation.lower()
    if interp_key not in INTERPOLATION_MAP:
        raise UnsupportedFormatError(
            f"Unsupported interpolation '{interpolation}'. Choose from {list(INTERPOLATION_MAP.keys())}."
        )
    interp_flag = INTERPOLATION_MAP[interp_key]

    if fx is not None or fy is not None:
        if width is not None or height is not None:
            raise ValidationError("Cannot specify both explicit dimensions (width/height) and scale factors (fx/fy).")
        scale_x = float(fx if fx is not None else (fy or 1.0))
        scale_y = float(fy if fy is not None else (fx or 1.0))
        validate_numeric_range(scale_x, "fx", min_val=0.0, inclusive_min=False)
        validate_numeric_range(scale_y, "fy", min_val=0.0, inclusive_min=False)
        return cv2.resize(image, (0, 0), fx=scale_x, fy=scale_y, interpolation=interp_flag)

    if width is None and height is None:
        raise ValidationError("Either (width, height) or scale factors (fx, fy) must be provided.")

    if width is not None:
        validate_numeric_range(width, "width", min_val=1)
        width = int(width)

    if height is not None:
        validate_numeric_range(height, "height", min_val=1)
        height = int(height)

    # Aspect-ratio preservation if only one dimension is specified
    if width is not None and height is None:
        scale = width / orig_w
        height = max(1, int(round(orig_h * scale)))
    elif height is not None and width is None:
        scale = height / orig_h
        width = max(1, int(round(orig_w * scale)))

    assert width is not None and height is not None
    return cv2.resize(image, (width, height), interpolation=interp_flag)


def flip(
    image: np.ndarray,
    direction: Union[str, int] = "horizontal",
) -> np.ndarray:
    """Flip an image horizontally, vertically, or along both axes.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    direction : {"horizontal", "vertical", "both", 0, 1, -1}
        Direction of flipping.
    """
    validate_image(image)
    if isinstance(direction, str):
        dir_lower = direction.lower()
        if dir_lower in ("horizontal", "h", "lr"):
            code = 1
        elif dir_lower in ("vertical", "v", "ud"):
            code = 0
        elif dir_lower in ("both", "hv"):
            code = -1
        else:
            raise ValidationError(
                f"Invalid flip direction '{direction}'. Expected 'horizontal', 'vertical', or 'both'."
            )
    else:
        if direction not in (0, 1, -1):
            raise ValidationError(f"Flip code must be 0, 1, or -1, got {direction}.")
        code = direction

    return cv2.flip(image, code)


def crop(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Crop a bounding box region from an image.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    x : int
        Starting x-coordinate (left column).
    y : int
        Starting y-coordinate (top row).
    width : int
        Width of the crop.
    height : int
        Height of the crop.
    """
    validate_image(image)
    validate_numeric_range(x, "x", min_val=0)
    validate_numeric_range(y, "y", min_val=0)
    validate_numeric_range(width, "width", min_val=1)
    validate_numeric_range(height, "height", min_val=1)

    h, w = image.shape[:2]
    if x >= w or y >= h:
        raise ValidationError(
            f"Crop start coordinates ({x}, {y}) are outside image bounds ({w}x{h})."
        )

    end_x = min(w, x + width)
    end_y = min(h, y + height)

    return image[y:end_y, x:end_x].copy()
