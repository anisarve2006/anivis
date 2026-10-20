"""Image input and output utilities."""

from pathlib import Path
from typing import Union
import cv2
import numpy as np

from visionkit.exceptions import ImageReadError, ImageWriteError, UnsupportedFormatError
from visionkit.types import ColorMode
from visionkit.validation import validate_image


def read(
    path: Union[str, Path],
    mode: ColorMode = "rgb",
) -> np.ndarray:
    """Read an image from disk.

    By default, images are loaded into RGB color space for consistent, modern
    Python semantics across libraries (NumPy, Matplotlib, PyTorch).

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the image file to read.
    mode : {"rgb", "bgr", "gray", "grayscale"}, default="rgb"
        Desired color representation.

    Returns
    -------
    np.ndarray
        Loaded image array (uint8).

    Raises
    ------
    ImageReadError
        If the file does not exist, cannot be opened, or cannot be decoded.
    UnsupportedFormatError
        If an invalid mode is specified.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise ImageReadError(f"Image file not found: '{path}'.")

    if not path_obj.is_file():
        raise ImageReadError(f"Specified path is not a file: '{path}'.")

    mode_lower = mode.lower()
    if mode_lower in ("gray", "grayscale"):
        img = cv2.imread(str(path_obj), cv2.IMREAD_GRAYSCALE)
    elif mode_lower in ("rgb", "bgr"):
        img = cv2.imread(str(path_obj), cv2.IMREAD_COLOR)
        if img is not None and mode_lower == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mode_lower in ("rgba", "bgra"):
        img = cv2.imread(str(path_obj), cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3 and img.shape[2] == 4 and mode_lower == "rgba":
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        raise UnsupportedFormatError(
            f"Unsupported read mode: '{mode}'. Expected one of ('rgb', 'bgr', 'gray', 'rgba')."
        )

    if img is None:
        raise ImageReadError(
            f"Failed to decode image from '{path}'. File may be corrupted, empty, "
            f"or in an unsupported format."
        )

    return img


def save(
    image: np.ndarray,
    path: Union[str, Path],
    mode: ColorMode = "rgb",
    create_dirs: bool = True,
) -> str:
    """Save an image array to disk.

    Parameters
    ----------
    image : np.ndarray
        Image array to save.
    path : str or pathlib.Path
        Output file path (extension determines format, e.g. .png, .jpg).
    mode : {"rgb", "bgr", "gray", "rgba"}, default="rgb"
        Color space of the input image array.
    create_dirs : bool, default=True
        Whether to create missing parent directories automatically.

    Returns
    -------
    str
        Resolved string path of the saved image.

    Raises
    ------
    InvalidImageError
        If the image array is invalid.
    ImageWriteError
        If writing fails due to permissions, disk space, or invalid extension.
    UnsupportedFormatError
        If an unsupported mode is passed.
    """
    validate_image(image, param_name="image")
    path_obj = Path(path)

    if create_dirs and path_obj.parent:
        try:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        except OSError as err:
            raise ImageWriteError(
                f"Failed to create directory '{path_obj.parent}': {err}"
            ) from err

    to_write = image
    mode_lower = mode.lower()

    if image.ndim == 3:
        channels = image.shape[2]
        if mode_lower == "rgb":
            if channels == 3:
                to_write = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif channels == 4:
                to_write = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
            elif channels == 1:
                to_write = image[:, :, 0]
        elif mode_lower == "bgr":
            to_write = image
        elif mode_lower in ("gray", "grayscale") and channels == 1:
            to_write = image[:, :, 0]
        elif mode_lower in ("rgba", "bgra"):
            if mode_lower == "rgba" and channels == 4:
                to_write = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
            else:
                to_write = image
        else:
            raise UnsupportedFormatError(
                f"Unsupported save mode '{mode}' for image with shape {image.shape}."
            )
    elif image.ndim == 2:
        # Grayscale 2D array
        to_write = image

    try:
        success = cv2.imwrite(str(path_obj), to_write)
    except Exception as err:
        raise ImageWriteError(f"Error writing image to '{path_obj}': {err}") from err

    if not success:
        raise ImageWriteError(
            f"Failed to write image to '{path_obj}'. Ensure the file extension is supported "
            f"(e.g., .jpg, .png, .bmp, .webp) and the path is writable."
        )

    return str(path_obj)
