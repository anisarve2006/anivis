"""Common type aliases and definitions for VisionKit."""

from typing import Literal, Tuple, Union
import numpy as np
import numpy.typing as npt

# Image array representation (NumPy ndarray)
ImageArray = npt.NDArray[Union[np.uint8, np.float32, np.float64, np.uint16]]

# Spatial kernel size (width, height) or single dimension
KernelSize = Union[int, Tuple[int, int]]

# Interpolation modes
InterpolationMode = Literal["linear", "nearest", "cubic", "area", "lanczos4"]

# Edge detection methods
EdgeMethod = Literal["canny", "sobel", "laplacian", "scharr"]

# Threshold methods
ThresholdMethod = Literal[
    "binary",
    "binary_inv",
    "trunc",
    "tozero",
    "tozero_inv",
    "otsu",
    "triangle",
]

# Color modes
ColorMode = Literal["rgb", "bgr", "gray", "grayscale", "rgba", "bgra"]
