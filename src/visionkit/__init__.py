"""VisionKit: A modular, reliable, and developer-friendly Python computer vision library.

Basic usage:
------------
    >>> import visionkit as vk
    >>> image = vk.read("input.jpg")
    >>> gray = vk.grayscale(image)
    >>> resized = vk.resize(gray, width=640)
    >>> edges = vk.detect_edges(resized)
    >>> vk.save(edges, "output.png")
"""

from visionkit._version import __version__
from visionkit.color import (
    bgr_to_rgb,
    grayscale,
    hsv_to_rgb,
    rgb_to_bgr,
    rgb_to_hsv,
    to_rgb,
)
from visionkit.edges import (
    detect_edges,
    sobel_gradients,
)
from visionkit.exceptions import (
    ImageIOError,
    ImageReadError,
    ImageWriteError,
    InvalidImageError,
    InvalidKernelError,
    UnsupportedFormatError,
    ValidationError,
    VisionKitError,
)
from visionkit.filter import (
    bilateral_filter,
    box_blur,
    gaussian_blur,
    median_blur,
)
from visionkit.io import (
    read,
    save,
)
from visionkit.metrics import (
    mean_squared_error,
    mse,
    psnr,
    ssim,
)
from visionkit.morphology import (
    black_hat,
    dilate,
    erode,
    get_structuring_element,
    morphology_close,
    morphology_gradient,
    morphology_open,
    top_hat,
)
from visionkit.threshold import (
    adaptive_threshold,
    threshold,
)
from visionkit.transform import (
    crop,
    flip,
    resize,
)
from visionkit.validation import (
    validate_image,
    validate_kernel_size,
)

__all__ = [
    # Version
    "__version__",
    # I/O
    "read",
    "save",
    # Color
    "grayscale",
    "to_rgb",
    "rgb_to_bgr",
    "bgr_to_rgb",
    "rgb_to_hsv",
    "hsv_to_rgb",
    # Transform
    "resize",
    "flip",
    "crop",
    # Filter
    "gaussian_blur",
    "median_blur",
    "box_blur",
    "bilateral_filter",
    # Edge detection
    "detect_edges",
    "sobel_gradients",
    # Thresholding
    "threshold",
    "adaptive_threshold",
    # Morphology
    "erode",
    "dilate",
    "morphology_open",
    "morphology_close",
    "morphology_gradient",
    "top_hat",
    "black_hat",
    "get_structuring_element",
    # Metrics
    "mean_squared_error",
    "mse",
    "psnr",
    "ssim",
    # Validation & Exceptions
    "validate_image",
    "validate_kernel_size",
    "VisionKitError",
    "ValidationError",
    "InvalidImageError",
    "InvalidKernelError",
    "ImageIOError",
    "ImageReadError",
    "ImageWriteError",
    "UnsupportedFormatError",
]
