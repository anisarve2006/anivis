"""Exception hierarchy for VisionKit."""

from typing import Optional


class VisionKitError(Exception):
    """Base exception for all VisionKit errors."""


class ValidationError(VisionKitError, ValueError):
    """Raised when input parameters fail validation checks."""


class InvalidImageError(ValidationError):
    """Raised when an image array is invalid (wrong type, shape, or dtype)."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.details = details or {}


class InvalidKernelError(ValidationError):
    """Raised when a kernel size or kernel matrix is invalid."""


class ImageIOError(VisionKitError, IOError):
    """Base exception for image input/output errors."""


class ImageReadError(ImageIOError):
    """Raised when reading/decoding an image from disk or stream fails."""


class ImageWriteError(ImageIOError):
    """Raised when writing/encoding an image to disk or stream fails."""


class UnsupportedFormatError(VisionKitError, ValueError):
    """Raised when an unsupported mode, format, or algorithm is requested."""
