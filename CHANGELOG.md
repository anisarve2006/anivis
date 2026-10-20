# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-20

### Added
- Core package architecture with `src/visionkit` layout.
- Custom exception hierarchy: `VisionKitError`, `ValidationError`, `InvalidImageError`, `InvalidKernelError`, `ImageReadError`, `ImageWriteError`, and `UnsupportedFormatError`.
- Input validation module for array dimensions, channel counts, dtypes, kernel sizes, and numeric ranges.
- Image I/O: `vk.read()` and `vk.save()` with native RGB color space default and automatic directory creation.
- Color transformations: `vk.grayscale()`, `vk.to_rgb()`, `vk.rgb_to_bgr()`, `vk.bgr_to_rgb()`, `vk.rgb_to_hsv()`, `vk.hsv_to_rgb()`.
- Geometric transformations: `vk.resize()` with single-dimension aspect-ratio preservation, `vk.flip()`, and `vk.crop()`.
- Smoothing & Filtering: `vk.gaussian_blur()`, `vk.median_blur()`, `vk.box_blur()`, and `vk.bilateral_filter()`.
- Edge Detection: `vk.detect_edges()` (supporting Canny, Sobel, Laplacian, Scharr) and `vk.sobel_gradients()`.
- Thresholding: `vk.threshold()` (binary, Otsu, triangle) and `vk.adaptive_threshold()`.
- Morphology: `vk.erode()`, `vk.dilate()`, `vk.morphology_open()`, `vk.morphology_close()`, `vk.morphology_gradient()`, `vk.top_hat()`, and `vk.black_hat()`.
- Quality Metrics: `vk.mean_squared_error()`, `vk.psnr()`, and `vk.ssim()`.
- Comprehensive automated test suite with synthetic image fixtures.
- Quickstart example script `example.py` with automatic sample generator.
