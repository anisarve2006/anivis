<div align="center">

# 📷 VisionKit

**A modular, reliable, and developer-friendly Python computer vision library.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

</div>

---

## 🎯 Mission

Make reusable computer vision and visual perception tools accessible through a clean, consistent Python API.

### 📐 Design Principles

- **Modular:** Install and use only the functionality you need.
- **Reliable:** Validated inputs, predictable outputs, automated tests.
- **RGB by Default:** Native NumPy arrays in RGB format—no more unexpected BGR color inversion headaches.
- **Efficient:** Measure performance before optimizing.
- **Extensible:** Support established backends (OpenCV / NumPy / SciPy) instead of reinventing every algorithm.
- **Developer-Friendly:** Typed signatures, helpful validation errors, and practical documentation.

---

## ⚡ Quickstart

### Installation

Install VisionKit in editable mode with development dependencies:

```bash
# Optional: create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

### 5-Line Example

```python
import visionkit as vk

image = vk.read("input.jpg")
gray = vk.grayscale(image)
resized = vk.resize(gray, width=640)
edges = vk.detect_edges(resized)
vk.save(edges, "output.png")

print("Image processing completed.")
```

Run the built-in example script:

```bash
python example.py
```

---

## 📦 Core Capabilities

### 1. Image I/O
- `vk.read(path, mode="rgb")`: Read image files into standard RGB or grayscale NumPy arrays with informative error reporting on missing or corrupt files.
- `vk.save(image, path, mode="rgb")`: Save NumPy arrays to disk with automatic parent directory creation.

### 2. Color Conversion
- `vk.grayscale(image, keep_dims=False)`: Convert RGB, BGR, or RGBA images to 2D or 3D grayscale arrays.
- `vk.to_rgb(image)`: Normalize any single-channel, grayscale, or RGBA image to 3-channel RGB.
- `vk.rgb_to_hsv(image)` / `vk.hsv_to_rgb(image)`: Fast HSV color space transformations.

### 3. Geometric Transformations
- `vk.resize(image, width=None, height=None, fx=None, fy=None, interpolation="linear")`: Resizing with automatic aspect ratio calculation when only one dimension is supplied.
- `vk.flip(image, direction="horizontal")`: Flip along horizontal, vertical, or both axes.
- `vk.crop(image, x, y, width, height)`: Safe bounding box cropping with bounds validation.

### 4. Smoothing and Filtering
- `vk.gaussian_blur(image, kernel_size=(5, 5), sigma=1.0)`: Gaussian smoothing with validated odd kernel dimensions.
- `vk.median_blur(image, kernel_size=5)`: Salt-and-pepper noise removal.
- `vk.box_blur(image, kernel_size=(5, 5))`: Normalized average box blur.
- `vk.bilateral_filter(image, diameter=9, sigma_color=75.0, sigma_space=75.0)`: Edge-preserving smoothing.

### 5. Edge and Gradient Detection
- `vk.detect_edges(image, method="canny", low_threshold=50, high_threshold=150)`: Clean edge masks (supports `canny`, `sobel`, `laplacian`, `scharr`).
- `vk.sobel_gradients(image)`: Returns directional gradients `(grad_x, grad_y)`, gradient magnitude, and angle in radians.

### 6. Thresholding & Binarization
- `vk.threshold(image, thresh=127, maxval=255, method="binary")`: Global thresholding, Otsu's thresholding, and Triangle method.
- `vk.adaptive_threshold(image, method="gaussian", block_size=11, c=2.0)`: Local adaptive thresholding for uneven lighting conditions.

### 7. Morphology
- `vk.erode`, `vk.dilate`: Fundamental morphological operators.
- `vk.morphology_open`, `vk.morphology_close`: Noise filtering and hole closure.
- `vk.morphology_gradient`, `vk.top_hat`, `vk.black_hat`: Advanced structural feature extraction.

### 8. Image Comparison Metrics
- `vk.mean_squared_error(im1, im2)` (or `vk.mse`): Pixel error.
- `vk.psnr(im1, im2)`: Peak Signal-to-Noise Ratio in dB.
- `vk.ssim(im1, im2)`: Structural Similarity Index.

---

## 🧪 Testing

Run the automated test suite using `pytest`:

```bash
pytest -v
```

Run test suite with code coverage:

```bash
pytest --cov=visionkit --cov-report=term-missing
```

---

## 🗺️ Development Roadmap

- [x] **01 Core Image Processing**
  - [x] Robust I/O (RGB default)
  - [x] Input validation & custom exception hierarchy
  - [x] Color space utilities & Grayscale
  - [x] Geometric transformations & Aspect-ratio resizing
  - [x] Filtering & Smoothing
  - [x] Edge detection (Canny, Sobel, Laplacian, Scharr)
  - [x] Thresholding (Binary, Otsu, Adaptive)
  - [x] Morphological operations
  - [x] Image quality metrics (MSE, PSNR, SSIM)
- [ ] **02 Feature Extraction**
  - Corner detection (Harris, Shi-Tomasi, FAST)
  - Keypoint descriptors & Feature matching (ORB, SIFT)
  - Hough transforms (lines and circles)
- [ ] **03 Visual Perception**
  - Optical flow (Lucas-Kanade, Farneback)
  - Visual object tracking (CSRT, KCF, ByteTrack wrapper)
  - Camera calibration & lens distortion correction
- [ ] **04 Quality and Performance**
  - Automated CI matrix (Python 3.9 - 3.13, Linux, macOS, Windows)
  - Benchmarks & performance regression tracking
- [ ] **05 Public Release**
  - Sphinx documentation site on ReadTheDocs
  - PyPI release configuration & GitHub Actions deployment

---

## 📄 License

VisionKit is open-source software licensed under the [MIT License](LICENSE).
