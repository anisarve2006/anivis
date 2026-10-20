"""Micro-benchmarks for core VisionKit operations."""

import time
from typing import Callable
import numpy as np
import visionkit as vk


def benchmark(func: Callable, name: str, iterations: int = 50) -> float:
    """Benchmark a function and print mean execution time in milliseconds."""
    # Warmup
    for _ in range(5):
        func()

    start = time.perf_counter()
    for _ in range(iterations):
        func()
    duration = time.perf_counter() - start

    avg_ms = (duration / iterations) * 1000.0
    print(f"[{name:<25}] {avg_ms:8.3f} ms / op ({iterations} runs)")
    return avg_ms


def main() -> None:
    print("=" * 60)
    print("VisionKit Core Performance Benchmark (1080p Image: 1920x1080)")
    print("=" * 60)

    # 1080p synthetic image
    h, w = 1080, 1920
    rgb = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    gray = vk.grayscale(rgb)

    benchmark(lambda: vk.grayscale(rgb), "Grayscale (1080p RGB)")
    benchmark(lambda: vk.resize(rgb, width=640), "Resize (1920 -> 640)")
    benchmark(lambda: vk.gaussian_blur(gray, kernel_size=(5, 5)), "Gaussian Blur (5x5)")
    benchmark(lambda: vk.detect_edges(gray, method="canny"), "Canny Edge Detection")
    benchmark(lambda: vk.threshold(gray, method="otsu"), "Otsu Thresholding")
    benchmark(lambda: vk.erode(gray, kernel_size=(3, 3)), "Erosion (3x3)")
    benchmark(lambda: vk.dilate(gray, kernel_size=(3, 3)), "Dilation (3x3)")
    benchmark(lambda: vk.mean_squared_error(gray, gray), "MSE Metric")
    print("=" * 60)


if __name__ == "__main__":
    main()
