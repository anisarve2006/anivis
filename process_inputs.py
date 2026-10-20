"""Batch processing script to apply VisionKit operations to all images in the input directory.

Creates structured outputs under output/<image_name>/ with each processed stage.
"""

from pathlib import Path
import json
import time
import numpy as np
import visionkit as vk


def process_image(input_path: Path, output_dir: Path) -> dict:
    """Apply all core VisionKit processes to a single image and save structured outputs."""
    stem = input_path.stem
    target_dir = output_dir / stem
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing '{input_path.name}' -> {target_dir}")

    # 1. Read image (RGB by default)
    t0 = time.perf_counter()
    image = vk.read(input_path, mode="rgb")
    orig_shape = image.shape
    orig_dtype = str(image.dtype)

    # Standardize working resolution for consistent inspection and fast processing
    # If image is very large (> 1280px wide), resize to max width 1280 preserving aspect ratio
    working_img = image
    if image.shape[1] > 1280:
        working_img = vk.resize(image, width=1280)
        print(f"  Resized from {orig_shape[1]}x{orig_shape[0]} to {working_img.shape[1]}x{working_img.shape[0]} for processing")

    vk.save(working_img, target_dir / "01_original.png")

    # 2. Grayscale conversion
    gray = vk.grayscale(working_img)
    vk.save(gray, target_dir / "02_grayscale.png")

    # 3. Filtering: Gaussian Blur
    gauss_blur = vk.gaussian_blur(working_img, kernel_size=(5, 5), sigma=1.2)
    vk.save(gauss_blur, target_dir / "03_gaussian_blur.png")

    # 4. Filtering: Bilateral Filter (edge-preserving)
    # bilateralFilter works on 8-bit RGB or Grayscale
    bilateral = vk.bilateral_filter(working_img, diameter=9, sigma_color=75.0, sigma_space=75.0)
    vk.save(bilateral, target_dir / "04_bilateral_filter.png")

    # 5. Edge Detection: Canny
    edges_canny = vk.detect_edges(gray, method="canny", low_threshold=50, high_threshold=150)
    vk.save(edges_canny, target_dir / "05_edges_canny.png")

    # 6. Edge Detection: Sobel
    edges_sobel = vk.detect_edges(gray, method="sobel")
    vk.save(edges_sobel, target_dir / "06_edges_sobel.png")

    # 7. Thresholding: Otsu
    otsu_val, thresh_otsu = vk.threshold(gray, method="otsu")
    vk.save(thresh_otsu, target_dir / "07_threshold_otsu.png")

    # 8. Thresholding: Adaptive Gaussian
    thresh_adaptive = vk.adaptive_threshold(gray, method="gaussian", block_size=15, c=3.0)
    vk.save(thresh_adaptive, target_dir / "08_threshold_adaptive.png")

    # 9. Morphology: Opening (removes noise/specks)
    morph_open = vk.morphology_open(gray, kernel_size=(3, 3))
    vk.save(morph_open, target_dir / "09_morphology_open.png")

    # 10. Morphology: Gradient (dilation - erosion)
    morph_grad = vk.morphology_gradient(gray, kernel_size=(3, 3))
    vk.save(morph_grad, target_dir / "10_morphology_gradient.png")

    # 11. Image Metrics
    # Compare original vs blurred and original vs bilateral
    mse_gauss = vk.mean_squared_error(working_img, gauss_blur)
    psnr_gauss = vk.psnr(working_img, gauss_blur)
    ssim_gauss = vk.ssim(working_img, gauss_blur)

    mse_bilateral = vk.mean_squared_error(working_img, bilateral)
    psnr_bilateral = vk.psnr(working_img, bilateral)
    ssim_bilateral = vk.ssim(working_img, bilateral)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    metadata = {
        "file_name": input_path.name,
        "original_shape": list(orig_shape),
        "working_shape": list(working_img.shape),
        "dtype": orig_dtype,
        "processing_time_ms": round(elapsed_ms, 2),
        "otsu_threshold": round(otsu_val, 2),
        "metrics": {
            "gaussian_blur": {
                "mse": round(mse_gauss, 3),
                "psnr_db": round(psnr_gauss, 2),
                "ssim": round(ssim_gauss, 4),
            },
            "bilateral_filter": {
                "mse": round(mse_bilateral, 3),
                "psnr_db": round(psnr_bilateral, 2),
                "ssim": round(ssim_bilateral, 4),
            },
        },
        "outputs": [
            "01_original.png",
            "02_grayscale.png",
            "03_gaussian_blur.png",
            "04_bilateral_filter.png",
            "05_edges_canny.png",
            "06_edges_sobel.png",
            "07_threshold_otsu.png",
            "08_threshold_adaptive.png",
            "09_morphology_open.png",
            "10_morphology_gradient.png",
        ]
    }

    with open(target_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Completed in {elapsed_ms:.1f}ms. Generated 10 processed stages.")
    return metadata


def main() -> None:
    input_dir = Path("input")
    output_dir = Path("output")

    if not input_dir.exists():
        print(f"Directory '{input_dir}' not found.")
        return

    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff")
    image_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions])

    if not image_files:
        print(f"No image files found in '{input_dir}'.")
        return

    print("=" * 60)
    print(f"VisionKit Batch Image Processor: Found {len(image_files)} images in 'input/'")
    print("=" * 60)

    all_results = []
    for img_path in image_files:
        res = process_image(img_path, output_dir)
        all_results.append(res)

    # Save global summary report
    with open(output_dir / "batch_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Batch processing finished! Structured results saved to '{output_dir.resolve()}'")
    print("=" * 60)


if __name__ == "__main__":
    main()
