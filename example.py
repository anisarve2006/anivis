"""VisionKit quickstart example.

Run:
    python example.py
"""

from pathlib import Path
import numpy as np
import visionkit as vk


def ensure_sample_image(filename: str = "input.jpg") -> None:
    """Create a sample input image if one doesn't exist."""
    path = Path(filename)
    if not path.exists():
        print(f"'{filename}' not found. Generating a sample test image...")
        # Create a visually rich test pattern with shapes and gradients
        h, w = 480, 640
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Background gradient
        for y in range(h):
            img[y, :, 0] = int(255 * (y / h))
            img[y, :, 1] = int(200 * (1 - y / h))
            img[y, :, 2] = 120

        # Draw geometric shapes (boxes, circles)
        img[100:250, 150:350] = [255, 255, 255]  # White rectangle
        img[200:380, 300:500] = [50, 220, 255]   # Yellowish-cyan box
        img[80:180, 420:560] = [230, 80, 80]     # Reddish-blue box

        vk.save(img, filename)
        print(f"Sample test image saved to '{filename}'.")


def main() -> None:
    # Ensure input image exists
    ensure_sample_image("input.jpg")

    # VisionKit workflow
    image = vk.read("input.jpg")
    gray = vk.grayscale(image)
    resized = vk.resize(gray, width=640)
    edges = vk.detect_edges(resized)
    vk.save(edges, "output.png")

    print("Image processing completed.")
    print("Output saved to 'output.png'.")


if __name__ == "__main__":
    main()
