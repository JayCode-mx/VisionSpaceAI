"""
Example usage of ImagePreprocessor and SSIM similarity scoring.
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from image_preprocessor import ImagePreprocessor, compute_ssim_similarity


def main():
    print("=== ImagePreprocessor & SSIM Demonstration ===\n")

    # 1. Initialize preprocessor
    preprocessor = ImagePreprocessor(
        target_size=(224, 224),
        clip_limit=2.0,
        tile_grid_size=(8, 8),
        pad_color=(0, 0, 0),  # Black letterbox padding
    )
    print(f"Initialized preprocessor with target_size={preprocessor.target_size}")

    # 2. Create sample images with different aspect ratios
    output_dir = Path("output_examples")
    output_dir.mkdir(exist_ok=True)

    # Image A: Landscape (320x160) gradient with shapes
    img_a = Image.new("RGB", (320, 160), color=(50, 80, 120))
    draw_a = ImageDraw.Draw(img_a)
    draw_a.ellipse((60, 30, 180, 130), fill=(200, 160, 90), outline=(255, 255, 255), width=3)
    draw_a.rectangle((200, 40, 290, 120), fill=(80, 180, 120), outline=(255, 255, 255), width=3)

    # Image B: Same content but slightly altered (simulating illumination/color shift)
    img_b = Image.new("RGB", (320, 160), color=(60, 85, 115))
    draw_b = ImageDraw.Draw(img_b)
    draw_b.ellipse((60, 30, 180, 130), fill=(195, 155, 95), outline=(250, 250, 250), width=3)
    draw_b.rectangle((200, 40, 290, 120), fill=(85, 175, 125), outline=(250, 250, 250), width=3)

    # Image C: Portrait (160x320) with completely different content
    img_c = Image.new("RGB", (160, 320), color=(140, 50, 50))
    draw_c = ImageDraw.Draw(img_c)
    draw_c.polygon([(80, 40), (140, 280), (20, 280)], fill=(220, 200, 80))

    # Save original samples
    img_a.save(output_dir / "sample_a_original.png")
    img_b.save(output_dir / "sample_b_original.png")
    img_c.save(output_dir / "sample_c_original.png")
    print(f"Saved original samples to {output_dir}/")

    # 3. Process images
    processed_a = preprocessor(img_a)
    processed_b = preprocessor(img_b)
    processed_c = preprocessor(img_c)

    processed_a.save(output_dir / "sample_a_preprocessed.png")
    processed_b.save(output_dir / "sample_b_preprocessed.png")
    processed_c.save(output_dir / "sample_c_preprocessed.png")
    print(f"Processed images: All standardized to size {processed_a.size}")

    # 4. Compute SSIM similarity scores
    ssim_a_self = compute_ssim_similarity(processed_a, processed_a)
    ssim_a_b = compute_ssim_similarity(processed_a, processed_b)
    ssim_a_c = compute_ssim_similarity(processed_a, processed_c)

    print("\n=== SSIM Structural Similarity Scores ===")
    print(f"SSIM(A, A) [Identical]:                {ssim_a_self:.4f}")
    print(f"SSIM(A, B) [Slight variations]:        {ssim_a_b:.4f}")
    print(f"SSIM(A, C) [Completely different]:     {ssim_a_c:.4f}")


if __name__ == "__main__":
    main()
