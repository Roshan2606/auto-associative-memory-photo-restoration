from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_image(file_obj):
    """Open a user-uploaded image and convert it to grayscale."""
    image = Image.open(file_obj)
    return ImageOps.grayscale(image.convert("RGB"))


def resize_to_size(image, target_size=(32, 32)):
    """Resize a grayscale image to a compact low-resolution pattern."""
    return image.resize(target_size, Image.Resampling.LANCZOS)


def image_to_bipolar_pattern(image, target_size=(32, 32)):
    """Convert a grayscale image into a bipolar pattern for Hopfield storage."""
    resized = resize_to_size(image, target_size=target_size)
    pixels = np.asarray(resized, dtype=np.float32)
    pattern = np.where(pixels >= 128.0, 1.0, -1.0)
    return pattern.reshape(-1)


def bipolar_to_grayscale_array(pattern):
    """Convert a bipolar vector back to a 0-255 grayscale image array."""
    vector = np.asarray(pattern, dtype=np.float32).reshape(-1)
    grayscale = ((vector + 1.0) / 2.0) * 255.0
    return grayscale.astype(np.uint8)


def pattern_to_image(pattern, size):
    """Create a PIL image from a bipolar pattern."""
    array = bipolar_to_grayscale_array(pattern).reshape(size)
    return Image.fromarray(array, mode="L")


def damage_pattern(pattern, damage_percent, seed=42):
    """Apply random damage to a pattern by zeroing a fraction of pixels."""
    if not 0 <= damage_percent <= 100:
        raise ValueError("Damage percentage must be between 0 and 100.")

    damaged = np.asarray(pattern, dtype=np.float32).copy()
    if damage_percent == 0:
        return damaged

    rng = np.random.default_rng(seed + int(damage_percent * 100))
    total_pixels = damaged.size
    missing_count = int(round((damage_percent / 100.0) * total_pixels))
    missing_count = max(1, min(missing_count, total_pixels))

    indices = rng.choice(total_pixels, size=missing_count, replace=False)
    damaged[indices] = 0.0
    return damaged


def compute_accuracy(original_pattern, recalled_pattern):
    """Pixel-by-pixel agreement between the original and recalled bipolar pattern."""
    original = np.asarray(original_pattern).reshape(-1)
    recalled = np.asarray(recalled_pattern).reshape(-1)
    if original.size != recalled.size:
        raise ValueError("Pattern sizes do not match for accuracy calculation.")
    return float(np.mean(original == recalled))


def ensure_results_dir(path):
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
