"""Crop equation images to ink content and remove blank margins."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

DEFAULT_MARGIN = 5
DEFAULT_BLOCK_SIZE = 31
DEFAULT_C = 15


def binarize_image(
    img: Image.Image,
    block_size: int = DEFAULT_BLOCK_SIZE,
    c: int = DEFAULT_C,
) -> Image.Image:
    """Adaptive binarization: black formula strokes, white background."""
    gray = np.array(img.convert("L"))
    if block_size % 2 == 0:
        block_size += 1
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )
    return Image.fromarray(binary, mode="L")


def equation_bbox(binary: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """Return the smallest box containing black (equation) pixels."""
    ink_mask = binary.point(lambda value: 255 if value == 0 else 0)
    return ink_mask.getbbox()


def binary_to_rgb(binary: Image.Image) -> Image.Image:
    """Convert black/white grayscale image to RGB (3 identical channels)."""
    return binary.convert("RGB")


def _save_binary(output_path: Path, binary: Image.Image) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    binary_to_rgb(binary).save(output_path, format="PNG")


try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    _LANCZOS = Image.LANCZOS


def scale_to_max_size(image_path: Path, max_image_size: float) -> Path:
    """Scale image down in-place only when ``max(width, height)`` exceeds ``max_image_size``."""
    max_size = int(round(max_image_size))
    if max_size <= 0:
        return image_path

    with Image.open(image_path) as img:
        width, height = img.size
        if max(width, height) <= max_size:
            return image_path

        scale = max_size / max(width, height)
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        img.convert("RGB").resize(new_size, _LANCZOS).save(image_path, format="PNG")
    return image_path


def crop_to_content(
    input_path: Path,
    output_path: Path,
    margin: int = DEFAULT_MARGIN,
    block_size: int = DEFAULT_BLOCK_SIZE,
    c: int = DEFAULT_C,
    max_image_size: float = 0,
) -> Path:
    """Binarize, find equation bounding box, crop binary image with margin, and save."""
    with Image.open(input_path) as img:
        rgb = img.convert("RGB")
        binary = binarize_image(rgb, block_size=block_size, c=c)
        bbox = equation_bbox(binary)
        if bbox is None:
            _save_binary(output_path, binary)
        else:
            left, top, right, bottom = bbox
            left = max(0, left - margin)
            top = max(0, top - margin)
            right = min(binary.width, right + margin)
            bottom = min(binary.height, bottom + margin)
            _save_binary(output_path, binary.crop((left, top, right, bottom)))

    if max_image_size > 0:
        scale_to_max_size(output_path, max_image_size)
    return output_path
