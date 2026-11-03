from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

CLASSES = ("peace", "stop", "thumbs_up")
IMAGE_SIZE = 128


ImageSource = bytes | BinaryIO | Image.Image | str | Path


def open_oriented_rgb_image(image_source: ImageSource) -> Image.Image:
    """Open an uploaded image, apply EXIF orientation, and convert it to RGB."""
    try:
        if isinstance(image_source, Image.Image):
            image = image_source
        elif isinstance(image_source, bytes):
            image = Image.open(BytesIO(image_source))
        else:
            image = Image.open(image_source)
        return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The selected file is not a supported or valid image.") from exc


def load_rgb_array(image_source: ImageSource) -> np.ndarray:
    """Apply EXIF orientation, then decode, resize, and return RGB pixels in 0-255."""
    try:
        image = open_oriented_rgb_image(image_source).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
        )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The selected file is not a supported or valid image.") from exc

    return np.asarray(image, dtype=np.float32)


def load_dataset(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load class-folder images without silently skipping unreadable files."""
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    images: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []

    for class_index, class_name in enumerate(CLASSES):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class folder '{class_name}' in dataset directory: {root}"
            )
        class_files = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not class_files:
            raise ValueError(f"No supported images found in class folder: {class_dir}")

        for image_path in class_files:
            try:
                images.append(load_rgb_array(image_path))
            except ValueError as exc:
                raise ValueError(f"Could not load dataset image '{image_path}': {exc}") from exc
            labels.append(class_index)
            paths.append(str(image_path.relative_to(root)))

    return np.stack(images), np.asarray(labels, dtype=np.int64), paths
