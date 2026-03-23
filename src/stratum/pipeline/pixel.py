"""Pixel-space crop pipeline — produce (3, H, W) float16 training tensors."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from stratum.config import PIXEL_FILE
from stratum.pipeline.bucket import load_bucketed_image, parse_bucket_dims


def _eprint(*args: object, **kwargs: object) -> None:
    print(*args, file=sys.stderr, **kwargs)


def process(image_path: Path, output_dir: Path, aspect_bucket: str) -> bool:
    """Load a bucketed image crop and save as ``pixel.npy``.

    The image is resized-to-cover + center-cropped to the bucket dimensions,
    converted to a ``(3, H, W)`` float16 array in ``[0, 1]``, and written to
    *output_dir* / :data:`~stratum.config.PIXEL_FILE`.

    Returns ``True`` on success, ``False`` on failure.
    """
    dims = parse_bucket_dims(aspect_bucket)
    if dims is None:
        _eprint(f"warning: invalid aspect_bucket '{aspect_bucket}' for {image_path}")
        return False

    bucket_w, bucket_h = dims

    try:
        img = load_bucketed_image(image_path, bucket_w, bucket_h)
    except Exception as e:
        _eprint(f"warning: failed to load image for pixel save {image_path}: {e}")
        return False

    # Convert to (3, H, W) float16 [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # (H, W, 3) → (3, H, W)
    arr = arr.astype(np.float16)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / PIXEL_FILE, arr)
    return True
