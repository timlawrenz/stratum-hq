"""Aspect ratio bucketing — assign images to resolution buckets and load cropped versions."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

from stratum.config import DEFAULT_ASPECT_BUCKETS


def compute_aspect_ratio(width: int, height: int) -> float:
    """Compute aspect ratio as width / height."""
    return width / height if height > 0 else 1.0


def assign_aspect_bucket(
    width: int,
    height: int,
    buckets: list[tuple[int, int]] | None = None,
) -> str:
    """Assign image to the closest aspect ratio bucket.

    Returns bucket name as ``"WxH"`` string (e.g. ``"1024x1024"``).
    Uses *DEFAULT_ASPECT_BUCKETS* when *buckets* is ``None``.
    """
    if buckets is None:
        buckets = DEFAULT_ASPECT_BUCKETS

    target_ratio = compute_aspect_ratio(width, height)

    best_bucket: tuple[int, int] | None = None
    best_diff = float("inf")

    for bucket_w, bucket_h in buckets:
        diff = abs(compute_aspect_ratio(bucket_w, bucket_h) - target_ratio)
        if diff < best_diff:
            best_diff = diff
            best_bucket = (bucket_w, bucket_h)

    assert best_bucket is not None
    return f"{best_bucket[0]}x{best_bucket[1]}"


def parse_bucket_dims(aspect_bucket: str) -> tuple[int, int] | None:
    """Parse ``"832x1216"`` or ``"bucket_832x1216"`` into ``(w, h)``."""
    if not aspect_bucket or not isinstance(aspect_bucket, str):
        return None
    s = aspect_bucket
    if s.startswith("bucket_"):
        s = s[len("bucket_"):]
    if "x" not in s:
        return None
    try:
        w_str, h_str = s.split("x", 1)
        return int(w_str), int(h_str)
    except Exception:
        return None


def load_bucketed_image(
    image_path: str | Path,
    bucket_w: int,
    bucket_h: int,
) -> Image.Image:
    """Resize-to-cover then center-crop to exact *bucket_w* × *bucket_h*.

    Returns an RGB :class:`PIL.Image.Image`.
    """
    with Image.open(image_path) as im:
        img = im.convert("RGB")
    w, h = img.size

    if w <= 0 or h <= 0:
        return img

    scale = max(bucket_w / w, bucket_h / h)
    new_w = int(math.ceil(w * scale))
    new_h = int(math.ceil(h * scale))

    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), resample=Image.BICUBIC)

    left = max(0, (new_w - bucket_w) // 2)
    top = max(0, (new_h - bucket_h) // 2)
    return img.crop((left, top, left + bucket_w, top + bucket_h))
