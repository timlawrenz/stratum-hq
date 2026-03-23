"""Pose-estimation pipeline — DWPose whole-body keypoints."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from stratum.config import NUM_POSE_KEYPOINTS, POSE_FILE
from stratum.pipeline.bucket import load_bucketed_image, parse_bucket_dims


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def process(
    image_path: Path,
    output_dir: Path,
    pose_model,
    aspect_bucket: str | None = None,
) -> bool:
    """Extract DWPose whole-body keypoints and save to *output_dir/pose.npy*.

    Keypoints are normalised to **[-1, 1]** relative to the bucket
    dimensions so the downstream model learns scale-invariant geometry.

    The saved array has shape ``(133, 3)`` with columns
    ``(x_norm, y_norm, confidence)`` stored as float16.

    Returns ``True`` on success, ``False`` on failure.
    """
    try:
        dims = parse_bucket_dims(aspect_bucket) if aspect_bucket else None

        if dims:
            bucket_w, bucket_h = dims
            img = load_bucketed_image(image_path, bucket_w, bucket_h)
        else:
            with Image.open(image_path) as im:
                img = im.convert("RGB")
            bucket_w, bucket_h = img.size

        img_arr = np.array(img)

        results = pose_model(img_arr, single_person=True)
        kpts_2d = results[0]  # (N, 133, 2)
        scores = results[1]   # (N, 133)

        if kpts_2d.shape[0] == 0:
            pose = np.zeros((NUM_POSE_KEYPOINTS, 3), dtype=np.float16)
        else:
            kpts = kpts_2d[0]   # (133, 2) pixel coords
            confs = scores[0]   # (133,)

            x_norm = (2.0 * kpts[:, 0] / bucket_w) - 1.0
            y_norm = (2.0 * kpts[:, 1] / bucket_h) - 1.0

            pose = np.stack([x_norm, y_norm, confs], axis=-1).astype(np.float16)

        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / POSE_FILE, pose)
        return True

    except Exception as exc:
        eprint(f"warning: pose extraction failed for {image_path}: {exc}")
        return False
