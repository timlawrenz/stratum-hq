"""Depth estimation pipeline — Sapiens relative depth."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from stratum.config import DEPTH_FILE, SEG_FILE
from stratum.pipeline.bucket import load_bucketed_image, parse_bucket_dims
from stratum.sapiens import postprocess_resize, preprocess


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def process(
    image_path: Path,
    output_dir: Path,
    depth_model,
    device: str,
    aspect_bucket: str | None = None,
) -> bool:
    """Estimate relative depth with Sapiens and save to *output_dir/depth.npy*.

    Depth values are normalised to **[0, 1]** over the foreground region
    defined by the segmentation mask (``seg.npy``).  Background pixels are
    set to zero.

    The saved array has shape ``(H, W)`` stored as float16.

    Returns ``True`` on success, ``False`` on failure.
    """
    try:
        seg_path = output_dir / SEG_FILE
        if not seg_path.exists():
            eprint(
                f"warning: depth skipped for {image_path}: "
                f"segmentation mask not found ({seg_path})"
            )
            return False

        seg = np.load(seg_path)
        fg_mask = seg > 0

        dims = parse_bucket_dims(aspect_bucket) if aspect_bucket else None

        if dims:
            bucket_w, bucket_h = dims
            img = load_bucketed_image(image_path, bucket_w, bucket_h)
        else:
            with Image.open(image_path) as im:
                img = im.convert("RGB")
            bucket_w, bucket_h = img.size

        img_arr = np.array(img)
        tensor = preprocess(img_arr)
        tensor = tensor.to(device)

        with torch.no_grad():
            result = depth_model(tensor)

        if isinstance(result, list):
            result = result[0]
        if result.ndim == 3:
            result = result.unsqueeze(0)

        result = postprocess_resize(result, bucket_h, bucket_w)

        depth = result[0, 0].cpu().numpy().astype(np.float32)

        depth[~fg_mask] = 0.0

        fg_values = depth[fg_mask]
        if fg_values.size > 0:
            lo = fg_values.min()
            hi = fg_values.max()
            if hi - lo > 0:
                depth[fg_mask] = (fg_values - lo) / (hi - lo)
            else:
                depth[fg_mask] = 0.0

        depth = depth.astype(np.float16)

        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / DEPTH_FILE, depth)
        return True

    except Exception as exc:
        eprint(f"warning: depth estimation failed for {image_path}: {exc}")
        return False
