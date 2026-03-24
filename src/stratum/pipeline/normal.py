"""Surface normal prediction pipeline — Sapiens per-pixel normals."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from stratum.config import NORMAL_FILE, SEG_FILE
from stratum.pipeline.bucket import load_bucketed_image, parse_bucket_dims
from stratum.sapiens import preprocess, postprocess_resize


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def process(
    image_path: Path,
    output_dir: Path,
    normal_model,
    device,
    aspect_bucket: str | None = None,
) -> bool:
    """Predict per-pixel surface normals and save to *output_dir/normal.npy*.

    Requires segmentation to have already run — the foreground mask from
    ``seg.npy`` is used to zero out background normals.

    The saved array has shape ``(H, W, 3)`` with XYZ normal components
    per pixel, L2-normalised and stored as float16.

    Returns ``True`` on success, ``False`` on failure.
    """
    try:
        seg_path = output_dir / SEG_FILE
        if not seg_path.exists():
            eprint(f"warning: seg file missing for {image_path}, skipping normals")
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
            result = normal_model(tensor)

        if isinstance(result, list):
            result = result[0]
        if result.ndim == 3:
            result = result.unsqueeze(0)

        result = postprocess_resize(result, bucket_h, bucket_w)

        normal_map = result.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)

        norms = np.linalg.norm(normal_map, axis=-1, keepdims=True)
        normal_map = normal_map / (norms + 1e-5)

        normal_map[~fg_mask] = 0.0
        normal_map = normal_map.astype(np.float16)

        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / NORMAL_FILE, normal_map)
        return True

    except Exception as exc:
        eprint(f"warning: normal estimation failed for {image_path}: {exc}")
        return False
