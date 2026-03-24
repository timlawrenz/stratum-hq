"""Body-part segmentation pipeline — Sapiens 28-class segmentation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from stratum.config import NUM_SEG_CLASSES, SEG_FILE
from stratum.pipeline.bucket import load_bucketed_image, parse_bucket_dims
from stratum.sapiens import postprocess_resize, preprocess


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def process(
    image_path: Path,
    output_dir: Path,
    seg_model,
    device,
    aspect_bucket: str | None = None,
) -> bool:
    """Run Sapiens segmentation and save to *output_dir/seg.npy*.

    The saved array has shape ``(H, W)`` with dtype uint8 where each
    pixel value is a class ID in ``[0, 27]`` (28 classes total).

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
        tensor = preprocess(img_arr).to(device)

        with torch.no_grad():
            raw = seg_model(tensor)

        # TorchScript may return a list of tensors or a single tensor.
        result = raw[0] if isinstance(raw, (list, tuple)) else raw

        # Ensure shape is (1, C, H, W) for postprocess_resize.
        if result.dim() == 3:
            result = result.unsqueeze(0)

        result = postprocess_resize(result, bucket_h, bucket_w)
        seg = result.squeeze(0).argmax(dim=0).cpu().numpy().astype(np.uint8)

        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / SEG_FILE, seg)
        return True

    except Exception as exc:
        eprint(f"warning: segmentation failed for {image_path}: {exc}")
        return False
