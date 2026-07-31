"""Sapiens2 pointmap estimation — per-pixel 3D XYZ coordinates.

Produces ``pointmap.npy`` — H×W×3 float16 array with XYZ coordinates in
camera frame.  Background pixels (from seg2.npy) are zeroed.

Model returns ``(pointmap, scale)`` — pointmap is divided by scale to get
metric coordinates.

Requires ``seg2.npy`` to already exist in the output directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from stratum2.config import POINTMAP_FILE, SEG2_FILE


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def process(
    image_path: Path,
    output_dir: Path,
    pointmap_model,
    device,
    aspect_bucket: str | None = None,
) -> bool:
    """Run Sapiens2 pointmap estimation and save ``pointmap.npy``.

    Returns ``True`` on success, ``False`` on failure.
    """
    try:
        # --- Dependency: seg2.npy must exist ---
        seg_path = output_dir / SEG2_FILE
        if not seg_path.exists():
            eprint(f"warning: pointmap skipped for {image_path}: seg2 not found")
            return False

        import cv2

        image = cv2.imread(str(image_path))  # BGR
        if image is None:
            eprint(f"warning: cannot read {image_path}")
            return False

        seg = np.load(seg_path)
        fg_mask = seg > 0

        # Sapiens2 pipeline
        data = pointmap_model.pipeline(dict(img=image))
        data = pointmap_model.data_preprocessor(data)
        inputs = data["inputs"].to(device)

        with torch.no_grad():
            pointmap, scale = pointmap_model(inputs)  # (1×3×H×W, scalar)

        # Convert to metric
        pointmap = pointmap / scale

        # Unpad
        ds = data["data_samples"]
        if isinstance(ds, list) and len(ds) > 0: ds = ds[0]
        metainfo = getattr(ds, "metainfo", ds.get("meta", {}) if isinstance(ds, dict) else {})
        padding_size = metainfo.get("padding_size", (0, 0, 0, 0))
        pad_left, pad_right, pad_top, pad_bottom = padding_size
        h_end = inputs.shape[2] - pad_bottom if pad_bottom > 0 else inputs.shape[2]
        w_end = inputs.shape[3] - pad_right if pad_right > 0 else inputs.shape[3]
        pointmap = pointmap[:, :, pad_top:h_end, pad_left:w_end]

        # Resize to original dimensions
        pointmap = F.interpolate(
            pointmap,
            size=image.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        pointmap_np = (
            pointmap.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float16)
        )

        # Zero out background
        pointmap_np[~fg_mask] = 0.0

        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(output_dir / POINTMAP_FILE), pointmap_np)
        return True

    except Exception as exc:
        eprint(f"warning: pointmap failed for {image_path}: {exc}")
        return False
