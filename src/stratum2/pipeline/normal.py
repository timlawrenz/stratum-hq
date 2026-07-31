"""Sapiens2 surface normal estimation — per-pixel unit vectors.

Produces ``normal2.npy`` — H×W×3 float16 array with L2-normalized XYZ
normal components in camera frame.  Background pixels (from seg2.npy) are zeroed.

Requires ``seg2.npy`` to already exist in the output directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from stratum2.config import NORMAL2_FILE, SEG2_FILE


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def process(
    image_path: Path,
    output_dir: Path,
    normal_model,
    device,
    aspect_bucket: str | None = None,
) -> bool:
    """Run Sapiens2 surface normal estimation and save ``normal2.npy``.

    Returns ``True`` on success, ``False`` on failure.
    """
    try:
        # --- Dependency: seg2.npy must exist ---
        seg_path = output_dir / SEG2_FILE
        if not seg_path.exists():
            eprint(f"warning: normal2 skipped for {image_path}: seg2 not found")
            return False

        import cv2

        image = cv2.imread(str(image_path))  # BGR
        if image is None:
            eprint(f"warning: cannot read {image_path}")
            return False

        seg = np.load(seg_path)
        fg_mask = seg > 0

        # Sapiens2 pipeline
        data = normal_model.pipeline(dict(img=image))
        data = normal_model.data_preprocessor(data)
        inputs = data["inputs"].to(device)

        with torch.no_grad():
            normal = normal_model(inputs)  # 1 × 3 × H_model × W_model

        # L2-normalize to unit vectors
        normal = normal / torch.norm(normal, dim=1, keepdim=True).clamp(min=1e-8)

        # Unpad
        pad_left, pad_right, pad_top, pad_bottom = data["data_samples"]["meta"][
            "padding_size"
        ]
        normal = normal[
            :,
            :,
            pad_top : inputs.shape[2] - pad_bottom,
            pad_left : inputs.shape[3] - pad_right,
        ]

        # Resize to original dimensions
        normal = F.interpolate(
            normal,
            size=image.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        normal_map = (
            normal.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float16)
        )

        # Zero out background
        normal_map[~fg_mask] = 0.0

        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(output_dir / NORMAL2_FILE), normal_map)
        return True

    except Exception as exc:
        eprint(f"warning: normal2 failed for {image_path}: {exc}")
        return False
