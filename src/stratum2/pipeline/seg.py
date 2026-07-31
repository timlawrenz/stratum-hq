"""Sapiens2 body-part segmentation — 29-class per-pixel labels.

Produces ``seg2.npy`` — H×W uint8 array with class IDs 0-28.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from stratum2.config import SEG2_FILE


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def process(
    image_path: Path,
    output_dir: Path,
    seg_model,
    device,
    aspect_bucket: str | None = None,
) -> bool:
    """Run Sapiens2 segmentation and save ``seg2.npy``.

    Returns ``True`` on success, ``False`` on failure.
    """
    try:
        import cv2

        image = cv2.imread(str(image_path))  # BGR — Sapiens2 expects this
        if image is None:
            eprint(f"warning: cannot read {image_path}")
            return False

        # Sapiens2 pipeline: model handles resize+pad+normalize
        data = seg_model.pipeline(dict(img=image))
        data = seg_model.data_preprocessor(data)
        inputs = data["inputs"].to(device)

        with torch.no_grad():
            seg_logits = seg_model(inputs)  # 1 × 29 × H_model × W_model

        # Resize to original image dimensions
        seg_logits = F.interpolate(
            seg_logits,
            size=image.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        pred_labels = (
            seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(output_dir / SEG2_FILE), pred_labels)
        return True

    except Exception as exc:
        eprint(f"warning: seg2 failed for {image_path}: {exc}")
        return False
