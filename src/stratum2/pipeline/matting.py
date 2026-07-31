"""Sapiens2 human matting — alpha matte extraction.

Produces ``matting.npy`` — H×W float16 array with alpha values in [0, 1].
No dependency on segmentation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from stratum2.config import MATTING_FILE


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def process(
    image_path: Path,
    output_dir: Path,
    matting_model,
    device,
    aspect_bucket: str | None = None,
) -> bool:
    """Run Sapiens2 human matting and save ``matting.npy``.

    Returns ``True`` on success, ``False`` on failure.
    """
    try:
        import cv2

        image = cv2.imread(str(image_path))  # BGR
        if image is None:
            eprint(f"warning: cannot read {image_path}")
            return False

        # Sapiens2 pipeline
        data = matting_model.pipeline(dict(img=image))
        data = matting_model.data_preprocessor(data)
        inputs = data["inputs"].to(device)

        with torch.no_grad():
            outputs = matting_model(inputs)  # 1 × 4 × H × W: [fgr_rgb, alpha]

        # Resize to original dimensions
        outputs = F.interpolate(
            outputs,
            size=image.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        # Extract alpha channel (index 3) and clamp to [0, 1]
        alpha = outputs[0, 3].clamp(0, 1).cpu().numpy().astype(np.float16)

        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(output_dir / MATTING_FILE), alpha)
        return True

    except Exception as exc:
        eprint(f"warning: matting failed for {image_path}: {exc}")
        return False
