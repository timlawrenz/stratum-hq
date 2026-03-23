"""Publish dataset to HuggingFace Hub."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from stratum.config import (
    CAPTION_FILE,
    DINOV3_CLS_FILE,
    DINOV3_PATCHES_FILE,
    METADATA_FILE,
    POSE_FILE,
    T5_HIDDEN_FILE,
    T5_MASK_FILE,
)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


LAYER_ARTIFACTS = {
    "caption": [CAPTION_FILE],
    "dinov3": [DINOV3_CLS_FILE, DINOV3_PATCHES_FILE],
    "t5": [T5_HIDDEN_FILE, T5_MASK_FILE],
    "pose": [POSE_FILE],
}


def publish_to_hub(
    dataset_dir: Path,
    hub_repo: str,
    layers: list[str],
    limit: int | None = None,
    offset: int = 0,
) -> int:
    """Publish dataset layers to HuggingFace Hub. Returns exit code."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        eprint("error: huggingface-hub not installed. Install with: pip install stratum-hq[publish]")
        return 1

    dataset_dir = dataset_dir.resolve()

    # Validate layers
    for layer in layers:
        if layer not in LAYER_ARTIFACTS:
            eprint(f"error: unknown layer '{layer}'. Available: {', '.join(LAYER_ARTIFACTS)}")
            return 2

    # Discover image directories
    image_dirs = sorted(
        p.parent for p in dataset_dir.rglob(METADATA_FILE)
    )

    if not image_dirs:
        eprint("error: no image directories found")
        return 1

    # Apply offset and limit
    image_dirs = image_dirs[offset:]
    if limit is not None:
        image_dirs = image_dirs[:limit]

    eprint(f"Publishing {len(image_dirs)} images, layers: {layers} to {hub_repo}")

    # TODO: Implement actual HF Hub upload logic
    # This is a stub that will be fleshed out in the publish phase
    eprint("error: publish command is not yet implemented")
    return 1
