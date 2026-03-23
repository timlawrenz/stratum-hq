"""Image discovery and dataset directory scanning."""

from __future__ import annotations

import sys
from pathlib import Path

from stratum.config import (
    CAPTION_FILE,
    DINOV3_CLS_FILE,
    DINOV3_PATCHES_FILE,
    IMAGE_EXTENSIONS,
    METADATA_FILE,
    PIXEL_FILE,
    POSE_FILE,
    T5_HIDDEN_FILE,
    T5_MASK_FILE,
)

# Artifacts to check in status/verify. Maps display name → filename.
ARTIFACT_FILES: dict[str, str] = {
    "metadata": METADATA_FILE,
    "caption": CAPTION_FILE,
    "dinov3_cls": DINOV3_CLS_FILE,
    "dinov3_patches": DINOV3_PATCHES_FILE,
    "t5_hidden": T5_HIDDEN_FILE,
    "t5_mask": T5_MASK_FILE,
    "pixel": PIXEL_FILE,
    "pose": POSE_FILE,
}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()


def discover_images(input_dir: Path, image_list_path: Path | None = None) -> list[Path]:
    """Find all images under input_dir, sorted by relative path.

    If image_list_path is provided, reads explicit paths from that file instead.
    Returns absolute paths.
    """
    if image_list_path is not None:
        paths = []
        with image_list_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    p = Path(line)
                    if not p.is_absolute():
                        p = input_dir / p
                    paths.append(p.resolve())
        return sorted(paths)

    input_dir = input_dir.resolve()
    return sorted(p.resolve() for p in input_dir.rglob("*") if _is_image(p))


def image_id_from_path(image_path: Path, input_dir: Path) -> str:
    """Derive image_id as relative path stem (mirroring subdirectory structure).

    Example: input_dir=/data/images, image_path=/data/images/ffhq/batch1/00001.png
             → image_id = "ffhq/batch1/00001"
    """
    input_dir = input_dir.resolve()
    image_path = image_path.resolve()
    rel = image_path.relative_to(input_dir)
    # Remove the file extension from the last component
    return str(rel.with_suffix(""))


def output_dir_for_image(image_path: Path, input_dir: Path, output_base: Path) -> Path:
    """Compute output directory for an image, mirroring source structure.

    Example: input_dir=/data/images, image_path=/data/images/ffhq/batch1/00001.png,
             output_base=/data/dataset
             → /data/dataset/ffhq/batch1/00001/
    """
    img_id = image_id_from_path(image_path, input_dir)
    return output_base / img_id


def shard_image_list(images: list[Path], worker: int, total: int) -> list[Path]:
    """Deterministic shard: take every total-th image starting at worker offset."""
    return images[worker::total]


def scan_dataset_status(dataset_dir: Path) -> dict[str, int]:
    """Scan dataset directory and count artifacts per type.

    Returns dict with 'total' key and one key per artifact type.
    """
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        return {"total": 0}

    counts: dict[str, int] = {name: 0 for name in ARTIFACT_FILES}
    total = 0

    # Walk dataset looking for metadata.json as indicator of image dirs
    for meta_path in sorted(dataset_dir.rglob(METADATA_FILE)):
        img_dir = meta_path.parent
        total += 1
        for name, filename in ARTIFACT_FILES.items():
            if (img_dir / filename).exists():
                counts[name] += 1

    counts["total"] = total
    return counts
