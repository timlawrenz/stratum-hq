"""Verify dataset integrity — check shapes, dtypes, completeness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from stratum.config import (
    CAPTION_FILE,
    DEPTH_FILE,
    DINOV3_CLS_FILE,
    DINOV3_PATCHES_FILE,
    METADATA_FILE,
    NORMAL_FILE,
    NUM_POSE_KEYPOINTS,
    PIXEL_FILE,
    POSE_FILE,
    SEG_FILE,
    T5_HIDDEN_FILE,
    T5_MASK_FILE,
)
from stratum.pipeline.bucket import parse_bucket_dims


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _check_npy(path: Path, expected_shape: tuple | None, expected_dtype: np.dtype | None) -> str | None:
    """Check an npy file. Returns error string or None if OK."""
    if not path.exists():
        return "missing"
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception as e:
        return f"corrupt ({e})"

    if expected_shape is not None and arr.shape != expected_shape:
        return f"shape {arr.shape}, expected {expected_shape}"
    if expected_dtype is not None and arr.dtype != expected_dtype:
        return f"dtype {arr.dtype}, expected {np.dtype(expected_dtype).name}"
    return None


def verify_image_dir(img_dir: Path) -> list[str]:
    """Verify all artifacts in a single image directory. Returns list of issues."""
    issues = []
    meta_path = img_dir / METADATA_FILE

    if not meta_path.exists():
        issues.append("metadata.json missing")
        return issues

    try:
        with meta_path.open() as f:
            meta = json.load(f)
    except Exception as e:
        issues.append(f"metadata.json corrupt: {e}")
        return issues

    aspect_bucket = meta.get("aspect_bucket")

    # Caption
    caption_path = img_dir / CAPTION_FILE
    if caption_path.exists():
        text = caption_path.read_text(encoding="utf-8").strip()
        if not text:
            issues.append("caption.txt is empty")
    # Don't flag missing — it's just not generated yet

    # DINOv3 CLS
    err = _check_npy(img_dir / DINOV3_CLS_FILE, (1024,), np.float16)
    if err and err != "missing":
        issues.append(f"dinov3_cls: {err}")

    # DINOv3 patches — variable shape, just check ndim and dim[1]
    patches_path = img_dir / DINOV3_PATCHES_FILE
    if patches_path.exists():
        try:
            arr = np.load(patches_path, mmap_mode="r")
            if arr.ndim != 2 or arr.shape[1] != 1024:
                issues.append(f"dinov3_patches: shape {arr.shape}, expected (N, 1024)")
            if arr.dtype != np.float16:
                issues.append(f"dinov3_patches: dtype {arr.dtype}, expected {np.dtype(np.float16).name}")
        except Exception as e:
            issues.append(f"dinov3_patches: corrupt ({e})")

    # T5 hidden
    err = _check_npy(img_dir / T5_HIDDEN_FILE, (512, 1024), np.float16)
    if err and err != "missing":
        issues.append(f"t5_hidden: {err}")

    # T5 mask
    err = _check_npy(img_dir / T5_MASK_FILE, (512,), np.uint8)
    if err and err != "missing":
        issues.append(f"t5_mask: {err}")

    # Pixel — shape depends on bucket
    pixel_path = img_dir / PIXEL_FILE
    if pixel_path.exists() and aspect_bucket:
        dims = parse_bucket_dims(aspect_bucket)
        if dims:
            bw, bh = dims
            err = _check_npy(pixel_path, (3, bh, bw), np.float16)
            if err:
                issues.append(f"pixel: {err}")

    # Pose
    err = _check_npy(img_dir / POSE_FILE, (NUM_POSE_KEYPOINTS, 3), np.float16)
    if err and err != "missing":
        issues.append(f"pose: {err}")

    # Segmentation — shape depends on bucket
    seg_path = img_dir / SEG_FILE
    if seg_path.exists() and aspect_bucket:
        dims = parse_bucket_dims(aspect_bucket)
        if dims:
            bw, bh = dims
            err = _check_npy(seg_path, (bh, bw), np.uint8)
            if err:
                issues.append(f"seg: {err}")

    # Depth — shape depends on bucket
    depth_path = img_dir / DEPTH_FILE
    if depth_path.exists() and aspect_bucket:
        dims = parse_bucket_dims(aspect_bucket)
        if dims:
            bw, bh = dims
            err = _check_npy(depth_path, (bh, bw), np.float16)
            if err:
                issues.append(f"depth: {err}")

    # Normal — shape depends on bucket
    normal_path = img_dir / NORMAL_FILE
    if normal_path.exists() and aspect_bucket:
        dims = parse_bucket_dims(aspect_bucket)
        if dims:
            bw, bh = dims
            err = _check_npy(normal_path, (bh, bw, 3), np.float16)
            if err:
                issues.append(f"normal: {err}")

    return issues


ARTIFACT_EXPECTED_DTYPE: dict[str, np.dtype] = {
    "dinov3_cls": np.dtype(np.float16),
    "dinov3_patches": np.dtype(np.float16),
    "t5_hidden": np.dtype(np.float16),
    "t5_mask": np.dtype(np.uint8),
    "pixel": np.dtype(np.float16),
    "pose": np.dtype(np.float16),
    "seg": np.dtype(np.uint8),
    "depth": np.dtype(np.float16),
    "normal": np.dtype(np.float16),
}

ARTIFACT_FILE_MAP: dict[str, str] = {
    "dinov3_cls": DINOV3_CLS_FILE,
    "dinov3_patches": DINOV3_PATCHES_FILE,
    "t5_hidden": T5_HIDDEN_FILE,
    "t5_mask": T5_MASK_FILE,
    "pixel": PIXEL_FILE,
    "pose": POSE_FILE,
    "seg": SEG_FILE,
    "depth": DEPTH_FILE,
    "normal": NORMAL_FILE,
}


def verify_dataset(dataset_dir: Path, fix: bool = False) -> int:
    """Verify all image directories in a dataset. Returns exit code."""
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        eprint(f"error: not a directory: {dataset_dir}")
        return 2

    total = 0
    total_issues = 0
    converted = 0
    deleted = 0

    for meta_path in sorted(dataset_dir.rglob(METADATA_FILE)):
        img_dir = meta_path.parent
        total += 1
        issues = verify_image_dir(img_dir)

        if issues:
            rel = img_dir.relative_to(dataset_dir)
            for issue in issues:
                eprint(f"  {rel}: {issue}")
                total_issues += 1

                if not fix:
                    continue

                artifact_name = issue.split(":")[0].strip()
                filename = ARTIFACT_FILE_MAP.get(artifact_name)
                if not filename:
                    continue
                target = img_dir / filename
                if not target.exists():
                    continue

                if "dtype" in issue:
                    expected = ARTIFACT_EXPECTED_DTYPE.get(artifact_name)
                    if expected is not None:
                        try:
                            arr = np.load(target)
                            np.save(target, arr.astype(expected))
                            eprint(f"    → converted {target.name} to {expected.name}")
                            converted += 1
                        except Exception as e:
                            eprint(f"    → conversion failed for {target.name}: {e}")
                elif "corrupt" in issue or "shape" in issue:
                    target.unlink()
                    eprint(f"    → deleted {target.name} for regeneration")
                    deleted += 1

    eprint(f"\nVerified {total} images: {total_issues} issue(s) found", end="")
    if fix and (converted or deleted):
        parts = []
        if converted:
            parts.append(f"{converted} converted")
        if deleted:
            parts.append(f"{deleted} deleted for regeneration")
        eprint(f", {', '.join(parts)}")
    else:
        eprint()

    return 0 if total_issues == 0 else 1
