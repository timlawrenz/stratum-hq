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
        return f"dtype {arr.dtype}, expected {expected_dtype}"
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
                issues.append(f"dinov3_patches: dtype {arr.dtype}, expected float16")
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


def verify_dataset(dataset_dir: Path, fix: bool = False) -> int:
    """Verify all image directories in a dataset. Returns exit code."""
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        eprint(f"error: not a directory: {dataset_dir}")
        return 2

    total = 0
    total_issues = 0
    fixed = 0

    for meta_path in sorted(dataset_dir.rglob(METADATA_FILE)):
        img_dir = meta_path.parent
        total += 1
        issues = verify_image_dir(img_dir)

        if issues:
            rel = img_dir.relative_to(dataset_dir)
            for issue in issues:
                eprint(f"  {rel}: {issue}")
                total_issues += 1

                if fix and "corrupt" in issue:
                    # Extract filename from issue prefix (before colon)
                    artifact_name = issue.split(":")[0].strip()
                    artifact_map = {
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
                    filename = artifact_map.get(artifact_name)
                    if filename:
                        target = img_dir / filename
                        if target.exists():
                            target.unlink()
                            eprint(f"    → deleted {target.name} for regeneration")
                            fixed += 1

    eprint(f"\nVerified {total} images: {total_issues} issue(s) found", end="")
    if fix and fixed:
        eprint(f", {fixed} corrupt file(s) deleted for regeneration")
    else:
        eprint()

    return 0 if total_issues == 0 else 1
