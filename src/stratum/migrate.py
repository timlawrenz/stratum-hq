"""Migrate from old prx-tg per-modality format to stratum per-image directories."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from stratum.config import (
    CAPTION_FILE,
    DINOV3_CLS_FILE,
    DINOV3_PATCHES_FILE,
    METADATA_FILE,
    PIXEL_FILE,
    POSE_FILE,
    T5_HIDDEN_FILE,
    T5_MASK_FILE,
)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# Old modality dir name → new artifact filename
_MODALITY_MAP: list[tuple[str, str]] = [
    ("dinov3", DINOV3_CLS_FILE),
    ("dinov3_patches", DINOV3_PATCHES_FILE),
    ("t5_hidden", T5_HIDDEN_FILE),
    ("images", PIXEL_FILE),
    ("pose", POSE_FILE),
]


def migrate_dataset(
    jsonl_path: Path,
    derived_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    progress_every: int = 1000,
    verbose: bool = False,
) -> int:
    """Migrate an old prx-tg dataset to stratum per-image directories.

    Reads the JSONL file line-by-line and for each record:
    1. Creates ``output_dir/{image_id}/``
    2. Writes ``metadata.json`` from record fields
    3. Writes ``caption.txt`` from record's ``caption`` field
    4. Hardlinks ``.npy`` files from modality directories (falls back to copy)
    5. Converts ``t5_attention_mask`` list to ``t5_mask.npy``

    Returns exit code (0 on success).
    """
    jsonl_path = jsonl_path.resolve()
    derived_dir = derived_dir.resolve()
    output_dir = output_dir.resolve()

    if not jsonl_path.exists():
        eprint(f"error: JSONL not found: {jsonl_path}")
        return 1
    if not derived_dir.is_dir():
        eprint(f"error: derived dir not found: {derived_dir}")
        return 1

    # Verify modality dirs exist
    for modality_dir, _ in _MODALITY_MAP:
        d = derived_dir / modality_dir
        if not d.is_dir():
            eprint(f"warning: modality dir not found, will skip: {d}")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    migrated = 0
    skipped = 0
    errors = 0
    npy_copied = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                eprint(f"warning: line {line_num}: malformed JSON ({e}), skipping")
                errors += 1
                continue

            image_id = record.get("image_id")
            if not image_id:
                # Derive from image_path
                image_path = record.get("image_path", "")
                image_id = Path(image_path).stem if image_path else None
            if not image_id:
                eprint(f"warning: line {line_num}: no image_id, skipping")
                errors += 1
                continue

            img_out = output_dir / image_id

            # Skip if already migrated (metadata.json exists)
            if (img_out / METADATA_FILE).exists():
                skipped += 1
                if verbose:
                    eprint(f"  skip {image_id} (already exists)")
                _report_progress(line_num, migrated, skipped, errors, npy_copied, started, progress_every)
                continue

            if dry_run:
                if verbose:
                    eprint(f"  dry-run: would create {img_out}/")
                migrated += 1
                _report_progress(line_num, migrated, skipped, errors, npy_copied, started, progress_every)
                continue

            img_out.mkdir(parents=True, exist_ok=True)

            # 1. Write metadata.json
            meta = {
                "image_id": image_id,
                "source_path": record.get("image_path", ""),
                "width": record.get("width"),
                "height": record.get("height"),
                "aspect_bucket": record.get("aspect_bucket", ""),
            }
            with (img_out / METADATA_FILE).open("w") as mf:
                json.dump(meta, mf, ensure_ascii=False, indent=2)

            # 2. Write caption.txt
            caption = record.get("caption", "")
            if caption:
                (img_out / CAPTION_FILE).write_text(caption, encoding="utf-8")

            # 3. Link or copy npy files from modality dirs
            for modality_dir, artifact_name in _MODALITY_MAP:
                src = derived_dir / modality_dir / f"{image_id}.npy"
                dst = img_out / artifact_name
                if src.exists():
                    try:
                        os.link(str(src), str(dst))
                        npy_copied += 1
                    except OSError:
                        # Cross-filesystem — fall back to copy
                        try:
                            shutil.copy2(str(src), str(dst))
                            npy_copied += 1
                        except Exception as e:
                            eprint(f"warning: {image_id}: failed to copy {src.name} → {artifact_name}: {e}")
                            errors += 1
                elif verbose:
                    eprint(f"  {image_id}: missing {modality_dir}/{image_id}.npy")

            # 4. Convert t5_attention_mask list → t5_mask.npy
            t5_mask = record.get("t5_attention_mask")
            if t5_mask and isinstance(t5_mask, list):
                try:
                    mask_arr = np.array(t5_mask, dtype=np.uint8)
                    np.save(img_out / T5_MASK_FILE, mask_arr)
                except Exception as e:
                    eprint(f"warning: {image_id}: failed to write t5_mask.npy: {e}")
                    errors += 1

            migrated += 1
            _report_progress(line_num, migrated, skipped, errors, npy_copied, started, progress_every)

    elapsed = time.time() - started
    eprint(
        f"\ndone: {migrated} migrated, {skipped} skipped, {errors} errors, "
        f"{npy_copied} npy files linked/copied in {elapsed:.1f}s"
    )
    if dry_run:
        eprint("(dry run — no files were written)")
    return 0


def _report_progress(
    line_num: int, migrated: int, skipped: int, errors: int, npy_copied: int,
    started: float, every: int,
):
    if every and line_num % every == 0:
        elapsed = time.time() - started
        rate = migrated / elapsed if elapsed > 0 else 0
        eprint(
            f"progress: line {line_num:,} | {migrated:,} migrated, {skipped:,} skipped, "
            f"{errors} errors, {npy_copied:,} npy copied | {rate:.0f} img/s"
        )
