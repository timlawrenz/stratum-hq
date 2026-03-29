"""Publish dataset to HuggingFace Hub with incremental width × depth support."""

from __future__ import annotations

import datetime
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from stratum import __version__
from stratum.config import (
    CAPTION_FILE,
    DEPTH_FILE,
    DINOV3_CLS_FILE,
    DINOV3_PATCHES_FILE,
    METADATA_FILE,
    NORMAL_FILE,
    POSE_FILE,
    SEG_FILE,
    T5_HIDDEN_FILE,
    T5_MASK_FILE,
)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


LAYER_ARTIFACTS: dict[str, list[str]] = {
    "caption": [CAPTION_FILE],
    "dinov3": [DINOV3_CLS_FILE, DINOV3_PATCHES_FILE],
    "t5": [T5_HIDDEN_FILE, T5_MASK_FILE],
    "pose": [POSE_FILE],
    "seg": [SEG_FILE],
    "depth": [DEPTH_FILE],
    "normal": [NORMAL_FILE],
}

MANIFEST_FILE = "manifest.json"

MAX_RETRIES = 5
INITIAL_BACKOFF = 10  # seconds


def _parse_range_label(filename: str) -> tuple[int, int] | None:
    """Extract (start, end) from a range-labeled filename like '00000-09999.parquet'."""
    m = re.match(r"(\d+)-(\d+)\.", filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _is_rate_limited(exc: Exception) -> bool:
    """Check if an exception is an HTTP 429 rate-limit response."""
    # huggingface_hub raises HfHubHTTPError with response attached
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True
    # Fall back to checking the string representation
    return "429" in str(exc)


def _is_transient_error(exc: Exception) -> bool:
    """Check if an exception is a transient network error worth retrying."""
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    try:
        import requests.exceptions as reqexc

        if isinstance(exc, (reqexc.ConnectionError, reqexc.Timeout,
                            reqexc.ChunkedEncodingError)):
            return True
    except ImportError:
        pass
    # Check chained exceptions (e.g., HfHubHTTPError wrapping ConnectionError)
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause is not None and cause is not exc:
        if _is_transient_error(cause):
            return True
    # Fall back to string matching for wrapped errors
    msg = str(exc).lower()
    if any(needle in msg for needle in (
        "connection reset", "connection abort", "broken pipe",
        "timed out", "timeout", "errno 104", "errno 110", "errno 32",
    )):
        return True
    return False


def _get_retry_after(exc: Exception) -> int | None:
    """Extract Retry-After seconds from a 429 response, if available."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", {})
    retry_after = headers.get("Retry-After") if headers else None
    if retry_after is not None:
        try:
            return max(1, int(retry_after))
        except (ValueError, TypeError):
            pass
    return None


def _retry_upload(fn, *args, verbose: bool = False, **kwargs):
    """Call *fn* with retry + exponential backoff on HTTP 429 or transient network errors.

    Respects the ``Retry-After`` header when present; otherwise falls
    back to exponential backoff starting at *INITIAL_BACKOFF* seconds.
    """
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.monotonic()
            result = fn(*args, **kwargs)
            if verbose:
                elapsed = time.monotonic() - t0
                eprint(f"  [verbose] {fn.__name__} succeeded in {elapsed:.1f}s")
            return result
        except Exception as exc:
            rate_limited = _is_rate_limited(exc)
            transient = _is_transient_error(exc)
            retriable = rate_limited or transient
            if not retriable or attempt == MAX_RETRIES:
                if verbose:
                    elapsed = time.monotonic() - t0
                    status = getattr(getattr(exc, "response", None), "status_code", "?")
                    eprint(f"  [verbose] {fn.__name__} failed after {elapsed:.1f}s — HTTP {status}: {exc}")
                raise
            if rate_limited:
                wait = _get_retry_after(exc) or backoff
                eprint(f"  rate-limited (429), retrying in {wait}s (attempt {attempt}/{MAX_RETRIES})...")
            else:
                wait = backoff
                eprint(f"  network error ({type(exc).__name__}), retrying in {wait}s "
                       f"(attempt {attempt}/{MAX_RETRIES})...")
            if verbose:
                elapsed = time.monotonic() - t0
                eprint(f"  [verbose] {fn.__name__} failed after {elapsed:.1f}s: {exc}")
            time.sleep(wait)
            backoff *= 2


# Keep old name as alias for backward compatibility
_retry_on_429 = _retry_upload


def _collect_image_dirs(dataset_dir: Path, offset: int, limit: int | None) -> list[Path]:
    """Find image directories (those containing metadata.json), sorted, with offset/limit."""
    dirs = sorted(p.parent for p in dataset_dir.rglob(METADATA_FILE))
    dirs = dirs[offset:]
    if limit is not None:
        dirs = dirs[:limit]
    return dirs


def _image_has_layer(img_dir: Path, layer: str) -> bool:
    """Check if an image directory has all artifacts for a layer."""
    return all((img_dir / f).exists() for f in LAYER_ARTIFACTS[layer])


def _build_metadata_records(image_dirs: list[Path], dataset_dir: Path) -> list[dict]:
    """Load metadata.json from each image directory."""
    records = []
    for img_dir in image_dirs:
        meta_path = img_dir / METADATA_FILE
        try:
            with meta_path.open() as f:
                meta = json.load(f)
            meta["_dir"] = str(img_dir)
            meta["_rel"] = str(img_dir.relative_to(dataset_dir))
            records.append(meta)
        except Exception as e:
            eprint(f"warning: skipping {img_dir}: {e}")
    return records


def _write_data_parquet(records: list[dict], image_dirs: list[Path], output_path: Path) -> int:
    """Write a unified parquet with metadata and captions.

    Columns: ``image_id``, ``width``, ``height``, ``aspect_bucket``, ``caption``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows: dict[str, list] = {
        "image_id": [],
        "width": [],
        "height": [],
        "aspect_bucket": [],
        "caption": [],
    }
    for rec, img_dir in zip(records, image_dirs):
        rows["image_id"].append(rec.get("image_id", ""))
        rows["width"].append(rec.get("width", 0))
        rows["height"].append(rec.get("height", 0))
        rows["aspect_bucket"].append(rec.get("aspect_bucket", ""))
        caption_path = img_dir / CAPTION_FILE
        if caption_path.exists():
            rows["caption"].append(caption_path.read_text(encoding="utf-8").strip())
        else:
            rows["caption"].append("")

    table = pa.table(rows)
    pq.write_table(table, output_path)
    return len(rows["image_id"])


def _pack_npy_tar(image_dirs: list[Path], records: list[dict], layer: str, output_path: Path) -> int:
    """Pack .npy artifacts for a layer into a tar archive."""
    import tarfile

    count = 0
    with tarfile.open(output_path, "w") as tar:
        for rec, img_dir in zip(records, image_dirs):
            image_id = rec.get("image_id", rec.get("_rel", "unknown"))
            if not _image_has_layer(img_dir, layer):
                continue
            for artifact_file in LAYER_ARTIFACTS[layer]:
                src = img_dir / artifact_file
                arcname = f"{image_id}/{artifact_file}"
                tar.add(str(src), arcname=arcname)
            count += 1
    return count


def _plan_tar_splits(
    image_dirs: list[Path], records: list[dict], layer: str, max_tar_bytes: int
) -> list[tuple[list[Path], list[dict]]]:
    """Group images so each group's tar stays under *max_tar_bytes*.

    Returns a list of ``(group_dirs, group_records)`` tuples.  Images that
    lack the layer's artifacts are silently skipped (they won't appear in
    any group).
    """
    groups: list[tuple[list[Path], list[dict]]] = []
    cur_dirs: list[Path] = []
    cur_recs: list[dict] = []
    cur_size = 0

    for img_dir, rec in zip(image_dirs, records):
        if not _image_has_layer(img_dir, layer):
            continue
        img_bytes = sum((img_dir / f).stat().st_size for f in LAYER_ARTIFACTS[layer])

        if cur_dirs and cur_size + img_bytes > max_tar_bytes:
            groups.append((cur_dirs, cur_recs))
            cur_dirs, cur_recs, cur_size = [], [], 0

        cur_dirs.append(img_dir)
        cur_recs.append(rec)
        cur_size += img_bytes

    if cur_dirs:
        groups.append((cur_dirs, cur_recs))

    return groups


def _load_manifest(api, hub_repo: str) -> dict:
    """Load existing manifest from HF repo, or return empty."""
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=hub_repo, filename=MANIFEST_FILE, repo_type="dataset")
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {
            "version": "0.0.0",
            "total_images": 0,
            "layers": {},
            "created_with": f"stratum-hq v{__version__}",
        }


def _bump_version(manifest: dict) -> str:
    """Bump the patch version in manifest."""
    parts = manifest.get("version", "0.0.0").split(".")
    parts = [int(p) for p in parts[:3]] + [0] * (3 - len(parts))
    parts[2] += 1
    return ".".join(str(p) for p in parts)


def _write_dataset_card(manifest: dict, hub_repo: str, output_path: Path,
                        license_id: str = "cc-by-nc-sa-4.0", attribution: str | None = None):
    """Generate a HuggingFace dataset card (README.md)."""
    layers = manifest.get("layers", {})
    total = manifest.get("total_images", 0)

    layer_table = ""
    for name, info in sorted(layers.items()):
        fmt = info.get("format", "unknown")
        count = info.get("count", 0)
        layer_table += f"| {name} | {count:,} | {fmt} |\n"

    attribution_section = ""
    if attribution:
        attribution_section = f"""
## Attribution & Provenance

{attribution}
"""

    card = f"""---
license: {license_id}
task_categories:
  - image-to-text
  - text-to-image
tags:
  - stratum-hq
  - embeddings
  - diffusion
  - dataset-enrichment
size_categories:
  - 10K<n<100K
---

# {hub_repo.split('/')[-1]}

Enriched image dataset generated by [stratum-hq](https://github.com/timlawrenz/stratum-hq).

## Dataset Summary

- **Total images**: {total:,}
- **Version**: {manifest.get('version', '0.0.0')}
- **Generated with**: {manifest.get('created_with', 'stratum-hq')}

## Available Layers

| Layer | Count | Format |
|-------|-------|--------|
{layer_table}
## Layer Formats

- **caption**: Included in the main data parquet (`data/`) with `image_id`, `width`, `height`, `aspect_bucket`, and `caption` columns
- **dinov3**: Tar archives with `dinov3_cls.npy` (1024, float16) and `dinov3_patches.npy` (N×1024, float16) per image
- **t5**: Tar archives with `t5_hidden.npy` (512×1024, float16) and `t5_mask.npy` (512, uint8) per image
- **pose**: Tar archives with `pose.npy` (133×3, float16) per image — COCO-WholeBody keypoints in [-1, 1]
- **seg**: Tar archives with `seg.npy` (H×W, uint8) per image — 28-class body-part segmentation (Sapiens)
- **depth**: Tar archives with `depth.npy` (H×W, float16) per image — relative depth, foreground-masked (Sapiens)
- **normal**: Tar archives with `normal.npy` (H×W×3, float16) per image — unit surface normals, foreground-masked (Sapiens)
{attribution_section}
## Reproduction

```bash
pip install stratum-hq[all]
stratum process ./your-images/ --output ./dataset/ --passes all --device cuda
stratum publish ./dataset/ --hub-repo {hub_repo} --layers caption,dinov3,t5,pose,seg,depth,normal
```
"""
    output_path.write_text(card, encoding="utf-8")


def _format_size(n_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}" if unit != "B" else f"{n_bytes} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def _log_upload_summary(folder: Path) -> None:
    """Log every file under *folder* with its repo-relative path and size."""
    eprint("  [verbose] files to upload:")
    total = 0
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            total += size
            rel = p.relative_to(folder)
            eprint(f"    {rel}  ({_format_size(size)})")
    eprint(f"  [verbose] total upload: {_format_size(total)}")


def publish_to_hub(
    dataset_dir: Path,
    hub_repo: str,
    layers: list[str],
    license_id: str = "cc-by-nc-sa-4.0",
    attribution: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    verbose: bool = False,
    max_tar_mb: int | None = None,
    _api=None,
) -> int:
    """Publish dataset layers to HuggingFace Hub. Returns exit code."""
    if _api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            eprint("error: huggingface-hub not installed. Install with: pip install stratum-hq[publish]")
            return 1
        _api = HfApi()
    api = _api

    dataset_dir = dataset_dir.resolve()

    for layer in layers:
        if layer not in LAYER_ARTIFACTS:
            eprint(f"error: unknown layer '{layer}'. Available: {', '.join(LAYER_ARTIFACTS)}")
            return 2

    image_dirs = _collect_image_dirs(dataset_dir, offset, limit)
    if not image_dirs:
        eprint("error: no image directories found")
        return 1

    eprint(f"Publishing {len(image_dirs)} images, layers: {layers} to {hub_repo}")

    # Ensure repo exists
    try:
        api.create_repo(repo_id=hub_repo, repo_type="dataset", exist_ok=True)
    except Exception as e:
        eprint(f"error: could not create/access repo {hub_repo}: {e}")
        return 1

    # Load records
    records = _build_metadata_records(image_dirs, dataset_dir)
    if not records:
        eprint("error: no valid metadata records found")
        return 1

    manifest = _load_manifest(api, hub_repo)
    max_tar_bytes = max_tar_mb * 1_000_000 if max_tar_mb else None
    has_caption = "caption" in layers

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        range_label = f"{offset:05d}-{offset + len(image_dirs) - 1:05d}"
        manifest["total_images"] = max(manifest.get("total_images", 0), offset + len(image_dirs))

        # Data parquet — only when publishing captions
        if has_caption:
            try:
                import pyarrow  # noqa: F401
            except ImportError:
                eprint("error: pyarrow not installed. Install with: pip install stratum-hq[publish]")
                return 1

            data_dir = tmp / "data"
            data_dir.mkdir()
            data_parquet = data_dir / f"{range_label}.parquet"
            n = _write_data_parquet(records, image_dirs, data_parquet)
            eprint(f"  data: {n} records -> data/{range_label}.parquet")

            layer_info = manifest["layers"].get("caption", {"format": "parquet", "chunks": {}})
            layer_info["chunks"] = layer_info.get("chunks", {})
            if isinstance(layer_info["chunks"], list):
                layer_info["chunks"] = {}
            layer_info["chunks"][range_label] = n
            layer_info["count"] = sum(layer_info["chunks"].values())
            manifest["layers"]["caption"] = layer_info
            eprint(f"  captions: {n} included in data parquet")

        for layer in layers:
            if layer == "caption":
                continue
            layer_dir = tmp / layer
            layer_dir.mkdir(exist_ok=True)

            layer_info = manifest["layers"].get(layer, {"format": "npy_tar", "chunks": {}})
            layer_info["chunks"] = layer_info.get("chunks", {})
            if isinstance(layer_info["chunks"], list):
                layer_info["chunks"] = {}

            if max_tar_bytes:
                groups = _plan_tar_splits(image_dirs, records, layer, max_tar_bytes)
            else:
                groups = [(image_dirs, records)]

            for group_dirs, group_recs in groups:
                # Find the index range this group covers in the original list
                first = image_dirs.index(group_dirs[0])
                last = image_dirs.index(group_dirs[-1])
                sub_label = f"{offset + first:05d}-{offset + last:05d}"

                tar_path = layer_dir / f"{sub_label}.tar"
                nc = _pack_npy_tar(group_dirs, group_recs, layer, tar_path)
                layer_info["chunks"][sub_label] = nc
                eprint(f"  {layer}: {nc} images -> {layer}/{sub_label}.tar")

            layer_info["count"] = sum(layer_info["chunks"].values())
            manifest["layers"][layer] = layer_info

        # Update manifest
        manifest["version"] = _bump_version(manifest)
        manifest["created_with"] = f"stratum-hq v{__version__}"
        manifest["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        manifest_path = tmp / MANIFEST_FILE
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # Generate dataset card
        card_path = tmp / "README.md"
        _write_dataset_card(manifest, hub_repo, card_path,
                            license_id=license_id, attribution=attribution)

        # Upload — upload_large_folder commits in batches and resumes
        if verbose:
            _log_upload_summary(tmp)
        eprint(f"Uploading to {hub_repo} (resumable multi-commit)...")
        try:
            api.upload_large_folder(
                folder_path=str(tmp),
                repo_id=hub_repo,
                repo_type="dataset",
            )
        except Exception as e:
            eprint(f"error uploading to {hub_repo}: {e}")
            return 1

    eprint(f"Published successfully. Manifest version: {manifest['version']}")
    return 0


def reconcile_hub_manifest(
    hub_repo: str,
    license_id: str = "cc-by-nc-sa-4.0",
    attribution: str | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    _api=None,
) -> int:
    """Reconcile manifest.json with actual files on HuggingFace Hub.

    Lists repo files, parses range labels from filenames, rebuilds
    per-layer chunk counts, and uploads a corrected manifest and dataset card.
    """
    if _api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            eprint("error: huggingface-hub not installed. Install with: pip install stratum-hq[publish]")
            return 1
        _api = HfApi()
    api = _api

    try:
        file_paths = api.list_repo_files(repo_id=hub_repo, repo_type="dataset")
    except Exception as e:
        eprint(f"error: could not list repo {hub_repo}: {e}")
        return 1

    # Parse data parquets → caption chunks
    caption_chunks: dict[str, int] = {}
    max_image_index = -1
    for fp in file_paths:
        if fp.startswith("data/") and fp.endswith(".parquet"):
            rng = _parse_range_label(fp.split("/")[-1])
            if rng:
                start, end = rng
                range_label = f"{start:05d}-{end:05d}"
                caption_chunks[range_label] = end - start + 1
                max_image_index = max(max_image_index, end)

    # Parse layer tars → per-layer chunks
    layer_chunks: dict[str, dict[str, int]] = {}
    for fp in file_paths:
        if not fp.endswith(".tar") or "/" not in fp:
            continue
        parts = fp.split("/")
        if len(parts) != 2:
            continue
        layer_name, basename = parts
        if layer_name == "data":
            continue
        rng = _parse_range_label(basename)
        if rng:
            start, end = rng
            range_label = f"{start:05d}-{end:05d}"
            layer_chunks.setdefault(layer_name, {})[range_label] = end - start + 1
            max_image_index = max(max_image_index, end)

    # Load existing manifest to preserve version lineage
    manifest = _load_manifest(api, hub_repo)

    # Rebuild layers entirely from the file listing
    new_layers: dict[str, dict] = {}
    if caption_chunks:
        new_layers["caption"] = {
            "format": "parquet",
            "chunks": caption_chunks,
            "count": sum(caption_chunks.values()),
        }
    for layer_name, chunks in sorted(layer_chunks.items()):
        new_layers[layer_name] = {
            "format": "npy_tar",
            "chunks": chunks,
            "count": sum(chunks.values()),
        }
    manifest["layers"] = new_layers

    manifest["total_images"] = max_image_index + 1 if max_image_index >= 0 else 0
    manifest["version"] = _bump_version(manifest)
    manifest["created_with"] = f"stratum-hq v{__version__}"
    manifest["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Report
    eprint(f"Reconciled manifest for {hub_repo}:")
    eprint(f"  total_images: {manifest['total_images']:,}")
    for name, info in sorted(new_layers.items()):
        eprint(f"  {name}: {info['count']:,} ({len(info['chunks'])} chunks)")

    if dry_run:
        eprint("\nDry run — manifest not uploaded.")
        eprint(json.dumps(manifest, indent=2))
        return 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        manifest_path = tmp / MANIFEST_FILE
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        card_path = tmp / "README.md"
        _write_dataset_card(manifest, hub_repo, card_path,
                            license_id=license_id, attribution=attribution)

        for local_path, remote_path in [(manifest_path, MANIFEST_FILE), (card_path, "README.md")]:
            if verbose:
                size = local_path.stat().st_size
                eprint(f"  [verbose] uploading {remote_path} ({size:,} bytes)")
            try:
                _retry_upload(
                    api.upload_file,
                    path_or_fileobj=str(local_path),
                    path_in_repo=remote_path,
                    repo_id=hub_repo,
                    repo_type="dataset",
                    verbose=verbose,
                )
            except Exception as e:
                eprint(f"error uploading {remote_path}: {e}")
                return 1

    eprint("Manifest reconciled successfully.")
    return 0
