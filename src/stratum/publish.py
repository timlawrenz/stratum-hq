"""Publish dataset to HuggingFace Hub with incremental width × depth support."""

from __future__ import annotations

import datetime
import json
import sys
import tempfile
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
- **dinov3**: Tar archives with `dinov3_cls.npy` (1024, float32) and `dinov3_patches.npy` (N×1024, float32) per image
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


def publish_to_hub(
    dataset_dir: Path,
    hub_repo: str,
    layers: list[str],
    license_id: str = "cc-by-nc-sa-4.0",
    attribution: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> int:
    """Publish dataset layers to HuggingFace Hub. Returns exit code."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        eprint("error: huggingface-hub not installed. Install with: pip install stratum-hq[publish]")
        return 1

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        eprint("error: pyarrow not installed. Install with: pip install stratum-hq[publish]")
        return 1

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

    api = HfApi()

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

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        uploads: list[tuple[Path, str]] = []

        # Unified data parquet — range-labeled for incremental uploads
        range_label = f"{offset:05d}-{offset + len(image_dirs) - 1:05d}"

        data_parquet = tmp / f"data_{range_label}.parquet"
        n = _write_data_parquet(records, image_dirs, data_parquet)
        uploads.append((data_parquet, f"data/{range_label}.parquet"))
        eprint(f"  data: {n} records -> data/{range_label}.parquet")
        manifest["total_images"] = max(manifest.get("total_images", 0), offset + len(image_dirs))

        for layer in layers:
            if layer == "caption":
                layer_info = manifest["layers"].get("caption", {"format": "parquet", "chunks": {}})
                layer_info["chunks"] = layer_info.get("chunks", {})
                if isinstance(layer_info["chunks"], list):
                    layer_info["chunks"] = {}
                layer_info["chunks"][range_label] = n
                layer_info["count"] = sum(layer_info["chunks"].values())
                manifest["layers"]["caption"] = layer_info
                eprint(f"  captions: {n} included in data parquet")
            else:
                tar_name = f"{layer}/{range_label}.tar"
                tar_path = tmp / f"{layer}_{range_label}.tar"
                n = _pack_npy_tar(image_dirs, records, layer, tar_path)
                uploads.append((tar_path, tar_name))

                layer_info = manifest["layers"].get(layer, {"format": "npy_tar", "chunks": {}})
                layer_info["chunks"] = layer_info.get("chunks", {})
                if isinstance(layer_info["chunks"], list):
                    layer_info["chunks"] = {}
                layer_info["chunks"][range_label] = n
                layer_info["count"] = sum(layer_info["chunks"].values())
                manifest["layers"][layer] = layer_info
                eprint(f"  {layer}: {n} images -> {tar_name}")

        # Update manifest
        manifest["version"] = _bump_version(manifest)
        manifest["created_with"] = f"stratum-hq v{__version__}"
        manifest["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        manifest_path = tmp / MANIFEST_FILE
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        uploads.append((manifest_path, MANIFEST_FILE))

        # Generate dataset card
        card_path = tmp / "README.md"
        _write_dataset_card(manifest, hub_repo, card_path,
                            license_id=license_id, attribution=attribution)
        uploads.append((card_path, "README.md"))

        # Upload all files
        eprint(f"Uploading {len(uploads)} files to {hub_repo}...")
        for local_path, remote_path in uploads:
            try:
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=remote_path,
                    repo_id=hub_repo,
                    repo_type="dataset",
                )
            except Exception as e:
                eprint(f"error uploading {remote_path}: {e}")
                return 1

    eprint(f"Published successfully. Manifest version: {manifest['version']}")
    return 0
