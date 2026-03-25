"""CLI entry point for stratum."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_PASSES = ["caption", "dinov3", "t5", "pose", "seg", "depth", "normal"]
ALL_PASSES = ["caption", "dinov3", "t5", "pose", "seg", "depth", "normal", "pixel"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="stratum",
        description="Enriched human image dataset pipeline",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # --- process ---
    proc = sub.add_parser("process", help="Generate dataset artifacts from images")
    proc.add_argument("input_dir", type=Path, help="Directory of source images (recursively scanned)")
    proc.add_argument("--output", type=Path, required=True, help="Output dataset directory")
    proc.add_argument(
        "--passes",
        default="all",
        help="Comma-separated list of passes to run (caption,dinov3,t5,pose,seg,depth,normal,pixel) or 'all' (excludes pixel)",
    )
    proc.add_argument("--device", default="auto", help="Compute device: auto, cpu, cuda, cuda:0, etc.")
    proc.add_argument("--shard", default=None, help="Worker shard as N/M (e.g., 0/4 = worker 0 of 4)")
    proc.add_argument("--image-list", type=Path, default=None, help="Explicit list of image paths (one per line)")
    proc.add_argument("--ollama-url", default="http://localhost:11434/api/generate", help="Ollama API endpoint")
    proc.add_argument("--ollama-model", default="gemma3:27b", help="Ollama model for captioning")
    proc.add_argument("--caption-max-tokens", type=int, default=500, help="Max tokens for caption generation")
    proc.add_argument("--progress-every", type=int, default=100, help="Print progress every N images (0 disables)")
    proc.add_argument("--verbose", action="store_true")

    # --- status ---
    st = sub.add_parser("status", help="Report dataset completeness")
    st.add_argument("dataset_dir", type=Path, help="Dataset directory to inspect")

    # --- verify ---
    vf = sub.add_parser("verify", help="Verify dataset integrity (shapes, dtypes)")
    vf.add_argument("dataset_dir", type=Path, help="Dataset directory to verify")
    vf.add_argument("--fix", action="store_true", help="Delete corrupt artifacts so they can be regenerated")

    # --- publish ---
    pub = sub.add_parser("publish", help="Publish dataset to HuggingFace Hub")
    pub.add_argument("dataset_dir", type=Path, help="Dataset directory")
    pub.add_argument("--hub-repo", required=True, help="HuggingFace repo (e.g., user/stratum-ffhq)")
    pub.add_argument("--layers", required=True, help="Comma-separated layers to publish (caption,dinov3,t5,pose,seg,depth,normal)")
    pub.add_argument("--license", default="cc-by-nc-sa-4.0",
                     help="SPDX license identifier for the dataset card (default: cc-by-nc-sa-4.0)")
    pub.add_argument("--attribution-file", type=Path, default=None,
                     help="Markdown file with attribution/provenance text to include in the dataset card")
    pub.add_argument("--limit", type=int, default=None, help="Max images to publish")
    pub.add_argument("--offset", type=int, default=0, help="Skip first N images")
    pub.add_argument("--verbose", action="store_true", help="Log upload requests, responses, and file sizes")

    # --- reconcile ---
    rec = sub.add_parser("reconcile", help="Reconcile HuggingFace manifest with actual repo files")
    rec.add_argument("--hub-repo", required=True, help="HuggingFace repo (e.g., user/stratum-ffhq)")
    rec.add_argument("--dry-run", action="store_true", help="Show reconciled manifest without uploading")
    rec.add_argument("--license", default="cc-by-nc-sa-4.0",
                     help="SPDX license identifier for the dataset card (default: cc-by-nc-sa-4.0)")
    rec.add_argument("--attribution-file", type=Path, default=None,
                     help="Markdown file with attribution/provenance text to include in the dataset card")
    rec.add_argument("--verbose", action="store_true", help="Log upload requests, responses, and file sizes")

    # --- migrate ---
    mig = sub.add_parser("migrate", help="Convert old prx-tg per-modality dataset to stratum per-image format")
    mig.add_argument("jsonl_path", type=Path, help="Path to approved_image_dataset.jsonl")
    mig.add_argument("--derived-dir", type=Path, default=None,
                     help="Directory containing modality subdirs (dinov3/, t5_hidden/, etc.). Defaults to parent of jsonl_path.")
    mig.add_argument("--output", type=Path, required=True, help="Output dataset directory")
    mig.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    mig.add_argument("--progress-every", type=int, default=1000, help="Print progress every N records")
    mig.add_argument("--verbose", action="store_true")

    return p.parse_args(argv)


def resolve_passes(passes_str: str) -> list[str]:
    """Resolve pass string into list of pass names."""
    if passes_str == "all":
        return list(DEFAULT_PASSES)
    return [p.strip() for p in passes_str.split(",") if p.strip()]


def parse_shard(shard_str: str | None) -> tuple[int, int] | None:
    """Parse 'N/M' into (worker_index, total_workers)."""
    if shard_str is None:
        return None
    parts = shard_str.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid shard format '{shard_str}', expected N/M (e.g., 0/4)")
    n, m = int(parts[0]), int(parts[1])
    if n < 0 or m < 1 or n >= m:
        raise ValueError(f"Invalid shard {n}/{m}: need 0 <= N < M")
    return (n, m)


def cmd_process(args: argparse.Namespace) -> int:
    from stratum.discovery import discover_images, shard_image_list
    from stratum.pipeline import run_passes

    passes = resolve_passes(args.passes)
    images = discover_images(args.input_dir, image_list_path=args.image_list)

    shard = parse_shard(args.shard)
    if shard is not None:
        images = shard_image_list(images, shard[0], shard[1])

    if not images:
        print("No images found.", file=sys.stderr)
        return 1

    print(f"Processing {len(images)} images, passes: {passes}", file=sys.stderr)

    return run_passes(
        images=images,
        input_dir=args.input_dir,
        output_dir=args.output,
        passes=passes,
        device=args.device,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        caption_max_tokens=args.caption_max_tokens,
        progress_every=args.progress_every,
        verbose=args.verbose,
    )


def cmd_status(args: argparse.Namespace) -> int:
    from stratum.discovery import scan_dataset_status

    status = scan_dataset_status(args.dataset_dir)
    total = status.pop("total", 0)
    if total == 0:
        print("No image directories found.", file=sys.stderr)
        return 1

    print(f"Total images: {total:,}")
    for artifact, count in sorted(status.items()):
        pct = (count / total) * 100 if total > 0 else 0
        print(f"  {artifact + ':':20s} {count:>7,} / {total:,} ({pct:.1f}%)")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from stratum.verify import verify_dataset

    return verify_dataset(args.dataset_dir, fix=args.fix)


def cmd_publish(args: argparse.Namespace) -> int:
    from stratum.publish import publish_to_hub

    layers = [l.strip() for l in args.layers.split(",")]
    attribution = None
    if args.attribution_file:
        attribution = args.attribution_file.read_text(encoding="utf-8")
    return publish_to_hub(
        dataset_dir=args.dataset_dir,
        hub_repo=args.hub_repo,
        layers=layers,
        license_id=args.license,
        attribution=attribution,
        limit=args.limit,
        offset=args.offset,
        verbose=args.verbose,
    )


def cmd_reconcile(args: argparse.Namespace) -> int:
    from stratum.publish import reconcile_hub_manifest

    attribution = None
    if args.attribution_file:
        attribution = args.attribution_file.read_text(encoding="utf-8")
    return reconcile_hub_manifest(
        hub_repo=args.hub_repo,
        license_id=args.license,
        attribution=attribution,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


def cmd_migrate(args: argparse.Namespace) -> int:
    from stratum.migrate import migrate_dataset

    derived_dir = args.derived_dir or args.jsonl_path.parent
    return migrate_dataset(
        jsonl_path=args.jsonl_path,
        derived_dir=derived_dir,
        output_dir=args.output,
        dry_run=args.dry_run,
        progress_every=args.progress_every,
        verbose=args.verbose,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    dispatch = {
        "process": cmd_process,
        "status": cmd_status,
        "verify": cmd_verify,
        "publish": cmd_publish,
        "reconcile": cmd_reconcile,
        "migrate": cmd_migrate,
    }

    try:
        return dispatch[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
