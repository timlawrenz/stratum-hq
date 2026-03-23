"""Pipeline orchestration — load models once, run passes over images."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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
from stratum.discovery import image_id_from_path, output_dir_for_image
from stratum.pipeline.bucket import assign_aspect_bucket


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _pick_device(device_str: str):
    """Resolve 'auto' to best available device."""
    import torch

    if device_str != "auto":
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_image_dimensions(image_path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        eprint(f"warning: cannot read dimensions for {image_path}: {e}")
        return None


def _ensure_metadata(image_path: Path, input_dir: Path, out_dir: Path) -> dict | None:
    """Create or load metadata.json. Returns metadata dict or None on failure."""
    meta_path = out_dir / METADATA_FILE
    if meta_path.exists():
        try:
            with meta_path.open() as f:
                return json.load(f)
        except Exception:
            pass  # Regenerate if corrupt

    dims = _get_image_dimensions(image_path)
    if dims is None:
        return None

    w, h = dims
    meta = {
        "image_id": image_id_from_path(image_path, input_dir),
        "source_path": str(image_path),
        "width": w,
        "height": h,
        "aspect_bucket": assign_aspect_bucket(w, h),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def _needs(out_dir: Path, filename: str) -> bool:
    """Check if an artifact needs generating (doesn't exist yet)."""
    return not (out_dir / filename).exists()


def run_passes(
    images: list[Path],
    input_dir: Path,
    output_dir: Path,
    passes: list[str],
    device: str = "auto",
    ollama_url: str = "http://localhost:11434/api/generate",
    ollama_model: str = "gemma3:27b",
    caption_max_tokens: int = 500,
    progress_every: int = 100,
    verbose: bool = False,
) -> int:
    """Run the specified passes over all images. Returns exit code."""
    run_caption = "caption" in passes
    run_dinov3 = "dinov3" in passes
    run_t5 = "t5" in passes
    run_pose = "pose" in passes
    run_pixel = "pixel" in passes

    # Load models lazily based on which passes are requested
    caption_backend = None
    dino = None
    t5_tokenizer = None
    t5_encoder = None
    pose_model = None
    torch_device = None

    if run_dinov3 or run_t5:
        torch_device = _pick_device(device)
        eprint(f"device: {torch_device}")

    if run_caption:
        from stratum.pipeline.caption import OllamaCaptionBackend

        caption_backend = OllamaCaptionBackend(url=ollama_url, model_name=ollama_model)
        eprint(f"caption backend: {ollama_model} at {ollama_url}")

    if run_dinov3:
        from stratum.pipeline.dinov3 import load_dinov3

        eprint("loading DINOv3 model...")
        dino = load_dinov3(torch_device)

    if run_t5:
        from stratum.pipeline.t5 import load_t5_encoder, load_t5_tokenizer

        eprint("loading T5 tokenizer + encoder...")
        t5_tokenizer = load_t5_tokenizer()
        t5_encoder = load_t5_encoder()
        t5_encoder = t5_encoder.to(torch_device)

    if run_pose:
        from stratum.dwpose import DWPoseDetector

        eprint("loading DWPose ONNX model...")
        pose_model = DWPoseDetector(device=device if device != "auto" else "cpu")

    # Process images
    counters = {"processed": 0, "skipped": 0, "errors": 0}
    started = time.time()

    for i, image_path in enumerate(images):
        out_dir = output_dir_for_image(image_path, input_dir, output_dir)

        # Always ensure metadata exists
        meta = _ensure_metadata(image_path, input_dir, out_dir)
        if meta is None:
            counters["errors"] += 1
            continue

        aspect_bucket = meta.get("aspect_bucket")
        did_work = False

        # Caption pass
        if run_caption and _needs(out_dir, CAPTION_FILE):
            from stratum.pipeline.caption import process as caption_process

            if verbose:
                eprint(f"  captioning {meta['image_id']}...")
            if caption_process(image_path, out_dir, caption_backend, aspect_bucket, caption_max_tokens):
                did_work = True
            else:
                counters["errors"] += 1

        # DINOv3 pass
        if run_dinov3 and (_needs(out_dir, DINOV3_CLS_FILE) or _needs(out_dir, DINOV3_PATCHES_FILE)):
            from stratum.pipeline.dinov3 import process as dinov3_process

            if verbose:
                eprint(f"  DINOv3 {meta['image_id']}...")
            if dinov3_process(image_path, out_dir, dino, torch_device, aspect_bucket):
                did_work = True
            else:
                counters["errors"] += 1

        # T5 pass (requires caption.txt to exist)
        if run_t5 and (_needs(out_dir, T5_HIDDEN_FILE) or _needs(out_dir, T5_MASK_FILE)):
            if (out_dir / CAPTION_FILE).exists():
                from stratum.pipeline.t5 import process as t5_process

                if verbose:
                    eprint(f"  T5 {meta['image_id']}...")
                if t5_process(out_dir, t5_tokenizer, t5_encoder, torch_device):
                    did_work = True
                else:
                    counters["errors"] += 1
            elif verbose:
                eprint(f"  T5 skipped {meta['image_id']} (no caption)")

        # Pose pass
        if run_pose and _needs(out_dir, POSE_FILE):
            from stratum.pipeline.pose import process as pose_process

            if verbose:
                eprint(f"  pose {meta['image_id']}...")
            if pose_process(image_path, out_dir, pose_model, aspect_bucket):
                did_work = True
            else:
                counters["errors"] += 1

        # Pixel pass (opt-in)
        if run_pixel and aspect_bucket and _needs(out_dir, PIXEL_FILE):
            from stratum.pipeline.pixel import process as pixel_process

            if verbose:
                eprint(f"  pixel {meta['image_id']}...")
            if pixel_process(image_path, out_dir, aspect_bucket):
                did_work = True
            else:
                counters["errors"] += 1

        if did_work:
            counters["processed"] += 1
        else:
            counters["skipped"] += 1

        # Progress
        total = i + 1
        if progress_every and total % progress_every == 0:
            elapsed = time.time() - started
            rate = counters["processed"] / elapsed if elapsed > 0 else 0
            eprint(
                f"progress: {total}/{len(images)} "
                f"({counters['processed']} processed, {counters['skipped']} skipped, "
                f"{counters['errors']} errors) {rate:.1f} img/s"
            )

    elapsed = time.time() - started
    eprint(
        f"done: {counters['processed']} processed, {counters['skipped']} skipped, "
        f"{counters['errors']} errors in {elapsed:.1f}s"
    )
    return 0
