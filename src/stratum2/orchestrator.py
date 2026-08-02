"""Pipeline orchestration — load models once, run passes over images.

Enforces dependency ordering: seg2 runs before normal2/pointmap (which need
the foreground mask).  Non-Sapiens passes (caption, dinov3, t5, pixel) come
from ``stratum.pipeline.*`` and run in parallel with everything else.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from stratum2.config import (
    MATTING_FILE,
    NORMAL2_FILE,
    POINTMAP_FILE,
    POSE2_FILE,
    SEG2_FILE,
)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _pick_device(device_str: str):
    """Resolve 'auto' to best available device."""
    import torch

    if device_str != "auto":
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _needs(out_dir: Path, filename: str) -> bool:
    """Check if an artifact needs generating (doesn't exist yet)."""
    return not (out_dir / filename).exists()


def run_passes(
    images: list[Path],
    input_dir: Path,
    output_dir: Path,
    passes: list[str],
    device: str = "auto",
    ollama_url: str = "http://192.168.86.137:11434/api/generate",
    ollama_model: str = "gemma3:27b",
    caption_max_tokens: int = 500,
    progress_every: int = 100,
    verbose: bool = False,
) -> int:
    """Run the specified passes over all images. Returns exit code.

    Supported stratum2 passes: seg2, normal2, pointmap, pose2, matting
    Non-Sapiens passes (from stratum): caption, dinov3, t5, pixel

    Dependency ordering:
      Phase 1: seg2, pose2, matting, caption, dinov3, t5, pixel (all parallel)
      Phase 2: normal2, pointmap (need seg2 foreground mask)
    """
    # --- Determine which passes to run ---
    run_caption = "caption" in passes
    run_dinov3 = "dinov3" in passes
    run_t5 = "t5" in passes
    run_pixel = "pixel" in passes
    run_seg2 = "seg2" in passes
    run_normal2 = "normal2" in passes
    run_pointmap = "pointmap" in passes
    run_pose2 = "pose2" in passes
    run_matting = "matting" in passes

    # --- Non-Sapiens passes (from stratum) ---
    caption_backend = None
    dino = None
    t5_tokenizer = None
    t5_encoder = None
    torch_device = None

    needs_torch = run_dinov3 or run_t5 or run_seg2 or run_normal2 or run_pointmap or run_pose2 or run_matting
    if needs_torch:
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
        t5_encoder = load_t5_encoder().to(torch_device)

    # --- Sapiens2 models ---
    seg2_model = None
    normal2_model = None
    pointmap_model = None
    pose2_model = None
    matting_model = None

    if run_seg2:
        from stratum2.loader import load_sapiens2_model

        eprint("loading Sapiens2 seg model...")
        seg2_model = load_sapiens2_model("seg", device=str(torch_device))

    if run_normal2:
        from stratum2.loader import load_sapiens2_model

        eprint("loading Sapiens2 normal model...")
        normal2_model = load_sapiens2_model("normal", device=str(torch_device))

    if run_pointmap:
        from stratum2.loader import load_sapiens2_model

        eprint("loading Sapiens2 pointmap model...")
        pointmap_model = load_sapiens2_model("pointmap", device=str(torch_device))

    if run_pose2:
        from stratum2.loader import load_sapiens2_model

        eprint("loading Sapiens2 pose model + DETR detector...")
        pose2_model = load_sapiens2_model("pose", device=str(torch_device))

    if run_matting:
        from stratum2.loader import load_sapiens2_model

        eprint("loading Sapiens2 matting model...")
        matting_model = load_sapiens2_model("matting", device=str(torch_device))

    # --- Process images ---
    from stratum.config import (
        CAPTION_FILE,
        DINOV3_CLS_FILE,
        DINOV3_PATCHES_FILE,
        METADATA_FILE,
        PIXEL_FILE,
        T5_HIDDEN_FILE,
        T5_MASK_FILE,
    )
    from stratum.discovery import image_id_from_path, output_dir_for_image
    from stratum.pipeline.bucket import assign_aspect_bucket

    def _ensure_metadata(image_path: Path, out_dir: Path) -> dict | None:
        from PIL import Image

        meta_path = out_dir / METADATA_FILE
        if meta_path.exists():
            try:
                with meta_path.open() as f:
                    return json.load(f)
            except Exception:
                pass

        try:
            with Image.open(image_path) as img:
                w, h = img.size
        except Exception as exc:
            eprint(f"warning: cannot read dimensions for {image_path}: {exc}")
            return None

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

    counters = {"processed": 0, "skipped": 0, "errors": 0}
    started = time.time()

    # Prefetch images in a background thread so the GPU never waits on CIFS
    from stratum2.prefetch import PrefetchReader

    with PrefetchReader(images, depth=8) as prefetcher:
        for i, (image_path, preloaded) in enumerate(prefetcher):
            if image_path is None:
                continue  # sentinel guard
            out_dir = output_dir_for_image(image_path, input_dir, output_dir)
            meta = _ensure_metadata(image_path, out_dir)
            if meta is None:
                counters["errors"] += 1
                continue

            aspect_bucket = meta.get("aspect_bucket")
            did_work = False

            # --- Phase 1: parallel-safe passes ---

            # Caption
            if run_caption and _needs(out_dir, CAPTION_FILE):
                from stratum.pipeline.caption import process as caption_process

                if verbose:
                    eprint(f"  captioning {meta['image_id']}...")
                if caption_process(
                    image_path, out_dir, caption_backend, aspect_bucket, caption_max_tokens
                ):
                    did_work = True
                else:
                    counters["errors"] += 1

            # DINOv3
            if run_dinov3 and (
                _needs(out_dir, DINOV3_CLS_FILE) or _needs(out_dir, DINOV3_PATCHES_FILE)
            ):
                from stratum.pipeline.dinov3 import process as dinov3_process

                if verbose:
                    eprint(f"  DINOv3 {meta['image_id']}...")
                if dinov3_process(image_path, out_dir, dino, torch_device, aspect_bucket):
                    did_work = True
                else:
                    counters["errors"] += 1

            # T5
            if run_t5 and (
                _needs(out_dir, T5_HIDDEN_FILE) or _needs(out_dir, T5_MASK_FILE)
            ):
                if (out_dir / CAPTION_FILE).exists():
                    from stratum.pipeline.t5 import process as t5_process

                    if verbose:
                        eprint(f"  T5 {meta['image_id']}...")
                    if t5_process(out_dir, t5_tokenizer, t5_encoder, torch_device):
                        did_work = True
                    else:
                        counters["errors"] += 1

            # Pixel
            if run_pixel and aspect_bucket and _needs(out_dir, PIXEL_FILE):
                from stratum.pipeline.pixel import process as pixel_process

                if verbose:
                    eprint(f"  pixel {meta['image_id']}...")
                if pixel_process(image_path, out_dir, aspect_bucket):
                    did_work = True
                else:
                    counters["errors"] += 1

            # Seg2
            if run_seg2 and _needs(out_dir, SEG2_FILE):
                from stratum2.pipeline.seg import process as seg2_process

                if verbose:
                    eprint(f"  seg2 {meta['image_id']}...")
                if seg2_process(
                    image_path, out_dir, seg2_model, torch_device, aspect_bucket,
                    image=preloaded,
                ):
                    did_work = True
                else:
                    counters["errors"] += 1

            # Pose2
            if run_pose2 and _needs(out_dir, POSE2_FILE):
                from stratum2.pipeline.pose import process as pose2_process

                if verbose:
                    eprint(f"  pose2 {meta['image_id']}...")
                if pose2_process(
                    image_path, out_dir, pose2_model, torch_device, aspect_bucket,
                    image=preloaded,
                ):
                    did_work = True
                else:
                    counters["errors"] += 1

            # Matting
            if run_matting and _needs(out_dir, MATTING_FILE):
                from stratum2.pipeline.matting import process as matting_process

                if verbose:
                    eprint(f"  matting {meta['image_id']}...")
                if matting_process(
                    image_path, out_dir, matting_model, torch_device, aspect_bucket,
                    image=preloaded,
                ):
                    did_work = True
                else:
                    counters["errors"] += 1

            # --- Phase 2: seg2-dependent passes ---

            # Normal2 (requires seg2.npy)
            if run_normal2 and _needs(out_dir, NORMAL2_FILE):
                if (out_dir / SEG2_FILE).exists():
                    from stratum2.pipeline.normal import process as normal2_process

                    if verbose:
                        eprint(f"  normal2 {meta['image_id']}...")
                    if normal2_process(
                        image_path, out_dir, normal2_model, torch_device, aspect_bucket,
                        image=preloaded,
                    ):
                        did_work = True
                    else:
                        counters["errors"] += 1
                elif verbose:
                    eprint(f"  normal2 skipped {meta['image_id']} (no seg2)")

            # Pointmap (requires seg2.npy)
            if run_pointmap and _needs(out_dir, POINTMAP_FILE):
                if (out_dir / SEG2_FILE).exists():
                    from stratum2.pipeline.pointmap import process as pointmap_process

                    if verbose:
                        eprint(f"  pointmap {meta['image_id']}...")
                    if pointmap_process(
                        image_path, out_dir, pointmap_model, torch_device, aspect_bucket,
                        image=preloaded,
                    ):
                        did_work = True
                    else:
                        counters["errors"] += 1
                elif verbose:
                    eprint(f"  pointmap skipped {meta['image_id']} (no seg2)")

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
