"""DINOv3 feature extraction pipeline — CLS + spatial patch embeddings."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from stratum.config import DINO_MODEL_ID, DINOV3_CLS_FILE, DINOV3_PATCHES_FILE
from stratum.pipeline.bucket import load_bucketed_image, parse_bucket_dims


def eprint(*args: object, **kwargs: object) -> None:
    print(*args, file=sys.stderr, **kwargs)


def load_dinov3(device, model_id: str | None = None):
    """Load DINOv3 model and processor.

    Uses :data:`~stratum.config.DINO_MODEL_ID` when *model_id* is ``None``.
    Returns a dict with ``"kind"``, ``"processor"``, and ``"model"`` keys.
    Falls back to a HuggingFace pipeline when ``AutoModel`` loading fails.
    """
    from transformers import AutoImageProcessor, AutoModel, pipeline

    if model_id is None:
        model_id = DINO_MODEL_ID

    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        model.to(device)
        model.eval()
        return {"kind": "automodel", "processor": processor, "model": model}
    except Exception:
        dev = 0 if device.type == "cuda" else -1
        fe = pipeline(model=model_id, task="image-feature-extraction", device=dev)
        return {"kind": "pipeline", "feature_extractor": fe}


def compute_dinov3_both(
    dino,
    device,
    image,
    target_width: int | None = None,
    target_height: int | None = None,
) -> tuple[list, np.ndarray | None]:
    """Extract CLS token and patch embeddings in a single forward pass.

    Args:
        dino: Model dict returned by :func:`load_dinov3`.
        device: Torch device.
        image: PIL Image.
        target_width: Bucket width — determines spatial resolution via RoPE.
        target_height: Bucket height — determines spatial resolution via RoPE.

    Returns:
        ``(cls_embedding, patches)`` where *cls_embedding* is a list of floats
        and *patches* is an ``(num_patches, 1024)`` float32 array (or ``None``
        when running the pipeline fallback).

    DINOv3 token sequence: ``[CLS, reg_1, …, reg_4, patch_1, …, patch_N]``.
    CLS is at index 0; the next 4 tokens are register tokens.  Spatial
    patches start at index 5.
    """
    import torch

    NUM_REGISTERS = 4

    if dino["kind"] == "pipeline":
        feats = dino["feature_extractor"](image)
        x = feats[0]
        while isinstance(x, list) and x and isinstance(x[0], list):
            seq = x
            hidden = len(seq[0])
            x = [sum(tok[i] for tok in seq) / len(seq) for i in range(hidden)]
        return x, None

    processor = dino["processor"]
    model = dino["model"]

    # Round target dims to nearest multiple of 16 (DINOv3 patch_size) for RoPE
    if target_width is not None and target_height is not None:
        dino_w = round(target_width / 16) * 16
        dino_h = round(target_height / 16) * 16

        inputs = processor(
            images=image,
            size={"height": dino_h, "width": dino_w},
            do_center_crop=False,  # preserve spatial alignment
            do_resize=True,
            return_tensors="pt",
        )
    else:
        inputs = processor(images=image, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    # CLS token
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        cls_emb = outputs.pooler_output[0]
    else:
        cls_emb = outputs.last_hidden_state[0, 0]

    cls_list = cls_emb.detach().cpu().float().tolist()

    # Patch tokens: skip CLS (idx 0) and 4 register tokens (idx 1-4).
    # Spatial patches begin at index 5.
    patches = outputs.last_hidden_state[0, 1 + NUM_REGISTERS :, :]
    patches_np = patches.detach().cpu().float().numpy()

    return cls_list, patches_np


def process(
    image_path: Path,
    output_dir: Path,
    dino,
    device,
    aspect_bucket: str | None = None,
) -> bool:
    """Load a bucketed image, compute DINOv3 embeddings, and save them.

    Saves :data:`~stratum.config.DINOV3_CLS_FILE` and
    :data:`~stratum.config.DINOV3_PATCHES_FILE` into *output_dir*.

    Returns ``True`` on success, ``False`` on failure.
    """
    target_w: int | None = None
    target_h: int | None = None

    if aspect_bucket is not None:
        dims = parse_bucket_dims(aspect_bucket)
        if dims is None:
            eprint(f"warning: invalid aspect_bucket '{aspect_bucket}' for {image_path}")
            return False
        target_w, target_h = dims

    # Load image — use bucketed crop when dimensions are available
    try:
        if target_w is not None and target_h is not None:
            image = load_bucketed_image(image_path, target_w, target_h)
        else:
            from PIL import Image

            with Image.open(image_path) as im:
                image = im.convert("RGB")
    except Exception as e:
        eprint(f"warning: failed to load image {image_path}: {e}")
        return False

    try:
        cls_list, patches_np = compute_dinov3_both(
            dino, device, image, target_width=target_w, target_height=target_h
        )
    except Exception as e:
        eprint(f"warning: DINOv3 forward pass failed for {image_path}: {e}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / DINOV3_CLS_FILE, np.asarray(cls_list, dtype=np.float32))

    if patches_np is not None:
        np.save(output_dir / DINOV3_PATCHES_FILE, patches_np.astype(np.float32))

    return True
