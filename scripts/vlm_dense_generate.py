#!/usr/bin/env python
"""Frozen dense-description generator for arm #47 (vlm-dense-description).

Runs the verified qwen3-vl:32b (digest ff2e46876908) over the frozen 24-item
cohort: one deterministic 2x2-collage image per item (full frame + seg2-derived
focal crops: face, upper body, lower/garment), a frozen six-subsection prompt,
[OBSERVED]/[INFERRED]/[ABSTAIN] tagging, scale-invariant prose only.

Writes per item:
  <stage>/<image_id>/vlm-dense.json      ({block_text, block_sha256, tags, ...})
plus <stage>/vlm-blocks.jsonl and a terminal <stage>/vlm-done.json containing
the per-item sha256 map (which the Stage-B plan freeze binds), model digest,
prompt fingerprint, and abstention/leak summary stats.

Intent: run on Strix (owned hardware) under a scheduler claim; OLM_* env
overrides the Ollama endpoint/model for a dry-run or a different host.
Reads only: canonical source (read-only) + derived pose2/seg2 (read-only).
Writes only under the noncanonical stage root. No corpus mutation.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError as exc:  # pragma: no cover - host check
    sys.stderr.write(f"vlm_dense_generate: missing dependency: {exc}\n")
    raise SystemExit(3)

# DOME-29 class indices (pinned, cross-checked in harness tests).
FACE_NECK = 3
HAIR = 4
LOWER_CLOTHING = 13
TORSO = 22
UPPER_CLOTHING = 23
APPAREL = 1

STAGE_ROOT = Path("/mnt/nas-ai-models/research/stratum/stage-b-vlm-dense-v1")
MANIFEST = Path("/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json")
SOURCE_ROOT = Path("/mnt/nas-ai-models/training-data/crawlr/approved")
DERIVED_ROOT = Path("/mnt/nas-ai-models/training-data/crawlr/stratum")

MODEL = os.environ.get("OLM_MODEL", "qwen3-vl:32b")
MODEL_DIGEST = os.environ.get("OLM_DIGEST", "ff2e46876908")
ENDPOINT = os.environ.get("OLM_ENDPOINT", "http://127.0.0.1:11434/api/generate")
SEED = int(os.environ.get("OLM_SEED", "20260806"))
NUM_PREDICT = int(os.environ.get("OLM_NPRED", "4096"))
NUM_CTX = int(os.environ.get("OLM_NCTX", "16384"))
TIMEOUT = int(os.environ.get("OLM_TIMEOUT", "900"))
KEEP_ALIVE = os.environ.get("OLM_KEEP_ALIVE", "5m")

PROMPT_FINGERPRINT_BODY = dict(
    sections=("SUBJECT", "GARMENTS", "HAIR", "SKIN", "POSE", "SETTING"),
    tags=("OBSERVED", "INFERRED", "ABSTAIN"),
    scale_invariant_only=True,
    absolute_pixel_claims=False,
    single_adult_female_subject=True,
    revival=1,
)

PROMPT = """You are the dense-description specialist for a single curated portrait photograph. The image below shows the FULL FRAME and three focal crops (face, upper body, lower body/garments) of the SAME photograph. Describe ONLY what is visible in these views of the photograph.

Emit a structured markdown block with exactly these six subsections, in this order:
## SUBJECT
## GARMENTS
## HAIR
## SKIN
## POSE
## SETTING

Rules (non-negotiable):
- Tag every factual statement with [OBSERVED] when it is directly visible in the image, [INFERRED] when it is a reasonable inference from visible evidence, or [ABSTAIN] followed by a one-line reason when a region is not visible enough (motion blur, occlusion, depth of field) to describe.
- NEVER invent details that are not visible. When uncertain, [ABSTAIN] with a reason.
- Use scale-invariant language only: relative sizes, ratios, and spatial relations. NEVER give absolute pixel measurements, absolute coordinates, or numeric dimensions in any unit.
- Be dense and specific where visible: garment types, colors, patterns, material hints, hair style and color, visible skin tone, pose, composition, background elements, lighting character.
- The photograph contains exactly one adult woman. Describe her appearance factually as a subject; do not name her and do not make any personal or biographical claims."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _prompt_fingerprint() -> str:
    payload = {**PROMPT_FINGERPRINT_BODY, "text": PROMPT}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(canonical.encode("utf-8"))


def _read_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"vlm_dense_generate: cannot read {label}: {exc}\n")
        raise SystemExit(3)
    if not isinstance(value, dict):
        sys.stderr.write(f"vlm_dense_generate: {label} must be a JSON object\n")
        raise SystemExit(3)
    return value


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _fit_cover(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Center-crop to the target aspect then resize (deterministic, covers)."""
    tw, th = size
    w, h = img.size
    scale = max(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = img.resize((nw, nh), Image.Resampling.BICUBIC)
    left = max(0, (nw - tw) // 2)
    top = max(0, (nh - th) // 2)
    return resized.crop((left, top, left + tw, top + th))


def build_collage(image: Image.Image, seg2: np.ndarray) -> Image.Image:
    """Deterministic 2x2 collage: full frame + face + upper body + lower/garment.

    Each cell is 512x512 on a 1024x1024 canvas. Empty crops leave a mid-gray
    cell labeled by position (the prompt tells the model crops may be absent).
    """
    cell = 512
    canvas = Image.new("RGB", (cell * 2, cell * 2), (128, 128, 128))
    canvas.paste(_fit_cover(image.convert("RGB"), (cell, cell)), (0, 0))

    def crop(mask: np.ndarray) -> Image.Image | None:
        box = _bbox(mask)
        if box is None:
            return None
        x0, y0, x1, y1 = box
        pad_x = int((x1 - x0) * 0.08) + 2
        pad_y = int((y1 - y0) * 0.08) + 2
        x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
        x1, y1 = min(image.width, x1 + pad_x), min(image.height, y1 + pad_y)
        region = image.crop((x0, y0, x1, y1))
        return _fit_cover(region, (cell, cell))

    face = crop(np.isin(seg2, (FACE_NECK, HAIR)))
    upper = crop(np.isin(seg2, (UPPER_CLOTHING, TORSO, APPAREL)))
    lower = crop(np.isin(seg2, (LOWER_CLOTHING,)))
    for idx, region in enumerate((face, upper, lower), start=1):
        if region is not None:
            canvas.paste(region, ((idx % 2) * cell, (idx // 2) * cell))
    return canvas


def _ollama_call(image: Image.Image, prompt: str) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95, subsampling=0)
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [base64.b64encode(buffer.getvalue()).decode("ascii")],
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": 0.0,
            "seed": SEED,
            "num_predict": NUM_PREDICT,
            "top_k": 1,
            "top_p": 1.0,
            "num_ctx": NUM_CTX,
        },
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        result = json.loads(response.read().decode("utf-8"))
    text = str(result.get("response", ""))
    if not text.strip():
        raise RuntimeError("empty ollama response")
    return text


def _attention_stats(block: str) -> dict:
    observed = len(re.findall(r"\[OBSERVED\]", block))
    inferred = len(re.findall(r"\[INFERRED\]", block))
    abstain = len(re.findall(r"\[ABSTAIN\]", block))
    total = observed + inferred + abstain
    return {
        "observed": observed,
        "inferred": inferred,
        "abstain": abstain,
        "abstain_rate": round(abstain / total, 4) if total else None,
        "tagged_claims": total,
    }


_ABS_PX_RE = re.compile(
    r"\b\d{2,}(?:\s*[x×*]\s*\d{2,})?\s*(?:px|pixels?|pixel\b|mm|cm|inches?|deg(?:rees)?)\b",
    re.IGNORECASE,
)


def _leak_stats(block: str) -> dict:
    hits = _ABS_PX_RE.findall(block)
    return {"absolute_px_hits": hits, "leak": bool(hits)}


def _run_item(item: dict, out_dir: Path) -> dict:
    image_id = item["image_id"]
    relative = item["source_relative_path"]
    source = SOURCE_ROOT / relative
    seg2_path = DERIVED_ROOT / image_id / "seg2.npy"

    seg2 = np.load(seg2_path, allow_pickle=False)
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        dims = item.get("source_dimensions") or {}
        if (image.width, image.height) != (dims.get("width"), dims.get("height")):
            raise RuntimeError(f"source dimensions drifted for {relative}")
    collage = build_collage(image, seg2)

    block = ""
    last_error = None
    for attempt in range(1, 4):
        try:
            block = _ollama_call(collage, PROMPT)
            break
        except Exception as exc:  # noqa: BLE001 - retry on transient host errors
            last_error = exc
            time.sleep(4 + 2 * attempt)
    if not block:
        raise RuntimeError(f"ollama call failed for {image_id}: {last_error}")

    block_sha = _sha256_text(block)
    meta = {
        "image_id": image_id,
        "source_relative_path": relative,
        "source_sha256": item["source_sha256"],
        "model": MODEL,
        "model_digest": MODEL_DIGEST,
        "prompt_fingerprint": _prompt_fingerprint(),
        "seed": SEED,
        "num_predict": NUM_PREDICT,
        "num_ctx": NUM_CTX,
        "block_sha256": block_sha,
        "block_character_count": len(block),
        "block_token_estimate": max(1, len(block) // 4),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "attention": _attention_stats(block),
        "leak": _leak_stats(block),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vlm-dense.json").write_text(
        json.dumps({"meta": meta, "block_text": block}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"image_id": image_id, **meta, "block_text": block}


def main() -> int:
    manifest = _read_json(MANIFEST, "candidate manifest")
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        sys.stderr.write("vlm_dense_generate: manifest has no items\n")
        return 3

    blocks_dir = STAGE_ROOT / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = STAGE_ROOT / "vlm-blocks.jsonl"
    done_path = STAGE_ROOT / "vlm-done.json"
    if done_path.exists():
        print(f"vlm_dense_generate: already complete at {done_path}")
        return 0

    # Resume: skip items whose per-item artifact already exists and is valid.
    summaries = []
    append_mode = jsonl_path.exists()
    with jsonl_path.open("a" if append_mode else "w", encoding="utf-8") as handle:
        for item in items:
            image_id = item["image_id"]
            out_dir = blocks_dir / image_id
            record_path = out_dir / "vlm-dense.json"
            if record_path.exists():
                try:
                    existing = json.loads(record_path.read_text(encoding="utf-8"))
                    block_text = existing.get("block_text") or ""
                    if existing["meta"]["block_sha256"] == _sha256_text(block_text):
                        summaries.append({"image_id": image_id, **existing["meta"]})
                        continue
                except (OSError, ValueError, KeyError, TypeError):
                    pass  # corrupt partial -> regenerate
            record = _run_item(item, out_dir)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            summaries.append({"image_id": image_id, **{k: v for k, v in record.items() if k != "block_text"}})
            print(f"vlm_dense_generate: {image_id} block {record['block_character_count']} chars "
                  f"tags={record['attention']} leak={record['leak']['leak']}", flush=True)

    by_id = {s["image_id"]: s["block_sha256"] for s in summaries}
    if len(by_id) != len(items):
        sys.stderr.write(f"vlm_dense_generate: expected {len(items)} blocks, got {len(by_id)}\n")
        return 3
    total_abstain = sum((s["attention"].get("abstain") or 0) for s in summaries)
    total_tagged = sum((s["attention"].get("tagged_claims") or 0) for s in summaries)
    leakers = [s["image_id"] for s in summaries if s["leak"]["leak"]]
    done = {
        "schema_version": 1,
        "stage": str(STAGE_ROOT),
        "model": MODEL,
        "model_digest": MODEL_DIGEST,
        "prompt_fingerprint": _prompt_fingerprint(),
        "prompt_sha256": _sha256_text(PROMPT),
        "seed": SEED,
        "num_predict": NUM_PREDICT,
        "num_ctx": NUM_CTX,
        "item_count": len(by_id),
        "vlm_blocks_sha256": by_id,
        "attention_summary": {
            "tagged_claims": total_tagged,
            "abstain": total_abstain,
            "abstain_rate": round(total_abstain / total_tagged, 4) if total_tagged else None,
        },
        "leak_summary": {"leaking_items": leakers},
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    done_path.write_text(json.dumps(done, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "stage": str(STAGE_ROOT),
                      "item_count": len(by_id), "attention": done["attention_summary"],
                      "leakers": leakers}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())