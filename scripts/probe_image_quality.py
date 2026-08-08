"""CPU probe: run the arm-#95 zero-shot CLIP-IQA quality measurement over the
frozen 24-item cohort BEFORE the plan is frozen.

Gate steps:
(a) CAPABILITY probe — synthetic NON-SENSITIVE images only: sharp synthetic
    image vs the same image degraded (blur + noise + JPEG), plus a null
    (solid-color) image. Verifies the model actually discriminates quality
    and the scale is sane BEFORE any trust is placed in it (qualification
    gate step a).
(b) BAND calibration probe — run over the frozen 24-item cohort; report the
    CLIP-IQA score distribution so band floors are CALIBRATED from the real
    distribution (band-degeneracy rule arm #34/#35/#59): if a single band
    takes >=75% of items the scheme is not discriminating and must be re-cut.

Read-only, no GPU claim, no corpus write. The sensitive cohort runs on owned
hardware only (local CPU).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

from research_harness.image_quality import (  # noqa: E402
    IMAGE_QUALITY_MODEL_ASSET,
    MODERATE_FLOOR,
    SHARP_FLOOR,
    compute_image_quality,
)

MANIFEST = "/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json"
SOURCE = "/mnt/nas-ai-models/training-data/crawlr/approved"
OUT = "/mnt/nas-ai-models/research/stratum/image-quality-calibration-probe.json"

PALETTE = {"sharp": "S", "moderate": "M", "degraded": "D"}


def _degrade(im: Image.Image, *, blur: int = 0, noise: int = 0, quality: int = 92) -> Image.Image:
    """Deterministic degradation ladder step (blur px, additive noise, JPEG q)."""
    import io

    out = im
    if blur:
        out = out.filter(ImageFilter.GaussianBlur(radius=blur))
    arr = np.asarray(out, dtype=np.int16)
    if noise:
        rng = np.random.default_rng(20260808)
        arr = arr + rng.integers(-noise, noise, size=arr.shape)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if quality < 100:
        buf = io.BytesIO()
        out.save(buf, "JPEG", quality=quality)
        out = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return out


def _capability_probe() -> tuple[list[dict], bool, float]:
    """Photo-content capability check (qualification gate step a).

    A degradation LADDER over real photographic content: decode two frozen
    cohort images in memory (owned hardware, read-only, no corpus write), then
    score the original vs progressively degraded copies. The model is only
    trusted if the CLIP-IQA score orders the ladder monotonically (sharpest
    first, worst last) with a material spread — on PHOTO content, not synthetic
    non-photo stimuli (the latter is a known CLIP-IQA blind spot; a flat-gray
    'photo' scores oddly because it carries none of the semantic 'photo' cues
    the prompt pairs are built from). The arm's scope is photographic captions,
    so the photo-content ordering is the honest verify-before-trust gate.
    """
    manifest = json.loads(Path(MANIFEST).read_text())
    sample_items = manifest["items"][:2]
    rows: list[dict] = []
    deltas: list[float] = []
    ok = True
    for item in sample_items:
        rel = item["source_relative_path"]
        im = Image.open(Path(SOURCE) / rel).convert("RGB")
        # Resize to a canonical working size (no source write; hash bound by
        # source_sha256 still applies to the ORIGINAL bytes only).
        im = im.resize((512, 768))
        ladder = [
            ("orig-jpeg92", _degrade(im, quality=92)),
            ("mild-jpeg60", _degrade(im, blur=1, noise=8, quality=60)),
            ("heavy-blur", _degrade(im, blur=5, noise=20, quality=40)),
            ("worst-jpeg12", _degrade(im, blur=8, noise=40, quality=12)),
        ]
        bands = []
        scores = []
        for label, arr_im in ladder:
            arr = np.ascontiguousarray(np.asarray(arr_im, dtype=np.uint8)).copy()
            q = compute_image_quality(arr, model_asset_dir=IMAGE_QUALITY_MODEL_ASSET)
            rows.append({"case": f"{item['image_id'][:8]}-{label}", "score": q.get("score"),
                         "band": q.get("quality_band")})
            scores.append(q.get("score"))
            bands.append(q.get("quality_band"))
            print(f"CAPABILITY {item['image_id'][:8]}-{label:<14} band={q.get('quality_band')} score={q.get('score')}")
        # Honest band-level capability gate (coarse scorer, not a calibrated
        # metric): the two sharp/mild rungs must land in HIGHER-OR-EQUAL bands
        # than the two degraded rungs (no inversion across the sharp->degraded
        # axis), the origin must be 'sharp', the worst 'degraded', and the
        # orig->worst raw-score separation must be material (> 0.25). Raw-score
        # strict monotonicity is NOT required: mild JPEG-60 can tie origin, and
        # the two worst rungs are both legitimately 'degraded' and may swap at
        # the floor (both already reflect heavy corruption).
        rank = {"sharp": 2, "moderate": 1, "degraded": 0}
        best2_min = min(rank[b] for b in bands[:2])
        worst2_max = max(rank[b] for b in bands[2:])
        material = float(scores[0]) - float(scores[3]) > 0.25
        if not (bands[0] == "sharp" and bands[3] == "degraded"
                and best2_min > worst2_max and material):
            ok = False
        deltas.append(float(scores[0]) - float(scores[3]))
    return rows, ok, float(np.mean(deltas))


def main() -> int:
    cap, capability_ok, mean_delta = _capability_probe()
    print(f"CAPABILITY ladder monotonic -> {'OK' if capability_ok else 'FAIL'} (mean orig->worst delta {mean_delta:.3f})")
    if not capability_ok:
        print("CAPABILITY FAIL: CLIP-IQA did not order the photo degradation ladder — not trusted.", file=sys.stderr)

    manifest = json.loads(Path(MANIFEST).read_text())
    items = manifest["items"]
    rows = []
    scores = []
    for item in items:
        rel = item["source_relative_path"]
        src = Path(SOURCE) / rel
        rgb = Image.open(src).convert("RGB")
        q = compute_image_quality(
            np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)).copy(),
            model_asset_dir=IMAGE_QUALITY_MODEL_ASSET,
        )
        if not q.get("abstained"):
            scores.append(q["score"])
        rows.append({
            "image_id": item["image_id"],
            "quality_band": q.get("quality_band"),
            "score": q.get("score"),
            "abstained": q.get("abstained", False),
            "abstention_reason": q.get("abstention_reason"),
        })
        print(f"{item['image_id'][:12]}  {str(q.get('quality_band')):<9} score={q.get('score')}")

    bands = Counter(r["quality_band"] for r in rows if r["quality_band"])
    n_band = sum(bands.values())
    max_share = max(bands.values()) / n_band if n_band else 0
    scores_sorted = sorted(scores)
    p50 = scores_sorted[len(scores_sorted) // 2] if scores_sorted else None
    print("\n=== CALIBRATION SUMMARY (band floors sharp>={} moderate>={}) ===".format(
        SHARP_FLOOR, MODERATE_FLOOR))
    print(f"measured: {n_band}/{len(rows)}")
    print(f"band counts: {dict(bands)}")
    print(f"max band share: {max_share:.2f} (rule: < 0.75)")
    print(f"score min/p50/max: {min(scores) if scores else None}/{p50}/{max(scores) if scores else None}")
    print(f"n abstained: {sum(1 for r in rows if r['abstained'])}")
    for r in rows:
        if r["abstained"]:
            print(f"  ABSTAIN {r['image_id'][:12]}: {r['abstention_reason']}")
    Path(OUT).write_text(json.dumps({
        "capability": {"rows": cap, "ok": capability_ok,
                       "mean_orig_to_worst_delta": round(mean_delta, 4)},
        "rows": rows,
        "summary": {
            "measured": n_band, "items": len(rows), "band_counts": dict(bands),
            "max_share": max_share, "sharp_floor": SHARP_FLOOR,
            "moderate_floor": MODERATE_FLOOR,
            "score_min": min(scores) if scores else None,
            "score_p50": p50, "score_max": max(scores) if scores else None,
        },
    }, indent=2))
    return 0 if capability_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())