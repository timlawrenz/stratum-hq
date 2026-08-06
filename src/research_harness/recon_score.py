"""Scoring/aggregation pass for the arm #37 run root (no GPU).

Loads the preserved generations from the run root, scores each against its
read-only source with CLIP ViT-L/14 (openai/clip-vit-large-patch14, HF cache
at $HF_HOME), aggregates the paired mean delta, and writes delta.json /
records.jsonl / run-provenance.json.

Usage:
  python -m research_harness.recon_score --run-root <root> --plan <plan>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .recon import aggregate_deltas, clip_cosine, item_seed, load_pilot_items

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
DEFAULT_RUN_ROOT = Path("/mnt/nas-ai-models/research/stratum/stage-b-reconstruction-v1")


class ReconScoreError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat()


def load_clip(device: str = "cpu"):
    from transformers import CLIPModel, CLIPProcessor

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    return model, processor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--plan", type=Path, default=Path("/home/tim/source/activity/stratum-hq-stage-b-experiment/research/stage-b-plans/stage-b-reconstruction-v1.json"))
    args = ap.parse_args()

    run_root = args.run_root
    plan = json.loads(args.plan.read_text())
    if plan.get("plan_id") != "stage-b-reconstruction-v1":
        raise ReconScoreError("unexpected plan identity")
    items = load_pilot_items()
    by_id = {i["image_id"]: i["source_relative_path"] for i in items}
    src_root = Path("/mnt/nas-ai-models/training-data/crawlr/approved")

    outputs = run_root / "outputs"
    ctx_dir = outputs / "recon-ctx4k"
    base_dir = outputs / "recon-baseline"
    null_dir = outputs / "recon-null"
    for d in (ctx_dir, base_dir):
        if not d.is_dir():
            raise ReconScoreError(f"missing outputs dir: {d}")

    # Verify the exact expected generation set exists (24 x 2 + 2 null).
    seeds = {i["image_id"]: item_seed(i["image_id"]) for i in items}
    expected_ctx = {f"{iid}_{seeds[iid]}.png" for iid in seeds}
    got_ctx = {p.name for p in ctx_dir.glob("*.png")}
    if got_ctx != expected_ctx:
        missing = expected_ctx - got_ctx
        raise ReconScoreError(f"ctx4k output set mismatch; missing {len(missing)}: {sorted(missing)[:5]}")
    got_base = {p.name for p in base_dir.glob("*.png")}
    if got_base != expected_ctx:
        raise ReconScoreError(f"baseline output set mismatch; missing {len(sorted(expected_ctx - got_base))}")

    model, processor = load_clip("cpu")
    sim_ctx: dict[str, float] = {}
    sim_base: dict[str, float] = {}
    for iid in seeds:
        sim_ctx[iid] = clip_cosine(model, processor, ctx_dir / f"{iid}_{seeds[iid]}.png", src_root / by_id[iid])
        sim_base[iid] = clip_cosine(model, processor, base_dir / f"{iid}_{seeds[iid]}.png", src_root / by_id[iid])
    null_paths = sorted(null_dir.glob("*.png")) if null_dir.is_dir() else []
    if len(null_paths) != 2:
        raise ReconScoreError(f"expected 2 null outputs, got {len(null_paths)}")
    null_sims = [clip_cosine(model, processor, p, src_root / by_id[list(by_id)[0]]) for p in null_paths]

    paired = [
        {"image_id": iid, "sim_ctx4k": round(sim_ctx[iid], 6), "sim_baseline": round(sim_base[iid], 6)}
        for iid in sorted(sim_ctx)
    ]
    agg = aggregate_deltas(paired)
    agg["null_floor_similarity_mean"] = round(sum(null_sims) / len(null_sims), 6)

    records = [
        {"image_id": iid, "condition": "recon-ctx4k", "seed": seeds[iid],
         "sim_clip_vitl14": round(sim_ctx[iid], 6)}
        for iid in sorted(sim_ctx)
    ]
    records += [
        {"image_id": iid, "condition": "recon-baseline", "seed": seeds[iid],
         "sim_clip_vitl14": round(sim_base[iid], 6)}
        for iid in sorted(sim_base)
    ]
    records += [
        {"image_id": f"null{k}", "condition": "recon-null", "seed": 9000 + k,
         "sim_clip_vitl14": round(null_sims[k], 6)}
        for k in range(len(null_sims))
    ]

    provenance = {
        "status": "PENDING_INDEPENDENT_REVIEW",
        "verdict_stamp": "PENDING_HUMAN_SPOT_CHECK",
        "arm": "reconstruction",
        "arm_issue": 37,
        "plan_id": plan["plan_id"],
        "generation": {
            "job_id": "stratum-stage-b-reconstruction-v1",
            "gpu": "4090",
            "checkpoint_sha256": plan["generation_settings"]["checkpoint_sha256"],
            "sampler": plan["generation_settings"]["sampler_name"],
            "steps": plan["generation_settings"]["steps"],
            "cfg": plan["generation_settings"]["cfg"],
            "size": f"{plan['generation_settings']['width']}x{plan['generation_settings']['height']}",
            "generated_png_count": len(records),
            "note": "generation stage completed under the scheduler claim 2026-08-06; the claim was RELEASED AS FAILED because the CLIP processor could not load (HF cache held only config.json). The generated artifacts are complete and deterministic; scoring/aggregation completed in a separate CPU pass after downloading the pre-registered openai/clip-vit-large-patch14 to $HF_HOME.",
        },
        "scoring": {
            "model": CLIP_MODEL_ID,
            "device": "cpu",
            "preprocess": "CLIPProcessor 224 center-crop",
            "similarity": "cosine of [CLS] embeddings",
            "aggregate": agg,
        },
        "completed_at": _now(),
    }
    (run_root / "delta.json").write_text(json.dumps({"aggregate": agg}, indent=1))
    (run_root / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (run_root / "run-provenance.json").write_text(json.dumps(provenance, indent=1))

    print(json.dumps(agg, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())