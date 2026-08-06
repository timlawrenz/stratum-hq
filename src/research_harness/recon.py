"""Arm #37 — generative reconstruction validation (ComfyUI round-trip).

Deterministic, artifact-driven measurement surface for the `reconstruction`
verdict method (`autonomous-tick --method reconstruction`).

Pre-registered design (see research/stage-b-plans/stage-b-reconstruction-v1.json):

- Items: the frozen 24-item pilot manifest from the arm-36 round-trip plan
  (`stage-b-roundtrip-context4k-v1`), source images read-only from
  ``/mnt/nas-ai-models/training-data/crawlr/approved``.
- Variant condition  : prompt = per-item context4k.md compact artifact
  (evidence-linked, ``dossier-context4k-v2/<image_id>/context4k.md``).
- Baseline condition : prompt = a fixed, item-independent degraded prompt
  (no context). Identical across all 24 items by construction.
- Null calibration   : prompt = meaningless tokens, 2 images (floor check).
- Generation         : one frozen SDXL checkpoint, fixed sampler/steps/cfg/
  cfg-scale/size, per-item seed (sha256 of image_id, SAME seed across both
  conditions of an item so the pair isolates the prompt axis).
- Scoring            : openai/clip-vit-large-patch14 (ViT-L/14, 224px),
  center-crop both images, cosine similarity of [CLS] embeddings.
- Delta rule         : reconstruction_delta = mean over 24 items of
  (sim(ctx4k) - sim(baseline)). Paired, scale-invariant to camera framing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

GOAL_ARM_PLAN = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment/research/stage-b-plans/stage-b-roundtrip-context4k-v1.json")
DOSSIER_V2_ROOT = Path("/mnt/nas-ai-models/research/stratum/dossier-context4k-v2")
CANONICAL_SOURCE_ROOT = Path("/mnt/nas-ai-models/training-data/crawlr/approved")

BASELINE_PROMPT = (
    "photorealistic studio portrait photograph of a person, "
    "neutral pose, plain background, soft diffuse lighting"
)
NULL_PROMPT = "zzzzzzzzzz"

# Frozen generation settings (one axis changed: the prompt text only).
CHECKPOINT_NAME = "Juggernaut_XL_v1759168.safetensors"
CHECKPOINT_SOURCE = "/mnt/nas-ai-models/checkpoints/SDXL/Juggernaut_XL_v1759168.safetensors"
SAMPLER_NAME = "dpmpp_2m"
SCHEDULER_NAME = "karras"
STEPS = 28
CFG = 7.0
WIDTH = 832
HEIGHT = 1216
NEGATIVE_PROMPT = ""
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"


class ReconError(RuntimeError):
    """Raised when the reconstruction measurement cannot be built honestly."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def item_seed(image_id: str) -> int:
    """Deterministic per-item seed, identical across conditions.

    sha256(image_id) truncated to 32 bits; never 0 (SDXL seed semantics).
    """
    seed = int(hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:8], 16)
    return seed if seed != 0 else 1


def load_pilot_items() -> list[dict[str, Any]]:
    """Return the frozen 24-item pilot manifest (read-only)."""
    plan = json.loads(GOAL_ARM_PLAN.read_text())
    manifest = plan.get("pilot_manifest")
    if not isinstance(manifest, Mapping) or not manifest.get("frozen"):
        raise ReconError("arm-36 pilot manifest is not frozen")
    items = manifest.get("items")
    if not isinstance(items, list) or len(items) != 24:
        raise ReconError(f"pilot manifest must contain exactly 24 items, got {len(items) if isinstance(items, list) else '?'}")
    return items


def load_context4k_artifact(image_id: str) -> str:
    """Return the evidence-linked compact context for one item (verbatim)."""
    path = DOSSIER_V2_ROOT / image_id / "context4k.md"
    if not path.is_file():
        raise ReconError(f"context4k.md missing for {image_id}: {path}")
    return path.read_text(encoding="utf-8")


def build_items() -> list[dict[str, Any]]:
    """Assemble the 24 paired rows with prompts, seeds and source paths."""
    rows: list[dict[str, Any]] = []
    for item in load_pilot_items():
        image_id = item["image_id"]
        src = CANONICAL_SOURCE_ROOT / item["source_relative_path"]
        if not src.is_file():
            raise ReconError(f"source image missing: {src}")
        ctx = load_context4k_artifact(image_id)
        rows.append(
            {
                "image_id": image_id,
                "source_relative_path": item["source_relative_path"],
                "source_sha256": item["source_sha256"],
                "source_abs": str(src),
                "seed": item_seed(image_id),
                "prompts": {
                    "recon-ctx4k": ctx,
                    "recon-baseline": BASELINE_PROMPT,
                },
                "prompt_sha256": {
                    "recon-ctx4k": _sha256_bytes(ctx.encode("utf-8")),
                    "recon-baseline": _sha256_bytes(BASELINE_PROMPT.encode("utf-8")),
                },
            }
        )
    return rows


def build_frozen_plan(repo_root: Path, *, checkpoint_sha256: str, plan_id: str = "stage-b-reconstruction-v1") -> dict[str, Any]:
    """Build the pre-registered, frozen plan JSON (hashes everything frozen)."""
    items = load_pilot_items()
    ctx_hashes = [load_context4k_artifact(i["image_id"]) for i in items]
    inputs_fingerprint = _sha256_bytes(
        json.dumps(
            {
                "context4k_md": [_sha256_bytes(c.encode("utf-8")) for c in ctx_hashes],
                "baseline_prompt": BASELINE_PROMPT,
                "null_prompt": NULL_PROMPT,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return {
        "schema_version": 1,
        "kind": "reconstruction",
        "program_id": "stratum-contextual-specialist-research",
        "plan_id": plan_id,
        "parent_issue": 37,
        "status": "preregistered",
        "hypothesis": "Generative reconstruction is a valid, non-LLM measure of per-asset information preserved by the evidence-linked compact context: images generated from context4k.md via a frozen SDXL checkpoint score higher in CLIP ViT-L/14 similarity against the source than images generated from an item-independent degraded baseline prompt.",
        "falsified_if": "Reconstruction similarity does not correlate with information content (mean per-item CLIP delta <= 0 over the frozen 24-item cohort with paired seeds), or the diffusion checkpoint fails to decode the relevant attributes at all (degenerate ceiling where both conditions saturate).",
        "deterministic_signal": "CLIP ViT-L/14 cosine similarity (source vs generated-from-context4k, and source vs generated-from-degraded-baseline) via local ComfyUI over one frozen SDXL checkpoint",
        "metric_version": "reconstruction-clip-v1",
        "data_snapshot": "first-500 core-covered cohort (24 frozen items); local ComfyUI /mnt/fscache/essdee/ComfyUI; checkpoint " + CHECKPOINT_NAME + "; CLIP openai/clip-vit-large-patch14",
        "conditions": [
            {
                "id": "recon-ctx4k",
                "role": "variant",
                "prompt_source": "dossier-context4k-v2/<image_id>/context4k.md verbatim",
                "prompt_text_sha256_fingerprint": inputs_fingerprint,
            },
            {
                "id": "recon-baseline",
                "role": "baseline",
                "prompt_source": "fixed item-independent degraded prompt (no context)",
            },
            {
                "id": "recon-null",
                "role": "calibration",
                "prompt_source": "meaningless tokens ('zzzzzzzzzz'), 2 images",
            },
        ],
        "contrasts": [
            {
                "id": "reconstruction-delta",
                "baseline_condition": "recon-baseline",
                "variant_condition": "recon-ctx4k",
                "changed_axes": ["prompt"],
                "delta_rule": "mean over 24 items of CLIP(ctx4k) - CLIP(baseline); paired same-seed",
            }
        ],
        "pilot_manifest": {
            "id": "first500-coverage-balanced-candidate-v1",
            "frozen": True,
            "source_root": str(CANONICAL_SOURCE_ROOT),
            "items": items,
        },
        "generation_settings": {
            "checkpoint_name": CHECKPOINT_NAME,
            "checkpoint_sha256": checkpoint_sha256,
            "sampler_name": SAMPLER_NAME,
            "scheduler_name": SCHEDULER_NAME,
            "steps": STEPS,
            "cfg": CFG,
            "width": WIDTH,
            "height": HEIGHT,
            "negative_prompt": NEGATIVE_PROMPT,
            "seed_per_item": "sha256(image_id) truncated to 32 bits; identical across conditions",
        },
        "scoring": {
            "model": CLIP_MODEL_ID,
            "preprocess": "224px center-crop both images",
            "similarity": "cosine of [CLS] embeddings",
        },
        "null_case": {"prompt": NULL_PROMPT, "expectation": "low similarity floor", "images": 2},
        "statistical_rule": "BETTER iff reconstruction_delta > 0.0 (harness verdict method=reconstruction); NOT_BETTER otherwise. delta reported as mean; paired positive count and median also recorded.",
        "review_protocol": "protocol independently reviewable: frozen plan, checkpoint sha256, per-item seeds and prompts hashes, outputs archived under /mnt/nas-ai-models/research/stratum/stage-b-reconstruction-v1; verdict advisory until human spot-check (PENDING_HUMAN_SPOT_CHECK stamp).",
        "representation_boundary": [
            "CLIP ViT-L/14 similarity is a non-LLM generative proxy, not identity verification",
            "scores depend on the single frozen checkpoint and sampler; changed settings invalidate the comparison",
            "only same-seed paired differences are interpreted; absolute similarity values are not evidence",
        ],
        "metric_self_audit": [
            "one frozen checkpoint; no seed search; no cherry-picking",
            "null-case generated under identical settings to bound the floor",
            "source images read-only; outputs additive under the approved research root",
        ],
        "repo_root": str(repo_root),
    }


def aggregate_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-item paired similarities into the measured delta.

    rows: list of {"image_id", "sim_ctx4k", "sim_baseline"} (sim in [0, 1]).
    Returns the pre-registered aggregate (mean delta + supporting stats +
    signed tally) plus per-item rows for the run record.
    """
    if len(rows) != 24:
        raise ReconError(f"reconstruction requires 24 paired items, got {len(rows)}")
    deltas = []
    for r in rows:
        d = r["sim_ctx4k"] - r["sim_baseline"]
        r = dict(r)
        r["delta"] = d
        deltas.append(r)
    mean_delta = sum(r["delta"] for r in deltas) / len(deltas)
    positive = sum(1 for r in deltas if r["delta"] > 0)
    ties = sum(1 for r in deltas if r["delta"] == 0)
    sorted_d = sorted(r["delta"] for r in deltas)
    median_delta = sorted_d[len(sorted_d) // 2]
    return {
        "reconstruction_delta": round(mean_delta, 6),
        "median_delta": round(median_delta, 6),
        "paired_positive": positive,
        "paired_negative": len(deltas) - positive - ties,
        "paired_ties": ties,
        "items": len(deltas),
        "per_item": deltas,
    }