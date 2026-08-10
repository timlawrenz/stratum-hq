#!/usr/bin/env python
"""Freeze the image-quality REVISION-2 Stage-B comparison plan + GPU manifest.

Arm #95, strike-1 retry (2026-08-08). Revision-2 rebuilds the SAME evidence
kind (``image-quality``) after an aspect-level band-degeneracy recovery: the
91.7%-degenerate "good/bad" CLIP-IQA aspect was excluded from the aggregate and
the band floors were re-calibrated to the 3-aspect score's lower absolute scale
(SHARP_FLOOR 0.60 -> 0.55; MODERATE_FLOOR stays 0.35). The revised specialist
passes the capability degradation ladder and the no-band->=75% calibration rule
(probe: 13 sharp / 8 moderate / 3 degraded, max_share 0.54; capability ladder
monotonic OK, mean orig->worst delta 0.492).

This is a NEW round-trip with a distinct plan/manifest/output-root/job-id
(revision-2), additive and non-overwriting of the revision-1 artifacts. The
``comparison_plan_id`` remains the evidence-kind slot label; the revision is
carried by the plan fingerprint (the evidence declaration hashes the revised
image_quality.py bytes) and by this script's distinct v2 file/run/job names.

CPU-only. No model invoked; no corpus write; outputs land only in the
noncanonical research tree (read-only crawlr/approved source).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

from research_harness.stage_b import (  # noqa: E402
    StageBGenerationSettings,
    freeze_stage_b_plan,
)
from research_harness.contracts import validate_gpu_manifest  # noqa: E402
from research_harness.labels import ContractError  # noqa: E402

PROGRAM = ROOT / "research/program.json"
CANDIDATE = Path("/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json")
PLAN_OUT = ROOT / "research/stage-b-plans/stage-b-image-quality-v2.json"
MANIFEST_OUT = ROOT / "research/gpu-manifests/stage-b-image-quality-v2.json"
OUTPUT_ROOT = Path("/mnt/nas-ai-models/research/stratum/stage-b-image-quality-v2")
JOB_ID = "stratum-stage-b-image-quality-v2"

# Arm-#4 approved settings (identical to revision-1), only evidence axis changes.
SETTINGS = StageBGenerationSettings(
    endpoint="http://127.0.0.1:11434/api/generate",
    model_name="gemma3:27b",
    model_digest="a418f5838eaf7fe2cfe0a3046c8384b68ba43a4435542c942f9db00a5f342203",
    temperature=0.0,
    seed=20260804,
    num_predict=384,
    top_k=1,
    top_p=1.0,
    context_window=4096,
    timeout_seconds=300,
)

# Frozen open-weight CLIP ViT-L/14 asset directory (arm #95, same asset as rev-1).
MODEL_DIR = Path("/mnt/nas-ai-models/research/stratum/models/image-quality")
MODEL_WEIGHTS = MODEL_DIR / "model.safetensors"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    import subprocess

    out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _runner_source_hash() -> str:
    return _sha256(ROOT / "src/research_harness/stage_b.py")


def _launcher_source_hash() -> str:
    return _sha256(ROOT / "src/research_harness/stage_b_launcher.py")


def main() -> int:
    program = json.loads(PROGRAM.read_text(encoding="utf-8"))
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    if not MODEL_WEIGHTS.exists():
        print(f"FAIL model asset missing: {MODEL_WEIGHTS}", file=sys.stderr)
        return 2
    model_sha = _sha256(MODEL_WEIGHTS)

    plan = freeze_stage_b_plan(program, candidate, SETTINGS, evidence_kind="image-quality")
    expected_id = "stage-b-first500-image-quality-v1"
    if plan.get("comparison_plan_id") != expected_id:
        print(f"FAIL plan id {plan.get('comparison_plan_id')!r} != {expected_id!r}", file=sys.stderr)
        return 2

    PLAN_OUT.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    from research_harness.stage_b import _canonical_json as stage_canonical

    plan_sha = hashlib.sha256(stage_canonical(plan).encode("utf-8")).hexdigest()

    manifest = {
        "schema_version": 1,
        "approved_issue": 18,
        "authorization": {
            "approval_issue": 18,
            "approved_by": "timlawrenz direct #18 approval and autonomous-decision delegation in authenticated Hermes WebUI, 2026-08-04",
            "mode": "human_reviewed",
        },
        "execution": {
            "candidate_manifest_fingerprint": candidate.get("manifest_fingerprint"),
            "candidate_manifest_path": str(CANDIDATE),
            "comparison_plan_fingerprint": plan["comparison_plan_fingerprint"],
            "comparison_plan_relative_path": "research/stage-b-plans/stage-b-image-quality-v2.json",
            "comparison_plan_sha256": plan_sha,
            "expected_record_count": 96,
            "generation": {
                "context_window": SETTINGS.context_window,
                "endpoint": SETTINGS.endpoint,
                "num_predict": SETTINGS.num_predict,
                "seed": SETTINGS.seed,
                "temperature": SETTINGS.temperature,
                "timeout_seconds": SETTINGS.timeout_seconds,
                "top_k": SETTINGS.top_k,
                "top_p": SETTINGS.top_p,
            },
            "generation_fingerprint": SETTINGS.fingerprint,
            "git_commit": _git_head(),
            "launcher_source_sha256": _launcher_source_hash(),
            "model_digest": SETTINGS.model_digest,
            "model_name": SETTINGS.model_name,
            "runner_module": "research_harness.stage_b",
            "runner_source_sha256": _runner_source_hash(),
        },
        "host_route": "local",
        "job_id": JOB_ID,
        "launcher_id": "registered-research-launcher",
        "manifest_state": "approved",
        "maximum_duration": "2h",
        "output_root": str(OUTPUT_ROOT),
        "requested_vram_gb": 22,
        "revision": 2,
        "scheduler_lifecycle": ["request", "poll_and_claim", "launch", "verify",
                                "activate", "heartbeat", "release"],
        "scheduler_project": "stratum-contextual-specialist-research",
        "target_gpu": "4090",
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        validate_gpu_manifest(manifest, program)
    except ContractError as exc:
        print(f"FAIL manifest validation: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({
        "plan_written": str(PLAN_OUT),
        "manifest_written": str(MANIFEST_OUT),
        "plan_id": plan.get("comparison_plan_id"),
        "plan_fingerprint": plan.get("comparison_plan_fingerprint"),
        "evidence_id": plan["conditions"][-1]["evidence"]["id"],
        "evidence_input_names": sorted(
            next(iter(plan["evidence_input_artifact_sha256"].values()))
        ),
        "model_asset_sha256": model_sha,
        "git_commit": manifest["execution"]["git_commit"],
        "revision": 2,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
