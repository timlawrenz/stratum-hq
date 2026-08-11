"""Freeze the hand-gesture Stage-B comparison plan + GPU manifest (CPU-only).

Arm #109 (NEW MODEL CLASS open-weight MediaPipe HandLandmarker + NEW evidence
part hand-gesture, registered 2026-08-10 via the gated propose-dimensions
channel, selected via EXPLOIT at selection_progress 33). Reuses the tested
`build_stage_b_plan` / `freeze_stage_b_plan` machinery with
evidence_kind="hand-gesture" and the exact frozen generation settings from the
arm-#4 manifest, so the only changed axis is the declared specialist
(per-visible-hand gesture class from within-hand finger-extension segment
ratios + hand-raised flag gated on the pose2 wrist/shoulder reference).

Capability + band calibration (measured 2026-08-10 frozen 24-item cohort
probe): coverage PASS 14/24 items with a visible hand (pre-registered floor
>= 14/24; full-frame-then-2x-upscale union policy is load-bearing —
full-frame alone measures 12/24); gesture distribution 15 hands: fist 11 /
relaxed-curl 4 (max_share 0.733, no band >= 75%); open-palm / pointing fire
0 in this cohort and remain declared; the hand-raised flag fired 2/14
measured (rare-positive claim); the one-vs-two-hands count axis is SILENCED
(payload-only; probe max_share one-hand 13/14 = 0.929 >= 0.75 degeneracy
gate). No corpus write; outputs land only in the noncanonical research tree.
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
    _canonical_json,
    freeze_stage_b_plan,
)
from research_harness.contracts import validate_gpu_manifest  # noqa: E402
from research_harness.labels import ContractError  # noqa: E402

PROGRAM = ROOT / "research/program.json"
CANDIDATE = Path("/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json")
PLAN_OUT = ROOT / "research/stage-b-plans/stage-b-hand-gesture-v1.json"
MANIFEST_OUT = ROOT / "research/gpu-manifests/stage-b-hand-gesture-v1.json"
OUTPUT_ROOT = Path("/mnt/nas-ai-models/research/stratum/stage-b-hand-gesture-v1")
JOB_ID = "stratum-stage-b-hand-gesture-v1"

# Arm-#4 approved settings (identical), only evidence axis changes.
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

    plan = freeze_stage_b_plan(program, candidate, SETTINGS, evidence_kind="hand-gesture")
    expected_id = "stage-b-first500-hand-gesture-v1"
    if plan.get("comparison_plan_id") != expected_id:
        print(f"FAIL plan id {plan.get('comparison_plan_id')!r} != {expected_id!r}", file=sys.stderr)
        return 2

    PLAN_OUT.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    plan_sha = hashlib.sha256(_canonical_json(plan).encode("utf-8")).hexdigest()

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
            "comparison_plan_relative_path": "research/stage-b-plans/stage-b-hand-gesture-v1.json",
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

    print(f"plan wrote:   {PLAN_OUT}  (fingerprint {plan.get('comparison_plan_fingerprint')})")
    print(f"manifest wrote: {MANIFEST_OUT}")
    print("runner_source_hash", _runner_source_hash())
    print("launcher_source_hash", _launcher_source_hash())
    print("git_head", _git_head())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())