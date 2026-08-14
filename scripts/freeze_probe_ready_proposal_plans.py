"""Pre-freeze Stage-B comparison plans + GPU manifests for the five
probe-ready proposals (lip-fullness #122, eye-shape #123, hand-face-ratio
#124, eyebrow-thickness #127, cheek-prominence #128). CPU-only.

Hold #132 (data authority) blocks every Stage-B LAUNCH because the launcher
preflights all 24 approved/ sources, but the plan/manifest FREEZE binds only
program.json + the frozen candidate manifest + the derived-tree evidence
inputs (seg2.npy per item, read-only) + generation settings. All five arms'
capability + band-calibration probes already PASSED on the full 24-item
cohort (2026-08-11, cohort-tercile cuts recorded in each module), so with
this pre-freeze the ONLY remaining step after the gate clears is the GPU
round-trip itself (request -> poll-and-launch -> review -> autonomous-tick).

Reuses the exact arm-#4 generation settings (gemma3:27b digest-pinned,
temperature 0.0, seed 20260804, 96 = 24 x 4 records) and the
post-GPU-reservation-lesson 21.0 GiB 4090 reserve. Each manifest pins the
repo git HEAD at freeze time (a commit that carries all five arm modules,
since the branches are stacked); the runner sha pins match this checkout
exactly, so execution checks out this branch/pin when the gate clears.

No corpus write; outputs land only in the noncanonical research tree and
this repo's research/ directory.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
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

# (evidence_kind, comparison_plan_id, job_id, plan file stem)
ARM_SPECS = [
    ("lip-fullness", "stage-b-first500-lip-fullness-v1", "stratum-stage-b-lip-fullness-v1", "stage-b-lip-fullness-v1"),
    ("eye-shape", "stage-b-first500-eye-shape-v1", "stratum-stage-b-eye-shape-v1", "stage-b-eye-shape-v1"),
    ("hand-face-ratio", "stage-b-first500-hand-face-ratio-v1", "stratum-stage-b-hand-face-ratio-v1", "stage-b-hand-face-ratio-v1"),
    ("eyebrow-thickness", "stage-b-first500-eyebrow-thickness-v1", "stratum-stage-b-eyebrow-thickness-v1", "stage-b-eyebrow-thickness-v1"),
    ("cheek-prominence", "stage-b-first500-cheek-prominence-v1", "stratum-stage-b-cheek-prominence-v1", "stage-b-cheek-prominence-v1"),
]

# Arm-#4 approved settings (identical for every arm); only the evidence axis changes.
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
    out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _runner_source_hash() -> str:
    return _sha256(ROOT / "src/research_harness/stage_b.py")


def _launcher_source_hash() -> str:
    return _sha256(ROOT / "src/research_harness/stage_b_launcher.py")


def freeze_one(program: dict, candidate: dict, spec: tuple[str, str, str, str]) -> int:
    evidence_kind, plan_id, job_id, stem = spec
    plan_out = ROOT / f"research/stage-b-plans/{stem}.json"
    manifest_out = ROOT / f"research/gpu-manifests/{stem}.json"
    output_root = Path(f"/mnt/nas-ai-models/research/stratum/{stem}")

    plan = freeze_stage_b_plan(program, candidate, SETTINGS, evidence_kind=evidence_kind)
    if plan.get("comparison_plan_id") != plan_id:
        print(f"FAIL plan id {plan.get('comparison_plan_id')!r} != {plan_id!r}", file=sys.stderr)
        return 2
    plan_out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
            "comparison_plan_relative_path": f"research/stage-b-plans/{stem}.json",
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
        "job_id": job_id,
        "launcher_id": "registered-research-launcher",
        "manifest_state": "approved",
        "maximum_duration": "2h",
        "output_root": str(output_root),
        "requested_vram_gb": 21.0,
        "scheduler_lifecycle": ["request", "poll_and_claim", "launch", "verify",
                                "activate", "heartbeat", "release"],
        "scheduler_project": "stratum-contextual-specialist-research",
        "target_gpu": "4090",
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        validate_gpu_manifest(manifest, program)
    except ContractError as exc:
        print(f"FAIL manifest validation: {exc}", file=sys.stderr)
        return 2

    print(f"[{evidence_kind}] plan wrote:   {plan_out.name}  (fingerprint {plan.get('comparison_plan_fingerprint')})")
    print(f"[{evidence_kind}] manifest wrote: {manifest_out.name}  (job {job_id})")
    return 0


def main() -> int:
    program = json.loads(PROGRAM.read_text(encoding="utf-8"))
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    rc = 0
    for spec in ARM_SPECS:
        rc = max(rc, freeze_one(program, candidate, spec))
    print("runner_source_hash", _runner_source_hash())
    print("launcher_source_hash", _launcher_source_hash())
    print("git_head (pinned)", _git_head())
    return rc


if __name__ == "__main__":
    raise SystemExit(main())