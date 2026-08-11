"""TDD coverage for the scheduler-bound Stage-B registered launcher."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from research_harness.stage_b import freeze_stage_b_plan
from research_harness.stage_b_launcher import (
    StageBLaunchError,
    _launcher_source_hash,
    _runner_source_hash,
    run_scheduler_bound_stage_b,
    validate_execution_binding,
)
from tests.test_stage_b_runner import _fixture


def _manifest(program: dict, candidate: dict, plan: dict, settings, output_root: Path) -> dict:
    return {
        "schema_version": 1,
        "job_id": "fixture-stage-b-job",
        "target_gpu": "4090",
        "requested_vram_gb": 22,
        "maximum_duration": "2h",
        "approved_issue": 18,
        "manifest_state": "approved",
        "authorization": {
            "mode": "human_reviewed",
            "approved_by": "timlawrenz direct Stage-B delegation",
            "approval_issue": 18,
        },
        "host_route": "local",
        "launcher_id": "registered-research-launcher",
        "scheduler_project": program["gpu_scheduler"]["scheduler_project"],
        "output_root": str(output_root),
        "scheduler_lifecycle": [
            "request",
            "poll_and_claim",
            "launch",
            "verify",
            "activate",
            "heartbeat",
            "release",
        ],
        "execution": {
            "runner_module": "research_harness.stage_b",
            "candidate_manifest_fingerprint": candidate["manifest_fingerprint"],
            "candidate_manifest_path": str(output_root.parent / "candidate.json"),
            "comparison_plan_fingerprint": plan["comparison_plan_fingerprint"],
            "comparison_plan_sha256": hashlib.sha256(
                json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            ).hexdigest(),
            "comparison_plan_relative_path": "research/templates/comparison-parity-plan.template.json",
            "model_name": settings.model_name,
            "model_digest": settings.model_digest,
            "generation_fingerprint": settings.fingerprint,
            "runner_source_sha256": _runner_source_hash(),
            "launcher_source_sha256": _launcher_source_hash(),
            "git_commit": "0" * 40,
            "expected_record_count": len(candidate["items"]) * 4,
            "generation": {
                "endpoint": settings.endpoint,
                "temperature": settings.temperature,
                "seed": settings.seed,
                "num_predict": settings.num_predict,
                "top_k": settings.top_k,
                "top_p": settings.top_p,
                "context_window": settings.context_window,
                "timeout_seconds": settings.timeout_seconds,
            },
        },
    }


def _materialize_candidate_manifest(manifest: dict, candidate: dict) -> None:
    path = Path(manifest["execution"]["candidate_manifest_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidate))


def test_execution_binding_rejects_model_or_plan_drift_before_scheduler(tmp_path: Path) -> None:
    program, candidate, settings, research_root = _fixture(tmp_path)
    plan = freeze_stage_b_plan(program, candidate, settings)
    manifest = _manifest(program, candidate, plan, settings, research_root / "run")
    manifest["execution"]["model_digest"] = "b" * 64

    with pytest.raises(StageBLaunchError, match="model_digest"):
        validate_execution_binding(manifest, program, candidate, plan, settings)


def test_registered_launcher_runs_request_poll_activate_heartbeat_verify_release_in_order(
    tmp_path: Path,
) -> None:
    program, candidate, settings, research_root = _fixture(tmp_path)
    plan = freeze_stage_b_plan(program, candidate, settings)
    output_root = research_root / "run"
    manifest = _manifest(program, candidate, plan, settings, output_root)
    _materialize_candidate_manifest(manifest, candidate)
    lifecycle: list[str] = []

    class FakeProcess:
        def __init__(self) -> None:
            self.wait_calls = 0

        def poll(self):
            return None

        def wait(self, timeout: int):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired("stage-b", timeout)
            return 0

    def scheduler(action: str, _args: list[str]) -> str:
        lifecycle.append(action)
        return {
            "request": manifest["job_id"],
            "poll": "claimed",
            "activate": "activated",
            "heartbeat": "ok",
            "release": "released",
        }[action]

    def launch(_command: list[str], log_path: Path):
        lifecycle.append("launch")
        log_path.write_text("launcher started\n")
        return FakeProcess()

    def verify_launch(_child) -> float:
        lifecycle.append("verify-launch")
        return 17.5

    def verify() -> None:
        lifecycle.append("verify")

    result = run_scheduler_bound_stage_b(
        manifest,
        program,
        candidate,
        plan,
        settings,
        scheduler_call=scheduler,
        launch_runner=launch,
        verify_launch=verify_launch,
        verify_run=verify,
        vram_used_gb=lambda: 17.5,
        heartbeat_interval_seconds=1,
        log_path=tmp_path / "scheduler.log",
    )

    assert result["status"] == "completed"
    assert lifecycle == [
        "request", "poll", "launch", "verify-launch", "activate", "heartbeat", "heartbeat", "verify", "release"
    ]


def test_registered_launcher_releases_its_own_claim_as_failed_when_child_exits_nonzero(
    tmp_path: Path,
) -> None:
    program, candidate, settings, research_root = _fixture(tmp_path)
    plan = freeze_stage_b_plan(program, candidate, settings)
    manifest = _manifest(program, candidate, plan, settings, research_root / "run")
    _materialize_candidate_manifest(manifest, candidate)
    lifecycle: list[str] = []

    class FailedProcess:
        def poll(self):
            return None

        def wait(self, timeout: int):
            return 2

    def scheduler(action: str, _args: list[str]) -> str:
        lifecycle.append(action)
        return {
            "request": manifest["job_id"],
            "poll": "claimed",
            "activate": "activated",
            "heartbeat": "ok",
            "release": "released",
        }[action]

    with pytest.raises(StageBLaunchError, match="exit code 2"):
        run_scheduler_bound_stage_b(
            manifest,
            program,
            candidate,
            plan,
            settings,
            scheduler_call=scheduler,
            launch_runner=lambda _command, log_path: (log_path.write_text("started\n"), FailedProcess())[1],
            verify_launch=lambda _child: 1.0,
            verify_run=lambda: None,
            vram_used_gb=lambda: 1.0,
            heartbeat_interval_seconds=1,
            log_path=tmp_path / "scheduler.log",
        )

    assert lifecycle == ["request", "poll", "activate", "heartbeat", "release"]


def test_launcher_refuses_to_request_when_manifest_binding_is_invalid(tmp_path: Path) -> None:
    program, candidate, settings, research_root = _fixture(tmp_path)
    plan = freeze_stage_b_plan(program, candidate, settings)
    manifest = _manifest(program, candidate, plan, settings, research_root / "run")
    manifest = copy.deepcopy(manifest)
    manifest["execution"]["comparison_plan_fingerprint"] = "0" * 64
    calls: list[str] = []

    with pytest.raises(StageBLaunchError, match="comparison_plan_fingerprint"):
        run_scheduler_bound_stage_b(
            manifest,
            program,
            candidate,
            plan,
            settings,
            scheduler_call=lambda action, _args: calls.append(action) or "unexpected",
            launch_runner=lambda _command, _log: None,
            verify_launch=lambda _child: 0.0,
            verify_run=lambda: None,
            vram_used_gb=lambda: 0.0,
            heartbeat_interval_seconds=1,
            log_path=tmp_path / "scheduler.log",
        )

    assert calls == []
