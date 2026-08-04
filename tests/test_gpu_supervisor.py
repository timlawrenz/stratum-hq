"""Tests for the intentionally fail-closed GPU supervisor."""

from __future__ import annotations

import json

import pytest

from research_harness import ContractError
from research_harness.gpu_supervisor import inspect_manifests, main, supervisor_message


def program() -> dict:
    return {
        "schema_version": 1,
        "program_id": "example-open-research",
        "canonical_source": {
            "path": "/mnt/datasets/example/approved",
            "subject_invariant": "exactly_one_curated_woman",
            "detector_disagreement": "quality_anomaly_not_semantic_content",
        },
        "content_policy": {
            "model_execution": "local_first",
            "autonomous_external_image_model_allowed": False,
            "reason": "The example corpus has a documented model-execution policy.",
            "external_model_requirement": "Require a reviewed policy decision.",
        },
        "artifact_policy": {
            "approved_output_roots": ["/mnt/nas-ai-models/research"],
            "canonical_source_write_allowed": False,
        },
        "representation": {
            "expanded_dossier_target_tokens": 100_000,
            "expanded_dossier_min_tokens": 100_000,
            "compact_context_target_tokens": 4_000,
            "compact_context_min_tokens": 4_000,
            "legacy_text_encoder_max_tokens": 512,
            "compact_artifacts": {
                "structured": "context4k.json",
                "human_readable": "context4k.md",
                "provenance": "compression.json",
            },
        },
        "specialists": {
            "policy": "open_world",
            "required_declaration_fields": [
                "scope",
                "inputs",
                "output_semantics",
                "provenance",
                "abstention_policy",
                "known_failure_modes",
                "qualification_gate",
            ],
        },
        "research_tree": {
            "require_program_root": False,
            "require_parent_issue": False,
            "require_selection_rationale": False,
        },
        "autonomy": {
            "mode": "draft_pr_only",
            "autonomous_merge_allowed": False,
            "autonomous_direct_main_push_allowed": False,
            "autonomous_gpu_execution_allowed": False,
            "autonomous_model_installation_allowed": False,
            "autonomous_canonical_source_write_allowed": False,
            "authorized_without_new_human_approval": ["tests"],
            "requires_hold": ["GPU activity"],
        },
        "gpu_scheduler": {
            "command": "/mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py",
            "execution_mode": "observer_only",
            "max_job_duration_hours": 24,
            "scheduler_project": "example-open-research",
            "allowed_launchers": ["registered-research-launcher"],
            "resources": {
                "4090": {
                    "host_route": "local",
                    "total_vram_gb": 24,
                    "usable_vram_gb": 24,
                },
                "strix": {
                    "host_route": "ssh:max395",
                    "total_vram_gb": 110,
                    "usable_vram_gb": 100,
                    "evergreen_reserved_vram_gb": 10,
                },
            },
        },
    }


def manifest() -> dict:
    return {
        "schema_version": 1,
        "job_id": "pilot-context-expansion-001",
        "target_gpu": "4090",
        "requested_vram_gb": 12,
        "maximum_duration": "2h",
        "approved_issue": 12,
        "manifest_state": "approved",
        "authorization": {
            "mode": "human_reviewed",
            "approved_by": "reviewer",
            "approval_issue": 12,
        },
        "host_route": "local",
        "launcher_id": "registered-research-launcher",
        "scheduler_project": "example-open-research",
        "output_root": "/mnt/nas-ai-models/research/pilot-context-expansion-001",
        "scheduler_lifecycle": [
            "request",
            "poll_and_claim",
            "launch",
            "verify",
            "activate",
            "heartbeat",
            "release",
        ],
    }


def test_supervisor_is_silent_when_no_approved_manifest_exists(tmp_path) -> None:
    assert inspect_manifests(program(), tmp_path) == []


def test_supervisor_ignores_draft_manifests(tmp_path) -> None:
    draft = manifest()
    draft["manifest_state"] = "draft"
    (tmp_path / "draft.json").write_text(json.dumps(draft))

    assert inspect_manifests(program(), tmp_path) == []


def test_supervisor_holds_an_approved_manifest_in_observer_only_mode(tmp_path) -> None:
    path = tmp_path / "approved.json"
    path.write_text(json.dumps(manifest()))

    manifests = inspect_manifests(program(), tmp_path)

    assert manifests == [path]
    message = supervisor_message(program(), manifests)
    assert "HOLD" in message
    assert "observer_only" in message
    assert "approved.json" in message


def test_supervisor_rejects_invalid_approved_manifest(tmp_path) -> None:
    broken = manifest()
    broken["host_route"] = "ssh:max395"
    (tmp_path / "broken.json").write_text(json.dumps(broken))

    with pytest.raises(ContractError, match="host_route"):
        inspect_manifests(program(), tmp_path)


def test_supervisor_rejects_invalid_utf8_manifest_as_contract_error(tmp_path) -> None:
    (tmp_path / "broken.json").write_bytes(b"\xff\xfe")

    with pytest.raises(ContractError, match="unable to decode"):
        inspect_manifests(program(), tmp_path)


def test_supervisor_cli_rejects_invalid_utf8_program_without_traceback(tmp_path, capsys) -> None:
    invalid_program = tmp_path / "program.json"
    invalid_program.write_bytes(b"\xff\xfe")

    assert main([str(invalid_program), str(tmp_path)]) == 2
    output = capsys.readouterr().out
    assert "unable to decode" in output
    assert "Traceback" not in output
