"""Fail-closed parsing behavior for research-harness command-line inputs."""

from __future__ import annotations

import json
import subprocess
import sys


def valid_program() -> dict:
    return {
        "schema_version": 1,
        "program_id": "example-open-research",
        "canonical_source": {
            "path": "/mnt/datasets/example/approved",
            "subject_invariant": "project-defined-invariant",
            "detector_disagreement": "quality anomaly",
        },
        "content_policy": {
            "model_execution": "local_first",
            "autonomous_external_image_model_allowed": False,
            "reason": "example policy",
            "external_model_requirement": "review required",
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
            "command": "/tmp/scheduler.py",
            "execution_mode": "observer_only",
            "max_job_duration_hours": 24,
            "scheduler_project": "example-open-research",
            "allowed_launchers": ["registered-research-launcher"],
            "resources": {
                "gpu-a": {
                    "host_route": "local",
                    "total_vram_gb": 24,
                    "usable_vram_gb": 24,
                }
            },
        },
    }


def test_cli_rejects_non_standard_nan_json_constant(tmp_path) -> None:
    path = tmp_path / "program.json"
    payload = json.dumps(valid_program()).replace("100000", "NaN", 1)
    path.write_text(payload)

    result = subprocess.run(
        [sys.executable, "-m", "research_harness", "validate-program", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "non-standard JSON constant" in result.stderr


def test_cli_rejects_invalid_utf8_without_traceback(tmp_path) -> None:
    path = tmp_path / "program.json"
    path.write_bytes(b"\xff\xfe")

    result = subprocess.run(
        [sys.executable, "-m", "research_harness", "validate-program", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "unable to decode" in result.stderr
    assert "Traceback" not in result.stderr
