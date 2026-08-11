"""GitHub-native issue-list input tests for the research-harness CLI."""

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
            "subject_invariant": "any-project-specific-invariant",
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


def active_issue() -> dict:
    return {
        "number": 7,
        "state": "OPEN",
        "labels": ["research", "research:active"],
        "body": """# Arm

<!-- research-harness:
{"kind":"arm","hypothesis":"A","falsified_if":"B","pre_registered_gate":"C","metric_version":"v1","data_snapshot":"snapshot","valid_non_improving_experiments":0}
-->
""",
    }


def test_cli_accepts_native_github_issue_list_json(tmp_path) -> None:
    program_path = tmp_path / "program.json"
    program_path.write_text(json.dumps(valid_program()))
    github_list_path = tmp_path / "issues.json"
    github_list_path.write_text(json.dumps([active_issue()]))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "research_harness",
            "validate-tree",
            str(program_path),
            str(github_list_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "valid"


def test_cli_rejects_non_object_non_list_tree_snapshot(tmp_path) -> None:
    program_path = tmp_path / "program.json"
    program_path.write_text(json.dumps(valid_program()))
    snapshot_path = tmp_path / "issues.json"
    snapshot_path.write_text('"not a snapshot"')

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "research_harness",
            "validate-tree",
            str(program_path),
            str(snapshot_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "issue-tree snapshot" in result.stderr
