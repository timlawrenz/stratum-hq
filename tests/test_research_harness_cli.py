"""End-to-end CLI checks for the research harness package."""

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


def test_module_cli_validates_a_program(tmp_path) -> None:
    path = tmp_path / "program.json"
    path.write_text(json.dumps(valid_program()))

    result = subprocess.run(
        [sys.executable, "-m", "research_harness", "validate-program", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "valid"


def test_module_cli_reports_contract_failure(tmp_path) -> None:
    invalid = valid_program()
    del invalid["canonical_source"]["subject_invariant"]
    path = tmp_path / "program.json"
    path.write_text(json.dumps(invalid))

    result = subprocess.run(
        [sys.executable, "-m", "research_harness", "validate-program", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "subject_invariant" in result.stderr
