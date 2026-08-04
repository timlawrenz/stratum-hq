"""Guard durable project-control assets against silent contract drift."""

from __future__ import annotations

import json
from pathlib import Path

from research_harness import validate_compression_bundle, validate_program

ROOT = Path(__file__).resolve().parents[1]


def test_project_program_contract_validates() -> None:
    program = json.loads((ROOT / "research" / "program.json").read_text())
    validate_program(program)


def test_generic_program_template_validates_without_stratum_hardware_or_subject_assumptions() -> None:
    template = json.loads((ROOT / "research" / "templates" / "program.template.json").read_text())
    validate_program(template)
    assert template["canonical_source"]["subject_invariant"] != "exactly_one_curated_woman"
    assert set(template["gpu_scheduler"]["resources"]) == {"gpu-a", "gpu-b"}


def test_stratum_program_keeps_its_specific_corpus_and_accelerator_contract() -> None:
    program = json.loads((ROOT / "research" / "program.json").read_text())
    source = program["canonical_source"]
    assert source["path"] == "/mnt/nas-ai-models/training-data/crawlr/approved"
    assert source["subject_invariant"] == "exactly_one_curated_woman"
    resources = program["gpu_scheduler"]["resources"]
    assert resources["4090"] == {
        "host_route": "local",
        "total_vram_gb": 24,
        "usable_vram_gb": 24,
        "notes": "Use the shared scheduler lifecycle before every GPU action.",
    }
    assert resources["strix"] == {
        "host_route": "ssh:max395",
        "total_vram_gb": 110,
        "usable_vram_gb": 100,
        "evergreen_reserved_vram_gb": 10,
        "notes": "The Crawlr labeling process is evergreen. Scheduler state, not a transient utilization snapshot, determines availability.",
    }
    assert program["policy_profile"] == "stratum-single-woman-v1"
    assert "known_failure_modes" in program["specialists"]["required_declaration_fields"]
    assert program["artifact_policy"] == {
        "approved_output_roots": ["/mnt/nas-ai-models/research"],
        "canonical_source_write_allowed": False,
    }
    assert program["research_tree"] == {
        "require_program_root": True,
        "require_parent_issue": True,
        "require_selection_rationale": True,
    }


def test_label_spec_is_unique_and_contains_hold_controls() -> None:
    labels = json.loads((ROOT / "research" / "labels.json").read_text())
    names = [label["name"] for label in labels]
    assert len(names) == len(set(names))
    assert {"research", "research:active", "research:hold", "research:postmortem"}.issubset(names)


def test_control_plane_documents_and_templates_exist() -> None:
    required = [
        ROOT / "AGENTS.md",
        ROOT / "PROJECT_STATUS.md",
        ROOT / "RESEARCH_CONTRACT.md",
        ROOT / "docs" / "EXPERIMENT_TREE.md",
        ROOT / "docs" / "EXPERIMENTS_AND_RESULTS.md",
        ROOT / "research" / "templates" / "program.template.json",
        ROOT / "research" / "templates" / "RESEARCH_CONTRACT.template.md",
        ROOT / "research" / "templates" / "research-arm-body.md",
        ROOT / ".github" / "ISSUE_TEMPLATE" / "research-arm.yml",
        ROOT / ".github" / "ISSUE_TEMPLATE" / "research-postmortem.yml",
        ROOT / ".github" / "ISSUE_TEMPLATE" / "research-harness-gap.yml",
        ROOT / ".github" / "workflows" / "test.yml",
    ]
    assert all(path.is_file() for path in required)
    forms = {path.name: path.read_text() for path in required[-4:-1]}
    assert "research-harness:" in forms["research-arm.yml"]
    assert all(key in forms["research-arm.yml"] for key in ("parent_issue", "selection_rationale", "surveyed_issue_numbers"))
    assert all(key in forms["research-postmortem.yml"] for key in ("parent_issue", "decision", "evidence_summary"))
    assert all(key in forms["research-harness-gap.yml"] for key in ("trigger", "risk", "decision"))


def test_program_keeps_compact_context_separate_from_legacy_t5() -> None:
    program = json.loads((ROOT / "research" / "program.json").read_text())
    representation = program["representation"]
    assert representation["compact_context_target_tokens"] > representation["legacy_text_encoder_max_tokens"]
    assert representation["compact_artifacts"] == {
        "structured": "context4k.json",
        "human_readable": "context4k.md",
        "provenance": "compression.json",
    }


def test_stratum_context_bundle_template_matches_the_project_contract() -> None:
    program = json.loads((ROOT / "research" / "program.json").read_text())
    bundle = json.loads((ROOT / "research" / "templates" / "context4k-bundle.json").read_text())
    validate_compression_bundle(bundle, program)
