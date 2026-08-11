"""Behavioral tests for the project-neutral autonomous research harness."""

from __future__ import annotations

import copy
import json

import pytest

from research_harness import (
    ContractError,
    validate_compression_bundle,
    validate_gpu_manifest,
    validate_program,
    validate_research_tree,
)

_GPU_LIFECYCLE = (
    "request",
    "poll_and_claim",
    "launch",
    "verify",
    "activate",
    "heartbeat",
    "release",
)


def program() -> dict:
    """Small valid program document shared by the contract tests."""
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
            "expanded_dossier_target_role": "aspiration",
            "expanded_dossier_min_tokens": 4_001,
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


def generic_program() -> dict:
    result = copy.deepcopy(program())
    result["canonical_source"]["subject_invariant"] = "declared_by_this_project"
    result["gpu_scheduler"]["resources"] = {
        "accelerator-a": {
            "host_route": "local",
            "total_vram_gb": 48,
            "usable_vram_gb": 48,
        },
        "accelerator-b": {
            "host_route": "ssh:compute-b",
            "total_vram_gb": 80,
            "usable_vram_gb": 72,
        },
    }
    return result


def program_issue(number: int = 1) -> dict:
    return {
        "number": number,
        "state": "OPEN",
        "labels": ["research"],
        "body": "<!-- research-harness: {\"kind\":\"program\"} -->",
    }


def arm_issue(
    number: int = 1,
    *,
    active: bool = True,
    failures: int = 0,
    parent_issue: int | None = None,
    selection_rationale: str | None = None,
    surveyed_issue_numbers: list[int] | None = None,
    postmortem_issue: int | None = None,
) -> dict:
    labels = ["research"]
    if active:
        labels.append("research:active")
    metadata: dict[str, object] = {
        "kind": "arm",
        "hypothesis": "A measured evidence bundle improves factual contextual representation.",
        "falsified_if": "The controlled evaluation does not improve against the baseline.",
        "pre_registered_gate": "PASS only after a valid controlled comparison and adversarial review.",
        "metric_version": "v1",
        "data_snapshot": "pilot-v1",
        "valid_non_improving_experiments": failures,
    }
    if parent_issue is not None:
        metadata["parent_issue"] = parent_issue
    if selection_rationale is not None:
        metadata["selection_rationale"] = selection_rationale
    if surveyed_issue_numbers is not None:
        metadata["surveyed_issue_numbers"] = surveyed_issue_numbers
    if postmortem_issue is not None:
        metadata["postmortem_issue"] = postmortem_issue
    return {
        "number": number,
        "state": "OPEN",
        "labels": labels,
        "body": f"<!-- research-harness: {json.dumps(metadata)} -->",
    }


def postmortem_issue(number: int, parent_issue: int) -> dict:
    return {
        "number": number,
        "state": "OPEN",
        "labels": ["research", "research:postmortem"],
        "body": (
            "<!-- research-harness: "
            + json.dumps(
                {
                    "kind": "postmortem",
                    "parent_issue": parent_issue,
                    "decision": "PARK",
                    "evidence_summary": "Three comparable trials did not improve the gate.",
                }
            )
            + " -->"
        ),
    }


def test_program_requires_canonical_source_and_subject_invariant() -> None:
    invalid = program()
    del invalid["canonical_source"]["subject_invariant"]

    with pytest.raises(ContractError, match="subject_invariant"):
        validate_program(invalid)


def test_generic_program_can_declare_its_own_subject_invariant_and_accelerators() -> None:
    validate_program(generic_program())


def test_program_requires_content_policy_and_autonomy_hold_boundary() -> None:
    missing_policy = program()
    del missing_policy["content_policy"]

    with pytest.raises(ContractError, match="content_policy"):
        validate_program(missing_policy)

    missing_hold_boundary = program()
    del missing_hold_boundary["autonomy"]

    with pytest.raises(ContractError, match="autonomy"):
        validate_program(missing_hold_boundary)


def test_program_rejects_external_image_execution_without_explicit_hold_policy() -> None:
    invalid = program()
    invalid["content_policy"]["model_execution"] = "external_allowed"

    with pytest.raises(ContractError, match="model_execution"):
        validate_program(invalid)

    invalid = program()
    invalid["content_policy"]["autonomous_external_image_model_allowed"] = True

    with pytest.raises(ContractError, match="autonomous_external_image_model_allowed"):
        validate_program(invalid)


def test_program_rejects_non_finite_token_budgets() -> None:
    invalid = program()
    invalid["representation"]["expanded_dossier_target_tokens"] = float("nan")

    with pytest.raises(ContractError, match="positive finite"):
        validate_program(invalid)


def test_tree_rejects_malformed_issue_state_with_contract_error() -> None:
    issue = arm_issue()
    issue["state"] = []

    with pytest.raises(ContractError, match="issue state"):
        validate_research_tree({"issues": [issue]}, program())


def test_gpu_manifest_rejects_request_above_declared_capacity() -> None:
    invalid = {
        "schema_version": 1,
        "job_id": "too-large",
        "target_gpu": "4090",
        "requested_vram_gb": 25,
        "maximum_duration": "2h",
        "approved_issue": 12,
        "manifest_state": "approved",
        "host_route": "local",
        "launcher_id": "registered-research-launcher",
        "output_root": "/mnt/nas-ai-models/research/too-large",
        "scheduler_lifecycle": list(_GPU_LIFECYCLE),
    }

    with pytest.raises(ContractError, match="exceeds usable_vram_gb"):
        validate_gpu_manifest(invalid, program())


def test_program_rejects_placeholder_preregistration_and_unsafe_autonomy_flags() -> None:
    invalid = program()
    invalid["autonomy"]["autonomous_merge_allowed"] = True

    with pytest.raises(ContractError, match="autonomous_merge_allowed"):
        validate_program(invalid)

    invalid = program()
    invalid["artifact_policy"]["canonical_source_write_allowed"] = True

    with pytest.raises(ContractError, match="canonical_source_write_allowed"):
        validate_program(invalid)


def test_program_rejects_context_that_silently_fits_legacy_encoder() -> None:
    invalid = program()
    invalid["representation"]["compact_context_target_tokens"] = 512
    invalid["representation"]["compact_context_min_tokens"] = 512

    with pytest.raises(ContractError, match="must exceed legacy_text_encoder_max_tokens"):
        validate_program(invalid)


def test_compression_rejects_dossier_below_the_structural_floor() -> None:
    """A dossier that does not exceed the 4K compact ceiling it compresses into
    is refused (structural floor), even though the 100K aspiration target is no
    longer a gate."""
    bundle = {
        "schema_version": 1,
        "image_id": "example-1",
        "expanded_dossier": {"token_count": 4_000, "evidence_ids": ["captioner:v1"]},
        "compact_context": {
            "token_count": 4_000,
            "claims": [{"text": "Supported.", "evidence_ids": ["captioner:v1"]}],
        },
        "artifacts": {
            "structured": "context4k.json",
            "human_readable": "context4k.md",
            "provenance": "compression.json",
        },
    }

    with pytest.raises(ContractError, match="below the structural minimum"):
        validate_compression_bundle(bundle, program())


def test_compression_accepts_honest_scale_dossier_below_100k_aspiration() -> None:
    """The reframe: an honest ~13.5K dossier (below the 100K aspiration target
    but above the structural floor) now VALIDATES — the aspiration target is
    metadata, not a pass gate. The claim->evidence honesty path is intact."""
    bundle = {
        "schema_version": 1,
        "image_id": "example-1",
        "expanded_dossier": {"token_count": 13_500, "evidence_ids": ["captioner:v1"]},
        "compact_context": {
            "token_count": 4_000,
            "claims": [{"text": "Supported.", "evidence_ids": ["captioner:v1"]}],
        },
        "artifacts": {
            "structured": "context4k.json",
            "human_readable": "context4k.md",
            "provenance": "compression.json",
        },
    }

    validate_compression_bundle(bundle, program())  # must not raise


def test_gpu_manifest_rejects_unapproved_output_root_and_unsafe_duration() -> None:
    manifest = {
        "schema_version": 1,
        "job_id": "unsafe-root",
        "target_gpu": "4090",
        "requested_vram_gb": 12,
        "maximum_duration": "forever",
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
        "output_root": "/mnt/nas-ai-models/training-data/crawlr/approved/run",
        "scheduler_lifecycle": list(_GPU_LIFECYCLE),
    }

    with pytest.raises(ContractError, match="maximum_duration"):
        validate_gpu_manifest(manifest, program())

    manifest["maximum_duration"] = "2h"
    with pytest.raises(ContractError, match="approved output root"):
        validate_gpu_manifest(manifest, program())


def test_program_rejects_an_unsupported_gpu_execution_mode() -> None:
    invalid = program()
    invalid["gpu_scheduler"]["execution_mode"] = "launch_without_review"

    with pytest.raises(ContractError, match="execution_mode"):
        validate_program(invalid)


def test_tree_requires_exactly_one_active_research_arm() -> None:
    snapshot = {"issues": [arm_issue(1), arm_issue(2)]}

    with pytest.raises(ContractError, match="exactly one active research arm"):
        validate_research_tree(snapshot, program())


def test_active_arm_requires_preregistration_before_selection() -> None:
    issue = arm_issue()
    metadata = json.loads(issue["body"].removeprefix("<!-- research-harness: ").removesuffix(" -->"))
    del metadata["pre_registered_gate"]
    issue["body"] = f"<!-- research-harness: {json.dumps(metadata)} -->"

    with pytest.raises(ContractError, match="pre_registered_gate"):
        validate_research_tree({"issues": [issue]}, program())


def test_active_arm_rejects_placeholder_preregistration_fields() -> None:
    issue = arm_issue()
    metadata = json.loads(issue["body"].removeprefix("<!-- research-harness: ").removesuffix(" -->"))
    metadata["hypothesis"] = "REPLACE"
    issue["body"] = f"<!-- research-harness: {json.dumps(metadata)} -->"

    with pytest.raises(ContractError, match="placeholder"):
        validate_research_tree({"issues": [issue]}, program())


def test_three_valid_non_improving_experiments_require_closed_arm_and_linked_postmortem() -> None:
    active = arm_issue(failures=3)

    with pytest.raises(ContractError, match="postmortem"):
        validate_research_tree({"issues": [active]}, program())

    postmortem = postmortem_issue(2, parent_issue=1)
    active_metadata = json.loads(active["body"].removeprefix("<!-- research-harness: ").removesuffix(" -->"))
    active_metadata["postmortem_issue"] = 2
    active["body"] = f"<!-- research-harness: {json.dumps(active_metadata)} -->"

    with pytest.raises(ContractError, match="must be closed"):
        validate_research_tree({"issues": [active, postmortem]}, program())

    active["state"] = "CLOSED"
    active["labels"].remove("research:active")
    successor = arm_issue(number=3)
    validate_research_tree({"issues": [active, postmortem, successor]}, program())


def test_strict_tree_requires_program_parent_and_full_tree_survey() -> None:
    strict = program()
    strict["research_tree"] = {
        "require_program_root": True,
        "require_parent_issue": True,
        "require_selection_rationale": True,
    }
    root = program_issue(1)
    active = arm_issue(
        2,
        parent_issue=1,
        selection_rationale="Highest expected information gain after surveying the tree.",
        surveyed_issue_numbers=[1, 2, 3],
    )
    proposal = arm_issue(3, active=False, parent_issue=1)
    proposal["labels"].append("research:proposal")
    validate_research_tree({"issues": [root, active, proposal]}, strict)

    active_without_survey = arm_issue(2, parent_issue=1)
    with pytest.raises(ContractError, match="selection_rationale"):
        validate_research_tree({"issues": [root, active_without_survey, proposal]}, strict)


def test_compression_bundle_requires_evidence_for_every_claim() -> None:
    bundle = {
        "schema_version": 1,
        "image_id": "example-1",
        "expanded_dossier": {
            "token_count": 100_000,
            "evidence_ids": ["pose2:v1", "captioner:v1"],
        },
        "compact_context": {
            "token_count": 4_000,
            "claims": [
                {"text": "The subject faces the camera.", "evidence_ids": ["pose2:v1"]},
                {"text": "Unsupported claim.", "evidence_ids": []},
            ],
        },
        "artifacts": {
            "structured": "context4k.json",
            "human_readable": "context4k.md",
            "provenance": "compression.json",
        },
    }

    with pytest.raises(ContractError, match="supporting evidence"):
        validate_compression_bundle(bundle, program())


def test_compression_bundle_accepts_provenance_preserving_context() -> None:
    bundle = {
        "schema_version": 1,
        "image_id": "example-1",
        "expanded_dossier": {
            "token_count": 100_000,
            "evidence_ids": ["pose2:v1", "captioner:v1"],
        },
        "compact_context": {
            "token_count": 4_000,
            "claims": [
                {"text": "The subject faces the camera.", "evidence_ids": ["pose2:v1"]},
                {"text": "Blue daylight is visible.", "evidence_ids": ["captioner:v1"]},
            ],
        },
        "artifacts": {
            "structured": "context4k.json",
            "human_readable": "context4k.md",
            "provenance": "compression.json",
        },
    }

    validate_compression_bundle(bundle, program())


def test_gpu_manifest_requires_scheduler_lifecycle_and_correct_strix_route() -> None:
    manifest = {
        "schema_version": 1,
        "job_id": "pilot-context-expansion-001",
        "target_gpu": "strix",
        "requested_vram_gb": 16,
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

    with pytest.raises(ContractError, match="ssh:max395"):
        validate_gpu_manifest(manifest, program())

    manifest["host_route"] = "ssh:max395"
    validate_gpu_manifest(manifest, program())


def test_gpu_manifest_rejects_an_unregistered_launcher() -> None:
    manifest = {
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
        "launcher_id": "arbitrary-shell-command",
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

    with pytest.raises(ContractError, match="registered launcher"):
        validate_gpu_manifest(manifest, program())


def test_valid_tree_can_have_open_research_branches_without_fifo_selection() -> None:
    active = arm_issue(1)
    proposed = arm_issue(2, active=False)
    proposed["labels"].append("research:proposal")
    snapshot = {"issues": [active, proposed]}

    validate_research_tree(snapshot, program())
