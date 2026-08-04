"""Fail-closed pre-compute contracts for controlled caption/context comparisons."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from research_harness import ContractError, validate_comparison_parity_plan

ROOT = Path(__file__).resolve().parents[1]


def _sha(char: str) -> str:
    return char * 64


def no_specialist_evidence() -> dict:
    return {
        "kind": "none",
        "id": "no-specialist-evidence-v1",
        "fingerprint": _sha("d"),
    }


def inline_geometry_evidence() -> dict:
    return {
        "kind": "specialist_bundle",
        "id": "geometry-v1",
        "fingerprint": _sha("0"),
        "specialists": [
            {
                "id": "geometry-determinations-v1",
                "scope": "Pose and segmentation-derived relational geometry only.",
                "inputs": "Frozen pose2 and seg2 artifacts for the selected input view.",
                "output_semantics": "Provenance-bearing measurements and relations, not caption facts.",
                "provenance": "Synthetic fixture declaration; no specialist model execution.",
                "abstention_policy": "Abstain when required artifacts are missing or contradictory.",
                "qualification_gate": "Must pass the pre-registered controlled comparison gate.",
            }
        ],
    }


def program() -> dict:
    return json.loads((ROOT / "research" / "program.json").read_text())


def parity_plan() -> dict:
    fixed_aggregator = {
        "model_id": "local-captioner-fixture-v1",
        "provenance": "synthetic fixture; no model execution",
        "generation_fingerprint": _sha("a"),
        "local_only": True,
    }

    def condition(
        condition_id: str,
        *,
        view_id: str,
        view_hash: str,
        prompt_id: str,
        prompt_hash: str,
        evidence: dict,
    ) -> dict:
        return {
            "id": condition_id,
            "pilot_manifest_id": "synthetic-parity-pilot-v1",
            "input_view": {"id": view_id, "fingerprint": view_hash},
            "prompt": {"id": prompt_id, "fingerprint": prompt_hash},
            "evidence": copy.deepcopy(evidence),
            "aggregator": copy.deepcopy(fixed_aggregator),
        }

    return {
        "schema_version": 1,
        "kind": "comparison-parity-plan",
        "program_id": "stratum-contextual-specialist-research",
        "status": "PENDING",
        "parent_issue": 4,
        "hypothesis": "Changing one declared condition axis at a time makes supported and unsupported claims attributable.",
        "falsified_if": "The plan cannot preserve a fixed pilot and fixed generation settings while isolating view, prompt, and evidence changes.",
        "metric_version": "caption-context-parity-fixture-v1",
        "pilot_manifest": {
            "id": "synthetic-parity-pilot-v1",
            "source_root": "/mnt/nas-ai-models/training-data/crawlr/approved",
            "frozen": True,
            "selection_rationale": "Synthetic fixture validates the pre-compute contract without selecting or reading a corpus image.",
            "coverage_notes": "No corpus coverage is claimed; the fixture only exercises hash, provenance, and paired-condition guards.",
            "items": [
                {
                    "image_id": "fixture-image-001",
                    "source_relative_path": "fixture-image-001.webp",
                    "source_sha256": _sha("1"),
                    "artifact_availability": {"geometry": True, "semantic_caption": False},
                },
                {
                    "image_id": "fixture-image-002",
                    "source_relative_path": "fixture-image-002.jpg",
                    "source_sha256": _sha("2"),
                    "artifact_availability": {"geometry": False, "semantic_caption": True},
                },
            ],
        },
        "conditions": [
            condition(
                "legacy-bucketed",
                view_id="bucketed",
                view_hash=_sha("b"),
                prompt_id="legacy",
                prompt_hash=_sha("c"),
                evidence=no_specialist_evidence(),
            ),
            condition(
                "legacy-raw",
                view_id="raw",
                view_hash=_sha("e"),
                prompt_id="legacy",
                prompt_hash=_sha("c"),
                evidence=no_specialist_evidence(),
            ),
            condition(
                "context-raw-no-evidence",
                view_id="raw",
                view_hash=_sha("e"),
                prompt_id="context",
                prompt_hash=_sha("f"),
                evidence=no_specialist_evidence(),
            ),
            condition(
                "context-raw-geometry",
                view_id="raw",
                view_hash=_sha("e"),
                prompt_id="context",
                prompt_hash=_sha("f"),
                evidence=inline_geometry_evidence(),
            ),
        ],
        "contrasts": [
            {
                "id": "view-only",
                "baseline_condition": "legacy-bucketed",
                "variant_condition": "legacy-raw",
                "changed_axes": ["input_view"],
            },
            {
                "id": "prompt-only",
                "baseline_condition": "legacy-raw",
                "variant_condition": "context-raw-no-evidence",
                "changed_axes": ["prompt"],
            },
            {
                "id": "evidence-only",
                "baseline_condition": "context-raw-no-evidence",
                "variant_condition": "context-raw-geometry",
                "changed_axes": ["evidence"],
            },
        ],
        "review_protocol": {
            "human_review_required": True,
            "sequence": [
                "selected_input_view",
                "provenance_evidence",
                "candidate_output_or_context",
                "decision_rubric",
            ],
            "fields": [
                "supported_claims",
                "unsupported_claims",
                "omissions",
                "contradictions",
                "abstentions",
            ],
            "detector_disagreement_handling": "quality_anomaly_not_caption_content",
        },
        "metric_self_audit": {
            "before_comparative_inference": True,
            "known_case_item_id": "fixture-image-001",
            "null_output_id": "empty-caption-null-v1",
            "evaluator_version": "claim-support-rubric-fixture-v1",
        },
        "adversarial_review": {
            "planned": True,
            "checks": [
                "metric_definition_stable",
                "fresh_process_or_second_review",
                "edge_case_inspection",
            ],
        },
        "representation_boundary": {
            "legacy_text_encoder_max_tokens": 512,
            "compact_context_routing": "out_of_scope",
            "no_silent_legacy_routing": True,
        },
    }


def test_comparison_parity_plan_accepts_controlled_synthetic_fixture() -> None:
    validate_comparison_parity_plan(parity_plan(), program())


def test_comparison_parity_plan_requires_frozen_hashed_canonical_pilot() -> None:
    invalid = parity_plan()
    invalid["pilot_manifest"]["source_root"] = "/tmp/not-canonical"

    with pytest.raises(ContractError, match="source_root"):
        validate_comparison_parity_plan(invalid, program())

    invalid = parity_plan()
    invalid["pilot_manifest"]["items"][0]["source_sha256"] = "not-a-sha"

    with pytest.raises(ContractError, match="source_sha256"):
        validate_comparison_parity_plan(invalid, program())


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../outside.webp",
        "/tmp/outside.webp",
        "nested/../../outside.webp",
        "nested/./outside.webp",
        r"nested\outside.webp",
    ],
)
def test_comparison_parity_plan_rejects_escaped_canonical_relative_paths(
    unsafe_path: str,
) -> None:
    invalid = parity_plan()
    invalid["pilot_manifest"]["items"][0]["source_relative_path"] = unsafe_path

    with pytest.raises(ContractError, match="source_relative_path"):
        validate_comparison_parity_plan(invalid, program())


def test_comparison_parity_plan_accepts_explicit_no_specialist_evidence() -> None:
    plan = parity_plan()

    validate_comparison_parity_plan(plan, program())


def test_comparison_parity_plan_rejects_opaque_non_null_evidence_bundle() -> None:
    invalid = parity_plan()
    invalid["conditions"][3]["evidence"] = {
        "kind": "specialist_bundle",
        "id": "opaque-unproven-specialist-bundle",
        "fingerprint": _sha("9"),
    }

    with pytest.raises(ContractError, match="evidence.specialists"):
        validate_comparison_parity_plan(invalid, program())


def test_comparison_parity_plan_requires_complete_inline_specialist_declarations() -> None:
    invalid = parity_plan()
    del invalid["conditions"][3]["evidence"]["specialists"][0]["abstention_policy"]

    with pytest.raises(ContractError, match="specialist.abstention_policy"):
        validate_comparison_parity_plan(invalid, program())


def test_comparison_parity_plan_rejects_non_explicit_evidence_kind() -> None:
    invalid = parity_plan()
    del invalid["conditions"][3]["evidence"]["kind"]

    with pytest.raises(ContractError, match="evidence.kind"):
        validate_comparison_parity_plan(invalid, program())


def test_comparison_parity_plan_rejects_declarations_on_no_evidence_baseline() -> None:
    invalid = parity_plan()
    invalid["conditions"][0]["evidence"]["specialists"] = [
        copy.deepcopy(inline_geometry_evidence()["specialists"][0])
    ]

    with pytest.raises(ContractError, match="evidence.specialists"):
        validate_comparison_parity_plan(invalid, program())


def test_comparison_parity_plan_rejects_duplicate_inline_specialist_ids() -> None:
    invalid = parity_plan()
    invalid["conditions"][3]["evidence"]["specialists"].append(
        copy.deepcopy(invalid["conditions"][3]["evidence"]["specialists"][0])
    )

    with pytest.raises(ContractError, match="specialist.id"):
        validate_comparison_parity_plan(invalid, program())


def test_comparison_parity_plan_requires_one_axis_contrasts_for_all_registered_axes() -> None:
    invalid = parity_plan()
    invalid["contrasts"][0]["changed_axes"] = ["input_view", "prompt"]

    with pytest.raises(ContractError, match="exactly one"):
        validate_comparison_parity_plan(invalid, program())

    invalid = parity_plan()
    invalid["contrasts"] = invalid["contrasts"][:2]

    with pytest.raises(ContractError, match="must cover"):
        validate_comparison_parity_plan(invalid, program())


def test_comparison_parity_plan_rejects_hidden_noncontrast_and_aggregator_changes() -> None:
    invalid = parity_plan()
    invalid["conditions"][1]["evidence"]["id"] = "geometry-v1"
    invalid["conditions"][1]["evidence"]["fingerprint"] = _sha("0")

    with pytest.raises(ContractError, match="non-contrast axis"):
        validate_comparison_parity_plan(invalid, program())

    invalid = parity_plan()
    invalid["conditions"][1]["aggregator"]["generation_fingerprint"] = _sha("9")

    with pytest.raises(ContractError, match="aggregator"):
        validate_comparison_parity_plan(invalid, program())


def test_comparison_parity_plan_requires_rubric_null_audit_and_adversarial_plan() -> None:
    invalid = parity_plan()
    invalid["review_protocol"]["detector_disagreement_handling"] = "caption_semantic"

    with pytest.raises(ContractError, match="detector_disagreement_handling"):
        validate_comparison_parity_plan(invalid, program())

    invalid = parity_plan()
    invalid["metric_self_audit"]["known_case_item_id"] = "missing-item"

    with pytest.raises(ContractError, match="known_case_item_id"):
        validate_comparison_parity_plan(invalid, program())

    invalid = parity_plan()
    invalid["adversarial_review"]["checks"] = ["metric_definition_stable"]

    with pytest.raises(ContractError, match="adversarial_review.checks"):
        validate_comparison_parity_plan(invalid, program())


def test_comparison_parity_plan_keeps_compact_context_out_of_legacy_encoder_path() -> None:
    invalid = parity_plan()
    invalid["representation_boundary"]["legacy_text_encoder_max_tokens"] = 4_000

    with pytest.raises(ContractError, match="legacy_text_encoder_max_tokens"):
        validate_comparison_parity_plan(invalid, program())

    invalid = parity_plan()
    invalid["representation_boundary"]["compact_context_routing"] = "legacy_t52"

    with pytest.raises(ContractError, match="compact_context_routing"):
        validate_comparison_parity_plan(invalid, program())


def test_module_cli_validates_comparison_parity_plan(tmp_path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(parity_plan()))
    program_path = ROOT / "research" / "program.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "research_harness",
            "validate-comparison-plan",
            str(program_path),
            str(plan_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "valid"


def test_comparison_template_is_explicitly_non_validating_until_filled() -> None:
    template_path = ROOT / "research" / "templates" / "comparison-parity-plan.template.json"
    template = json.loads(template_path.read_text())

    assert template["kind"] == "comparison-parity-plan"
    assert template["template_status"] == "fill_before_validation"
    assert "without-dotdot" in template["pilot_manifest"]["items"][0]["source_relative_path"]
    evidence = template["conditions"][0]["evidence"]
    assert evidence["kind"] == "specialist_bundle"
    assert {
        "id",
        "scope",
        "inputs",
        "output_semantics",
        "provenance",
        "abstention_policy",
        "qualification_gate",
    }.issubset(evidence["specialists"][0])
