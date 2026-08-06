"""TDD coverage for the arm-36 round-trip surface (context4k evidence kind).

The round-trip claim-support audit compares captions generated FROM the
evidence-linked <=4K compact context (`context-raw-context4k`) against the
matched plain-4K summarization baseline (`context-raw-no-evidence`). The
aggregate machinery (`_derive_conditions_from_plan`) derives baseline/evidence
exactly: baseline = condition with `no-specialist-evidence-v1` whose id
contains "context"; evidence = the condition with a real specialist id.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from research_harness import validate_comparison_parity_plan
from research_harness.autonomous import _derive_conditions_from_plan
from research_harness.stage_b import (
    StageBGenerationSettings,
    StageBRunError,
    _render_condition,
    build_stage_b_plan,
    execute_stage_b,
    freeze_stage_b_plan,
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_source(path: Path, *, width: int = 80, height: int = 80) -> bytes:
    image = Image.new("RGB", (width, height), color=(31, 47, 59))
    image.save(path, format="JPEG")
    return path.read_bytes()


def _write_geometry(derived_item: Path) -> None:
    derived_item.mkdir(parents=True)
    pose = np.zeros((1, 308, 3), dtype=np.float32)
    pose[0, 5] = (20, 20, 0.95)
    pose[0, 6] = (60, 20, 0.95)
    pose[0, 9] = (25, 50, 0.95)
    pose[0, 10] = (55, 50, 0.95)
    pose[0, 41] = (70, 10, 0.95)
    pose[0, 62] = (10, 10, 0.95)
    pose[0, 69] = (40, 15, 0.95)
    np.save(derived_item / "pose2.npy", pose)

    seg = np.zeros((80, 80), dtype=np.uint8)
    seg[10:55, 15:65] = 22
    seg[10:30, 5:75] = 20
    np.save(derived_item / "seg2.npy", seg)

    # context4k binds normal2 as an evidence input (the dossier consumes it for
    # the lighting dimension), so the frozen fixture must provide it.
    normal = np.zeros((80, 80, 3), dtype=np.float16)
    normal[..., 2] = 1.0
    np.save(derived_item / "normal2.npy", normal)

    pointmap = np.zeros((80, 80, 3), dtype=np.float16)
    pointmap[..., 2] = 1.0
    np.save(derived_item / "pointmap.npy", pointmap)


def _candidate_manifest(source_root: Path, derived_root: Path, source_name: str, source_bytes: bytes) -> dict:
    manifest = {
        "schema_version": 1,
        "kind": "first500-coverage-balanced-candidate-manifest",
        "status": "PENDING_PRE_COMPUTE_NON_EXECUTING",
        "manifest_id": "fixture-first500-v1",
        "program_id": "stratum-contextual-specialist-research",
        "parent_issue": 4,
        "canonical_source_root": str(source_root),
        "derived_artifact_root": str(derived_root),
        "items": [
            {
                "image_id": "fixture",
                "source_relative_path": source_name,
                "source_sha256": _sha256(source_bytes),
                "source_dimensions": {"width": 80, "height": 80},
                "source_format": "JPEG",
                "source_byte_read_count": 1,
                "selection": {"aspect_bucket": "squareish", "rank_sha256": "a" * 64},
                "artifact_availability": {
                    "pose2.npy": True,
                    "seg2.npy": True,
                    "normal2.npy": True,
                    "pointmap.npy": True,
                    "caption.txt": True,
                },
                "artifact_readability_status": {
                    "pose2.npy": "readable",
                    "seg2.npy": "readable",
                    "normal2.npy": "readable",
                    "pointmap.npy": "readable",
                    "caption.txt": "readable",
                },
                "quality_status": {
                    "pose2_detection_count": 1,
                    "detector_disagreement": False,
                    "caption_semantics": "excluded",
                },
            }
        ],
    }
    manifest["manifest_fingerprint"] = _sha256(_canonical_json(manifest).encode("utf-8"))
    return manifest


def _program(source_root: Path, derived_root: Path, research_root: Path) -> dict:
    return {
        "schema_version": 1,
        "program_id": "stratum-contextual-specialist-research",
        "canonical_source": {
            "path": str(source_root),
            "derived_tree": str(derived_root),
            "subject_invariant": "exactly_one_curated_woman",
            "detector_disagreement": "quality_anomaly_not_semantic_content",
        },
        "content_policy": {
            "model_execution": "local_first",
            "autonomous_external_image_model_allowed": False,
            "reason": "Fixture local-only policy.",
            "external_model_requirement": "External execution is prohibited.",
        },
        "artifact_policy": {
            "approved_output_roots": [str(research_root)],
            "canonical_source_write_allowed": False,
        },
        "representation": {
            "expanded_dossier_target_tokens": 100000,
            "expanded_dossier_target_role": "aspiration",
            "expanded_dossier_min_tokens": 4001,
            "compact_context_target_tokens": 4000,
            "compact_context_min_tokens": 4000,
            "legacy_text_encoder_max_tokens": 512,
            "compact_artifacts": {
                "structured": "context4k.json",
                "human_readable": "context4k.md",
                "provenance": "compression.json",
            },
            "rule": "Fixture representation contract.",
        },
        "specialists": {
            "policy": "open_world",
            "required_declaration_fields": [
                "scope", "inputs", "output_semantics", "provenance",
                "abstention_policy", "known_failure_modes", "qualification_gate",
            ],
        },
        "research_tree": {
            "require_program_root": False,
            "require_parent_issue": False,
            "require_selection_rationale": False,
        },
        "gpu_scheduler": {
            "command": "/tmp/scheduler.py",
            "execution_mode": "observer_only",
            "max_job_duration_hours": 24,
            "scheduler_project": "fixture-research",
            "allowed_launchers": ["registered-research-launcher"],
            "resources": {"4090": {"host_route": "local", "total_vram_gb": 24, "usable_vram_gb": 24}},
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
    }


def _settings() -> StageBGenerationSettings:
    return StageBGenerationSettings(
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


def _fixture(tmp_path: Path) -> tuple[dict, dict, StageBGenerationSettings, Path]:
    source_root = tmp_path / "approved"
    derived_root = tmp_path / "stratum"
    research_root = tmp_path / "research"
    source_root.mkdir()
    derived_root.mkdir()
    research_root.mkdir()
    source_name = "fixture.jpg"
    source_bytes = _write_source(source_root / source_name)
    _write_geometry(derived_root / "fixture")
    return (
        _program(source_root, derived_root, research_root),
        _candidate_manifest(source_root, derived_root, source_name, source_bytes),
        _settings(),
        research_root,
    )


def test_roundtrip_plan_builds_with_context4k_evidence_kind(tmp_path: Path) -> None:
    program, candidate, settings, _ = _fixture(tmp_path)
    plan = build_stage_b_plan(program, candidate, settings, evidence_kind="context4k")

    assert plan["comparison_plan_id"] == "stage-b-roundtrip-context4k-v1"
    assert [c["id"] for c in plan["conditions"]] == [
        "legacy-bucketed-no-evidence",
        "legacy-raw-no-evidence",
        "context-raw-no-evidence",
        "context-raw-context4k",
    ]
    evidence = plan["conditions"][-1]["evidence"]
    assert evidence["id"] == "in-memory-context4k-compact-v1"
    assert any(
        s.get("scope", "").startswith("Per-item evidence-linked compact context")
        for s in evidence["specialists"]
    )
    assert plan["contrasts"][-1] == {
        "id": "evidence-only",
        "baseline_condition": "context-raw-no-evidence",
        "variant_condition": "context-raw-context4k",
        "changed_axes": ["evidence"],
    }
    validate_comparison_parity_plan(plan, program)


def test_roundtrip_frozen_plan_binds_pose2_seg2_normal2(tmp_path: Path) -> None:
    program, candidate, settings, _ = _fixture(tmp_path)
    plan = freeze_stage_b_plan(program, candidate, settings, evidence_kind="context4k")

    assert plan["comparison_plan_id"] == "stage-b-roundtrip-context4k-v1"
    assert plan["evidence_input_artifact_sha256"]["fixture"] == {
        "pose2.npy": _sha256((Path(program["canonical_source"]["derived_tree"]) / "fixture" / "pose2.npy").read_bytes()),
        "seg2.npy": _sha256((Path(program["canonical_source"]["derived_tree"]) / "fixture" / "seg2.npy").read_bytes()),
        "normal2.npy": _sha256((Path(program["canonical_source"]["derived_tree"]) / "fixture" / "normal2.npy").read_bytes()),
    }


def test_render_context4k_condition_emits_evidence_linked_compact(tmp_path: Path) -> None:
    program, candidate, settings, research_root = _fixture(tmp_path)
    frozen = freeze_stage_b_plan(program, candidate, settings, evidence_kind="context4k")

    # Render via the frozen-plan execution path (validates the rebuild mapping too).
    captured: list[dict] = []

    def generate(image: Image.Image, prompt: str, generation: StageBGenerationSettings) -> str:
        captured.append({"prompt": prompt})
        return f"caption-{len(captured)}"

    result = execute_stage_b(
        program, candidate, settings,
        output_root=research_root / "run",
        expected_plan=frozen,
        generate=generate,
    )
    assert result["record_count"] == 4

    records = [json.loads(line) for line in (research_root / "run" / "records.jsonl").read_text().splitlines()]
    roundtrip_record = next(r for r in records if r["condition_id"] == "context-raw-context4k")
    prompt = roundtrip_record["prompt"]["rendered_text"]
    assert "DECLARED SPECIALIST EVIDENCE:" in prompt
    # Evidence-linked compact: claims carry their evidence ids, no absolute pixels.
    payload = roundtrip_record["evidence_payload"]
    assert payload["compact_claim_count"] >= 1
    assert payload["dossier_evidence_ids"] == [
        "body-type-proportions:v1",
        "clothing-apparel:v1",
        "hair:v1",
        "skin-color-tone:v1",
        "lighting:v1",
        "relational-determinations:v1",
    ]
    for key in ("between_shoulders", "between_hips", "torso_length"):
        assert key not in prompt
    assert roundtrip_record["selected_derived_reads"] == ["pose2.npy", "seg2.npy", "normal2.npy"]

    # Baseline condition stays evidence-free (plain-4K summarization baseline).
    baseline_record = next(r for r in records if r["condition_id"] == "context-raw-no-evidence")
    assert "no specialist evidence declared" in baseline_record["prompt"]["rendered_text"]


def test_aggregate_derivation_picks_roundtrip_conditions_from_plan(tmp_path: Path) -> None:
    program, candidate, settings, research_root = _fixture(tmp_path)
    frozen = freeze_stage_b_plan(program, candidate, settings, evidence_kind="context4k")

    # Publish the plan beside a review root exactly as the run does.
    review_root = research_root / "run-review"
    review_root.mkdir(parents=True)
    (review_root / "stage-b-plan.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    derived = _derive_conditions_from_plan(str(review_root))
    assert derived is not None
    baseline, evidence = derived
    assert baseline == "context-raw-no-evidence"
    assert evidence == "context-raw-context4k"


def test_roundtrip_rejects_unknown_evidence_kind_still(tmp_path: Path) -> None:
    program, candidate, settings, _ = _fixture(tmp_path)
    with pytest.raises(StageBRunError, match="unsupported Stage-B evidence_kind"):
        build_stage_b_plan(program, candidate, settings, evidence_kind="not-a-kind")
