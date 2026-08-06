"""TDD coverage for the arm-47 vlm-dense-description surface.

The vlm-dense plan is FIVE conditions: the three neutral controls, the
deterministic dossier compact condition (`context-raw-context4k`) as the matched
baseline, and the variant (`context-raw-vlm-dense`) that blends the deterministic
compact with the pre-generated VLM dense block (pinned byte-for-byte by
`vlm_blocks_sha256`). The verdict isolates the VLM marginal via EXPLICIT
baseline/evidence conditions in the tick:
  --baseline-condition context-raw-context4k --evidence-condition context-raw-vlm-dense
(plain plan-derived derivation would pair no-evidence -> context4k, which is the
arm-36 result, not the VLM marginal.)
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from research_harness import validate_comparison_parity_plan
from research_harness.autonomous import (
    _derive_conditions_from_plan,
    aggregate_claim_support,
    run_tick,
)
from research_harness.stage_b import (
    StageBGenerationSettings,
    StageBRunError,
    _sha256,
    _validate_frozen_execution_plan,
    build_stage_b_plan,
    execute_stage_b,
    freeze_stage_b_plan,
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "source_dimensions": {"width": 80, "height": 80},
                "source_format": "JPEG",
                "source_byte_read_count": 1,
                "selection": {"aspect_bucket": "squareish", "rank_sha256": "a" * 64},
                "artifact_availability": {
                    "pose2.npy": True, "seg2.npy": True, "normal2.npy": True,
                    "pointmap.npy": True, "caption.txt": True,
                },
                "artifact_readability_status": {
                    "pose2.npy": "readable", "seg2.npy": "readable", "normal2.npy": "readable",
                    "pointmap.npy": "readable", "caption.txt": "readable",
                },
                "quality_status": {
                    "pose2_detection_count": 1,
                    "detector_disagreement": False,
                    "caption_semantics": "excluded",
                },
            }
        ],
    }
    manifest["manifest_fingerprint"] = _canonical_json(manifest)
    manifest["manifest_fingerprint"] = hashlib.sha256(manifest["manifest_fingerprint"].encode("utf-8")).hexdigest()
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


def _fake_block(image_id: str = "fixture") -> str:
    return (
        "## SUBJECT\n- [OBSERVED] single adult woman, standing.\n"
        "## GARMENTS\n- [OBSERVED] dark upper garment covering most of the torso.\n"
        "## HAIR\n- [ABSTAIN] hair details not clearly visible.\n"
        "## SKIN\n- [OBSERVED] exposed skin tone light.\n"
        "## POSE\n- [OBSERVED] arms relaxed at sides.\n"
        "## SETTING\n- [INFERRED] shallow depth of field behind the subject.\n"
    )


def test_vlm_dense_plan_builds_five_conditions(tmp_path: Path) -> None:
    program, candidate, settings, _ = _fixture(tmp_path)
    block = _fake_block()
    plan = build_stage_b_plan(program, candidate, settings, evidence_kind="vlm-dense",
                              vlm_blocks_sha256={"fixture": _sha256(block.encode("utf-8"))})

    assert plan["comparison_plan_id"] == "stage-b-vlm-dense-v1"
    assert [c["id"] for c in plan["conditions"]] == [
        "legacy-bucketed-no-evidence",
        "legacy-raw-no-evidence",
        "context-raw-no-evidence",
        "context-raw-context4k",
        "context-raw-vlm-dense",
    ]
    assert plan["vlm_blocks_sha256"]["fixture"] == _sha256(block.encode("utf-8"))
    assert plan["vlm_model_digest"] == "ff2e46876908"
    # The VLM marginal contrast is the one the tick must use explicitly.
    assert {
        "id": "vlm-marginal",
        "baseline_condition": "context-raw-context4k",
        "variant_condition": "context-raw-vlm-dense",
        "changed_axes": ["evidence"],
    } in plan["contrasts"]
    validate_comparison_parity_plan(plan, program)


def test_vlm_dense_plan_requires_block_hashes(tmp_path: Path) -> None:
    program, candidate, settings, _ = _fixture(tmp_path)
    with pytest.raises(StageBRunError, match="vlm-dense evidence_kind requires vlm_blocks_sha256"):
        build_stage_b_plan(program, candidate, settings, evidence_kind="vlm-dense")


def test_vlm_dense_plan_rejects_partial_hash_coverage(tmp_path: Path) -> None:
    program, candidate, settings, _ = _fixture(tmp_path)
    with pytest.raises(StageBRunError, match="must cover exactly the frozen items"):
        build_stage_b_plan(program, candidate, settings, evidence_kind="vlm-dense",
                           vlm_blocks_sha256={"other": "a" * 64})


def test_vlm_dense_frozen_plan_replays_via_validate(tmp_path: Path) -> None:
    program, candidate, settings, _ = _fixture(tmp_path)
    block = _fake_block()
    frozen = freeze_stage_b_plan(program, candidate, settings, evidence_kind="vlm-dense",
                                 vlm_blocks_sha256={"fixture": _sha256(block.encode("utf-8"))})
    # The frozen-execution rebuild must reproduce the plan exactly (vlm-dense
    # rebuild mapping), else the launcher refuses to run.
    plan, hashes = _validate_frozen_execution_plan(frozen, program, candidate, settings)
    assert plan["comparison_plan_id"] == "stage-b-vlm-dense-v1"
    assert set(hashes["fixture"]) == {"pose2.npy", "seg2.npy", "normal2.npy"}


def test_vlm_dense_execution_blends_compact_with_block(tmp_path: Path) -> None:
    program, candidate, settings, research_root = _fixture(tmp_path)
    block = _fake_block()
    block_sha = _sha256(block.encode("utf-8"))
    frozen = freeze_stage_b_plan(program, candidate, settings, evidence_kind="vlm-dense",
                                 vlm_blocks_sha256={"fixture": block_sha})
    vlm_blocks = {"fixture": {"block_text": block, "block_sha256": block_sha}}

    captured: list[dict] = []

    def generate(image: Image.Image, prompt: str, generation: StageBGenerationSettings) -> str:
        captured.append({"prompt": prompt})
        return f"caption-{len(captured)}"

    result = execute_stage_b(program, candidate, settings, output_root=research_root / "run",
                             expected_plan=frozen, generate=generate, vlm_blocks=vlm_blocks)
    assert result["record_count"] == 5

    records = [json.loads(line) for line in (research_root / "run" / "records.jsonl").read_text().splitlines()]
    vlm_record = next(r for r in records if r["condition_id"] == "context-raw-vlm-dense")
    prompt = vlm_record["prompt"]["rendered_text"]
    assert "VLM DENSE DESCRIPTION" in prompt
    assert "# GARMENTS" in prompt  # the frozen block text is rendered as evidence
    assert "[OBSERVED] dark upper garment covering most of the torso" in prompt
    assert "compact" not in prompt or "DECLARED SPECIALIST EVIDENCE:" in prompt
    payload = vlm_record["evidence_payload"]
    assert payload["vlm_block_present"] is True
    assert payload["vlm_block_sha256"] == block_sha
    assert payload["compact_claim_count"] >= 1
    # The deterministic dossier compact remains its own condition (matched baseline).
    base_record = next(r for r in records if r["condition_id"] == "context-raw-context4k")
    assert "VLM DENSE DESCRIPTION" not in base_record["prompt"]["rendered_text"]


def test_vlm_dense_execution_fails_closed_on_block_drift(tmp_path: Path) -> None:
    program, candidate, settings, research_root = _fixture(tmp_path)
    block = _fake_block()
    frozen = freeze_stage_b_plan(program, candidate, settings, evidence_kind="vlm-dense",
                                 vlm_blocks_sha256={"fixture": _sha256(block.encode("utf-8"))})
    wrong = {"fixture": {"block_text": block + "\n- tampered", "block_sha256": "x" * 64}}

    def generate(image: Image.Image, prompt: str, generation: StageBGenerationSettings) -> str:
        return "caption"

    with pytest.raises(StageBRunError, match="SHA-256 drifted"):
        execute_stage_b(program, candidate, settings, output_root=research_root / "run-fail",
                        expected_plan=frozen, generate=generate, vlm_blocks=wrong)


def test_vlm_dense_plain_derivation_picks_last_real_evidence_condition(tmp_path: Path) -> None:
    """The plan-derived evidence condition is the LAST real-evidence one.

    Because the vlm-dense plan carries TWO real-evidence conditions
    (context-raw-context4k and context-raw-vlm-dense) and the derivation loop
    overwrites `evidence` on every real-evidence condition, plain derivation
    yields context-raw-vlm-dense vs context-raw-no-evidence — the COMBINED
    record vs the plain baseline (a re-measurement of arm #36's direction).
    To isolate the VLM MARGINAL (dossier compact -> compact + VLM block) the
    tick MUST pass explicit --baseline-condition/--evidence-condition; this
    test documents that requirement.
    """
    program, candidate, settings, research_root = _fixture(tmp_path)
    block = _fake_block()
    frozen = freeze_stage_b_plan(program, candidate, settings, evidence_kind="vlm-dense",
                                 vlm_blocks_sha256={"fixture": _sha256(block.encode("utf-8"))})
    review_root = research_root / "run-review"
    review_root.mkdir(parents=True)
    (review_root / "stage-b-plan.json").write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")

    baseline, evidence = _derive_conditions_from_plan(str(review_root))
    assert baseline == "context-raw-no-evidence"
    assert evidence == "context-raw-vlm-dense"  # last real-evidence condition; not the marginal


def _write_synthetic_review(review_root: Path, plan: dict) -> None:
    review_root.mkdir(parents=True, exist_ok=True)
    (review_root / "stage-b-plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    rows = []
    n_items = 6
    for i in range(n_items):
        image_id = f"item{i}"
        rows.append({"image_id": image_id, "condition_id": "context-raw-context4k",
                     "model": "reviewer-qwen3vl-32b", "supported": ["c1"],
                     "unsupported": ["u1", "u2", "u3"], "omissions": [], "contradictions": [], "abstentions": []})
        rows.append({"image_id": image_id, "condition_id": "context-raw-vlm-dense",
                     "model": "reviewer-qwen3vl-32b", "supported": ["a", "b", "c", "d"],
                     "unsupported": ["u1", "u2"], "omissions": [], "contradictions": [], "abstentions": []})
    (review_root / "reviews.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")


def test_aggregate_explicit_conditions_isolate_vlm_marginal(tmp_path: Path) -> None:
    program, candidate, settings, research_root = _fixture(tmp_path)
    block = _fake_block()
    plan = build_stage_b_plan(program, candidate, settings, evidence_kind="vlm-dense",
                              vlm_blocks_sha256={"fixture": _sha256(block.encode("utf-8"))})
    review_root = research_root / "run-review"
    _write_synthetic_review(review_root, plan)

    agg = aggregate_claim_support(str(review_root), baseline_condition="context-raw-context4k",
                                  evidence_condition="context-raw-vlm-dense")
    # vlm-marginal: 6 items, baseline 1 supported/3 unsupported each -> 6/18;
    # variant 4 supported/2 unsupported each -> 24/12.
    assert agg["baseline_supported"] == 6
    assert agg["evidence_supported"] == 24
    assert agg["paired_items"] == 6
    assert agg["positive_delta_count"] == 6
    assert agg["sign_test_p_supported"] <= 0.05


def _live_registry(tmp_path: Path) -> dict:
    """A registry with vlm-dense-description as the sole active arm."""
    import json as _json

    registry_path = Path(__file__).resolve().parent.parent / "research/dimensions/evidence-dimension-registry-v1.json"
    registry = _json.loads(registry_path.read_text(encoding="utf-8"))
    registry = copy.deepcopy(registry)
    for dim in registry["dimensions"]:
        if dim["id"] != "vlm-dense-description":
            dim["state"] = "validated" if dim["state"] == "active" else dim["state"]
        else:
            dim["state"] = "active"
    return registry


def test_tick_with_explicit_conditions_returns_better(tmp_path: Path) -> None:
    program, candidate, settings, research_root = _fixture(tmp_path)
    block = _fake_block()
    plan = build_stage_b_plan(program, candidate, settings, evidence_kind="vlm-dense",
                              vlm_blocks_sha256={"fixture": _sha256(block.encode("utf-8"))})
    review_root = research_root / "run-review"
    _write_synthetic_review(review_root, plan)

    registry = _live_registry(tmp_path)
    outcome = run_tick(
        registry,
        review_dir=str(review_root),
        baseline_condition="context-raw-context4k",
        evidence_condition="context-raw-vlm-dense",
    )
    assert outcome["verdict"]["verdict"] == "BETTER"
    assert outcome["advanced_arm"] == "vlm-dense-description"
    assert outcome["verdict"]["support_ratio_base"] == 0.25
    assert outcome["verdict"]["support_ratio_variant"] == 0.6667
