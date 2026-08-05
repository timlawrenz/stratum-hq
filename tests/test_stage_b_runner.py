"""TDD coverage for the noncanonical Stage-B caption-comparison runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from research_harness import validate_comparison_parity_plan
from research_harness.stage_b import (
    StageBGenerationSettings,
    StageBRunError,
    _serialize_proportions,
    build_stage_b_plan,
    execute_stage_b,
    freeze_stage_b_plan,
    main,
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
    # shoulders, hips, neck, and wrists are sufficient for the deterministic fixture.
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
    seg[10:30, 5:75] = 11
    seg[10:30, 5:75] = 20
    np.save(derived_item / "seg2.npy", seg)

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
                    "pointmap.npy": True,
                    "caption.txt": True,
                },
                "artifact_readability_status": {
                    "pose2.npy": "readable",
                    "seg2.npy": "readable",
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
            "expanded_dossier_min_tokens": 100000,
            "compact_context_target_tokens": 4000,
            "compact_context_min_tokens": 4000,
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
        model_digest="a" * 64,
        temperature=0.0,
        seed=20260804,
        num_predict=384,
        top_k=1,
        top_p=1.0,
        context_window=4096,
        timeout_seconds=120,
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


def test_stage_b_plan_is_a_valid_four_condition_one_axis_contract(tmp_path: Path) -> None:
    program, candidate, settings, _ = _fixture(tmp_path)

    plan = freeze_stage_b_plan(program, candidate, settings)

    validate_comparison_parity_plan(plan, program)
    assert plan["evidence_input_artifact_sha256"]["fixture"] == {
        "pose2.npy": _sha256((Path(program["canonical_source"]["derived_tree"]) / "fixture" / "pose2.npy").read_bytes()),
        "seg2.npy": _sha256((Path(program["canonical_source"]["derived_tree"]) / "fixture" / "seg2.npy").read_bytes()),
    }
    assert [contrast["changed_axes"] for contrast in plan["contrasts"]] == [
        ["input_view"],
        ["prompt"],
        ["evidence"],
    ]
    assert {condition["aggregator"]["generation_fingerprint"] for condition in plan["conditions"]} == {
        settings.fingerprint
    }
    assert settings.request_options()["num_ctx"] == 4096


def test_stage_b_rejects_protected_output_before_generator_or_source_read(tmp_path: Path) -> None:
    program, candidate, settings, _ = _fixture(tmp_path)
    calls: list[object] = []

    with pytest.raises(StageBRunError, match="protected corpus root"):
        execute_stage_b(
            program,
            candidate,
            settings,
            output_root=Path(program["canonical_source"]["derived_tree"]) / "forbidden-run",
            generate=lambda *_args, **_kwargs: calls.append(True) or "unexpected",
        )

    assert calls == []


def test_stage_b_rejects_source_hash_drift_before_generator(tmp_path: Path) -> None:
    program, candidate, settings, research_root = _fixture(tmp_path)
    source = Path(program["canonical_source"]["path"]) / "fixture.jpg"
    source.write_bytes(b"tampered")
    calls: list[object] = []

    with pytest.raises(StageBRunError, match="source SHA-256 drift"):
        execute_stage_b(
            program,
            candidate,
            settings,
            output_root=research_root / "run",
            generate=lambda *_args, **_kwargs: calls.append(True) or "unexpected",
        )

    assert calls == []


def test_stage_b_writes_only_noncanonical_outputs_and_keeps_axes_separate(tmp_path: Path) -> None:
    program, candidate, settings, research_root = _fixture(tmp_path)
    source_root = Path(program["canonical_source"]["path"])
    derived_root = Path(program["canonical_source"]["derived_tree"])
    source_before = (source_root / "fixture.jpg").read_bytes()
    derived_before = {path.relative_to(derived_root): path.read_bytes() for path in derived_root.rglob("*") if path.is_file()}
    captured: list[dict] = []

    def generate(image: Image.Image, prompt: str, generation: StageBGenerationSettings) -> str:
        captured.append({"size": image.size, "prompt": prompt, "generation": generation})
        return f"caption-{len(captured)}"

    frozen_plan = freeze_stage_b_plan(program, candidate, settings)
    result = execute_stage_b(
        program,
        candidate,
        settings,
        output_root=research_root / "run",
        expected_plan=frozen_plan,
        generate=generate,
    )

    assert result["record_count"] == 4
    assert len(captured) == 4
    assert (research_root / "run" / "stage-b-plan.json").is_file()
    run_provenance = json.loads((research_root / "run" / "run-provenance.json").read_text())
    assert run_provenance["metric_self_audit"]["status"] == "PENDING_HUMAN_SELF_AUDIT"
    assert run_provenance["metric_self_audit"]["known_case_item_id"] == "fixture"
    assert (research_root / "run" / "records.jsonl").is_file()
    assert {path.name for path in (research_root / "run" / "outputs").rglob("*.txt")} == {"fixture.txt"}
    assert len(list((research_root / "run" / "outputs").rglob("*.txt"))) == 4

    # Legacy bucketed transform and raw source views are observably distinct for this 80x80 source.
    assert captured[0]["size"] == (1024, 1024)
    assert captured[1]["size"] == (80, 80)
    assert "DECLARED SPECIALIST EVIDENCE:" in captured[2]["prompt"]
    assert "no specialist evidence declared" in captured[2]["prompt"]
    assert "exactly one primary subject detected" not in captured[2]["prompt"]
    assert "DECLARED SPECIALIST EVIDENCE:" in captured[3]["prompt"]
    assert "geometric relations" in captured[3]["prompt"]
    assert all(call["generation"] == settings for call in captured)

    records = [json.loads(line) for line in (research_root / "run" / "records.jsonl").read_text().splitlines()]
    geometry_record = next(record for record in records if record["condition_id"] == "context-raw-geometry")
    assert "camera" not in geometry_record["evidence_payload"]
    assert geometry_record["selected_derived_reads"] == ["pose2.npy", "seg2.npy"]

    assert (source_root / "fixture.jpg").read_bytes() == source_before
    assert {path.relative_to(derived_root): path.read_bytes() for path in derived_root.rglob("*") if path.is_file()} == derived_before
    assert not (derived_root / "fixture" / "determinations.json").exists()


def test_stage_b_rejects_derived_artifact_hash_drift_before_generator(tmp_path: Path) -> None:
    program, candidate, settings, research_root = _fixture(tmp_path)
    frozen_plan = freeze_stage_b_plan(program, candidate, settings)
    pose_path = Path(program["canonical_source"]["derived_tree"]) / "fixture" / "pose2.npy"
    pose_path.write_bytes(b"tampered-pose2")
    calls: list[object] = []

    with pytest.raises(StageBRunError, match="artifact SHA-256 drift"):
        execute_stage_b(
            program,
            candidate,
            settings,
            output_root=research_root / "run",
            expected_plan=frozen_plan,
            generate=lambda *_args, **_kwargs: calls.append(True) or "unexpected",
        )

    assert calls == []


def test_stage_b_refuses_to_overwrite_a_prior_run_root(tmp_path: Path) -> None:
    program, candidate, settings, research_root = _fixture(tmp_path)
    output_root = research_root / "run"
    output_root.mkdir()

    with pytest.raises(StageBRunError, match="already exists"):
        execute_stage_b(
            program,
            candidate,
            settings,
            output_root=output_root,
            generate=lambda *_args, **_kwargs: "unexpected",
        )


def test_stage_b_rejects_a_drifted_precomputed_plan_before_generation(tmp_path: Path) -> None:
    program, candidate, settings, research_root = _fixture(tmp_path)
    plan = build_stage_b_plan(program, candidate, settings)
    plan["conditions"][0]["aggregator"]["model_id"] = "tampered"
    calls: list[object] = []

    with pytest.raises(StageBRunError, match="expected comparison plan"):
        execute_stage_b(
            program,
            candidate,
            settings,
            output_root=research_root / "run",
            expected_plan=plan,
            generate=lambda *_args, **_kwargs: calls.append(True) or "unexpected",
        )

    assert calls == []


def test_stage_b_cli_loads_and_binds_an_expected_plan(tmp_path: Path, monkeypatch) -> None:
    program, candidate, settings, research_root = _fixture(tmp_path)
    plan = build_stage_b_plan(program, candidate, settings)
    program_path = tmp_path / "program.json"
    candidate_path = tmp_path / "candidate.json"
    plan_path = tmp_path / "expected-plan.json"
    program_path.write_text(json.dumps(program))
    candidate_path.write_text(json.dumps(candidate))
    plan_path.write_text(json.dumps(plan))
    captured: dict[str, object] = {}

    def fake_execute(*args, **kwargs):
        captured["expected_plan"] = kwargs["expected_plan"]
        captured["output_root"] = kwargs["output_root"]
        return {"status": "PENDING_INDEPENDENT_REVIEW"}

    monkeypatch.setattr("research_harness.stage_b.execute_stage_b", fake_execute)

    assert main(
        [
            str(program_path),
            str(candidate_path),
            "--output",
            str(research_root / "run"),
            "--model",
            settings.model_name,
            "--model-digest",
            settings.model_digest,
            "--expected-plan",
            str(plan_path),
        ]
    ) == 0
    assert captured["expected_plan"] == plan
    assert captured["output_root"] == research_root / "run"


def test_stage_b_unloads_local_ollama_model_after_successful_real_mode_run(tmp_path: Path, monkeypatch) -> None:
    program, candidate, settings, research_root = _fixture(tmp_path)
    frozen_plan = freeze_stage_b_plan(program, candidate, settings)
    post_payloads: list[dict] = []

    class Response:
        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self.payload

    monkeypatch.setattr(
        "research_harness.stage_b.requests.get",
        lambda *_args, **_kwargs: Response({"models": [{"name": settings.model_name, "digest": settings.model_digest}]}),
    )

    def post(_url, *, json, **_kwargs):
        post_payloads.append(json)
        if "images" in json:
            return Response({"response": "local caption"})
        return Response({})

    monkeypatch.setattr("research_harness.stage_b.requests.post", post)

    execute_stage_b(
        program,
        candidate,
        settings,
        output_root=research_root / "run",
        expected_plan=frozen_plan,
    )

    generation_payloads = [payload for payload in post_payloads if "images" in payload]
    assert len(generation_payloads) == 4
    assert all(payload["keep_alive"] == "10m" for payload in generation_payloads)
    assert post_payloads[-1] == {"model": settings.model_name, "keep_alive": 0, "stream": False}


def test_stage_b_checks_installed_ollama_digest_before_actual_generation(tmp_path: Path, monkeypatch) -> None:
    program, candidate, settings, research_root = _fixture(tmp_path)
    calls: list[object] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"models": [{"name": "gemma3:27b", "digest": "b" * 64}]}

    monkeypatch.setattr("research_harness.stage_b.requests.get", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(
        "research_harness.stage_b.requests.post",
        lambda *_args, **_kwargs: calls.append(True),
    )

    with pytest.raises(StageBRunError, match="digest drift"):
        execute_stage_b(
            program,
            candidate,
            settings,
            output_root=research_root / "run",
        )

    assert calls == []


def test_stage_b_bodytype_plan_is_a_valid_four_condition_contract(tmp_path: Path) -> None:
    program, candidate, settings, _ = _fixture(tmp_path)

    plan = build_stage_b_plan(program, candidate, settings, evidence_kind="body-type")

    validate_comparison_parity_plan(plan, program)
    assert plan["comparison_plan_id"] == "stage-b-first500-bodytype-v1"
    condition_ids = [condition["id"] for condition in plan["conditions"]]
    assert condition_ids == [
        "legacy-bucketed-no-evidence",
        "legacy-raw-no-evidence",
        "context-raw-no-evidence",
        "context-raw-body-type",
    ]
    evidence_condition = next(c for c in plan["conditions"] if c["id"] == "context-raw-body-type")
    assert evidence_condition["evidence"]["id"] == "in-memory-body-type-proportions-v1"
    evidence_only = next(c for c in plan["contrasts"] if c["id"] == "evidence-only")
    assert evidence_only["variant_condition"] == "context-raw-body-type"
    assert evidence_only["baseline_condition"] == "context-raw-no-evidence"
    assert "anthropometric" in plan["hypothesis"]
    specialist = evidence_condition["evidence"]["specialists"][0]
    for field in ("scope", "inputs", "output_semantics", "provenance", "abstention_policy", "known_failure_modes", "qualification_gate"):
        assert specialist[field]


def test_stage_b_bodytype_plan_default_geometry_is_unchanged(tmp_path: Path) -> None:
    program, candidate, settings, _ = _fixture(tmp_path)

    plan = build_stage_b_plan(program, candidate, settings)

    assert plan["comparison_plan_id"] == "stage-b-first500-parity-v1"
    assert [condition["id"] for condition in plan["conditions"]][-1] == "context-raw-geometry"
    evidence_only = next(c for c in plan["contrasts"] if c["id"] == "evidence-only")
    assert evidence_only["variant_condition"] == "context-raw-geometry"


def test_stage_b_bodytype_freeze_binds_pose_hashes_and_validates(tmp_path: Path) -> None:
    program, candidate, settings, _ = _fixture(tmp_path)

    plan = freeze_stage_b_plan(program, candidate, settings, evidence_kind="body-type")

    validate_comparison_parity_plan(plan, program)
    assert plan["evidence_input_artifact_sha256"]["fixture"] == {
        "pose2.npy": _sha256((Path(program["canonical_source"]["derived_tree"]) / "fixture" / "pose2.npy").read_bytes()),
        "seg2.npy": _sha256((Path(program["canonical_source"]["derived_tree"]) / "fixture" / "seg2.npy").read_bytes()),
    }
    assert plan["status"] == "PENDING"


def test_stage_b_bodytype_serialize_proportions_abstains_and_reports(tmp_path: Path) -> None:
    program, candidate, settings, _ = _fixture(tmp_path)
    del program, candidate, settings

    abstained = _serialize_proportions({"subject_present": False})
    assert "abstain from body-type claims" in abstained

    measured = _serialize_proportions(
        {
            "subject_present": True,
            "between_shoulders": 40.0,
            "between_hips": 30.0,
            "shoulder_hip_ratio": 1.3226,
            "torso_length": 30.0,
            "left_leg_length": None,
            "right_leg_length": None,
            "leg_torso_ratio": None,
        }
    )
    assert "shoulder:hip width ratio: 1.3226" in measured
    assert "left leg length (px): not measurable" in measured


def test_stage_b_bodytype_execution_writes_evidence_condition(tmp_path: Path) -> None:
    program, candidate, settings, research_root = _fixture(tmp_path)
    captured: list[str] = []

    def generate(image: Image.Image, prompt: str, generation: StageBGenerationSettings) -> str:
        captured.append(prompt)
        return f"bodytype-caption-{len(captured)}"

    frozen_plan = freeze_stage_b_plan(program, candidate, settings, evidence_kind="body-type")
    result = execute_stage_b(
        program,
        candidate,
        settings,
        output_root=research_root / "run",
        expected_plan=frozen_plan,
        generate=generate,
    )

    assert result["record_count"] == 4
    assert len(captured) == 4
    records = [json.loads(line) for line in (research_root / "run" / "records.jsonl").read_text().splitlines()]
    body_record = next(record for record in records if record["condition_id"] == "context-raw-body-type")
    assert "shoulder:hip width ratio" in captured[3]
    assert captured[3].startswith("You are an expert descriptive captioner")
    assert body_record["evidence_payload"]["subject_present"] is True
    assert body_record["evidence_payload"]["between_shoulders"] == 40.0
    assert body_record["selected_derived_reads"] == ["pose2.npy", "seg2.npy"]
    assert body_record["caption_sha256"]
    # legacy/context conditions remain geometry-free (no determinations text leaked)
    assert "geometric relations" not in captured[2]
    assert "geometric relations" not in captured[3]


def test_stage_b_bodytype_rejects_unknown_evidence_kind(tmp_path: Path) -> None:
    program, candidate, settings, _ = _fixture(tmp_path)

    with pytest.raises(StageBRunError, match="unsupported Stage-B evidence_kind"):
        build_stage_b_plan(program, candidate, settings, evidence_kind="not-a-kind")
