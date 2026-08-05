"""TDD coverage for `propose-dimensions` — first-class, gated idea generation.

The loop's idea generation is deterministic + gated: candidates must carry the
full declaration (scope/inputs/output_semantics/provenance/abstention_policy/
qualification_gate), a hypothesis, a falsification condition, and either a NEW
evidence part or a NEW model class before they are registered as `proposal`
and can be selected. No strategist discretion bypasses the gate.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from research_harness.dimension_registry import load_registry, validate_registry


def _registry() -> dict:
    return {
        "schema_version": 1,
        "program_id": "stratum-contextual-specialist-research",
        "goal": "iterate",
        "sweep_terms": {
            "terminal_states": ["validated", "falsified", "exhausted"],
            "per_dimension_strike_limit": 3,
            "brainstorm_states": ["brainstorm-new-data"],
        },
        "dimensions": [
            {
                "id": "clothing",
                "name": "clothing/apparel",
                "arm_issue": 29,
                "state": "validated",
                "valid_non_improving_experiments": 0,
                "hypothesis": "h-clothing",
                "falsified_if": "f-clothing",
                "deterministic_signal": "seg2 DOME-29 apparel classes",
                "metric_version": "claim-support-rubric-v1",
                "data_snapshot": "first-500",
                "selection_rationale": "common hallucination",
                "evidence_parts": ["seg2-apparel"],
            },
            {
                "id": "hair",
                "name": "hair",
                "arm_issue": 30,
                "state": "proposal",
                "valid_non_improving_experiments": 0,
                "hypothesis": "h-hair",
                "falsified_if": "f-hair",
                "deterministic_signal": "seg2 Hair(4)",
                "metric_version": "claim-support-rubric-v1",
                "data_snapshot": "first-500",
                "selection_rationale": "high hallucination risk",
                "evidence_parts": ["seg2-hair"],
            },
        ],
    }


def _valid_candidate() -> dict:
    return {
        "id": "relational-interaction",
        "name": "relational/interaction evidence",
        "arm_issue": 40,
        "hypothesis": "Declared co-subject relational geometry improves interaction claims.",
        "falsified_if": "Relational evidence does not reduce unsupported interaction claims vs baseline.",
        "deterministic_signal": "multi-person pose relational graph + gaze/contact vectors",
        "metric_version": "claim-support-rubric-v1",
        "data_snapshot": "first-500 cohort",
        "selection_rationale": "Interaction is a genuinely new evidence axis vs attribute taggers.",
        "scope": "interpersonal geometry and contact between the subject and any co-present actors",
        "inputs": "pose2 keypoints across all detected persons + source pixels",
        "output_semantics": "open-set relational phrases (contact, distance, orientation) with abstention",
        "provenance": "deterministic multi-person keypoint graph from pose2",
        "abstention_policy": "abstain when only one person is detected or keypoints are missing",
        "qualification_gate": ">=90% of cohort yields a non-abstained relational measurement",
        "evidence_parts": ["multi-person-interaction-graph"],
        "prior_evidence_strength": 0.5,
        "measurability": "medium",
        "cost_bucket": "medium",
    }


def _propose(registry, candidates, *, count=1, require_new_evidence_part=False) -> dict:
    from research_harness.proposals import propose_dimensions

    return propose_dimensions(
        registry,
        candidates,
        count=count,
        require_new_evidence_part=require_new_evidence_part,
    )


def test_propose_registers_gated_candidate_as_proposal() -> None:
    reg = _registry()
    out = _propose(reg, [_valid_candidate()], count=1)
    assert len(out["registered"]) == 1
    assert out["rejected"] == []
    added = [d for d in reg["dimensions"] if d["id"] == "relational-interaction"]
    assert len(added) == 1
    assert added[0]["state"] == "proposal"
    assert added[0]["valid_non_improving_experiments"] == 0
    assert added[0]["qualification_gate"] == ">=90% of cohort yields a non-abstained relational measurement"
    validate_registry(reg)  # the augmented registry must still validate


def test_propose_requires_declared_scope_abstention_gate() -> None:
    candidate = _valid_candidate()
    del candidate["qualification_gate"]
    with pytest.raises(Exception, match="qualification_gate"):
        _propose(_registry(), [candidate], count=1)


def test_propose_requires_count_new_dimensions() -> None:
    reg = _registry()
    with pytest.raises(Exception, match="count"):
        _propose(reg, [_valid_candidate()], count=2)


def test_propose_rejects_duplicate_id() -> None:
    candidate = _valid_candidate()
    candidate["id"] = "clothing"  # collides with an existing dimension
    with pytest.raises(Exception, match="duplicate|already exists"):
        _propose(_registry(), [candidate], count=1)


def test_propose_require_new_evidence_part_rejects_redundant_axis() -> None:
    """A an idea that only reuses an already-validated evidence part (yet another
    attribute tagger over seg2-apparel) must be rejected when
    require_new_evidence_part is set — seed diversity is deliberate."""
    candidate = _valid_candidate()
    candidate["id"] = "overcoat-tagger"
    candidate["name"] = "overcoat tagger"
    candidate["evidence_parts"] = ["seg2-apparel"]  # validated => redundant
    with pytest.raises(Exception, match="new evidence part|new model class|redundant"):
        _propose(_registry(), [candidate], count=1, require_new_evidence_part=True)


def test_propose_accepts_new_evidence_part_axis() -> None:
    reg = _registry()
    candidate = _valid_candidate()  # evidence_parts is new
    candidate["id"] = "temporal-sequence"
    candidate["name"] = "temporal/sequence evidence"
    out = _propose(reg, [candidate], count=1, require_new_evidence_part=True)
    assert len(out["registered"]) == 1
    assert reg["dimensions"][-1]["id"] == "temporal-sequence"


def test_propose_accepts_new_model_class_axis_even_with_known_part() -> None:
    """seed diversity also accepts a NEW MODEL CLASS over a known evidence part:
    e.g. a reconstruction arm (ComfyUI round-trip) even though it consumes the
    validated dossier artifacts."""
    reg = _registry()
    candidate = _valid_candidate()
    candidate["id"] = "generative-reconstruction"
    candidate["name"] = "generative reconstruction validation"
    candidate["evidence_parts"] = ["dossier-context4k"]
    candidate["model_candidates"] = ["local ComfyUI diffusion checkpoints + CLIP ViT-L/14"]
    out = _propose(reg, [candidate], count=1, require_new_evidence_part=True)
    assert len(out["registered"]) == 1
    assert reg["dimensions"][-1]["id"] == "generative-reconstruction"


def test_cli_propose_dimensions_registers_and_writes(tmp_path: Path) -> None:
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(json.dumps(_registry()), encoding="utf-8")
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(json.dumps([_valid_candidate()]), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "research_harness.cli", "propose-dimensions",
         str(reg_path), "--candidates", str(candidates_path), "--count", "1",
         "--require-new-evidence-part", "--write"],
        cwd=".",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert len(payload["registered"]) == 1
    loaded = load_registry(reg_path)
    assert any(d["id"] == "relational-interaction" for d in loaded["dimensions"])


def test_cli_propose_dimensions_rejects_when_count_unmet(tmp_path: Path) -> None:
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(json.dumps(_registry()), encoding="utf-8")
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(json.dumps([_valid_candidate()]), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "research_harness.cli", "propose-dimensions",
         str(reg_path), "--candidates", str(candidates_path), "--count", "2"],
        cwd=".",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2


def test_cli_propose_dimensions_requires_candidates_file() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "research_harness.cli", "propose-dimensions",
         "registry.json"],
        cwd=".",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "candidates" in result.stderr
