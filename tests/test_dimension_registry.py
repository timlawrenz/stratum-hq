"""TDD coverage for the evidence-dimension registry + convex-landscape sweep."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_harness.dimension_registry import (
    DimensionRegistryError,
    load_registry,
    render_arm_issue,
    sweep_status,
    validate_registry,
)


def _registry() -> dict:
    return {
        "schema_version": 1,
        "program_id": "stratum-contextual-specialist-research",
        "goal": "Enumicro squash: full roster.",
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
                "state": "proposal",
                "valid_non_improving_experiments": 0,
                "hypothesis": "h-clothing",
                "falsified_if": "f-clothing",
                "deterministic_signal": "seg2 apparel classes + pixel colors",
                "metric_version": "claim-support-rubric-v1",
                "data_snapshot": "first-500 cohort",
                "selection_rationale": "common hallucination, measurable",
            },
            {
                "id": "hair",
                "name": "hair",
                "arm_issue": 30,
                "state": "validated",
                "valid_non_improving_experiments": 0,
                "hypothesis": "h-hair",
                "falsified_if": "f-hair",
                "deterministic_signal": "seg2 Hair(4)",
                "metric_version": "claim-support-rubric-v1",
                "data_snapshot": "first-500 cohort",
                "selection_rationale": "high hallucination risk",
            },
        ],
    }


def test_validate_registry_accepts_well_formed() -> None:
    validate_registry(_registry())  # must not raise


def test_validate_registry_rejects_bad_state(tmp_path: Path) -> None:
    reg = _registry()
    reg["dimensions"][0]["state"] = "running"  # invalid terminal/proposal/active state
    with pytest.raises(DimensionRegistryError):
        validate_registry(reg)


def test_validate_registry_rejects_strikes_over_limit(tmp_path: Path) -> None:
    reg = _registry()
    reg["sweep_terms"]["per_dimension_strike_limit"] = 3
    reg["dimensions"][0]["valid_non_improving_experiments"] = 4
    with pytest.raises(DimensionRegistryError, match="strikes"):
        validate_registry(reg)


def test_validate_registry_rejects_nonterminal_when_striked_out(tmp_path: Path) -> None:
    reg = _registry()
    reg["sweep_terms"]["per_dimension_strike_limit"] = 3
    reg["dimensions"][0]["valid_non_improving_experiments"] = 3
    reg["dimensions"][0]["state"] = "proposal"  # must be terminal after strike limit
    with pytest.raises(DimensionRegistryError, match="terminal"):
        validate_registry(reg)


def test_validate_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    reg = _registry()
    reg["dimensions"].append(dict(reg["dimensions"][0]))
    with pytest.raises(DimensionRegistryError, match="duplicate"):
        validate_registry(reg)


def test_validate_registry_rejects_missing_field(tmp_path: Path) -> None:
    reg = _registry()
    del reg["dimensions"][0]["deterministic_signal"]
    with pytest.raises(DimensionRegistryError, match="deterministic_signal"):
        validate_registry(reg)


def test_sweep_status_counts_and_exhaustion() -> None:
    status = sweep_status(_registry())
    assert status["total"] == 2
    assert status["by_state"]["proposal"] == 1
    assert status["terminal"] == 1
    assert status["exhausted"] is False
    assert status["next_action"] == "none"  # still proposals to sweep


def test_sweep_status_exhausted_triggers_brainstorm() -> None:
    reg = _registry()
    reg["dimensions"][0]["state"] = "falsified"
    status = sweep_status(reg)
    assert status["exhausted"] is True
    assert status["next_action"] == "brainstorm-new-data"


def test_sweep_status_exhausted_offers_brainstorm_widening_options() -> None:
    """Exhaustion must not mean 'stop': it routes to brainstorm options that
    include new data sources, new evidence parts, AND new model candidates
    (literature/arXiv scan) — the owner's widening directive."""
    reg = _registry()
    reg["dimensions"][0]["state"] = "falsified"
    status = sweep_status(reg)
    assert status["exhausted"] is True
    opts = status["brainstorm_options"]
    assert "new-model-candidate research (literature/arXiv scan)" in " ".join(opts)
    assert any("new-data-source" in o for o in opts)
    assert any("new-evidence-part" in o for o in opts)
    # not exhausted -> no brainstorm options offered
    assert sweep_status(_registry())["brainstorm_options"] == []


def test_render_arm_issue_is_deterministic_and_contains_meta() -> None:
    reg = _registry()
    body = render_arm_issue(reg, "clothing")
    for token in ("research-harness:", '"id":"clothing"', "clothing/apparel", "arm_issue"):
        assert token in body
    # deterministic: identical output for identical input
    assert render_arm_issue(reg, "clothing") == body


def test_render_arm_issue_rejects_unknown_id() -> None:
    reg = _registry()
    with pytest.raises(DimensionRegistryError, match="clothing2"):
        render_arm_issue(reg, "clothing2")


def test_registry_freezes_after_exhaustion_from_live_file(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    reg = _registry()
    for dim in reg["dimensions"]:
        dim["state"] = dim["state"] if dim["state"] in ("validated", "falsified") else "falsified"
    path.write_text(json.dumps(reg), encoding="utf-8")
    loaded = load_registry(path)
    s = sweep_status(loaded)
    assert s["exhausted"] is True
    assert s["next_action"] == "brainstorm-new-data"


def test_specialists_and_validation_methods_accepted() -> None:
    reg = _registry()
    reg["dimensions"][0]["specialists"] = [
        {
            "name": "Florence-2-Flux-Large",
            "source": "local ComfyUI /mnt/fscache/essdee/ComfyUI/models/LLM",
            "scope": "open-set clothing/attribute tagging",
            "known_failure_modes": ["hallucinated attributes on tight crops"],
        }
    ]
    reg["dimensions"][0]["validation_methods"] = ["claim-support", "reconstruction"]
    validate_registry(reg)  # must not raise


def test_specialists_reject_missing_field() -> None:
    reg = _registry()
    reg["dimensions"][0]["specialists"] = [{"name": "X", "source": "local"}]
    with pytest.raises(DimensionRegistryError, match="scope"):
        validate_registry(reg)


def test_validation_methods_reject_unknown() -> None:
    reg = _registry()
    reg["dimensions"][0]["validation_methods"] = ["ledger-aggregation"]
    with pytest.raises(DimensionRegistryError, match="ledger-aggregation"):
        validate_registry(reg)
