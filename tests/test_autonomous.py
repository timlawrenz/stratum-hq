"""TDD coverage for the autonomous research loop (select-next-arm + verdict)."""

from __future__ import annotations

import pytest

from research_harness.autonomous import (
    AutonomousError,
    better_or_not,
    select_next_arm,
)


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
                "id": "body-type",
                "name": "body-type",
                "arm_issue": 32,
                "state": "proposal",
                "valid_non_improving_experiments": 0,
                "hypothesis": "h body",
                "falsified_if": "f body",
                "deterministic_signal": "pose2 Goliath-308 keypoints",
                "metric_version": "claim-support-rubric-v1",
                "data_snapshot": "s",
                "selection_rationale": "pose evidence already helped in arm #4",
                "prior_evidence_strength": 0.8,
                "measurability": "high",
                "cost_bucket": "low",
            },
            {
                "id": "mood",
                "name": "mood",
                "arm_issue": 38,
                "state": "proposal",
                "valid_non_improving_experiments": 0,
                "hypothesis": "h mood",
                "falsified_if": "f mood",
                "deterministic_signal": "open-world relational language only",
                "metric_version": "claim-support-rubric-v1",
                "data_snapshot": "s",
                "selection_rationale": "mood is hard to measure deterministically",
                "prior_evidence_strength": 0.2,
                "measurability": "low",
                "cost_bucket": "medium",
            },
        ],
    }


def test_select_next_arm_prefers_high_prior_high_measurability() -> None:
    reg = _registry()
    selected = select_next_arm(reg)
    assert selected["id"] == "body-type"


def test_select_next_arm_rejects_terminal_registry() -> None:
    reg = _registry()
    for dim in reg["dimensions"]:
        dim["state"] = "validated"
    with pytest.raises(AutonomousError, match="no actionable proposal"):
        select_next_arm(reg)


def test_select_next_arm_skips_struck_out() -> None:
    reg = _registry()
    reg["dimensions"][0]["valid_non_improving_experiments"] = (
        reg["sweep_terms"]["per_dimension_strike_limit"]
    )
    reg["dimensions"][0]["state"] = "exhausted"
    selected = select_next_arm(reg)
    assert selected["id"] == "mood"


def test_better_or_not_returns_better_on_strong_supported_gain() -> None:
    result = better_or_not(
        supported_base=47,
        supported_variant=156,
        unsupported_base=99,
        unsupported_variant=40,
        items=24,
        sign_test_p_supported=0.003,
        method="claim-support",
    )
    assert result["verdict"] == "BETTER"
    assert result["sign_test_p_supported"] <= 0.05


def test_better_or_not_returns_not_better_on_weak_evidence() -> None:
    result = better_or_not(
        supported_base=47,
        supported_variant=52,
        unsupported_base=40,
        unsupported_variant=38,
        items=24,
        sign_test_p_supported=0.41,
        method="claim-support",
    )
    assert result["verdict"] == "NOT_BETTER"


def test_better_or_not_requires_items() -> None:
    with pytest.raises(AutonomousError, match="items"):
        better_or_not(
            supported_base=0,
            supported_variant=0,
            unsupported_base=0,
            unsupported_variant=0,
            items=0,
            sign_test_p_supported=1.0,
            method="claim-support",
        )


def _write_reviews(tmp_path, *, geometry_supported=6, base_supported=2, base_unsupported=0) -> str:
    import json as _json

    review_dir = tmp_path / "review"
    review_dir.mkdir()
    lines = []
    for i in range(24):
        row = {
            "image_id": f"img-{i}",
            "condition_id": "context-raw-geometry",
            "supported": ["s"] * geometry_supported,
            "unsupported": [],
            "omissions": [],
            "contradictions": [],
            "abstentions": [],
        }
        lines.append(_json.dumps(row))
        row = {
            "image_id": f"img-{i}",
            "condition_id": "context-raw-no-evidence",
            "supported": ["s"] * base_supported,
            "unsupported": ["u"] * base_unsupported,
            "omissions": [],
            "contradictions": [],
            "abstentions": [],
        }
        lines.append(_json.dumps(row))
    (review_dir / "reviews.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(review_dir)


def test_run_tick_activates_when_none_active() -> None:
    from research_harness.autonomous import run_tick

    reg = _registry()
    out = run_tick(reg)
    assert out["next_action"] == "activate"
    assert out["next_arm"] == "body-type"
    assert reg["dimensions"][0]["state"] == "active"
    assert reg["dimensions"][1]["state"] == "proposal"


def test_run_tick_concludes_and_advances_with_results(tmp_path) -> None:
    from research_harness.autonomous import run_tick

    reg = _registry()
    reg["dimensions"][0]["state"] = "active"
    review_dir = _write_reviews(tmp_path, geometry_supported=6, base_supported=2, base_unsupported=2)
    out = run_tick(reg, review_dir=review_dir)
    assert out["next_action"] == "activate-next"
    assert out["verdict"]["verdict"] == "BETTER"
    # body-type advanced to validated; mood becomes the next active arm
    assert reg["dimensions"][0]["state"] == "validated"
    assert reg["dimensions"][1]["state"] == "active"


def test_run_tick_strikes_on_not_better(tmp_path) -> None:
    from research_harness.autonomous import run_tick

    reg = _registry()
    # flatten priors so selection doesn't re-pick body-type after it fails
    reg["dimensions"][0]["prior_evidence_strength"] = 0.2
    reg["dimensions"][1]["prior_evidence_strength"] = 0.2
    reg["dimensions"][0]["state"] = "active"
    # no ratio change + not significant -> NOT_BETTER, strike 1, still active
    review_dir = _write_reviews(tmp_path, geometry_supported=2, base_supported=2, base_unsupported=2)
    out = run_tick(reg, review_dir=review_dir)
    assert out["verdict"]["verdict"] == "NOT_BETTER"
    assert reg["dimensions"][0]["valid_non_improving_experiments"] == 1
    assert reg["dimensions"][0]["state"] == "active"  # still under strike limit


def test_run_tick_pending_when_no_results(tmp_path) -> None:
    from research_harness.autonomous import run_tick

    reg = _registry()
    reg["dimensions"][0]["state"] = "active"
    out = run_tick(reg, review_dir=str(tmp_path / "missing"))
    assert out["next_action"] == "research-pending"
    assert reg["dimensions"][0]["state"] == "active"
