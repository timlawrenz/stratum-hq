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


def _twin(id_: str) -> dict:
    """A proposal scoring EXACTLY like body-type (identical EIG tuple), so any
    dict-comparison in the selector sort would crash with a TypeError."""
    return {
        "id": id_,
        "name": id_,
        "arm_issue": 90,
        "state": "proposal",
        "valid_non_improving_experiments": 0,
        "hypothesis": f"h {id_}",
        "falsified_if": f"f {id_}",
        "deterministic_signal": f"signal {id_}",
        "metric_version": "claim-support-rubric-v1",
        "data_snapshot": "s",
        "selection_rationale": f"twin of body-type {id_}",
        "prior_evidence_strength": 0.8,
        "measurability": "high",
        "cost_bucket": "low",
    }


def test_select_next_arm_ties_broken_by_id_never_compare_dicts() -> None:
    """A selector sort that falls back to comparing dimension dicts must never
    happen: when two actionable proposals tie on the full EIG tuple the
    selector breaks ties by id instead of crashing with
    `TypeError: '<' not supported between instances of 'dict'`. Regression for
    the bug hit on 2026-08-06 after registering proposal #47 (vlm-dense-
    description) revealed the exploit-path sort had no id tiebreaker."""
    reg = _registry()
    reg["dimensions"] = [
        _twin("body-type"),
        _twin("alpha"),
        _twin("beta"),
    ]
    # All three score identically -> a plain sorted((score, dim)) would compare
    # dicts. Expect deterministic id-ordered result (no crash).
    selected = select_next_arm(reg)
    assert selected["id"] == "body-type"  # max over (score, id) = id asc
    assert selected["selected_via"] == "exploit"
    # All_scores are present and also id-ordered.
    ids = [s["id"] for s in selected["all_scores"]]
    assert ids == sorted(ids, reverse=True)
    assert selected["all_scores"][0]["expected_information_gain"] == (
        selected["all_scores"][-1]["expected_information_gain"]
    )


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


def _write_reviews(
    tmp_path,
    *,
    geometry_supported=6,
    base_supported=2,
    base_unsupported=0,
    evidence_condition="context-raw-geometry",
    evidence_evidence_id="in-memory-pose2-seg2-geometry-v1",
    with_plan=True,
) -> str:
    import json as _json

    run_root = tmp_path / "run-root"
    run_root.mkdir(exist_ok=True)
    review_dir = run_root.parent / (run_root.name + "-review")
    review_dir.mkdir(exist_ok=True)
    if with_plan:
        plan = {
            "comparison_plan_id": "test-plan",
            "conditions": [
                {"id": "legacy-bucketed-no-evidence",
                 "evidence": {"id": "no-specialist-evidence-v1"}},
                {"id": "legacy-raw-no-evidence",
                 "evidence": {"id": "no-specialist-evidence-v1"}},
                {"id": "context-raw-no-evidence",
                 "evidence": {"id": "no-specialist-evidence-v1"}},
                {"id": evidence_condition,
                 "evidence": {"id": evidence_evidence_id}},
            ],
        }
        (run_root / "stage-b-plan.json").write_text(
            _json.dumps(plan) + "\n", encoding="utf-8")
    lines = []
    for i in range(24):
        row = {
            "image_id": f"img-{i}",
            "condition_id": evidence_condition,
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


def test_aggregate_auto_derives_non_geometry_evidence_condition(tmp_path) -> None:
    """Auto-derivation must pick up ANY specialist evidence condition (e.g.
    body-type proportions), not just the hardcoded geometry one — regression
    guard for the bug that silently flipped body-type to NOT_BETTER."""
    from research_harness.autonomous import aggregate_claim_support

    review_dir = _write_reviews(
        tmp_path,
        geometry_supported=9,
        base_supported=4,
        base_unsupported=2,
        evidence_condition="context-raw-body-type",
        evidence_evidence_id="in-memory-body-type-proportions-v1",
    )
    agg = aggregate_claim_support(review_dir)
    assert agg["evidence_supported"] == 9 * 24
    assert agg["baseline_supported"] == 4 * 24
    assert "context-raw-body-type" in agg["per_condition"]


def test_aggregate_requires_conditions_without_plan(tmp_path) -> None:
    """Without a plan and without explicit conditions, fail closed rather than
    silently assuming the geometry condition names."""
    from research_harness.autonomous import AutonomousError, aggregate_claim_support

    review_dir = _write_reviews(tmp_path, with_plan=False)
    with pytest.raises(AutonomousError):
        aggregate_claim_support(review_dir)


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
