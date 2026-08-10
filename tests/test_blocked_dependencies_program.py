"""TDD coverage for selector improvements #1-#4 (2026-08-06):

1. `blocked` dimension state — non-terminal, non-actionable; arms gated on a
   policy/authority decision (not a measurement) are excluded from selector
   scoring so the loop stops re-electing a stuck integrator.
2. Dependency graph (`feeds`/`unblocks`) + downstream-value weighting — an arm
   that feeds/unblocks a blocked arm gets `fraction * value(blocked_arm)` added
   to its EIG, so the selector prefers the globally-useful move.
3. `goal_unreachable` auto-detection — when the goal arm's pre-registered floor
   is structurally unreachable from the measured evidence budget,
   sweep_status emits goal_unreachable + measured gap, routes to the AUTONOMOUS
   "grow-evidence-supply" action, and flags floor-renegotiation as the HUMAN
   decision.
4. `program_overview` readout — budget vs floor, % goal-arm inputs validated,
   blocked count, dependency frontier — the strategist's step-back.
"""

from __future__ import annotations

import json

import pytest

from research_harness.dimension_registry import (
    DimensionRegistryError,
    dependency_frontier,
    goal_reachability,
    mark_dimension_blocked,
    mark_dimension_unblocked,
    program_overview,
    reachable_blocked,
    sweep_status,
    validate_registry,
)
from research_harness.autonomous import (
    AutonomousError,
    select_next_arm,
    run_tick,
)


def _proposal(
    id_: str,
    *,
    prior: float = 0.5,
    measurability: str = "medium",
    cost: str = "medium",
    feeds: list[str] | None = None,
    evidence_parts: list[str] | None = None,
    arm_issue: int = 90,
) -> dict:
    d = {
        "id": id_,
        "name": id_,
        "arm_issue": arm_issue,
        "state": "proposal",
        "valid_non_improving_experiments": 0,
        "hypothesis": f"h {id_}",
        "falsified_if": f"f {id_}",
        "deterministic_signal": f"signal {id_}",
        "metric_version": "claim-support-rubric-v1",
        "data_snapshot": "s",
        "selection_rationale": f"rationale {id_}",
        "prior_evidence_strength": prior,
        "measurability": measurability,
        "cost_bucket": cost,
    }
    if feeds is not None:
        d["feeds"] = feeds
    if evidence_parts is not None:
        d["evidence_parts"] = evidence_parts
    return d


def _registry(**sweep_overrides: dict) -> dict:
    reg = {
        "schema_version": 1,
        "program_id": "stratum-contextual-specialist-research",
        "goal": "dossier to context4k",
        "goal_arm": "goal-assembly",
        "goal_floors": {
            "expanded_dossier_min_tokens": 100000,
            "compact_context_min_tokens": 4000,
        },
        "evidence_budget": {
            "basis": "measured audit",
            "deterministic_min_tokens_per_item": 2040,
            "deterministic_median_tokens_per_item": 2700,
            "deterministic_max_tokens_per_item": 3489,
            "honest_ceiling_max_tokens_per_item": 13500,
        },
        "sweep_terms": {
            "terminal_states": ["validated", "falsified", "exhausted"],
            "per_dimension_strike_limit": 3,
            "brainstorm_states": ["brainstorm-new-data"],
            "exploration": {"every_n": 0, "novelty_bonus": 0},
        },
        "dimensions": [
            # A blocked GOAL arm: prior 0.9, meas high, cost high -> EIG 0.19;
            # novel evidence_parts -> +0.0 (novelty disabled here) => 0.19.
            {
                "id": "goal-assembly",
                "name": "goal assembly",
                "arm_issue": 36,
                "state": "blocked",
                "valid_non_improving_experiments": 0,
                "blocked_reason": "gate is a human A/B ruling, not a measurement",
                "blocked_by_issue": 46,
                "hypothesis": "h goal",
                "falsified_if": "f goal",
                "deterministic_signal": "assembled dossier",
                "metric_version": "m1",
                "data_snapshot": "s",
                "selection_rationale": "goal",
                "prior_evidence_strength": 0.9,
                "measurability": "high",
                "cost_bucket": "high",
                "evidence_parts": ["assembled-dossier"],
            },
            # A feeder: prior 0.5, meas medium, cost medium -> 0.5*0.6-0.15=0.15.
            _proposal("feeder", feeds=["goal-assembly"], evidence_parts=["vlm-prose"]),
            # A non-feeder with the SAME base score: 0.15, no downstream edge.
            _proposal("other", evidence_parts=["mood-prose"]),
        ],
    }
    reg["sweep_terms"].update(sweep_overrides)
    return reg


# ---------------------------------------------------------------------------
# #1 blocked state
# ---------------------------------------------------------------------------


def test_blocked_is_a_valid_dimension_state() -> None:
    validate_registry(_registry())  # must not raise


def test_blocked_requires_reason() -> None:
    reg = _registry()
    del reg["dimensions"][0]["blocked_reason"]
    with pytest.raises(DimensionRegistryError, match="blocked_reason"):
        validate_registry(reg)


def test_blocked_rejects_bad_issue_number() -> None:
    reg = _registry()
    reg["dimensions"][0]["blocked_by_issue"] = -3
    with pytest.raises(DimensionRegistryError, match="blocked_by_issue"):
        validate_registry(reg)


def test_selector_skips_blocked_arm() -> None:
    """The highest-value arm in the registry is BLOCKED; the selector must pick
    a best *proposal* instead of re-electing the stuck integrator."""
    reg = _registry()
    # downstream boost disabled so we test pure exclusion:
    reg["sweep_terms"]["downstream_boost"] = {"enabled": False, "fraction": 0.0}
    selected = select_next_arm(reg)
    assert selected["id"] != "goal-assembly"
    assert selected["id"] in ("feeder", "other")


def test_run_tick_activates_next_when_only_blocked_remains() -> None:
    """With no active arm and the goal blocked, the tick activates the best
    proposal instead of stalling on the gate."""
    reg = _registry()
    reg["sweep_terms"]["downstream_boost"] = {"enabled": False, "fraction": 0.0}
    out = run_tick(reg)
    assert out["next_action"] == "activate"
    assert out["next_arm"] != "goal-assembly"
    assert next(d for d in reg["dimensions"] if d["id"] == "goal-assembly")["state"] == "blocked"


def test_mark_blocked_and_unblocked_transitions() -> None:
    reg = _registry()
    reg["dimensions"][1] = _proposal("feeder")
    reg["dimensions"][1]["state"] = "active"
    mark_dimension_blocked(reg, "feeder", "waiting on authority", issue=77)
    d = next(x for x in reg["dimensions"] if x["id"] == "feeder")
    assert d["state"] == "blocked"
    assert d["blocked_reason"] == "waiting on authority"
    assert d["blocked_by_issue"] == 77
    mark_dimension_unblocked(reg, "feeder")
    d = next(x for x in reg["dimensions"] if x["id"] == "feeder")
    assert d["state"] == "proposal"
    assert "blocked_reason" not in d
    assert "blocked_by_issue" not in d


def test_mark_blocked_fails_closed() -> None:
    reg = _registry()
    with pytest.raises(DimensionRegistryError, match="reason"):
        mark_dimension_blocked(reg, "feeder", "  ", issue=77)
    with pytest.raises(DimensionRegistryError, match="unknown"):
        mark_dimension_blocked(reg, "nope", "reason", issue=77)
    with pytest.raises(DimensionRegistryError, match="issue"):
        mark_dimension_blocked(reg, "feeder", "reason", issue=-1)


def test_mark_unblocked_requires_blocked() -> None:
    reg = _registry()
    with pytest.raises(DimensionRegistryError, match="not blocked"):
        mark_dimension_unblocked(reg, "feeder")


# ---------------------------------------------------------------------------
# #2 dependency graph + downstream weighting
# ---------------------------------------------------------------------------


def test_validate_rejects_self_reference() -> None:
    reg = _registry()
    reg["dimensions"][1]["feeds"] = ["feeder"]
    with pytest.raises(DimensionRegistryError, match="itself"):
        validate_registry(reg)


def test_validate_rejects_unknown_dependency_ref() -> None:
    reg = _registry()
    reg["dimensions"][1]["feeds"] = ["ghost"]
    with pytest.raises(DimensionRegistryError, match="unknown dimension"):
        validate_registry(reg)


def test_validate_rejects_dependency_cycle() -> None:
    reg = _registry()
    reg["dimensions"][1]["feeds"] = ["other"]
    reg["dimensions"][2]["feeds"] = ["feeder"]  # feeder <-> other cycle
    with pytest.raises(DimensionRegistryError, match="cycle"):
        validate_registry(reg)


def test_reachable_blocked_traverses_edges() -> None:
    reg = _registry()
    validator = _proposal("validator", feeds=["feeder"], evidence_parts=["vp"])
    reg["dimensions"].append(validator)
    # validator -> feeder -> goal-assembly (blocked)
    assert reachable_blocked(reg, "validator") == {"goal-assembly"}
    assert reachable_blocked(reg, "feeder") == {"goal-assembly"}
    assert reachable_blocked(reg, "other") == set()


def _boosted_value(reg: dict, dim_id: str) -> float:
    from research_harness.autonomous import _downstream_boost_for, _downstream_config, _exploration_config

    _, novelty = _exploration_config(reg)
    _, fraction = _downstream_config(reg)
    parts = {
        d["id"] for d in reg["dimensions"]
        if d["state"] in ("validated", "falsified", "exhausted")
    }
    for d in reg["dimensions"]:
        if d["state"] in ("validated", "falsified", "exhausted"):
            parts.update(d.get("evidence_parts") or [])
    return _downstream_boost_for(reg, next(d for d in reg["dimensions"] if d["id"] == dim_id),
                                 set(), set(), novelty, fraction)


def test_downstream_boost_proportional_to_blocked_value() -> None:
    """Feeding arm earns fraction * value(blocked goal). Blocked value here is
    prior 0.9 * meas 1.0 - cost 0.35 = 0.55 (no novelty, disabled); fraction 0.5
    => boost 0.275; non-feeder gets 0."""
    reg = _registry()
    reg["sweep_terms"]["downstream_boost"] = {"enabled": True, "fraction": 0.5}
    # blocked goal value: 0.9*1.0 - 0.35 = 0.55
    assert _boosted_value(reg, "feeder") == pytest.approx(0.5 * 0.55)
    assert _boosted_value(reg, "other") == pytest.approx(0.0)


def test_selector_prefers_feeder_of_blocked_arm_with_boost() -> None:
    """feeder and other tie on base EIG (0.15); with downstream boost enabled
    the feeder (0.15 + 0.275 = 0.425) beats other (0.15)."""
    reg = _registry()
    reg["sweep_terms"]["downstream_boost"] = {"enabled": True, "fraction": 0.5}
    selected = select_next_arm(reg)
    assert selected["id"] == "feeder"
    assert selected["downstream_boost_applied"] == pytest.approx(0.5 * 0.55, abs=0.001)


def test_selector_downstream_boost_flips_choice_from_higher_local_eig() -> None:
    """Without the boost, a higher-local-EIG non-feeder wins; with the boost the
    feeder's downstream value (fraction * blocked goal EIG) overtakes it."""
    reg = _registry()
    # other scores 0.6*0.6 - 0.0 = 0.36; feeder base is 0.15 -> +0.275 = 0.425.
    reg["dimensions"][2] = _proposal("other", prior=0.6, measurability="medium", cost="low")
    reg["sweep_terms"]["downstream_boost"] = {"enabled": False, "fraction": 0.5}
    assert select_next_arm(reg)["id"] == "other"
    assert select_next_arm(reg)["downstream_boost_applied"] == 0.0

    reg["sweep_terms"]["downstream_boost"] = {"enabled": True, "fraction": 0.5}
    selected = select_next_arm(reg)
    assert selected["id"] == "feeder"
    assert selected["expected_information_gain"] == pytest.approx(0.15 + 0.5 * 0.55, abs=0.002)


def test_dependency_frontier_lists_only_actionable_feeders() -> None:
    reg = _registry()
    reg["sweep_terms"]["downstream_boost"] = {"enabled": True, "fraction": 0.5}
    frontier = dependency_frontier(reg)
    ids = [f["id"] for f in frontier]
    assert "feeder" in ids
    assert "other" not in ids  # no edge to the blocked arm
    assert "goal-assembly" not in ids  # blocked is not actionable
    entry = next(f for f in frontier if f["id"] == "feeder")
    assert entry["downstream_blocked"] == ["goal-assembly"]


# ---------------------------------------------------------------------------
# #3 goal unreachability
# ---------------------------------------------------------------------------


def test_goal_reachability_unreachable_fires_with_gap() -> None:
    reg = _registry()
    # budget max 3489, ceiling 13500, floor 100000 -> unreachable, gap 96511
    s = goal_reachability(reg)
    assert s["declared"] is True
    assert s["goal_unreachable"] is True
    assert s["floor_tokens"] == 100000
    assert s["validated_budget_tokens"] == 3489.0
    assert s["honest_ceiling_tokens"] == 13500.0
    assert s["measured_gap_tokens"] == pytest.approx(96511.0)
    assert s["route_to"] == "grow-evidence-supply"   # option B: AUTONOMOUS
    assert s["requires_human"] == ["floor-renegotiation"]  # option A: HUMAN


def test_goal_reachability_not_unreachable_when_budget_meets_floor() -> None:
    reg = _registry()
    reg["evidence_budget"]["honest_ceiling_max_tokens_per_item"] = 200000
    s = goal_reachability(reg)
    assert s["goal_unreachable"] is False
    assert s["route_to"] == "none"
    assert s["requires_human"] == []


def test_goal_reachability_not_declared_without_measurement() -> None:
    reg = _registry()
    reg.pop("evidence_budget")
    s = goal_reachability(reg)
    assert s["declared"] is False
    assert s["goal_unreachable"] is False


def test_sweep_status_includes_blocked_count_and_reachability() -> None:
    s = sweep_status(_registry())
    assert s["blocked"] == 1
    assert s["goal_reachability"]["goal_unreachable"] is True
    assert s["dependency_frontier"]


# ---------------------------------------------------------------------------
# #4 program overview
# ---------------------------------------------------------------------------


def test_program_overview_aggregates_program_state() -> None:
    reg = _registry()
    # second feeder (evidence toward the goal) so the inputs-validated % is meaningfully partial
    reg["dimensions"].append(_proposal("feeder2", feeds=["goal-assembly"], evidence_parts=["vlm2"]))
    # validate exactly one of the two feeders -> 50%
    for d in reg["dimensions"]:
        if d["id"] == "feeder":
            d["state"] = "validated"
    ov = program_overview(reg)
    assert ov["total"] == 4
    assert ov["goal_arm"] == "goal-assembly"
    assert ov["blocked_count"] == 1
    assert ov["by_state"]["blocked"] == 1
    assert ov["goal_inputs_validated_pct"] == pytest.approx(50.0)  # 1 of 2 feeders
    assert set(ov["goal_feeders"]) == {"feeder", "feeder2"}
    assert ov["goal_reachability"]["goal_unreachable"] is True
    assert [f["id"] for f in ov["dependency_frontier"]] == ["feeder2"]  # feeder validated, feeder2 actionable


def test_program_overview_pct_none_without_goal() -> None:
    reg = _registry()
    reg.pop("goal_arm")
    ov = program_overview(reg)
    assert ov["goal_arm"] is None
    assert ov["goal_inputs_validated_pct"] is None
    assert ov["goal_feeders"] == []


# ---------------------------------------------------------------------------
# CLI wiring (mark-blocked / mark-unblocked / program-overview)
# ---------------------------------------------------------------------------


def test_cli_mark_blocked_and_unblocked_write_path(tmp_path) -> None:
    from research_harness.cli import main

    path = tmp_path / "registry.json"
    reg = _registry()
    path.write_text(json.dumps(reg) + "\n", encoding="utf-8")

    assert main(["mark-blocked", str(path), "feeder", "--reason", "waiting", "--issue", "9", "--write"]) == 0
    loaded = json.loads(path.read_text(encoding="utf-8"))
    d = next(x for x in loaded["dimensions"] if x["id"] == "feeder")
    assert d["state"] == "blocked"
    assert d["blocked_reason"] == "waiting"
    assert d["blocked_by_issue"] == 9

    assert main(["mark-unblocked", str(path), "feeder", "--write"]) == 0
    loaded = json.loads(path.read_text(encoding="utf-8"))
    d = next(x for x in loaded["dimensions"] if x["id"] == "feeder")
    assert d["state"] == "proposal"


def test_cli_program_overview(tmp_path) -> None:
    from research_harness.cli import main

    path = tmp_path / "registry.json"
    reg = _registry()
    path.write_text(json.dumps(reg) + "\n", encoding="utf-8")
    out = main(["program-overview", str(path)])
    assert out == 0
