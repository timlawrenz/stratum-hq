"""TDD coverage for issue-label synchronization driven by the dimension registry."""

from __future__ import annotations

import pytest

from research_harness.issue_labels import (
    IssueLabelError,
    compute_desired_labels,
    plan_issue_label_sync,
)


def _registry() -> dict:
    return {
        "schema_version": 1,
        "program_id": "stratum-contextual-specialist-research",
        "goal": "x",
        "sweep_terms": {
            "terminal_states": ["validated", "falsified", "exhausted"],
            "per_dimension_strike_limit": 3,
            "brainstorm_states": ["brainstorm-new-data"],
        },
        "dimensions": [
            {
                "id": "clothing", "name": "clothing", "arm_issue": 29,
                "state": "active", "valid_non_improving_experiments": 0,
                "hypothesis": "h", "falsified_if": "f",
                "deterministic_signal": "s", "metric_version": "m",
                "data_snapshot": "d", "selection_rationale": "r",
            },
            {
                "id": "body-type", "name": "body-type", "arm_issue": 32,
                "state": "validated", "valid_non_improving_experiments": 0,
                "hypothesis": "h", "falsified_if": "f",
                "deterministic_signal": "s", "metric_version": "m",
                "data_snapshot": "d", "selection_rationale": "r",
            },
            {
                "id": "hair", "name": "hair", "arm_issue": 30,
                "state": "proposal", "valid_non_improving_experiments": 0,
                "hypothesis": "h", "falsified_if": "f",
                "deterministic_signal": "s", "metric_version": "m",
                "data_snapshot": "d", "selection_rationale": "r",
            },
            {
                "id": "texture", "name": "texture", "arm_issue": 35,
                "state": "falsified", "valid_non_improving_experiments": 3,
                "hypothesis": "h", "falsified_if": "f",
                "deterministic_signal": "s", "metric_version": "m",
                "data_snapshot": "d", "selection_rationale": "r",
            },
        ],
    }


def test_compute_desired_labels_maps_states() -> None:
    reg = _registry()
    desired = compute_desired_labels(reg)
    # issue -> state label
    assert desired[29] == {"research:active"}
    assert desired[32] == {"research:validated"}
    assert desired[30] == {"research:proposal"}
    assert desired[35] == {"research:postmortem"}


def test_plan_sync_adds_missing_and_removes_stale_state() -> None:
    reg = _registry()
    current = {
        29: {"research", "research:proposal", "research:high-priority"},
        32: {"research", "research:proposal"},
        30: {"research", "research:proposal"},
        35: {"research", "research:proposal", "research:hold"},
    }
    operations = plan_issue_label_sync(reg, current)
    ops_by_issue: dict[int, dict[str, set[str]]] = {}
    for op in operations:
        issue = op["issue"]
        bucket = ops_by_issue.setdefault(issue, {"add": set(), "remove": set()})
        if op["action"] == "add":
            bucket["add"].add(op["label"])
        else:
            bucket["remove"].add(op["label"])
    # issue 29: proposal should be removed, active added; high-priority preserved
    assert ops_by_issue[29]["remove"] == {"research:proposal"}
    assert ops_by_issue[29]["add"] == {"research:active"}
    assert "research:high-priority" not in ops_by_issue[29]["remove"]
    # issue 32: stale proposal removed, validated added
    assert ops_by_issue[32]["remove"] == {"research:proposal"}
    assert ops_by_issue[32]["add"] == {"research:validated"}
    # issue 30: already correct -> no changes
    assert 30 not in ops_by_issue
    # issue 35: postmortem added, stale proposal + hold removed (hold is a state label)
    assert ops_by_issue[35]["add"] == {"research:postmortem"}
    assert "research:proposal" in ops_by_issue[35]["remove"]
    assert "research:hold" in ops_by_issue[35]["remove"]


def test_plan_sync_is_idempotent() -> None:
    reg = _registry()
    current = {
        29: {"research", "research:active"},
        32: {"research", "research:validated"},
        30: {"research", "research:proposal"},
        35: {"research", "research:postmortem"},
    }
    assert plan_issue_label_sync(reg, current) == []


def test_plan_sync_rejects_missing_issue() -> None:
    reg = dict(_registry())
    reg["dimensions"][0]["arm_issue"] = -5
    with pytest.raises(RuntimeError):  # registry validator rejects invalid arm_issue
        plan_issue_label_sync(reg, {})
