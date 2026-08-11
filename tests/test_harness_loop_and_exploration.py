"""TDD coverage for change 1 (deterministic conclude+advance via run_tick,
method-aware verdicts, atomic registry write) and change 2 (exploration /
novelty in the selector, brainstorm-on-stall).

Synthetic fixtures only — no live models, no GPU, no corpus access.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from research_harness.autonomous import (
    AutonomousError,
    better_or_not,
    run_tick,
    select_next_arm,
)
from research_harness.dimension_registry import (
    DimensionRegistryError,
    load_registry,
    validate_registry,
    write_registry,
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
                "evidence_parts": ["pose2"],
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
                "evidence_parts": ["vlm-relational"],
            },
        ],
    }


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
        (run_root / "stage-b-plan.json").write_text(json.dumps(plan) + "\n", encoding="utf-8")
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
        lines.append(json.dumps(row))
        row = {
            "image_id": f"img-{i}",
            "condition_id": "context-raw-no-evidence",
            "supported": ["s"] * base_supported,
            "unsupported": ["u"] * base_unsupported,
            "omissions": [],
            "contradictions": [],
            "abstentions": [],
        }
        lines.append(json.dumps(row))
    (review_dir / "reviews.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(review_dir)


# ---------------------------------------------------------------------------
# Change 2a: exploration slot (ε-greedy) in the selector.
# ---------------------------------------------------------------------------


def test_select_exploration_slot_forces_lowest_prior_when_configured() -> None:
    reg = _registry()
    reg["sweep_terms"]["exploration"] = {"every_n": 2, "novelty_bonus": 0.0}
    reg["selection_progress"] = 1  # next selection is the (progress+1)th => slot fired
    sel = select_next_arm(reg)
    # body-type has the highest prior (0.8); exploration must force mood (0.2).
    assert sel["id"] == "mood"
    assert sel["selected_via"] == "explore"
    assert sel["exploration_slot"] is True


def test_select_exploit_when_not_on_slot() -> None:
    reg = _registry()
    reg["sweep_terms"]["exploration"] = {"every_n": 2, "novelty_bonus": 0.0}
    reg["selection_progress"] = 2  # (2+1)%2 == 1 => not a slot, normal exploit
    sel = select_next_arm(reg)
    assert sel["id"] == "body-type"
    assert sel["selected_via"] == "exploit"
    assert sel["exploration_slot"] is False


def test_select_novelty_bonus_recovers_new_evidence_part() -> None:
    """An arm that declares a NEW evidence part (not used by any validated arm)
    gets a novelty bonus; with a large enough bonus it beats a higher-prior
    arm that reuses already-validated artifacts."""
    reg = _registry()
    reg["sweep_terms"]["exploration"] = {"every_n": 0, "novelty_bonus": 0.9}
    # body-type is now validated; its evidence_parts {pose2} are established.
    reg["dimensions"][0]["state"] = "validated"
    # mood (prior 0.2) declares a new evidence part -> gets +0.9
    sel = select_next_arm(reg)
    assert sel["id"] == "mood"
    assert sel["novelty_bonus_applied"] == 0.9


def test_select_no_novelty_bonus_for_reuse() -> None:
    reg = _registry()
    reg["sweep_terms"]["exploration"] = {"every_n": 0, "novelty_bonus": 0.9}
    reg["dimensions"][0]["state"] = "validated"  # pose2 established
    reg["dimensions"][1]["evidence_parts"] = ["pose2"]  # mood now reuses pose2
    sel = select_next_arm(reg)
    assert sel["novelty_bonus_applied"] == 0.0


# ---------------------------------------------------------------------------
# Change 1: run_tick is method-aware and records history.
# ---------------------------------------------------------------------------


def test_run_tick_reconstruction_method_better(tmp_path) -> None:
    reg = _registry()
    reg["dimensions"][0]["state"] = "active"
    reg["dimensions"][0]["validation_methods"] = ["reconstruction"]
    out = run_tick(
        reg,
        review_dir=None,
        method="reconstruction",
        reconstruction_delta=0.05,
        items=24,
    )
    assert out["verdict"]["verdict"] == "BETTER"
    assert reg["dimensions"][0]["state"] == "validated"


def test_run_tick_reconstruction_requires_delta() -> None:
    reg = _registry()
    reg["dimensions"][0]["state"] = "active"
    with pytest.raises(AutonomousError, match="reconstruction_delta"):
        run_tick(reg, method="reconstruction", reconstruction_delta=None)


def test_run_tick_not_better_strike_keeps_one_active() -> None:
    """A valid NOT_BETTER below the falsification limit records a strike and
    keeps the SAME arm the sole research:active one (one-active invariant).
    It must NOT activate a second arm while the struck arm is still active."""
    reg = _registry()
    reg["dimensions"][0]["state"] = "active"
    out = run_tick(
        reg,
        method="reconstruction",
        reconstruction_delta=-0.01,
        items=24,
    )
    assert out["verdict"]["verdict"] == "NOT_BETTER"
    assert out["next_action"] == "research-pending"
    assert out["advanced_arm"] == "body-type"
    assert out["active_arm"] == "body-type"
    acts = [d["state"] for d in reg["dimensions"]]
    assert acts == ["active", "proposal"]  # mood NEVER activated
    assert reg["dimensions"][0]["valid_non_improving_experiments"] == 1
    assert reg["dimensions"][1]["state"] == "proposal"
    assert reg.get("selection_progress", 0) == 0  # no selection happened
    hist = reg["conclusion_history"]
    assert hist[-1]["verdict"] == "NOT_BETTER"
    assert hist[-1]["state"] == "active"


def test_run_tick_third_strike_falsifies_then_activates_next() -> None:
    """At the falsification limit a NOT_BETTER closes the arm (falsified) and
    only then the next proposal is activated — still exactly one active."""
    reg = _registry()
    reg["dimensions"][0]["state"] = "active"
    reg["dimensions"][0]["valid_non_improving_experiments"] = 2
    out = run_tick(
        reg,
        method="reconstruction",
        reconstruction_delta=-0.01,
        items=24,
    )
    assert out["verdict"]["verdict"] == "NOT_BETTER"
    assert out["next_action"] == "activate-next"
    assert out["advanced_arm"] == "body-type"
    assert out["next_arm"] == "mood"
    assert reg["dimensions"][0]["state"] == "falsified"
    assert reg["dimensions"][0]["valid_non_improving_experiments"] == 3
    assert reg["dimensions"][1]["state"] == "active"
    acts = [d["state"] for d in reg["dimensions"]]
    assert sorted(acts) == ["active", "falsified"]


def test_run_tick_records_conclusion_history_and_progress(tmp_path) -> None:
    reg = _registry()
    reg["dimensions"][0]["state"] = "active"
    review_dir = _write_reviews(tmp_path, geometry_supported=6, base_supported=2, base_unsupported=2)
    out = run_tick(reg, review_dir=review_dir)
    assert out["next_action"] == "activate-next"
    assert reg["dimensions"][0]["state"] == "validated"
    history = reg.get("conclusion_history", [])
    assert len(history) == 1
    assert history[0]["arm_id"] == "body-type"
    assert history[0]["verdict"] == "BETTER"
    assert reg["selection_progress"] == 1  # one selection happened


def test_validate_accepts_exploration_stall_and_history_fields() -> None:
    reg = _registry()
    reg["sweep_terms"]["exploration"] = {"every_n": 3, "novelty_bonus": 0.2}
    reg["sweep_terms"]["stall"] = {"no_validation_in_last": 2, "selector_top_score_below": 0.3}
    reg["selection_progress"] = 4
    reg["conclusion_history"] = [
        {"arm_id": "x", "verdict": "NOT_BETTER", "state": "active", "cycle": 1}
    ]
    reg["dimensions"][0]["evidence_parts"] = ["pose2", "seg2"]
    validate_registry(reg)  # must not raise


# ---------------------------------------------------------------------------
# Change 2c: brainstorm on stall (history-based and top-score-based).
# ---------------------------------------------------------------------------


def test_sweep_status_stalled_when_no_validation_in_last_k() -> None:
    from research_harness.dimension_registry import sweep_status

    reg = _registry()
    reg["sweep_terms"]["stall"] = {"no_validation_in_last": 2}
    reg["conclusion_history"] = [
        {"arm_id": "a", "verdict": "NOT_BETTER", "state": "active", "cycle": 1},
        {"arm_id": "b", "verdict": "NOT_BETTER", "state": "active", "cycle": 2},
    ]
    status = sweep_status(reg)
    assert status["stalled"] is True
    assert status["stall_reason"] == "no validation in last 2 concluded cycles"


def test_run_tick_routes_brainstorm_on_stall_not_activate_next(tmp_path) -> None:
    """After a NOT_BETTER that triggers the stall window, run_tick must route to
    brainstorm-on-stall instead of activating the next arm — new ideas surface
    while the safe menu is still being worked."""
    reg = _registry()
    reg["sweep_terms"]["stall"] = {"no_validation_in_last": 1}
    reg["conclusion_history"] = [
        {"arm_id": "prev", "verdict": "NOT_BETTER", "state": "active", "cycle": 0},
    ]
    reg["dimensions"][0]["state"] = "active"
    review_dir = _write_reviews(tmp_path, geometry_supported=2, base_supported=2, base_unsupported=2)
    out = run_tick(reg, review_dir=review_dir)
    assert out["verdict"]["verdict"] == "NOT_BETTER"
    assert out["next_action"] == "brainstorm-on-stall"
    # the failed arm got its strike but the next arm was NOT activated
    assert reg["dimensions"][0]["valid_non_improving_experiments"] == 1
    assert reg["dimensions"][1]["state"] == "proposal"


def test_run_tick_stall_by_top_score_below_threshold(tmp_path) -> None:
    """selector_top_score_below: if the best actionable EIG is below the
    threshold, surface brainstorm even when there is no history at all."""
    reg = _registry()
    reg["sweep_terms"]["stall"] = {"selector_top_score_below": 0.95}  # body-type EIG is lower
    reg["dimensions"][0]["state"] = "active"
    review_dir = _write_reviews(tmp_path, geometry_supported=2, base_supported=2, base_unsupported=2)
    out = run_tick(reg, review_dir=review_dir)
    assert out["verdict"]["verdict"] == "NOT_BETTER"
    assert out["next_action"] == "brainstorm-on-stall"


def test_run_tick_no_stall_still_activates_next(tmp_path) -> None:
    reg = _registry()
    reg["dimensions"][0]["state"] = "active"
    reg["dimensions"][1]["state"] = "proposal"
    review_dir = _write_reviews(tmp_path, geometry_supported=6, base_supported=2, base_unsupported=2)
    out = run_tick(reg, review_dir=review_dir)
    assert out["next_action"] == "activate-next"
    assert out["next_arm"] == "mood"
    assert reg["dimensions"][1]["state"] == "active"


# ---------------------------------------------------------------------------
# Change 1: atomic registry write (guard against stale/clobbered writes).
# ---------------------------------------------------------------------------


def test_write_registry_atomic_ok(tmp_path: Path) -> None:
    reg = _registry()
    path = tmp_path / "registry.json"
    write_registry(path, reg)
    loaded = load_registry(path)
    assert loaded["dimensions"][0]["id"] == "body-type"


def test_write_registry_refuses_when_file_changed_under_us(tmp_path: Path) -> None:
    reg = _registry()
    path = tmp_path / "registry.json"
    write_registry(path, reg)
    before = path.read_bytes()
    # someone else edits the registry between our load and write
    path.write_text(path.read_text(encoding="utf-8").replace("body-type", "body-type-2"), encoding="utf-8")
    with pytest.raises(DimensionRegistryError, match="changed on disk"):
        write_registry(path, reg, expected_sha256=before)


def test_cli_autonomous_tick_write_guard_reports_stale(tmp_path, monkeypatch, capsys) -> None:
    """A concurrent registry advance between our tick and our write must fail
    closed (exit 2, changed-on-disk) instead of clobbering newer state."""
    import research_harness.autonomous as autonomous
    from research_harness.cli import main as cli_main

    reg = _registry()
    reg["dimensions"][0]["state"] = "active"
    review_dir = _write_reviews(tmp_path, geometry_supported=6, base_supported=2, base_unsupported=2)
    registry_path = tmp_path / "registry.json"
    write_registry(registry_path, reg)

    real_run_tick = autonomous.run_tick

    def _concurrent_writer(*args, **kwargs) -> dict:
        outcome = real_run_tick(*args, **kwargs)
        # a concurrent writer advances the registry after our tick computed
        registry_path.write_text(
            registry_path.read_text(encoding="utf-8").replace("h body", "h body edited"),
            encoding="utf-8",
        )
        return outcome

    monkeypatch.setattr(autonomous, "run_tick", _concurrent_writer)
    rc = cli_main(["autonomous-tick", str(registry_path), "--review-dir", review_dir, "--write"])
    assert rc == 2
    assert "changed on disk" in capsys.readouterr().err


def test_cli_autonomous_tick_empty_review_dir_is_research_pending(tmp_path) -> None:
    reg = _registry()
    reg["dimensions"][0]["state"] = "active"
    registry_path = tmp_path / "registry.json"
    write_registry(registry_path, reg)
    result = subprocess.run(
        [sys.executable, "-m", "research_harness.cli", "autonomous-tick",
         str(registry_path), "--review-dir", str(tmp_path / "missing-review")],
        cwd=".",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["next_action"] == "research-pending"


def test_cli_autonomous_tick_consumes_tick_ready_marker(tmp_path, capsys) -> None:
    """Change 1 wiring: the tick must conclude against the review root named by
    the wrapper's tick-ready marker — deterministic, no path guessing."""
    import research_harness.cli as cli

    reg = _registry()
    reg["dimensions"][0]["state"] = "active"
    review_dir = _write_reviews(tmp_path, geometry_supported=6, base_supported=2, base_unsupported=2)
    registry_path = tmp_path / "registry.json"
    write_registry(registry_path, reg)

    marker_path = tmp_path / "tick-ready.json"
    marker_path.write_text(json.dumps({
        "schema_version": 1,
        "status": "completed",
        "review_root": review_dir,
        "run_root": str(tmp_path / "run-root"),
        "job_id": "stratum-test",
    }) + "\n", encoding="utf-8")

    rc = cli.main(["autonomous-tick", str(registry_path),
                   "--review-dir-from", str(marker_path)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    # the marker's review root was consumed: verdict computed, arm advanced, next selected
    assert payload["next_action"] == "activate-next"
    assert payload["verdict"]["verdict"] == "BETTER"
    assert payload["advanced_arm"] == "body-type"
    assert payload["next_arm"] == "mood"


def test_cli_tick_refuses_ambiguous_review_dir_sources(tmp_path) -> None:
    import research_harness.cli as cli

    reg = _registry()
    registry_path = tmp_path / "registry.json"
    write_registry(registry_path, reg)
    marker_path = tmp_path / "tick-ready.json"
    marker_path.write_text(json.dumps({
        "status": "completed", "review_root": str(tmp_path / "x-review")}) + "\n",
        encoding="utf-8")
    rc = cli.main(["autonomous-tick", str(registry_path),
                   "--review-dir", str(tmp_path / "r-review"),
                   "--review-dir-from", str(marker_path)])
    assert rc == 2


def test_cli_tick_rejects_incomplete_marker(tmp_path) -> None:
    import research_harness.cli as cli

    reg = _registry()
    registry_path = tmp_path / "registry.json"
    write_registry(registry_path, reg)
    marker_path = tmp_path / "tick-ready.json"
    marker_path.write_text(json.dumps({
        "status": "in-flight", "review_root": str(tmp_path / "x-review")}) + "\n",
        encoding="utf-8")
    rc = cli.main(["autonomous-tick", str(registry_path),
                   "--review-dir-from", str(marker_path)])
    assert rc == 2


def test_better_or_not_reconstruction_signature_unchanged() -> None:
    v = better_or_not(
        supported_base=0,
        supported_variant=0,
        unsupported_base=0,
        unsupported_variant=0,
        items=24,
        sign_test_p_supported=0.5,
        method="reconstruction",
        reconstruction_delta=0.01,
    )
    assert v["verdict"] == "BETTER"
