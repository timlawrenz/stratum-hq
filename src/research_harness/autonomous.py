"""Autonomous research loop: select the next highest-impact arm + verdict.

This is the decision layer of the harness. Given the dimension registry (and,
optionally, a tree snapshot / measured results), it:

1. `select_next_arm` — scores every actionable proposal by expected
   information gain and returns the single highest-impact candidate as the
   next `research:active` arm, with a recorded reasoning trace.
2. `better_or_not` — turns a measured one-axis comparison (base vs variant,
   e.g. no-evidence vs declared-evidence) into an explicit verdict on whether
   the ~4K asset description is BETTER, using a pre-registered claim-support
   significance rule (paired sign-test p-value + direction check) or a
   reconstruction similarity rule.
3. `run_tick` — the autonomous loop body: select next arm -> mark it active
   -> (research runs via the launchers / measurement engines) -> read measured
   results -> conclude -> advance registry state -> return the next action.

Both basics are deterministic and fail-closed. The purpose: the harness *decides*
its next step, *researches* it (execution lives in the launchers/registry
arms), and *concludes* better-or-not — then the sweep state advances.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Mapping

from .dimension_registry import (
    BRAINSTORM_OPTIONS,
    reachable_blocked,
    validate_registry,
    validated_evidence_parts,
)

ACTIONABLE = ("proposal", "active")

# Expected-information-gain prior: an arm whose deterministic signal closely
# follows a measurement already validated in another arm (e.g. pose2 helped in
# arm #4) earns a higher prior. Values are normalized weights.
_PRIOR_WEIGHTS = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.2,
}
_MEASURABILITY_WEIGHTS = {
    "high": 1.0,  # deterministic, existing artifacts, no new model
    "medium": 0.6,
    "low": 0.3,   # open-world relational only / needs new inference
}
_COST_PENALTY = {
    "low": 0.0,
    "medium": 0.15,
    "high": 0.35,
}
STRIKE_PENALTY = 0.45  # each prior valid non-improving experiment cuts value


class AutonomousError(RuntimeError):
    pass


def _weighted(value: Any, table: Mapping[str, float], field: str) -> float:
    if not isinstance(value, str):
        raise AutonomousError(f"arm field {field!r} must be a string, got {type(value).__name__}")
    if value not in table:
        raise AutonomousError(f"arm field {field!r} has unsupported value {value!r}")
    return table[value]


def _arm_value(dim: Mapping[str, Any]) -> tuple[float, str]:
    """Expected information gain per arm; lower can-tie-break by id for determinism."""
    prior = dim.get("prior_evidence_strength")
    if prior is None:
        prior = _weighted(dim.get("prior_evidence_strength_str", "low"),
                          _PRIOR_WEIGHTS, "prior_evidence_strength_str")
    if not isinstance(prior, (int, float)) or isinstance(prior, bool) or not (0.0 <= prior <= 1.0):
        raise AutonomousError(f"arm {dim['id']!r} prior_evidence_strength must be in [0,1]")
    meas = _weighted(dim.get("measurability", "medium"), _MEASURABILITY_WEIGHTS, "measurability")
    cost = _weighted(dim.get("cost_bucket", "medium"), _COST_PENALTY, "cost_bucket")
    strikes = dim.get("valid_non_improving_experiments", 0)
    if not isinstance(strikes, int) or isinstance(strikes, bool) or strikes < 0:
        raise AutonomousError(f"arm {dim['id']!r} strikes must be a non-negative integer")
    value = float(prior) * float(meas) - cost - strikes * STRIKE_PENALTY
    return value, dim["id"]


def _prior_number(dim: Mapping[str, Any]) -> float:
    """Numeric prior for exploration tie-breaking (higher = more established)."""
    prior = dim.get("prior_evidence_strength")
    if prior is None:
        prior = _PRIOR_WEIGHTS[dim.get("prior_evidence_strength_str", "low")]
    return float(prior)


def _established_models(registry: Mapping[str, Any]) -> set[str]:
    """Model/specialist identities already established by terminal-state arms."""
    terminal = set(registry.get("sweep_terms", {}).get("terminal_states", ("validated", "falsified", "exhausted")))
    models: set[str] = set()
    for dim in registry.get("dimensions", []):
        if dim.get("state") not in terminal:
            continue
        models.update(dim.get("model_candidates") or [])
        for spec in dim.get("specialists") or []:
            models.add(spec.get("name", ""))
    return {m for m in models if m}


def _novelty_for(
    dim: Mapping[str, Any],
    established_parts: set[str],
    established_models: set[str],
) -> bool:
    """A proposal is novel if it names an evidence part or model class NOT
    already established by a terminal (validated/falsified/exhausted) arm.
    Arms that declare neither evidence_parts nor model_candidates are treated
    as non-novel (conservative — no bonus).
    """
    parts = set(dim.get("evidence_parts") or [])
    models = set(dim.get("model_candidates") or [])
    return bool((parts - established_parts) or (models - established_models))


def _eig_with_novelty(
    dim: Mapping[str, Any],
    established_parts: set[str],
    established_models: set[str],
    novelty_bonus: float,
) -> tuple[float, bool, float]:
    """EIG with an optional novelty bonus; returns (value, is_novel, applied_bonus)."""
    base = _arm_value(dim)[0]
    novel = _novelty_for(dim, established_parts, established_models)
    applied = novelty_bonus if novel else 0.0
    return base + applied, novel, applied


def _exploration_config(registry: Mapping[str, Any]) -> tuple[int, float]:
    sweep = registry.get("sweep_terms", {})
    expl = sweep.get("exploration") if isinstance(sweep, Mapping) else None
    if not isinstance(expl, Mapping):
        return 0, 0.0
    every_n = expl.get("every_n", 0)
    bonus = expl.get("novelty_bonus", 0.0)
    return (int(every_n) if isinstance(every_n, int) and not isinstance(every_n, bool) else 0,
            float(bonus) if isinstance(bonus, (int, float)) and not isinstance(bonus, bool) else 0.0)


def _downstream_config(registry: Mapping[str, Any]) -> tuple[bool, float]:
    """sweep_terms.downstream_boost = {enabled, fraction}: dependency-graph
    weighting (#2). Returns (enabled, fraction)."""
    sweep = registry.get("sweep_terms", {})
    cfg = sweep.get("downstream_boost") if isinstance(sweep, Mapping) else None
    if not isinstance(cfg, Mapping):
        return False, 0.0
    enabled = cfg.get("enabled", False)
    fraction = cfg.get("fraction", 0.5)
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
        fraction = 0.0
    # `enabled` gates the weight: disabled means zero downstream value.
    return bool(enabled), float(fraction) if enabled else 0.0


def _downstream_boost_for(
    registry: Mapping[str, Any],
    dim: Mapping[str, Any],
    established_parts: set[str],
    established_models: set[str],
    novelty_bonus: float,
    fraction: float,
) -> float:
    """Downstream-value boost (#2): an arm whose evidence feeds/unblocks a
    blocked arm earns `fraction * value(blocked_arm)` per reachable blocked
    arm. This lets the selector choose the globally-useful move (growing the
    evidence supply toward the goal) instead of the locally-highest one, while
    the blocked goal arm sits non-actionable waiting on a human ruling."""
    if fraction <= 0.0:
        return 0.0
    dims = {d["id"]: d for d in registry["dimensions"]}
    total = 0.0
    for bid in reachable_blocked(registry, dim["id"]):
        bdim = dims.get(bid)
        if bdim is None:
            continue
        total += fraction * _eig_with_novelty(
            bdim, established_parts, established_models, novelty_bonus
        )[0]
    return total


def _score_dim(
    registry: Mapping[str, Any],
    dim: Mapping[str, Any],
    established_parts: set[str],
    established_models: set[str],
    novelty_bonus: float,
    downstream_fraction: float,
) -> tuple[float, float, float]:
    """Full selector score = base EIG + novelty bonus + downstream-value boost.

    Returns (value, novelty_applied, downstream_applied) — deterministic and
    consistent across the exploit path, the explore path and stall detection.
    """
    value, novel, applied = _eig_with_novelty(
        dim, established_parts, established_models, novelty_bonus
    )
    downstream = (
        _downstream_boost_for(
            registry, dim, established_parts, established_models,
            novelty_bonus, downstream_fraction,
        )
        if downstream_fraction > 0.0
        else 0.0
    )
    return value + downstream, applied, downstream


def select_next_arm(
    registry: Mapping[str, Any],
    *,
    at_selection_index: int | None = None,
) -> dict[str, Any]:
    """Return the highest-impact actionable proposal as the next active arm.

    Two exploration affordances (owner directive 2026-08-05):
    - **ε-greedy slot**: when `sweep_terms.exploration.every_n` > 0, every N-th
      selection forces the *lowest-prior / highest-uncertainty* actionable
      proposal instead of the max-EIG one (`selected_via: 'explore'`); all other
      selections exploit (`selected_via: 'exploit'`). The index defaults to the
      registry's `selection_progress` (bumped by run_tick after each selection).
    - **novelty bonus**: when `sweep_terms.exploration.novelty_bonus` > 0, a
      proposal whose deterministic signal names an evidence part *or* model class
      not already established by a terminal arm gets the bonus added to its EIG.
      This rewards genuinely-new axes (relational, temporal, reconstruction)
      instead of double-dipping validated artifacts.

    Still deterministic (ties broken by id) and fail-closed.
    """
    validate_registry(registry)
    actionable = [d for d in registry["dimensions"] if d["state"] in ACTIONABLE]
    if not actionable:
        raise AutonomousError("no actionable proposal — registry is terminal; run brainstorm-new-data")

    every_n, novelty_bonus = _exploration_config(registry)
    _, downstream_fraction = _downstream_config(registry)
    index = (
        int(registry.get("selection_progress", 0))
        if at_selection_index is None
        else at_selection_index
    )
    exploration_slot = every_n > 0 and (index + 1) % every_n == 0
    established_parts = validated_evidence_parts(registry)
    established_models = _established_models(registry)

    def _score(dim: Mapping[str, Any]) -> tuple[float, float, float]:
        # (value, novelty_applied, downstream_applied)
        return _score_dim(
            registry, dim, established_parts, established_models,
            novelty_bonus, downstream_fraction,
        )

    if exploration_slot:
        # Force the highest-uncertainty (lowest-prior) actionable proposal.
        chosen = min(actionable, key=lambda d: (_prior_number(d), d["id"]))
        value, applied, downstream = _score(chosen)
        return {
            "id": chosen["id"],
            "name": chosen["name"],
            "arm_issue": chosen["arm_issue"],
            "expected_information_gain": round(value, 4),
            "state": chosen["state"],
            "selection_rationale_recorded": True,
            "ties_broken_by": "id",
            "selected_via": "explore",
            "exploration_slot": True,
            "novelty_bonus_applied": round(applied, 4),
            "downstream_boost_applied": round(downstream, 4),
            "all_scores": [
                {"id": d["id"],
                 "expected_information_gain": round(_score(d)[0], 4),
                 "novelty_bonus_applied": round(_score(d)[1], 4),
                 "downstream_boost_applied": round(_score(d)[2], 4)}
                for d in sorted(actionable, key=lambda d: (_score(d)[0], d["id"]), reverse=True)
            ],
        }

    scored = sorted(
        ((_score(d), d) for d in actionable),
        key=lambda t: (t[0], t[1]["id"]),  # deterministic: ties broken by id, never compare dicts
    )
    (value, applied, downstream), chosen = scored[-1]
    chosen_id = chosen["id"]
    return {
        "id": chosen_id,
        "name": chosen["name"],
        "arm_issue": chosen["arm_issue"],
        "expected_information_gain": round(value, 4),
        "state": chosen["state"],
        "selection_rationale_recorded": True,
        "ties_broken_by": "id",
        "selected_via": "exploit",
        "exploration_slot": False,
        "novelty_bonus_applied": round(applied, 4),
        "downstream_boost_applied": round(downstream, 4),
        "all_scores": [
            {"id": d["id"],
             "expected_information_gain": round(_score(d)[0], 4),
             "novelty_bonus_applied": round(_score(d)[1], 4),
             "downstream_boost_applied": round(_score(d)[2], 4)}
            for (s, d) in sorted(scored, key=lambda t: (t[0][0], t[1]["id"]), reverse=True)
        ],
    }


def _top_actionable_eig(registry: Mapping[str, Any]) -> float:
    """Best (novelty+downstream-adjusted) EIG among actionable proposals — used
    by the selector-top-score-below stall trigger."""
    every_n, novelty_bonus = _exploration_config(registry)
    _, downstream_fraction = _downstream_config(registry)
    established_parts = validated_evidence_parts(registry)
    established_models = _established_models(registry)
    best = 0.0
    for d in registry.get("dimensions", []):
        if d.get("state") not in ACTIONABLE:
            continue
        value = _score_dim(
            registry, d, established_parts, established_models,
            novelty_bonus, downstream_fraction,
        )[0]
        best = max(best, value)
    return best


def _tick_stall_reason(registry: Mapping[str, Any]) -> str | None:
    """Combine history-based stall (sweep_status) and selector top-score-below.

    Returns a human reason when the loop should pause to brainstorm even though
    the safe menu is not yet terminal.
    """
    from .dimension_registry import _stall_reason_from_history

    reason = _stall_reason_from_history(registry)
    if reason is not None:
        return reason
    stall = registry.get("sweep_terms", {}).get("stall")
    if isinstance(stall, Mapping) and isinstance(stall.get("selector_top_score_below"), (int, float)) \
            and not isinstance(stall.get("selector_top_score_below"), bool):
        threshold = float(stall["selector_top_score_below"])
        top = _top_actionable_eig(registry)
        if top < threshold:
            return f"selector top score {top:.3f} below threshold {threshold:.3f}"
    return None


def _support_ratio(supported: int, unsupported: int) -> float:
    denom = supported + unsupported
    return supported / denom if denom else 0.0


def better_or_not(
    *,
    supported_base: int,
    supported_variant: int,
    unsupported_base: int,
    unsupported_variant: int,
    items: int,
    sign_test_p_supported: float,
    method: str = "claim-support",
    reconstruction_delta: float | None = None,
) -> dict[str, Any]:
    """Verdict: is the variant (declared evidence / context4k) better on the asset description?

    Rule (pre-registered, deterministic):
    - Must have a valid item count and method.
    - `claim-support`: BETTER iff sign_test_p_supported <= 0.05 AND the support
      ratio improved above a small epsilon AND unsupported did not balloon.
      Otherwise NOT_BETTER. Ambiguity (improvement but p > 0.05) is NOT_BETTER
      with an `inconclusive: true` note, never a fabricated PASS.
    - `reconstruction`: BETTER iff reconstruction_delta > 0.0 (CLIP similarity
      of variant-generated > base-generated) ; NOT_BETTER otherwise.
    """
    if items <= 0:
        raise AutonomousError("items must be a positive integer for a verdict")
    if not isinstance(sign_test_p_supported, (int, float)) or isinstance(sign_test_p_supported, bool):
        raise AutonomousError("sign_test_p_supported must be a number")
    if not (0.0 <= sign_test_p_supported <= 1.0):
        raise AutonomousError("sign_test_p_supported must be in [0,1]")
    if method not in ("claim-support", "reconstruction"):
        raise AutonomousError(f"unsupported method {method!r}")

    if method == "reconstruction":
        if reconstruction_delta is None:
            raise AutonomousError("reconstruction method requires reconstruction_delta")
        better = reconstruction_delta > 0.0
        return {
            "verdict": "BETTER" if better else "NOT_BETTER",
            "method": method,
            "reconstruction_delta": reconstruction_delta,
            "items": items,
            "note": "reconstruction similarity is a non-LLM generative proxy",
        }

    base_ratio = _support_ratio(supported_base, unsupported_base)
    variant_ratio = _support_ratio(supported_variant, unsupported_variant)
    improved = variant_ratio - base_ratio > 0.02
    ratio_unsup_sane = unsupported_variant <= unsupported_base + max(2, int(0.15 * unsupported_base))
    significant = sign_test_p_supported <= 0.05
    better = improved and significant and ratio_unsup_sane
    note = None
    if improved and not significant:
        note = "support ratio improved but sign test not significant (p>0.05)"
    return {
        "verdict": "BETTER" if better else "NOT_BETTER",
        "method": method,
        "items": items,
        "support_ratio_base": round(base_ratio, 4),
        "support_ratio_variant": round(variant_ratio, 4),
        "delta_support_ratio": round(variant_ratio - base_ratio, 4),
        "sign_test_p_supported": sign_test_p_supported,
        "significant": significant,
        "inconclusive": bool(improved and not significant),
        "note": note,
    }


def _binom_right_tail(n: int, k: int, p: float = 0.5) -> float:
    """One-sided binomial survival P(X >= k) with p=0.5 (sign test)."""
    from math import comb

    if n <= 0:
        return 1.0
    return sum(comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(max(0, k), n + 1))


def _derive_conditions_from_plan(review_dir: str) -> tuple[str, str] | None:
    """Find baseline/evidence condition ids from the run plan beside the review root.

    The run publishes `stage-b-plan.json` in the run root; the review root is
    typically `<run-root>-review`. If that plan is present, derive: baseline =
    the condition with the null evidence id (`no-specialist-evidence-v1`),
    evidence = the condition with a real specialist evidence id. Returns None
    if the plan cannot be located or parsed usefully.
    """
    from pathlib import Path

    review_root = Path(review_dir)
    candidates = [
        review_root / "stage-b-plan.json",
        review_root.parent / (review_root.name + "-plan.json"),
        review_root.parent / "stage-b-plan.json",
    ]
    # The run root is usually the parent sibling without the `-review` suffix.
    if review_root.name.endswith("-review"):
        run_root = review_root.parent / review_root.name[: -len("-review")]
        candidates.insert(0, run_root / "stage-b-plan.json")

    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            plan = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        conditions = plan.get("conditions") if isinstance(plan, dict) else None
        if not isinstance(conditions, list):
            continue
        baseline = None
        evidence = None
        for cond in conditions:
            if not isinstance(cond, dict):
                continue
            cid = cond.get("id")
            ev = cond.get("evidence")
            evid = ev.get("id") if isinstance(ev, dict) else None
            if not isinstance(cid, str):
                continue
            if evid == "no-specialist-evidence-v1" and "context" in cid:
                baseline = cid
            elif isinstance(evid, str) and evid and evid != "no-specialist-evidence-v1":
                evidence = cid
        if baseline and evidence:
            return baseline, evidence
    return None


def aggregate_claim_support(
    review_dir: str,
    *,
    baseline_condition: str | None = None,
    evidence_condition: str | None = None,
) -> dict[str, Any]:
    """Aggregate claim-support counts + paired sign-test from review files.

    Expects per-row JSONL with keys: condition_id, image_id, and the five
    score lists (supported / unsupported / omissions / contradictions /
    abstentions). Returns per-condition counts and a paired sign-test p-value
    for the evidence-vs-baseline supported-claim delta (matches the measured
    arm-#4 protocol). The baseline/evidence conditions are matched **exactly**
    so a legacy `*-raw-no-evidence` row can never substitute for the matched
    context baseline.

    When the conditions are not given explicitly, they are **derived from the
    run plan** (`stage-b-plan.json`, found next to the review root): the
    baseline is the condition with the null evidence id
    (`no-specialist-evidence-v1`) and the evidence condition is the one with a
    real specialist evidence id. This removes the per-arm hardcoding that once
    silently flipped body-type/other arms to NOT_BETTER by aggregating the
    wrong (geometry-only) condition columns.
    """
    from pathlib import Path

    if baseline_condition is None or evidence_condition is None:
        derived = _derive_conditions_from_plan(review_dir)
        if derived is not None:
            baseline_condition, evidence_condition = derived

    if baseline_condition is None or evidence_condition is None:
        raise AutonomousError(
            "aggregate_claim_support: could not determine baseline/evidence "
            "conditions; pass them explicitly or provide stage-b-plan.json "
            "beside the review root"
        )

    path = Path(review_dir) / "reviews.jsonl"
    if not path.is_file():
        raise AutonomousError(f"no reviews.jsonl found under {review_dir}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise AutonomousError(f"empty reviews.jsonl under {review_dir}")

    per_condition: dict[str, dict[str, int]] = defaultdict(
        lambda: {"items": 0, "supported": 0, "unsupported": 0, "omissions": 0,
                 "contradictions": 0, "abstentions": 0}
    )
    for r in rows:
        cid = r.get("condition_id", "?")
        a = per_condition[cid]
        a["items"] += 1
        a["supported"] += len(r.get("supported", []) or [])
        a["unsupported"] += len(r.get("unsupported", []) or [])
        a["omissions"] += len(r.get("omissions", []) or [])
        a["contradictions"] += len(r.get("contradictions", []) or [])
        a["abstentions"] += len(r.get("abstentions", []) or [])

    # Paired sign test on supported claims: evidence vs the exact matched baseline.
    by_item: dict[str, dict[str, int]] = defaultdict(dict)
    for r in rows:
        by_item[r.get("image_id")][r.get("condition_id")] = len(r.get("supported", []) or [])
    deltas = [conds[evidence_condition] - conds[baseline_condition]
              for item, conds in by_item.items()
              if evidence_condition in conds and baseline_condition in conds]
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    n_paired = pos + neg
    p = _binom_right_tail(n_paired, max(pos, neg)) if n_paired else 1.0

    return {
        "per_condition": {k: dict(v) for k, v in per_condition.items()},
        "baseline_supported": per_condition[baseline_condition]["supported"],
        "evidence_supported": per_condition[evidence_condition]["supported"],
        "baseline_unsupported": per_condition[baseline_condition]["unsupported"],
        "evidence_unsupported": per_condition[evidence_condition]["unsupported"],
        "paired_items": n_paired,
        "positive_delta_count": pos,
        "sign_test_p_supported": round(p, 6),
    }


def advance_dimension(registry: dict[str, Any], dim_id: str, *, state: str, strikes: int) -> dict[str, Any]:
    """Return a copy of the registry with one dimension's state/strikes updated."""
    validate_registry(registry)
    found = False
    for dim in registry["dimensions"]:
        if dim["id"] == dim_id:
            dim["state"] = state
            dim["valid_non_improving_experiments"] = strikes
            found = True
    if not found:
        raise AutonomousError(f"dimension {dim_id!r} not in registry")
    validate_registry(registry)
    return registry


def run_tick(
    registry: dict[str, Any],
    *,
    review_dir: str | None = None,
    method: str = "claim-support",
    reconstruction_delta: float | None = None,
    items: int | None = None,
) -> dict[str, Any]:
    """One loop iteration. Returns a next_action + (optionally) verdict.

    Flow:
    - If a dimension is already active: look for measured results in
      `review_dir` (`claim-support` method) or use a reconstruction CLIP delta
      (`reconstruction` method). When present, aggregate -> better_or_not ->
      advance the registry (BETTER => validated; NOT_BETTER => +1 strike,
      falsified at the limit) and then select the next arm. If absent,
      next_action is research-pending (the launchers are the executor) and we
      do not advance.
    - Every conclude records a `conclusion_history` entry and every selection
      bumps `selection_progress`, so the ε-greedy slot and stall detection are
      deterministic and atomic with the persisted registry.
    - If the sweep is stalled (see `_tick_stall_reason`) but not terminal, the
      next_action is `brainstorm-on-stall` — new ideas surface while the safe
      menu is still being worked, instead of only on full exhaustion.
    - If nothing is active: if the registry is terminal, next_action is
      brainstorm-new-data; otherwise activate the highest-impact proposal.
    """
    if method not in ("claim-support", "reconstruction"):
        raise AutonomousError(f"unsupported tick method {method!r}")
    if method == "reconstruction" and reconstruction_delta is None:
        raise AutonomousError("reconstruction method requires reconstruction_delta")
    validate_registry(registry)
    strike_limit: int = registry["sweep_terms"]["per_dimension_strike_limit"]
    terminal = set(registry["sweep_terms"]["terminal_states"])
    active = [d for d in registry["dimensions"] if d["state"] == "active"]
    if len(active) > 1:
        raise AutonomousError("more than one research:active arm — routing invariant violated")

    def _record_conclusion(arm_id: str, verdict: str, state: str, agg: Any) -> None:
        history = registry.setdefault("conclusion_history", [])
        history.append({
            "arm_id": arm_id,
            "verdict": verdict,
            "state": state,
            "cycle": len(history) + 1,
        })

    def _bump_progress() -> int:
        registry["selection_progress"] = int(registry.get("selection_progress", 0)) + 1
        return registry["selection_progress"]

    if active:
        arm = active[0]
        if review_dir is None and method == "claim-support":
            return {"next_action": "research-pending", "active_arm": arm["id"]}
        try:
            if method == "claim-support":
                agg = aggregate_claim_support(review_dir or "")
                verdict = better_or_not(
                    supported_base=agg["baseline_supported"],
                    supported_variant=agg["evidence_supported"],
                    unsupported_base=agg["baseline_unsupported"],
                    unsupported_variant=agg["evidence_unsupported"],
                    items=items or agg["paired_items"] or 24,
                    sign_test_p_supported=agg["sign_test_p_supported"],
                    method="claim-support",
                )
            else:
                agg = None
                verdict = better_or_not(
                    supported_base=0,
                    supported_variant=0,
                    unsupported_base=0,
                    unsupported_variant=0,
                    items=items or 24,
                    sign_test_p_supported=1.0,
                    method="reconstruction",
                    reconstruction_delta=reconstruction_delta,
                )
        except AutonomousError:
            return {"next_action": "research-pending", "active_arm": arm["id"]}
        strikes = arm["valid_non_improving_experiments"]
        if verdict["verdict"] == "BETTER":
            advance_dimension(registry, arm["id"], state="validated", strikes=strikes)
            _record_conclusion(arm["id"], verdict["verdict"], "validated", agg)
        else:
            strikes += 1
            new_state = "falsified" if strikes >= strike_limit else "active"
            advance_dimension(registry, arm["id"], state=new_state, strikes=strikes)
            _record_conclusion(arm["id"], verdict["verdict"], new_state, agg)
        # Select the next arm from the updated registry.
        proposals = [d for d in registry["dimensions"]
                     if d["state"] in ("proposal",) and d["id"] != arm["id"]]
        if proposals:
            stall_reason = _tick_stall_reason(registry)
            if stall_reason is not None:
                return {"next_action": "brainstorm-on-stall", "verdict": verdict,
                        "advanced_arm": arm["id"], "aggregate": agg,
                        "stall_reason": stall_reason,
                        "brainstorm_options": BRAINSTORM_OPTIONS}
            selection = select_next_arm(registry)
            advance_dimension(registry, selection["id"], state="active", strikes=0)
            _bump_progress()
            return {"next_action": "activate-next", "verdict": verdict,
                    "advanced_arm": arm["id"], "next_arm": selection["id"],
                    "aggregate": agg,
                    "selected_via": selection.get("selected_via", "exploit"),
                    "selection_progress": registry["selection_progress"]}
        if all(d["state"] in terminal for d in registry["dimensions"]):
            return {"next_action": "brainstorm-new-data", "verdict": verdict,
                    "advanced_arm": arm["id"], "aggregate": agg}
        return {"next_action": "complete", "verdict": verdict,
                "advanced_arm": arm["id"], "aggregate": agg}

    if all(d["state"] in terminal for d in registry["dimensions"]):
        return {"next_action": "brainstorm-new-data"}
    stall_reason = _tick_stall_reason(registry)
    if stall_reason is not None:
        return {"next_action": "brainstorm-on-stall", "stall_reason": stall_reason,
                "brainstorm_options": BRAINSTORM_OPTIONS}
    selection = select_next_arm(registry)
    advance_dimension(registry, selection["id"], state="active", strikes=0)
    _bump_progress()
    return {"next_action": "activate", "next_arm": selection["id"],
            "expected_information_gain": selection["expected_information_gain"],
            "selected_via": selection.get("selected_via", "exploit"),
            "selection_progress": registry["selection_progress"]}
