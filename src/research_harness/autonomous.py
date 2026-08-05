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

from .dimension_registry import validate_registry

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


def select_next_arm(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Return the highest-impact actionable proposal as the next active arm."""
    validate_registry(registry)
    actionable = [d for d in registry["dimensions"] if d["state"] in ACTIONABLE]
    if not actionable:
        raise AutonomousError("no actionable proposal — registry is terminal; run brainstorm-new-data")
    scored = sorted((_arm_value(d), d) for d in actionable)
    (value, _chosen_id), chosen = scored[-1]
    return {
        "id": chosen["id"],
        "name": chosen["name"],
        "arm_issue": chosen["arm_issue"],
        "expected_information_gain": round(value, 4),
        "state": chosen["state"],
        "selection_rationale_recorded": True,
        "ties_broken_by": "id",
        "all_scores": [
            {"id": d["id"], "expected_information_gain": round(_arm_value(d)[0], 4)}
            for _, d in sorted(scored, key=lambda t: t[0], reverse=True)
        ],
    }


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


def aggregate_claim_support(
    review_dir: str,
    *,
    baseline_condition: str = "context-raw-no-evidence",
    evidence_condition: str = "context-raw-geometry",
) -> dict[str, Any]:
    """Aggregate claim-support counts + paired sign-test from review files.

    Expects per-row JSONL with keys: condition_id, image_id, and the five
    score lists (supported / unsupported / omissions / contradictions /
    abstentions). Returns per-condition counts and a paired sign-test p-value
    for the evidence-vs-baseline supported-claim delta (matches the measured
    arm-#4 protocol). The baseline/evidence conditions are matched **exactly**
    so a legacy `*-raw-no-evidence` row can never substitute for the matched
    context baseline.
    """
    from pathlib import Path

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


def run_tick(registry: dict[str, Any], *, review_dir: str | None = None) -> dict[str, Any]:
    """One loop iteration. Returns a next_action + (optionally) verdict.

    Flow:
    - If a dimension is already active: look for measured results in
      `review_dir`. If present, aggregate -> better_or_not -> advance the
      registry (BETTER => validated; NOT_BETTER => +1 strike, falsified at the
      limit) and then select the next arm. If absent, next_action is
      research-pending (the launchers are the executor) and we do not advance.
    - If nothing is active: if the registry is terminal, next_action is
      brainstorm-new-data; otherwise activate the highest-impact proposal.
    """
    validate_registry(registry)
    strike_limit: int = registry["sweep_terms"]["per_dimension_strike_limit"]
    terminal = set(registry["sweep_terms"]["terminal_states"])
    active = [d for d in registry["dimensions"] if d["state"] == "active"]
    if len(active) > 1:
        raise AutonomousError("more than one research:active arm — routing invariant violated")

    if active:
        arm = active[0]
        if review_dir is None:
            return {"next_action": "research-pending", "active_arm": arm["id"]}
        try:
            agg = aggregate_claim_support(review_dir)
        except AutonomousError:
            return {"next_action": "research-pending", "active_arm": arm["id"]}
        verdict = better_or_not(
            supported_base=agg["baseline_supported"],
            supported_variant=agg["evidence_supported"],
            unsupported_base=agg["baseline_unsupported"],
            unsupported_variant=agg["evidence_unsupported"],
            items=agg["paired_items"] or 24,
            sign_test_p_supported=agg["sign_test_p_supported"],
            method="claim-support",
        )
        strikes = arm["valid_non_improving_experiments"]
        if verdict["verdict"] == "BETTER":
            advance_dimension(registry, arm["id"], state="validated", strikes=strikes)
        else:
            strikes += 1
            new_state = "falsified" if strikes >= strike_limit else "active"
            advance_dimension(registry, arm["id"], state=new_state, strikes=strikes)
        # Select the next arm from the updated registry.
        proposals = [d for d in registry["dimensions"]
                     if d["state"] in ("proposal",) and d["id"] != arm["id"]]
        if proposals:
            selection = select_next_arm(registry)
            advance_dimension(registry, selection["id"], state="active", strikes=0)
            return {"next_action": "activate-next", "verdict": verdict,
                    "advanced_arm": arm["id"], "next_arm": selection["id"],
                    "aggregate": agg}
        if all(d["state"] in terminal for d in registry["dimensions"]):
            return {"next_action": "brainstorm-new-data", "verdict": verdict,
                    "advanced_arm": arm["id"], "aggregate": agg}
        return {"next_action": "complete", "verdict": verdict,
                "advanced_arm": arm["id"], "aggregate": agg}

    if all(d["state"] in terminal for d in registry["dimensions"]):
        return {"next_action": "brainstorm-new-data"}
    selection = select_next_arm(registry)
    advance_dimension(registry, selection["id"], state="active", strikes=0)
    return {"next_action": "activate", "next_arm": selection["id"],
            "expected_information_gain": selection["expected_information_gain"]}
