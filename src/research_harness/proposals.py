"""Gated idea generation: `propose_dimensions` (propose-dimensions CLI).

Turns idea generation into a first-class, deterministic step instead of an LLM
aside. The loop drafts candidate dimensions (the open-world creative act), but
they only enter the sweep as `proposal` arms if they pass a structural gate:

- every candidate must declare the structural fields the registry requires AND
  the specialist declaration contract (scope / inputs / output_semantics /
  provenance / abstention_policy / qualification_gate);
- a `count` of new dimensions is required — generation fails closed if fewer
  pass;
- with `require_new_evidence_part`, candidates must name a NEW evidence part
  (not already established by a validated/falsified/exhausted arm) or a NEW
  model class — seed diversity is deliberate, so the loop pushes toward
  relational/interaction, temporal/sequence, or reconstruction axes rather than
  "yet another attribute tagger".

The candidates are registered as `proposal` with the required fields BEFORE
`select_next_arm` runs, so generation is a gated artifact, not strategist
discretion.
"""

from __future__ import annotations

from typing import Any, Mapping

from .dimension_registry import validate_registry, validated_evidence_parts

# Registry-structural fields every dimension needs (state + strikes are set
# by the gate itself — a proposal is never admitted by hand-editing).
REGISTRY_PROPOSAL_FIELDS = (
    "id",
    "name",
    "arm_issue",
    "hypothesis",
    "falsified_if",
    "deterministic_signal",
    "metric_version",
    "data_snapshot",
    "selection_rationale",
)

# Specialist declaration contract (program.json specialists.required_declaration_fields).
DECLARATION_FIELDS = (
    "scope",
    "inputs",
    "output_semantics",
    "provenance",
    "abstention_policy",
    "qualification_gate",
)

# Optional-but-scored fields for the selector's EIG model.
SCORED_FIELDS = (
    "prior_evidence_strength",
    "measurability",
    "cost_bucket",
)

REQUIRED_PROPOSAL_FIELDS = REGISTRY_PROPOSAL_FIELDS + DECLARATION_FIELDS


class ProposalGateError(RuntimeError):
    pass


def _missing_fields(candidate: Mapping[str, Any]) -> list[str]:
    return [field for field in REQUIRED_PROPOSAL_FIELDS if field not in candidate]


def _established_models(registry: Mapping[str, Any]) -> set[str]:
    from .autonomous import _established_models as _autonomous_established

    return _autonomous_established(registry)


def propose_dimensions(
    registry: dict[str, Any],
    candidates: Any,
    *,
    count: int,
    require_new_evidence_part: bool = False,
) -> dict[str, Any]:
    """Validate + register `count` new candidate dimensions as `proposal` arms.

    Raises `ProposalGateError` (fail-closed) when fewer than `count` candidates
    pass the gate, so `select_next_arm` can never run on an admitted-but-empty
    idea channel. Returns registered/rejected ids (rejected carries reasons).
    """
    validate_registry(registry)
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ProposalGateError(f"propose-dimensions: count must be a positive integer, got {count!r}")
    if not isinstance(candidates, list):
        raise ProposalGateError("propose-dimensions: candidates must be a JSON array of dimension objects")
    if len(candidates) < count:
        raise ProposalGateError(
            f"propose-dimensions: required {count} new dimension(s) but only "
            f"{len(candidates)} candidate(s) provided (count unmet)"
        )

    existing_ids = {d["id"] for d in registry["dimensions"]}
    existing_names = {d["name"] for d in registry["dimensions"]}
    established_parts = validated_evidence_parts(registry)
    established_models = _established_models(registry)

    registered: list[str] = []
    rejected: list[dict[str, Any]] = []

    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            rejected.append({"id": None, "reason": "candidate is not an object"})
            continue
        cid = candidate.get("id")
        missing = _missing_fields(candidate)
        if missing:
            rejected.append({"id": cid, "reason": f"missing required field(s): {', '.join(missing)}"})
            continue
        if candidate["id"] in existing_ids:
            rejected.append({"id": cid, "reason": "duplicate id already in registry"})
            continue
        if candidate["name"] in existing_names:
            rejected.append({"id": cid, "reason": "duplicate name already in registry"})
            continue
        if require_new_evidence_part:
            new_parts = set(candidate.get("evidence_parts") or [])
            new_models = set(candidate.get("model_candidates") or [])
            is_novel = bool((new_parts - established_parts) or (new_models - established_models))
            if not is_novel:
                rejected.append({
                    "id": cid,
                    "reason": (
                        "candidate does not name a new evidence part or a new model class "
                        "(redundant-axis seed-diversity gate); reuse of already-validated axes is rejected"
                    ),
                })
                continue

        dim = dict(candidate)
        dim["state"] = "proposal"
        dim["valid_non_improving_experiments"] = 0
        registry["dimensions"].append(dim)
        existing_ids.add(dim["id"])
        existing_names.add(dim["name"])
        registered.append(dim["id"])

    validate_registry(registry)
    if len(registered) < count:
        raise ProposalGateError(
            f"propose-dimensions: required {count} new dimension(s) but only "
            f"{len(registered)} registered: {registered}; rejected: {rejected}"
        )
    return {"registered": registered, "rejected": rejected, "count": count}
