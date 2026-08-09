"""Synchronize GitHub issue labels with the dimension-registry state.

The registry is the source of truth for arm state (proposal / active /
validated / falsified / exhausted). Each arm maps to a GitHub issue via
`arm_issue`. This module computes the *desired lifetime state label* per issue
and returns a minimal, idempotent diff against the labels currently on the
issue.

State label mapping:
  proposal   -> research:proposal
  active     -> research:active
  blocked    -> research:needs-human  (gate is a policy/authority decision)
  validated  -> research:validated
  falsified  -> research:postmortem
  exhausted  -> research:postmortem

Non-state labels (research, research:high-priority, research:metric-risk, ...)
are preserved as-is: only the state labels managed here are ever added/removed.
"""

from __future__ import annotations

from typing import Any, Mapping

from .dimension_registry import validate_registry

STATE_LABELS: set[str] = {
    "research:proposal",
    "research:active",
    "research:validated",
    "research:postmortem",
    "research:hold",
    "research:needs-human",
    "research:blocked",
}
_STATE_TO_LABEL: dict[str, str] = {
    "proposal": "research:proposal",
    "active": "research:active",
    "blocked": "research:needs-human",
    "validated": "research:validated",
    "falsified": "research:postmortem",
    "exhausted": "research:postmortem",
}


class IssueLabelError(RuntimeError):
    pass


def compute_desired_labels(registry: Mapping[str, Any]) -> dict[int, set[str]]:
    """Return {issue_number: {desired state label}} derived from the registry."""
    validate_registry(registry)
    desired: dict[int, set[str]] = {}
    for dim in registry["dimensions"]:
        issue = dim.get("arm_issue")
        if not isinstance(issue, int) or issue <= 0:
            raise IssueLabelError(
                f"dimension {dim['id']!r} has invalid arm_issue {issue!r}"
            )
        state = dim["state"]
        desired[issue] = {_STATE_TO_LABEL[state]}
    return desired


def plan_issue_label_sync(
    registry: Mapping[str, Any],
    current_by_issue: Mapping[int, set[str]],
) -> list[dict[str, Any]]:
    """Compute {add|remove} label ops for each arm issue; idempotent and minimal.

    `current_by_issue` maps issue number -> set of label names currently on it.
    Only labels in STATE_LABELS are ever removed; other labels (priority,
    research, metric-risk, ...) are preserved.
    """
    desired = compute_desired_labels(registry)
    operations: list[dict[str, Any]] = []
    for issue, wanted in desired.items():
        current = set(current_by_issue.get(issue) or set())
        state_labels = current & STATE_LABELS
        to_add = wanted - current
        to_remove = state_labels - wanted
        if to_add or to_remove:
            for label in sorted(to_add):
                operations.append({"issue": issue, "action": "add", "label": label})
            for label in sorted(to_remove):
                operations.append({"issue": issue, "action": "remove", "label": label})
    return operations
