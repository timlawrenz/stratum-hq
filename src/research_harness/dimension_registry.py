"""Evidence-dimension registry + convex-landscape sweep.

The registry is the machine-readable source of truth for the "endless list" of
evidence dimensions (clothing, hair, makeup, skin color, texture, mood,
lighting, setting, body type, ...). Sweeping it enumerates candidate arms in a
bounded, falsifiable space; when every dimension reaches a terminal state
(validated / falsified / exhausted) the swing reports EXHAUSTED and routes to
`brainstorm-new-data` — a harness state for proposing *new* dimensions or new
data sources rather than inventing variants of the same space.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

DIMENSION_STATES = ("proposal", "active", "validated", "falsified", "exhausted")
REQUIRED_DIMENSION_FIELDS = (
    "id",
    "name",
    "arm_issue",
    "state",
    "valid_non_improving_experiments",
    "hypothesis",
    "falsified_if",
    "deterministic_signal",
    "metric_version",
    "data_snapshot",
    "selection_rationale",
)


class DimensionRegistryError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_registry(registry: Mapping[str, Any]) -> None:
    if registry.get("schema_version") != 1:
        raise DimensionRegistryError("dimension registry schema_version must be 1")
    if registry.get("program_id") != "stratum-contextual-specialist-research":
        raise DimensionRegistryError("dimension registry program_id must match the research program")
    sweep_terms = registry.get("sweep_terms")
    if not isinstance(sweep_terms, Mapping):
        raise DimensionRegistryError("registry sweep_terms must be an object")
    terminal = sweep_terms.get("terminal_states")
    if not isinstance(terminal, list) or not all(isinstance(t, str) for t in terminal):
        raise DimensionRegistryError("sweep_terms.terminal_states must be a string list")
    raw_strike = sweep_terms.get("per_dimension_strike_limit")
    if not isinstance(raw_strike, int) or isinstance(raw_strike, bool) or raw_strike <= 0:
        raise DimensionRegistryError("sweep_terms.per_dimension_strike_limit must be a positive integer")
    strike_limit: int = raw_strike
    brainstorm = sweep_terms.get("brainstorm_states")
    if not isinstance(brainstorm, list) or "brainstorm-new-data" not in brainstorm:
        raise DimensionRegistryError("sweep_terms.brainstorm_states must include brainstorm-new-data")

    dims = registry.get("dimensions")
    if not isinstance(dims, list) or not dims:
        raise DimensionRegistryError("registry dimensions must be a non-empty list")
    seen: set[str] = set()
    for dim in dims:
        if not isinstance(dim, Mapping):
            raise DimensionRegistryError("each dimension must be a JSON object")
        for field in REQUIRED_DIMENSION_FIELDS:
            if field not in dim:
                raise DimensionRegistryError(f"dimension missing required field {field!r}")
        dim_id = dim["id"]
        if not isinstance(dim_id, str) or not dim_id.strip():
            raise DimensionRegistryError("dimension id must be a non-empty string")
        if dim_id in seen:
            raise DimensionRegistryError(f"duplicate dimension id {dim_id!r}")
        seen.add(dim_id)
        if dim["state"] not in DIMENSION_STATES:
            raise DimensionRegistryError(f"dimension {dim_id!r} has invalid state {dim['state']!r}")
        strikes = dim["valid_non_improving_experiments"]
        if not isinstance(strikes, int) or isinstance(strikes, bool) or strikes < 0:
            raise DimensionRegistryError(f"dimension {dim_id!r} strikes must be a non-negative integer")
        if strikes > strike_limit:
            raise DimensionRegistryError(f"dimension {dim_id!r} has {strikes} strikes > limit {strike_limit}")
        if strikes == strike_limit and dim["state"] not in terminal:
            raise DimensionRegistryError(
                f"dimension {dim_id!r} is striked-out but state {dim['state']!r} is not terminal"
            )
        if not isinstance(dim.get("arm_issue"), int) or dim["arm_issue"] <= 0:
            raise DimensionRegistryError(f"dimension {dim_id!r} arm_issue must be a positive issue number")


def load_registry(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DimensionRegistryError(f"unable to read registry {path}: {exc}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DimensionRegistryError(f"invalid JSON registry {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DimensionRegistryError("dimension registry must be a JSON object")
    validate_registry(value)
    return value


def sweep_status(registry: Mapping[str, Any]) -> dict[str, Any]:
    validate_registry(registry)
    terminal = set(registry["sweep_terms"]["terminal_states"])
    dims = registry["dimensions"]
    by_state: dict[str, int] = {}
    for dim in dims:
        by_state[dim["state"]] = by_state.get(dim["state"], 0) + 1
    exhausted = all(dim["state"] in terminal for dim in dims)
    return {
        "total": len(dims),
        "by_state": by_state,
        "terminal": sum(by_state.get(s, 0) for s in terminal),
        "exhausted": exhausted,
        "next_action": "brainstorm-new-data" if exhausted else "none",
    }


def _meta(registry: Mapping[str, Any]) -> str:
    header = "<!-- research-harness:\n"
    footer = "\n-->"
    return header + _canonical_json(registry) + footer


def render_arm_issue(registry: Mapping[str, Any], dim_id: str) -> str:
    validate_registry(registry)
    dim = next((d for d in registry["dimensions"] if d["id"] == dim_id), None)
    if dim is None:
        raise DimensionRegistryError(f"unknown dimension id {dim_id!r}")
    body = (
        f"# Arm: {dim['name']} evidence specialist\n\n"
        f"## Machine-readable metadata\n{_meta(registry)}\n\n"
        f"## Hypothesis\n{dim['hypothesis']}\n\n"
        f"## Falsified if\n{dim['falsified_if']}\n\n"
        f"## Deterministic signal\n{dim['deterministic_signal']}\n\n"
        f"## Metric / data snapshot\n- metric_version: {dim['metric_version']}\n"
        f"- data_snapshot: {dim['data_snapshot']}\n\n"
        f"## Selection rationale\n{dim['selection_rationale']}\n"
    )
    return body
