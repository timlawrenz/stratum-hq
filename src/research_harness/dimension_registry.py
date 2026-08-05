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

import hashlib
import json
import os
import tempfile
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

# Optional per-dimension fields (non-stratum specialists + validation methods)
# are validated structurally but are not required for a proposal.
VALIDATION_METHODS = ("claim-support", "reconstruction", "roundtrip-audit")


class DimensionRegistryError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _check_validation_methods(value: Any, dim_id: str) -> None:
    if value is None:
        return
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise DimensionRegistryError(f"dimension {dim_id!r} validation_methods must be a string list")
    unknown = set(value) - set(VALIDATION_METHODS)
    if unknown:
        raise DimensionRegistryError(
            f"dimension {dim_id!r} validation_methods has unsupported entries {sorted(unknown)}"
        )


def _check_specialists(value: Any, dim_id: str) -> None:
    if value is None:
        return
    if not isinstance(value, list) or not value:
        raise DimensionRegistryError(f"dimension {dim_id!r} specialists must be a non-empty list")
    for spec in value:
        if not isinstance(spec, Mapping):
            raise DimensionRegistryError(f"dimension {dim_id!r} specialist must be an object")
        for field in ("name", "source", "scope", "known_failure_modes"):
            if field not in spec:
                raise DimensionRegistryError(
                    f"dimension {dim_id!r} specialist missing field {field!r}"
                )


def _check_string_list(value: Any, dim_id: str, field: str) -> None:
    if value is None:
        return
    if not isinstance(value, list) or not all(
        isinstance(v, str) and v.strip() for v in value
    ):
        raise DimensionRegistryError(
            f"dimension {dim_id!r} {field} must be a list of non-empty strings"
        )


def _checked_float(value: Any, field: str, low: float, high: float, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DimensionRegistryError(f"{field} must be a number")
    if not (low <= value <= high):
        raise DimensionRegistryError(f"{field} must be in [{low}, {high}]")


def _check_exploration(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise DimensionRegistryError("sweep_terms.exploration must be an object")
    if "every_n" in value:
        v = value["every_n"]
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise DimensionRegistryError("sweep_terms.exploration.every_n must be a non-negative integer")
    if "novelty_bonus" in value:
        _checked_float(value["novelty_bonus"], "sweep_terms.exploration.novelty_bonus", 0.0, 10.0)


def _check_stall(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise DimensionRegistryError("sweep_terms.stall must be an object")
    if "no_validation_in_last" in value:
        v = value["no_validation_in_last"]
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise DimensionRegistryError("sweep_terms.stall.no_validation_in_last must be a positive integer")
    if "selector_top_score_below" in value:
        _checked_float(value["selector_top_score_below"], "sweep_terms.stall.selector_top_score_below", 0.0, 1.0)


def _check_conclusion_history(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        raise DimensionRegistryError("registry conclusion_history must be a list")
    for entry in value:
        if not isinstance(entry, Mapping):
            raise DimensionRegistryError("conclusion_history entries must be objects")
        for key in ("arm_id", "verdict", "state", "cycle"):
            if key not in entry:
                raise DimensionRegistryError(f"conclusion_history entry missing field {key!r}")
        if not isinstance(entry["cycle"], int) or isinstance(entry["cycle"], bool) or entry["cycle"] < 0:
            raise DimensionRegistryError("conclusion_history entry cycle must be a non-negative integer")


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
    _check_exploration(sweep_terms.get("exploration"))
    _check_stall(sweep_terms.get("stall"))

    progress = registry.get("selection_progress")
    if progress is not None and (
        not isinstance(progress, int) or isinstance(progress, bool) or progress < 0
    ):
        raise DimensionRegistryError("registry selection_progress must be a non-negative integer")
    _check_conclusion_history(registry.get("conclusion_history"))

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
        _check_validation_methods(dim.get("validation_methods"), dim_id)
        _check_specialists(dim.get("specialists"), dim_id)
        _check_string_list(dim.get("evidence_parts"), dim_id, "evidence_parts")
        _check_string_list(dim.get("model_candidates"), dim_id, "model_candidates")
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


TERMINAL_DEFAULTS = ("validated", "falsified", "exhausted")

BRAINSTORM_OPTIONS = [
    "new-data-source candidacy",
    "new-evidence-part candidacy",
    "new-model-candidate research (literature/arXiv scan): better model "
    "for an existing task, or a model covering an entirely new part",
]


def validated_evidence_parts(registry: Mapping[str, Any]) -> set[str]:
    """Evidence parts already established by terminal-state arms.

    Used by the selector's novelty bonus: a proposal whose deterministic
    signal reuses only these parts is 'known land'; naming a part outside
    this set is a genuinely new evidence axis.
    """
    terminal: set[str] = set(registry.get("sweep_terms", {}).get("terminal_states", TERMINAL_DEFAULTS))
    parts: set[str] = set()
    for dim in registry.get("dimensions", []):
        if dim.get("state") in terminal:
            parts.update(dim.get("evidence_parts") or [])
    return parts


def _stall_reason_from_history(registry: Mapping[str, Any]) -> str | None:
    """Return a stall reason if the last K conclusions contained no BETTER."""
    stall = registry.get("sweep_terms", {}).get("stall")
    if not isinstance(stall, Mapping):
        return None
    k = stall.get("no_validation_in_last")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        return None
    history = registry.get("conclusion_history") or []
    if len(history) < k:
        return None
    tail = history[-k:]
    if all(entry.get("verdict") != "BETTER" for entry in tail):
        return f"no validation in last {k} concluded cycles"
    return None


def sweep_status(registry: Mapping[str, Any]) -> dict[str, Any]:
    validate_registry(registry)
    terminal = set(registry["sweep_terms"]["terminal_states"])
    dims = registry["dimensions"]
    by_state: dict[str, int] = {}
    for dim in dims:
        by_state[dim["state"]] = by_state.get(dim["state"], 0) + 1
    exhausted = all(dim["state"] in terminal for dim in dims)
    stall_reason = _stall_reason_from_history(registry)
    stalled = stall_reason is not None and not exhausted
    if exhausted:
        next_action = "brainstorm-new-data"
        brainstorming = True
    elif stalled:
        next_action = "brainstorm-on-stall"
        brainstorming = True
    else:
        next_action = "none"
        brainstorming = False
    return {
        "total": len(dims),
        "by_state": by_state,
        "terminal": sum(by_state.get(s, 0) for s in terminal),
        "exhausted": exhausted,
        "stalled": stalled,
        "stall_reason": stall_reason,
        "next_action": next_action,
        # (owner directive 2026-08-05) exhaustion must not mean "stop": the
        # brainstorm step is the widening move — new data sources, new evidence
        # parts, AND new model candidates (literature/arXiv scan) for existing
        # tasks or entirely new parts. Stall (no validation in last K cycles)
        # also fires brainstorm-on-stall while the safe menu is still open.
        "brainstorm_options": BRAINSTORM_OPTIONS if brainstorming else [],
    }


def registry_sha256(path: Path) -> str:
    """Content hash of a registry file — used as an optimistic-concurrency token."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise DimensionRegistryError(f"unable to read registry {path} for hashing: {exc}") from exc
    return hashlib.sha256(raw).hexdigest()


def write_registry(
    path: Path,
    registry: Mapping[str, Any],
    *,
    expected_sha256: str | None = None,
) -> None:
    """Atomically persist a validated registry, optionally guarded against
    concurrent modification.

    When `expected_sha256` is provided, the write is refused (fail-closed)
    if the file on disk no longer hashes to that value — i.e. someone else
    advanced the registry between our load and this write. This makes the
    conclude+advance transition deterministic and atomic: a stale tick can
    never clobber a newer registry state.
    """
    validate_registry(registry)
    path = Path(path)
    if expected_sha256 is not None:
        if registry_sha256(path) != expected_sha256:
            raise DimensionRegistryError(
                f"registry {path} changed on disk since load; refusing to overwrite "
                "concurrent state (re-run tick against the latest registry)"
            )
    payload = json.dumps(dict(registry), ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


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
