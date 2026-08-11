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

DIMENSION_STATES = (
    "proposal",
    "active",
    "blocked",
    "validated",
    "falsified",
    "exhausted",
)
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


def _check_downstream_boost(value: Any) -> None:
    """sweep_terms.downstream_boost = {enabled, fraction}: the selector adds
    `fraction * value(blocked_arm)` to any actionable arm whose evidence
    feeds/unblocks a blocked arm (dependency-graph weighting, #2)."""
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise DimensionRegistryError("sweep_terms.downstream_boost must be an object")
    if "enabled" in value and not isinstance(value["enabled"], bool):
        raise DimensionRegistryError("sweep_terms.downstream_boost.enabled must be a boolean")
    if "fraction" in value:
        _checked_float(value["fraction"], "sweep_terms.downstream_boost.fraction", 0.0, 1.0)


def _check_goal_floors(value: Any) -> None:
    """Top-level goal_floors: the goal arm's pre-registered token floors
    (mirrors program.json representation); used by goal_reachability (#3)."""
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise DimensionRegistryError("registry goal_floors must be an object")
    for key in ("expanded_dossier_min_tokens", "compact_context_min_tokens"):
        if key not in value:
            raise DimensionRegistryError(f"registry goal_floors missing {key!r}")
        v = value[key]
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise DimensionRegistryError(f"registry goal_floors.{key} must be a positive integer")


def _check_evidence_budget(value: Any) -> None:
    """Top-level evidence_budget: the MEASURED per-item token budget the
    validated evidence can honestly produce (from the expansion-ceiling audit).
    goal_reachability (#3) compares it against goal_floors deterministically."""
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise DimensionRegistryError("registry evidence_budget must be an object")
    if "basis" not in value or not isinstance(value["basis"], str) or not value["basis"].strip():
        raise DimensionRegistryError("registry evidence_budget.basis must be a non-empty string")
    for key in (
        "deterministic_min_tokens_per_item",
        "deterministic_median_tokens_per_item",
        "deterministic_max_tokens_per_item",
        "honest_ceiling_max_tokens_per_item",
    ):
        if key not in value:
            raise DimensionRegistryError(f"registry evidence_budget missing {key!r}")
        _checked_float(value[key], f"registry evidence_budget.{key}", 0.0, 10_000_000.0)


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
    _check_downstream_boost(sweep_terms.get("downstream_boost"))

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
        _check_string_list(dim.get("feeds"), dim_id, "feeds")
        _check_string_list(dim.get("unblocks"), dim_id, "unblocks")
        if dim["state"] == "blocked":
            reason = dim.get("blocked_reason")
            if not isinstance(reason, str) or not reason.strip():
                raise DimensionRegistryError(
                    f"dimension {dim_id!r} is blocked but has no non-empty blocked_reason"
                )
            issue = dim.get("blocked_by_issue")
            if issue is not None and (
                not isinstance(issue, int) or isinstance(issue, bool) or issue <= 0
            ):
                raise DimensionRegistryError(
                    f"dimension {dim_id!r} blocked_by_issue must be a positive issue number"
                )
        budget = dim.get("token_budget_per_item")
        if budget is not None and (
            not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget < 0
        ):
            raise DimensionRegistryError(
                f"dimension {dim_id!r} token_budget_per_item must be a non-negative number"
            )

    # Dependency graph (feeds/unblocks): refs must exist, never self-refer, and
    # must not form a cycle — the selector traverses these edges, so a cycle
    # would make downstream weighting ill-defined.
    edges: dict[str, set[str]] = {}
    for dim in dims:
        d_id = dim["id"]
        edges[d_id] = set((dim.get("feeds") or []) + (dim.get("unblocks") or []))
        for ref in edges[d_id]:
            if ref == d_id:
                raise DimensionRegistryError(
                    f"dimension {d_id!r} dependency edges cannot reference itself"
                )
            if ref not in seen:
                raise DimensionRegistryError(
                    f"dimension {d_id!r} dependency edge references unknown dimension {ref!r}"
                )
    _check_dependency_acyclic(edges)

    goal_arm = registry.get("goal_arm")
    if goal_arm is not None:
        if not isinstance(goal_arm, str) or goal_arm not in seen:
            raise DimensionRegistryError("registry goal_arm must be a registered dimension id")
    _check_goal_floors(registry.get("goal_floors"))
    _check_evidence_budget(registry.get("evidence_budget"))


def _check_dependency_acyclic(edges: Mapping[str, set[str]]) -> None:
    """Raise DimensionRegistryError if the feeds/unblocks edges contain a cycle
    (iterative DFS with WHITE/GRAY/BLACK coloring — no recursion-depth issues)."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in edges}

    def _visit(node: str) -> None:
        color[node] = GRAY
        for nxt in sorted(edges.get(node, ())):
            if color[nxt] == GRAY:
                raise DimensionRegistryError(
                    f"dependency graph contains a cycle through {node!r} -> {nxt!r}"
                )
            if color[nxt] == WHITE:
                _visit(nxt)
        color[node] = BLACK

    for node in sorted(edges):
        if color[node] == WHITE:
            _visit(node)


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
        "blocked": by_state.get("blocked", 0),
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
        # (#3) program-level gate: when the goal arm's registered expanded-floor
        # is structurally unreachable from the measured evidence budget, emit
        # goal_unreachable + the measured gap and route to the AUTONOMOUS
        # "grow-evidence-supply" action, while flagging the floor
        # renegotiation (option A) as the HUMAN decision (needs-human hold).
        "goal_reachability": goal_reachability(registry),
        # (#2) the actionable arms that would feed/unblock a blocked arm — the
        # globally-useful moves the selector should prefer.
        "dependency_frontier": dependency_frontier(registry),
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


# ---------------------------------------------------------------------------
# #1 - #4: blocked state, dependency graph, goal unreachability, program overview
# ---------------------------------------------------------------------------


def blocked_dimensions(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Dimensions parked in the non-actionable `blocked` state (their gate is a
    policy/authority decision, not a measurement — e.g. waiting on a human
    A/B ruling). Blocked arms are excluded from selector scoring, so the loop
    stops re-electing a stuck integrator and picks the best proposal instead."""
    validate_registry(registry)
    return [d for d in registry["dimensions"] if d["state"] == "blocked"]


def goal_arm_id(registry: Mapping[str, Any]) -> str | None:
    """The registered program-level goal arm (e.g. dossier-context4k), if any."""
    validate_registry(registry)
    return registry.get("goal_arm")


def _downstream_edges(registry: Mapping[str, Any]) -> dict[str, set[str]]:
    """feeds/unblocks edges as {dim_id: {downstream dim ids}} (a validated DAG)."""
    edges: dict[str, set[str]] = {}
    for d in registry["dimensions"]:
        edges[d["id"]] = set((d.get("feeds") or []) + (d.get("unblocks") or []))
    return edges


def reachable_blocked(registry: Mapping[str, Any], dim_id: str) -> set[str]:
    """Blocked dimensions reachable from `dim_id` along feeds/unblocks edges
    (BFS — the graph is validated acyclic). Used by the selector's downstream
    boost (#2) and the dependency frontier."""
    validate_registry(registry)
    known = {d["id"] for d in registry["dimensions"]}
    if dim_id not in known:
        raise DimensionRegistryError(f"unknown dimension id {dim_id!r}")
    blocked = {d["id"] for d in registry["dimensions"] if d["state"] == "blocked"}
    if dim_id in blocked:
        return {dim_id}
    edges = _downstream_edges(registry)
    seen: set[str] = set()
    stack = [dim_id]
    while stack:
        node = stack.pop()
        for nxt in edges.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen & blocked


def goal_reachability(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Program-level signal (#3): is the goal arm's registered expanded-dossier
    floor structurally reachable from the measured evidence budget?

    Only fires when BOTH `goal_floors` and a measured `evidence_budget` are
    registered; otherwise `declared: false` (fail-soft — never invent a
    measurement). Routing split (owner directive 2026-08-06):
      - grow-evidence-supply  (option B) is AUTONOMOUS work — the loop may
        pursue it on its own (the blocked goal arm is excluded + feeders are
        boosted by the selector);
      - floor-renegotiation   (option A) is a HUMAN decision (needs-human hold).
    """
    validate_registry(registry)
    floors = registry.get("goal_floors")
    budget = registry.get("evidence_budget")
    if not isinstance(floors, Mapping) or not isinstance(budget, Mapping):
        return {"declared": False, "goal_unreachable": False}
    floor = int(floors["expanded_dossier_min_tokens"])
    validated_budget = float(budget.get("deterministic_max_tokens_per_item", 0.0))
    ceiling = float(budget.get("honest_ceiling_max_tokens_per_item", validated_budget))
    unreachable = ceiling < float(floor)
    return {
        "declared": True,
        "goal_arm": goal_arm_id(registry),
        "floor_tokens": floor,
        "validated_budget_tokens": validated_budget,
        "honest_ceiling_tokens": ceiling,
        "measured_gap_tokens": round(float(floor) - validated_budget, 1),
        "goal_unreachable": bool(unreachable),
        "route_to": "grow-evidence-supply" if unreachable else "none",
        "requires_human": ["floor-renegotiation"] if unreachable else [],
        "basis": budget.get("basis", ""),
    }


def dependency_frontier(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Actionable arms whose evidence feeds/unblocks a blocked arm — the moves
    that keep the loop productive while the gate waits on a human decision."""
    validate_registry(registry)
    actionable = {"proposal", "active"}
    blocked_ids = {d["id"] for d in blocked_dimensions(registry)}
    if not blocked_ids:
        return []
    frontier: list[dict[str, Any]] = []
    for d in registry["dimensions"]:
        if d["state"] not in actionable:
            continue
        downstream = reachable_blocked(registry, d["id"]) & blocked_ids
        if downstream:
            frontier.append({
                "id": d["id"],
                "name": d.get("name", ""),
                "state": d["state"],
                "downstream_blocked": sorted(downstream),
            })
    return sorted(frontier, key=lambda f: f["id"])


def program_overview(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Program-state readout (#4) for the strategist's step-back: validated
    evidence budget vs the floor, % of the goal arm's inputs validated, blocked
    count, dependency frontier, and goal reachability — so every cycle starts
    from the whole picture instead of rediscovering it from issue prose."""
    validate_registry(registry)
    terminal = set(registry["sweep_terms"]["terminal_states"])
    dims = registry["dimensions"]
    by_state: dict[str, int] = {}
    for d in dims:
        by_state[d["state"]] = by_state.get(d["state"], 0) + 1
    blocked = blocked_dimensions(registry)
    goal = goal_arm_id(registry)
    goal_feeders: list[str] = []
    inputs_validated_pct = None
    if goal is not None:
        for d in dims:
            if goal in set((d.get("feeds") or []) + (d.get("unblocks") or [])):
                goal_feeders.append(d["id"])
        if goal_feeders:
            validated_feeder_count = sum(
                1
                for d in dims
                if d["id"] in set(goal_feeders) and d["state"] in terminal
            )
            inputs_validated_pct = round(100.0 * validated_feeder_count / len(goal_feeders), 1)
    return {
        "total": len(dims),
        "goal_arm": goal,
        "by_state": by_state,
        "terminal": sum(by_state.get(s, 0) for s in terminal),
        "blocked_count": len(blocked),
        "blocked_arms": [
            {
                "id": d["id"],
                "arm_issue": d.get("arm_issue"),
                "blocked_reason": d.get("blocked_reason"),
                "blocked_by_issue": d.get("blocked_by_issue"),
            }
            for d in blocked
        ],
        "goal_feeders": sorted(goal_feeders),
        "goal_inputs_validated_pct": inputs_validated_pct,
        "dependency_frontier": dependency_frontier(registry),
        "goal_reachability": goal_reachability(registry),
    }


def mark_dimension_blocked(
    registry: dict[str, Any],
    dim_id: str,
    reason: str,
    issue: int | None = None,
) -> dict[str, Any]:
    """Transition a dimension to the non-actionable `blocked` state: its gate is
    a policy/authority decision, NOT a measurement (e.g. a research:needs-human
    ruling). Requires a human-readable reason and (optionally) the issue number
    the arm is waiting on. Preserves strikes and all other fields."""
    validate_registry(registry)
    if not isinstance(reason, str) or not reason.strip():
        raise DimensionRegistryError("mark_dimension_blocked requires a non-empty reason")
    if issue is not None and (not isinstance(issue, int) or isinstance(issue, bool) or issue <= 0):
        raise DimensionRegistryError("blocked_by_issue must be a positive issue number")
    found = next((d for d in registry["dimensions"] if d["id"] == dim_id), None)
    if found is None:
        raise DimensionRegistryError(f"unknown dimension id {dim_id!r}")
    found["state"] = "blocked"
    found["blocked_reason"] = reason
    if issue is not None:
        found["blocked_by_issue"] = issue
    else:
        found.pop("blocked_by_issue", None)
    validate_registry(registry)
    return registry


def mark_dimension_unblocked(
    registry: dict[str, Any],
    dim_id: str,
    *,
    state: str = "proposal",
) -> dict[str, Any]:
    """Return a blocked dimension to an actionable state (default `proposal`,
    so the selector re-evaluates it on its merits once the gate resolves)."""
    validate_registry(registry)
    if state not in ("proposal", "active"):
        raise DimensionRegistryError("mark dim unblocked state must be 'proposal' or 'active'")
    found = next((d for d in registry["dimensions"] if d["id"] == dim_id), None)
    if found is None:
        raise DimensionRegistryError(f"unknown dimension id {dim_id!r}")
    if found["state"] != "blocked":
        raise DimensionRegistryError(f"dimension {dim_id!r} is not blocked")
    found["state"] = state
    found.pop("blocked_reason", None)
    found.pop("blocked_by_issue", None)
    validate_registry(registry)
    return registry
