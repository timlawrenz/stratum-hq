# Selector improvements: blocked state, dependency graph, goal-unreachability, program overview (2026-08-06)

**Status:** draft PR — harness-code branch, base = experiment-worktree HEAD lineage.
**Decision record for improvement set #1-#4.**

## Problem (the stall mechanism, exactly)

`research:active` was a *blocking binary*. A `research:active` arm whose gate is a
policy/authority decision (not a measurement) was still scored by `select_next_arm`
because `ACTIONABLE = ("proposal", "active")`. Arm #36 (dossier-context4k, EIG 0.34)
was therefore re-elected every cycle while decision #46 (the A/B ruling: reframe the
round-trip scale vs grow the evidence supply) sat as `research:needs-human`. The loop
correctly produced no-op cycles, but it could not *make progress on the option-B path*
(#47 VLM dense description, #34 setting, #35 texture) because selection always returned
the stuck integrator.

The defect is program-level: it can be read as "waiting for the human ruling" while the
measured facts (honest ceiling 13.5K tok/item << 100K floor) already point at a useful
autonomous action. Four improvements, per owner direction 2026-08-06.

## #1 — `blocked` state (non-terminal, non-actionable)

- `DIMENSION_STATES` gains `"blocked"`. A blocked arm's gate is a policy/authority
  decision, NOT a measurement, so it is excluded from `_arm_value` scoring.
- Validator requires `blocked_reason` (non-empty) when `state == "blocked"`; optional
  `blocked_by_issue` (positive int).
- Issue-label mapping: `blocked -> research:needs-human` (`issue_labels.py`).
- Transitions via deterministic CLI: `mark-blocked <registry> <dim> --reason "..."
  [--issue N] [--write]` and `mark-unblocked <registry> <dim> [--state proposal|active]
  [--write]` (atomic, SHA-guarded).
- One-active invariant preserved: blocked is not active, so `run_tick` re-selects the
  best *proposal* instead of returning `research-pending` on the stuck arm.

**Live effect:** #36 → `blocked` (gate = #46 ruling, `blocked_by_issue: 46`). The next
`autonomous-tick` activates the best proposal instead of re-electing #36.

## #2 — dependency graph (`feeds`/`unblocks`) + downstream weighting

- Optional per-dimension `feeds: [dim_ids]` (this arm's evidence feeds those arms) and
  `unblocks: [dim_ids]` (this arm resolves their gate). Validated: refs must exist,
  no self-ref, graph must be a DAG (cycle check).
- `sweep_terms.downstream_boost = {enabled, fraction}` (default fraction 0.5,
  disabled when flag off). Selector adds
  `fraction * value(blocked_arm)` per blocked arm reachable along feeds/unblocks edges
  (transitive). Reported as `downstream_boost_applied` in the selection + `all_scores`.
- An arm that feeds the blocked goal arm now scores above its isolated EIG — the
  selector prefers the globally-useful move (grow evidence toward the floor).

**Live effect:** #47 (vlm-dense-description), #34 setting, #35 texture declare
`feeds: ["dossier-context4k"]`; selector picks **setting EIG 0.47** (0.30 + 0.17
downstream) > texture 0.41 > vlm 0.27 > reconstruction 0.10, versus re-electing #36.

## #3 — `goal_unreachable` first-class signal (auto-detect, auto-route)

- Registry registers the measurement + the floors:
  - `goal_floors {expanded_dossier_min_tokens, compact_context_min_tokens}` (mirrors
    program.json representation);
  - `evidence_budget {basis, deterministic_min/median/max_tokens_per_item,
    honest_ceiling_max_tokens_per_item}` — numbers from the honest expansion-ceiling
    audit (commit 2bd3292, run `.../dossier-expansion-audit-v1/`).
- `goal_reachability(registry)` returns `declared`, `goal_unreachable`,
  `measured_gap_tokens`, `route_to`, `requires_human`, `basis`. Unreachable when
  `honest_ceiling_max < expanded_dossier_min`.
- `sweep_status` merges `goal_reachability` + `blocked` count + `dependency_frontier`.
- Routing split is the contract: **grow-evidence-supply** (option B) is AUTONOMOUS work
  (the loop pursues it: blocked goal excluded, feeders boosted); **floor-renegotiation**
  (option A) is a HUMAN decision surfaced as `requires_human` — the harness holds only
  on A, never on B.

**Live effect:** `goal_unreachable: true`, gap 96,511 tok/item, `route_to:
grow-evidence-supply`, `requires_human: [floor-renegotiation]`, basis = measured audit.

## #4 — `program-overview` readout (strategist step-back)

- New CLI: `program-overview <registry> [--program <program.json>]` — JSON readout:
  total/terminal/blocked counts + blocked-arm details, goal arm, list of goal feeders,
  `goal_inputs_validated_pct` (% of feeding arms in a terminal state), dependency
  frontier, and the full `goal_reachability` block. With `--program`, also reports
  `program_floor_matches_registry`.
- This is the "step back and see the whole picture" capability: each strategist cycle
  starts from program health (budget vs floor, inputs validated, blocked count,
  frontier) instead of rediscovering it from issue prose.

**Live effect:** `goal_inputs_validated_pct: 62.5` (5 of 8 feeders validated),
`program_floor_matches_registry: true`.

## Verification

- 480 pytest tests pass (new `tests/test_blocked_dependencies_program.py` covers all
  four: blocked validation/transitions/selector-exclusion/tick behavior; dependency
  DAG validation + reachability + proportional boost + flips-selection; goal
  unreachability fire/clear/undeclared; program overview aggregation).
- `validate-program` and `validate-dimension-registry` pass on the updated registry.
- Live smoke (real CLI): sweep-status `blocked:1 goal_unreachable:true`; selector picks
  `setting` 0.47 w/ downstream 0.17; program-overview matches.

## Strategist prompt (cron `jobs.json`, applied POST-merge)

The live cron prompt drives the loop from the experiment worktree. To use these
improvements, the prompt's step 1 gains (only after this harness PR is merged into the
live worktree, else keep fallbacks):

1. `program-overview` before sweeping: read `goal_reachability`, `blocked_count`,
   `dependency_frontier` as the opening picture.
2. On `goal_unreachable`: route to `grow-evidence-supply` autonomously (run the
   dependency-frontier arm); keep the floor-renegotiation as a needs-human note — do
   NOT hold the loop on option A.
3. When an arm's gate becomes a policy/authority ruling (not a measurement), mark it:
   `mark-blocked ... --reason <ruling> --issue <N> --write`, then let the selector pick
   the next feeder; `mark-unblocked` when a ruling or evidence change resolves it.

## PR base note

Harness code lives only on the experiment-worktree lineage; this PR must be based on the
experiment-worktree HEAD (not `feat/autonomous-research-harness`, which holds only the
parsing CLI). Verify `gh pr diff --name-only` shows only the intended files.
