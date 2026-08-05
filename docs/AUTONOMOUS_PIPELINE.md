# Autonomous Research Pipeline — decide → research → conclude

The Stratum harness runs a closed loop that (1) decides the next highest-impact
branch, (2) researches it through the scheduler/model machinery, and (3) returns
an explicit **BETTER / NOT_BETTER** verdict on whether the ~4K asset description
improved — then repeats, advancing registry state each cycle.

## The loop

```
┌────────────────────────────────────────────────────────────────┐
│  1. DECIDE   autonomous-select  →  next `research:active` arm   │
│  2. RESEARCH proportions / launcher / review (scheduler GPU)    │
│  3. CONCLUDE autonomous-verdict →  BETTER / NOT_BETTER          │
│  4. ADVANCE  registry state (validated / falsified / strikes)   │
│  5. REPEAT   dimension-sweep-status → exhausted? brainstorm     │
└────────────────────────────────────────────────────────────────┘
```

## Decision layer (`src/research_harness/autonomous.py`)

- `select_next_arm(registry)` — deterministic expected-information-gain
  scoring over actionable proposals: prior evidence strength × measurability,
  minus cost weight and strike penalty. Returns the single highest-impact arm
  with a full score table + rationale trace. Report exhaustion as
  `AutonomousError` so the loop knows to stop and brainstorm instead.
- `better_or_not(...)` — pre-registered verdict rule (claim-support):
  **BETTER iff sign-test p ≤ 0.05 AND support ratio improves AND unsupported
  does not balloon**. `reconstruction` method uses a positive CLIP delta.
  Inconclusive-but-improved is reported NOT_BETTER with an `inconclusive`
  note — never a fabricated PASS.
- `aggregate_claim_support(review_dir)` — reads `reviews.jsonl`, computes
  per-condition counts and the paired sign-test p-value. Conditions are matched
  **exactly** (baseline `context-raw-no-evidence`, evidence
  `context-raw-geometry`) so a legacy `*-no-evidence` row can never act as the
  baseline (this exact bug was caught by the integration check and fixed).
- `run_tick(registry, review_dir=...)` — the loop body: if an arm is active,
  aggregate → verdict → advance (BETTER→validated; NOT_BETTER→+1 strike,
  falsified at the 3-strike limit) → select next; if nothing is active,
  activate the highest-impact proposal or report `brainstorm-new-data` when
  terminal.

## CLI

```
research-harness autonomous-select <registry>
research-harness autonomous-verdict --base-supported N --variant-supported N \
    --base-unsupported N --variant-unsupported N --items N --p-supported P \
    [--method reconstruction --reconstruction-delta D]
research-harness autonomous-tick <registry> [--review-dir DIR] [--write]
```

## Research (arm execution)

- Deterministic measurement (CPU, existing artifacts only): e.g. body-type via
  `research_harness.proportions.compute_proportions(pose2)`.
- Caption generation + independent review via the existing launchers
  (`stage_b_launcher`, `stage_b_review_launcher`) through the GPU scheduler
  lifecycle (claim/poll/launch/activate/heartbeat/release).
- Reconstruction (arm #37): local ComfyUI → CLIP ViT-L/14 vs original.

## Integration

The `Stratum research strategist` cron (every 60m) now drives this loop: it runs
the deterministic selector, executes the arm, computes the verdict, records the
result in `docs/EXPERIMENTS_AND_RESULTS.md` + the arm issue, advances the
registry, and reports. Delivery is `origin,all` so reports land in the active
session or connected channels.

## Rules

- Exactly one `research:active` arm; keep the before/after parity convention.
- BETTER requires significance + ratio gain + unsupported-sane. Never relax.
- Three valid non-improving experiments force a terminal state.
- Exhausted sweep → `brainstorm-new-data` harness state (new data
  sources/dimensions), not variants of the same space.
- Local-first, scheduler-managed GPU, noncanonical outputs, additive artifacts,
  PR-only.
