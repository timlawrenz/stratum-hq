# Stage-B Self-Audit Fixture Readiness — Observer-Only Check

**Date:** 2026-08-04
**Arm / parent:** #4 / #2 (held by #18)
**Status:** `PENDING / PRE-COMPUTE / NON-EXECUTING` — a metric-readiness finding, not a PASS/FAIL.

## Purpose

The pre-registered claim-support protocol for the completed Stage-B run
(`/mnt/nas-ai-models/research/stratum/stage-b-first500-parity-v1`) includes a
`metric_self_audit` declaring:

- `known_case_item_id: 0yo0gxbfflugqp205k128kktigl5`
- `null_output_id: empty-caption-null-v1`

`run-provenance.json` lists the required self-audit checks, including "Score the
declared empty-caption null output and confirm it is recorded as an abstention
rather than a supported claim." A human self-audit can therefore only execute as
pre-registered if those fixtures are **materialized by the run's records**.

This document records the additive, observer-only readiness check that reports
whether those fixtures exist, applying it to the completed output root. It is a
**finding**, not an authorization, model, GPU, or empirical verdict.

## Read-only evidence (this round)

The output root is structurally valid (`verify-stage-b-output` → `valid`; 96
records = 24 frozen images × 4 conditions; all binding checks ok). On top of
that, the new observer-only readiness check reports:

| Metric self-audit fixture | Declared | Materialized? |
|---|---|---|
| `known_case_item_id` `0yo0gxbfflugqp205k128kktigl5` | yes | **yes** — 4 records (one per condition) bind image_id |
| `null_output_id` `empty-caption-null-v1` | yes | **no** — not a record_id; 0 empty-caption records |

`readiness_verdict: NOT_READY`; missing fixture message:

```
null_output_id 'empty-caption-null-v1' is neither a record_id nor materialized
as an empty-caption record
```

## Interpretation / boundary

- **The run remains structurally sound and unreviewed.** Structural verification and
  self-audit-fixture readiness are deliberately separate checks. A run can bind every
  fingerprint, file, and review row and still lack a declared self-audit fixture.
- **The null/abstention self-audit step cannot execute as pre-registered** against this
  run's records because the declared empty-caption null output was not materialized.
  The known-case step can execute (4 records bind the declared item).
- **No empirical PASS/FAIL, authorization, or readiness-to-claim follows.** This is a
  metric-precondition observation that sharpens the decision the owner must make on #18:
  whether to accept this output for the self-audit protocol knowing the null fixture is
  missing (and how to resolve that), or treat the run as incomplete for that step.

## Tooling (additive, observer-only)

- `research_harness.stage_b_verify.check_stage_b_self_audit_readiness(root)` returns a
  structured report and never fabricates a verdict.
- CLI: `research-harness check-stage-b-self-audit-readiness <root>`.
- Synthetic fixtures cover: missing-null → `NOT_READY`; materialized null record → `READY`;
  empty-caption abstention record → `READY`; undeclared fixture → `ContractError`.
- No model invocation, GPU/scheduler action, corpus/derived-tree mutation, backfill,
  Stage-B execution, merge, or direct `main` push occurred for this artifact.

## Smallest exact next decision (unchanged in scope, now with a precise readiness gap)

The owner must decide on #18 whether to (a) accept the completed 96-record output
for the claim-support self-audit + adversarial review given the missing
`empty-caption-null-v1` null fixture and record how its null/abstention self-audit
step should be satisfied, or (b) treat the run (or that step) as invalid and require a
re-run under a durable approved manifest that materializes the declared null fixture.
All existing boundaries hold; nothing here authorizes execution.
