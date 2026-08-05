# Stage-B Evidence-Axis Integrity — Observer-Only Finding

**Date:** 2026-08-04
**Arm / parent:** #4 / #2 (held by #18)
**Status:** `PENDING / PRE-COMPUTE / NON-EXECUTING` — a metric-readiness finding, not a PASS/FAIL.

## Purpose

The completed Stage-B run
(`/mnt/nas-ai-models/research/stratum/stage-b-first500-parity-v1`) declares four
conditions whose only intended difference on the **evidence-only** contrast is
whether specialist evidence is present:

| Condition | Evidence kind | Expected payload |
|---|---|---|
| `legacy-bucketed-no-evidence` | `none` | `null` |
| `legacy-raw-no-evidence` | `none` | `null` |
| `context-raw-no-evidence` | `none` | `null` |
| `context-raw-geometry` | `specialist_bundle` | non-empty per-image geometry object |

The frozen coverage-balanced cohort has **0 / 24** items with a *materialized*
`determinations.json → caption2.txt → t52_*` later chain, so a reviewer could
reasonably ask: was the evidence axis actually exercised, or are the
evidence-bearing records empty/boilerplate? This round answers that question
structurally, read-only.

## Read-only evidence (this round)

The new additive observer-only check
(`research_harness.stage_b_verify.check_stage_b_evidence_axis`, CLI
`research-harness check-stage-b-evidence-axis <root>`) reports on the completed
output root:

```text
evidence_axis_ok: true   (197 checks, 0 failed)
evidence_condition_ids:   ["context-raw-geometry"]
evidence_record_count:    24
no_evidence_condition_ids:["legacy-bucketed-no-evidence",
                           "legacy-raw-no-evidence",
                           "context-raw-no-evidence"]
no_evidence_record_count: 72
summary: evidence axis isolated and materialized
```

Concretely verified on the run's records:

- **24 / 24** evidence-bearing records (`context-raw-geometry`) carry a
  non-empty, per-image `evidence_payload` object; the payloads are **distinct per
  image** (24 distinct canonical payloads — not a shared boilerplate blob).
- The payload is a real `in-memory-geometry-determinations-v1` specialist bundle
  with `subject`, `relations`, `body_parts_visible`, `orientation`,
  `subject_extent`, and `schema_version` members, declared to be computed by
  `stratum2.pipeline.determinations.derive_determinations` (source SHA-256
  `3d4d35ecfaa534df9b1ed0608991f6573e0af15ff0a04ce86c705e92628d60e6`) **in
  memory** from the selected item's `pose2.npy` and `seg2.npy` only.
- **24 / 24** records bind their `selected_evidence_input_artifact_sha256`
  (`pose2.npy`, `seg2.npy`) to the actual on-disk derived files for the same
  source image (both hashes matched byte-for-byte for all 24 items).
- **72 / 72** no-evidence records carry `evidence_payload: null`.

## Interpretation / boundary

- **The evidence axis is structurally real and isolated on the completed run.**
  The geometry presented to the aggregator in the `context-raw-geometry`
  condition is not empty, not a shared placeholder, and is bound to existing core
  artifacts (`pose2.npy`/`seg2.npy` — 500/500 readable per the first-500 audit).
  The immutable freeze record's "0/24 complete later chains" refers to the
  *materialized* `determinations.json → caption2.txt → t52_*` files; the Stage-B
  run did not need those and instead derived geometry in memory from the existing
  core artifacts.
- **This is structural, not semantic.** The verifier confirms content is present,
  per-image distinct, and input-bound; it does **not** judge whether any geometry
  claim is accurate, supported, or a good caption-conditioning cue. Claim-support,
  known-case/null self-audit, and adversarial review remain reserved human steps
  (`run-provenance.json` still declares `PENDING_INDEPENDENT_REVIEW`, `semantic_verdict:
  PENDING`, metric self-audit `PENDING_HUMAN_SELF_AUDIT`; 96/96 review rows remain
  `unreviewed` / `PENDING`).
- **No authorization, model, GPU, corpus, or derived-tree action occurred.** This
  finding does not validate the run for claim-support purposes and does not
  change the #18 hold: the run still lacks a durable owner authorization and its
  pre-registered null-output self-audit fixture
  (`empty-caption-null-v1`) is still not materialized.

## Tooling (additive, observer-only)

- `research_harness.stage_b_verify.check_stage_b_evidence_axis(root)` returns a
  structured report and never fabricates a verdict.
- CLI: `research-harness check-stage-b-evidence-axis <root>`.
- Synthetic fixtures cover: isolated+materialized → `true`; boilerplate/shared
  payload → rejected; payload on a no-evidence record → rejected; missing core
  input hash → rejected; empty payload → rejected. Full suite: **287 passed**
  (6 new tests).

## Smallest exact next decision (unchanged in scope, now with the evidence-axis question answered)

The owner must decide on #18 whether to (a) accept the completed 96-record output
for the claim-support self-audit + adversarial review given the missing
`empty-caption-null-v1` null fixture (recording how that step should be
satisfied), or (b) treat the run (or that step) as invalid and require a re-run
under a durable approved manifest. The remaining owner decisions are unchanged:
confirm/deny the asserted WebUI approval for frozen manifest fingerprint
`b18843c759a8b93165a1261350ac46feea7cc62df787d44d4beb0ef9bc4b132d`, name the
persisting geometry-derivation provenance if a re-run is chosen, and freeze the
claim-support/adversarial review protocol. No model, GPU/scheduler, corpus
mutation, backfill, Stage-B execution, merge, or direct `main` push occurred for
this artifact.
