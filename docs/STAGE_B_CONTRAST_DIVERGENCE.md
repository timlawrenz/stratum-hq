# Stage-B Contrast Divergence — Observer-Only Finding

**Date:** 2026-08-05
**Arm / parent:** #4 / #2 (held by #18)
**Status:** `PENDING / PRE-COMPUTE / NON-EXECUTING` — a metric-readiness finding, not a PASS/FAIL.

## Purpose

The completed Stage-B run
(`/mnt/nas-ai-models/research/stratum/stage-b-first500-parity-v1`) declares
exactly three one-axis contrasts in its frozen plan:

| Contrast id | Baseline condition | Variant condition | Only changed axis |
|---|---|---|---|
| `input-view-only` | `legacy-bucketed-no-evidence` | `legacy-raw-no-evidence` | `input_view` |
| `prompt-only` | `legacy-raw-no-evidence` | `context-raw-no-evidence` | `prompt` |
| `evidence-only` | `context-raw-no-evidence` | `context-raw-geometry` | `evidence` |

Structural binding (96/96 `verify-stage-b-output`) proves the *inputs* were
bound correctly, and `check-stage-b-evidence-axis` proves the evidence payload
was real and isolated. But nothing has verified the **output** side: did the
aggregator actually produce different captions for each declared contrast, or
did it collapse every pair to the same text (which would make the contrast
vacuous no matter how the inputs were bound)? This round answers that question
read-only, with no model, GPU, or semantic judgment.

## Read-only evidence (this round)

The new additive observer-only check
(`research_harness.stage_b_verify.check_stage_b_contrast_divergence`, CLI
`research-harness check-stage-b-contrast-divergence <root>`) reports on the
completed output root:

```text
contrast_divergence_ok: true   (20 checks, 0 failed)
contrast_count:               3
condition_boilerplate_ids:    []
summary: all declared one-axis contrasts produced distinguishable captions
```

Per declared contrast (24 images, 24 baseline/variant pairs each):

| Contrast id | identical pairs | token-Jaccard (min / median / max) |
|---|---:|---:|
| `input-view-only` | **0 / 24** | 0.362 / 0.491 / 0.594 |
| `prompt-only` | **0 / 24** | 0.236 / 0.308 / 0.457 |
| `evidence-only` | **0 / 24** | 0.294 / 0.380 / 0.460 |

Also verified: no condition emitted one boilerplate caption across all 24
images (each condition has 24 distinct per-image captions).

## Interpretation / boundary

- **The completed run's declared one-axis contrasts are expressed in its
  outputs.** None of the 72 cross-axis pairs collapsed to identical text, and
  per-condition outputs are per-image distinct. A reviewer can therefore treat
  the run as having *attempted* all three declared axes at the output level.
- **This is structural, not semantic.** The check measures presence and
  statistical divergence of the recorded caption text only. It does **not**
  judge which output is better, whether any claim is supported by the image, or
  whether an observed difference is attributable to the declared axis rather
  than preprocessing drift. Claim-support scoring, known-case/null self-audit,
  and adversarial review remain reserved human steps
  (`run-provenance.json` still declares `PENDING_INDEPENDENT_REVIEW`,
  `semantic_verdict: PENDING`, metric self-audit `PENDING_HUMAN_SELF_AUDIT`;
  96/96 review rows remain `unreviewed` / `PENDING`).
- **No authorization, model, GPU, corpus, or derived-tree action occurred.**
  This finding does not validate the run for claim-support purposes and does
  not change the #18 hold: the run still lacks a durable owner authorization
  and its pre-registered null-output self-audit fixture
  (`empty-caption-null-v1`) is still not materialized.

## Tooling (additive, observer-only)

- `research_harness.stage_b_verify.check_stage_b_contrast_divergence(root)`
  returns a structured report and never fabricates a verdict.
- CLI: `research-harness check-stage-b-contrast-divergence <root>`.
- Synthetic fixtures cover: distinguishable outputs → `true`; a wholly
  collapsed (vacuous) contrast → rejected; a single-boilerplate condition →
  rejected; missing root → rejected. Full suite: **291 passed** (4 new tests);
  `validate-program` and fresh-open-snapshot `validate-tree` remain `valid`.

## Smallest exact next decision (unchanged scope, with the output question answered)

The owner must still decide on #18 whether to (a) accept the completed 96-record
output for the claim-support self-audit + adversarial review given the missing
`empty-caption-null-v1` null fixture (recording how that step should be
satisfied), or (b) treat the run (or that step) as invalid and require a re-run
under a durable approved manifest. The remaining owner decisions are unchanged:
confirm/deny the asserted WebUI approval for frozen manifest fingerprint
`b18843c759a8b93165a1261350ac46feea7cc62df787d44d4beb0ef9bc4b132d`, name the
persisting geometry-derivation provenance if a re-run is chosen, and freeze the
claim-support/adversarial review protocol. No model, GPU/scheduler, corpus
mutation, backfill, Stage-B execution, merge, or direct `main` push occurred for
this artifact.
