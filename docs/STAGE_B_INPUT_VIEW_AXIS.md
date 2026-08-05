# Stage-B Input-View Axis Integrity (Input-Level, Observer-Only)

**Date:** 2026-08-05
**Arm / parent:** #4 / #2 (held by #18)
**Status:** `METRIC-READINESS FINDING / NON-EXECUTING`

## Purpose and provenance boundary

The earlier observer-only checks on the completed Stage-B run
(`stage-b-first500-parity-v1`) covered three distinct boundaries:

1. `verify-stage-b-output` — structural binding of records, plan, evidence
   fingerprints, rendered text, output files, and review rows.
2. `check-stage-b-evidence-axis` — the **evidence payload** side: the recorded
   `evidence_payload` field is non-empty, per-image distinct, and bound to
   on-disk `pose2.npy`/`seg2.npy` inputs, while every no-evidence record
   carries `evidence_payload: null`.
3. `check-stage-b-contrast-divergence` — the **output** side: recorded caption
   text differs across every declared one-axis contrast (0/24 byte-identical
   pairs on every axis).

None of these inspected the **input-view side**: whether the run's own records
demonstrate that the declared input-view-only contrast was actually exercised
— i.e. that the `legacy-bucketed` condition fed a bucketed-crop view and the
`legacy-raw` condition fed the raw source, per image, at the model-input
boundary. A run whose records bind the view *declaration* but never document
the per-image view *bytes* cannot show from its own records that the two
conditions differed at all on the input axis.

## Question answered

Did the completed run's records isolate and materialize the declared
**input-view-only** contrast at the input level?

## Evidence (read-only, no model/GPU/scheduler/corpus action)

New additive observer-only check:
`research_harness.stage_b_verify.check_stage_b_input_view_axis` (+ CLI
`research-harness check-stage-b-input-view-axis <root>`). It verifies three
layers: declaration, per-record binding, and per-image input materialization.

Applied to `/mnt/nas-ai-models/research/stratum/stage-b-first500-parity-v1`:

- `input_view_axis_declared: **true**` (104 checks passed):
  - The plan declares exactly **two distinct view components**:
    `legacy-bucketed-crop-view-v1` (renderer `stage_b.resize_to_cover_center_crop`,
    Pillow.BICUBIC, aspect buckets) used by exactly **1** condition
    (`legacy-bucketed-no-evidence`), and `raw-source-view-v1` (renderer
    `stage_b.decoded_source_rgb`, no crop/resize) shared by the other **3**
    conditions (`legacy-raw-no-evidence`, `context-raw-no-evidence`,
    `context-raw-geometry`) — with distinct fingerprints.
  - The `input-view-only` contrast pairs those two views with
    `changed_axes: ["input_view"]`.
  - Every record's `input_view` `{id, fingerprint}` binds its condition's
    declaration (96/96).
- `input_view_axis_materialized: **false**` (0/96 records): **no record in the
  run carries a per-image view-content digest** (no `input_view_sha256`,
  `view_sha256`, `view_content_sha256`, or equivalent on the record or inside
  the `input_view` object).
- Therefore `input_view_axis_ok: **false**`: the input-view-only contrast is
  **declared and bound, but not input-documented** in the run's own records.
  The run cannot demonstrate from its records that the bucketed and raw
  conditions fed different view bytes — in contrast to the evidence axis,
  which stores per-image `selected_evidence_input_artifact_sha256` values bound
  byte-for-byte to on-disk `pose2.npy`/`seg2.npy`.

Executor-level context (from draft PR #20's `stage_b.py`, not from the run
records): the runner *does* implement two different views —
`_bucketed_view(raw)` (resize-to-cover + center-crop to the legacy aspect
bucket) vs `raw.copy()` — and the plan's frozen plan file records distinct
renderers. So the absence of per-image digests is an evidentiary gap in the
run's records, not proof that the views were identical.

## Boundary and interpretation

This is a **structural, input-level metric-readiness finding**, not a semantic
verdict:

- It does not fault the plan declaration, the per-record view binding, the
  structural bindings (`verify-stage-b-output` → valid), the evidence axis
  (`evidence_axis_ok: true`), or the output-level divergence
  (`contrast_divergence_ok: true`, input-view-only token-Jaccard median 0.491).
- It does NOT mean the views were identical (the executor implements and the
  plan declares two renderers); it means **the run's own records cannot certify
  the per-image view materialization**, so a reviewer cannot verify the
  input-view-only contrast's input side from the run alone.
- It does not authorize anything. `run-provenance.json` remains
  `PENDING_INDEPENDENT_REVIEW`, `semantic_verdict: PENDING`, metric self-audit
  `PENDING_HUMAN_SELF_AUDIT`; 96/96 review rows remain `unreviewed`/`PENDING`.
- No model invocation, GPU/scheduler use, corpus/derived-tree mutation,
  backfill, Stage-B execution, merge, or direct `main` push occurred.

## Decision impact for #18

This sharpens (does not replace) the existing #18 decisions. If the owner is
weighing whether to accept `stage-b-first500-parity-v1` for the claim-support
self-audit + adversarial review, they should now know that **two of the three
declared one-axis contrasts carry an input-side or executor-level caveat**:

- **Input-view-only** (this finding): declared, bound, output-divergent, but
  **not input-materialized** — no per-image view-content digest in the records.
- **Evidence-only** (prior finding, `evidence_prompt_clean: false`): the
  evidence slot embeds the CAPTION2 role/task instruction block, so the
  rendered-input boundary changes evidence *and* instructions.

The smallest exact owner decisions are unchanged in kind, with one addition:

1. Confirm **or deny** the asserted WebUI approval for frozen manifest
   fingerprint `b18843c759a8b93165a1261350ac46feea7cc62df787d44d4beb0ef9bc4b132d`.
2. Decide how the declared-but-unmaterialized `empty-caption-null-v1` null
   self-audit fixture should be satisfied.
3. Decide whether the evidence-only contrast may only be interpreted as
   "evidence + instructions", or whether any re-run must use a data-only
   evidence renderer.
4. Decide whether the input-view-only contrast may be interpreted with the
   declared-but-not-input-documented caveat, or whether any re-run must record
   a **per-image view-content digest** (e.g. `input_view_sha256` of the exact
   encoded bytes sent to the aggregator) so the view axis is certifiable from
   the run's own records.
5. Freeze the claim-support known-case/null self-audit + adversarial-review
   protocol.

## Smallest next action

No model/GPU/Stage-B action is authorized. The additive observer-only check is
the artifact of this round; the next decision is entirely the owner's, as above.
A future re-run under a durable approved manifest should record per-image view
digests so `check-stage-b-input-view-axis` can report
`input_view_axis_materialized: true`.
