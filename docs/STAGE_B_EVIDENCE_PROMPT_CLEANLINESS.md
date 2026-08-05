# Stage-B Evidence-Prompt Cleanliness (Executor-Level, Observer-Only)

**Date:** 2026-08-05
**Arm / parent:** #4 / #2 (held by #18)
**Status:** `METRIC-READINESS FINDING / NON-EXECUTING`

## Purpose and provenance boundary

The earlier observer-only checks on the completed Stage-B run
(`stage-b-first500-parity-v1`) verified three separate things:

1. `verify-stage-b-output` — structural binding of records, plan, evidence
   fingerprints, rendered text, output files, and review rows.
2. `check-stage-b-evidence-axis` — the **recorded** `evidence_payload` field is
   non-empty, per-image distinct, and bound to on-disk `pose2.npy`/`seg2.npy`
   inputs, while every no-evidence record carries `evidence_payload: null`.
3. `check-stage-b-contrast-divergence` — the **recorded output captions**
   actually differ across each declared one-axis contrast (0/24 byte-identical
   pairs on every axis).

None of these inspected the **exact rendered prompt** that was sent to the local
aggregator. That is the executor-level boundary the executor audit requires:
a data-only evidence slot must contain only the evidence content itself, never
role text, task instructions, semantic-expansion guidance, or detector/evaluator
metadata. An evidence-only contrast that changes *both* the evidence payload
*and* embedded instructions is not cleanly isolated at the model-input boundary.

## Question answered

Did the completed run's evidence-bearing rendered prompts keep their declared
specialist-evidence slot data-only, or did they smuggle instruction-bearing text
into the evidence slot?

## Evidence (read-only, no model/GPU/scheduler/corpus action)

New additive observer-only check:
`research_harness.stage_b_verify.check_stage_b_evidence_prompt_clean` (+ CLI
`research-harness check-stage-b-evidence-prompt-clean <root>`).

Applied to `/mnt/nas-ai-models/research/stratum/stage-b-first500-parity-v1`:

- `evidence_prompt_clean: **false**` (99 checks passed, **24 failed**).
- All **24 / 24** `context-raw-geometry` records carry a readable evidence slot
  whose content is per-image distinct (24 distinct slots → the data block is not
  boilerplate).
- In **every one of those 24 evidence slots**, the rendered prompt embeds the
  full CAPTION2 role/task instruction block:
  - "Your job is to VERBALIZE the geometry and ADD what the determinations omit"
  - "Name the posture or activity if obvious"
  - "Translate the measured relations"
  - "Describe mood, lighting quality, color palette"
  - "Describe the setting and environment"
  - "Subject & Pose", "Semantics:", "Visuals:", "Background:"
- The 72 no-evidence records carry no such instruction text in their evidence
  slots.

Because the evidence slot of `context-raw-geometry` contains the CAPTION2
instruction block verbatim, the declared **evidence-only** contrast actually
differs on two axes at the rendered-input boundary: the evidence payload **and**
the embedded task/semantic-expansion instructions. The evidence axis is therefore
**not cleanly isolated** in this run's rendered prompts, independent of the
recorded `evidence_payload` field being clean.

## Boundary and interpretation

This is a **structural, executor-level metric-readiness finding**, not a
semantic verdict:

- It does not fault the `evidence_payload` field, the per-image data, the
  structural bindings (`verify-stage-b-output` → valid), or the output-level
  divergence (`contrast_divergence_ok: true`).
- It does NOT mean the data block is missing (24 distinct slots were read) and
  it does NOT mean the instructions caused the caption differences — that
  attribution is exactly what a cleanly isolated design must avoid asserting.
- It does not authorize anything. `run-provenance.json` remains
  `PENDING_INDEPENDENT_REVIEW`, `semantic_verdict: PENDING`, metric self-audit
  `PENDING_HUMAN_SELF_AUDIT`; 96/96 review rows remain `unreviewed`/`PENDING`.
- No model invocation, GPU/scheduler use, corpus/derived-tree mutation, backfill,
  Stage-B execution, merge, or direct `main` push occurred.

## Decision impact for #18

This sharpens (does not replace) the existing #18 decisions. If the owner is
weighing whether to accept `stage-b-first500-parity-v1` for the claim-support
self-audit + adversarial review, they should know that **the evidence-only
contrast is confounded at the rendered-input level**: the `context-raw-geometry`
prompts carry the instructions from `CAPTION2_PROMPT_TEMPLATE` (the
`stratum2.pipeline.caption2` `build_prompt` output) inside their evidence slot,
because the Stage-B runner extracted the evidence block with
`build_prompt(determinations).split("DETERMINATIONS:\n", 1)[-1].strip()`, which
retains the template's trailing role/task block.

The smallest exact owner decisions remain:

1. Confirm **or deny** the asserted WebUI approval for frozen manifest
   fingerprint `b18843c759a8b93165a1261350ac46feea7cc62df787d44d4beb0ef9bc4b132d`.
2. Decide how the declared-but-unmaterialized `empty-caption-null-v1` null
   self-audit fixture should be satisfied.
3. Decide whether the evidence-only contrast should be interpreted only as
   "evidence + instructions" (with the confound documented), or whether any
   re-run must use a **data-only evidence renderer** that inserts only the
   determinations block into the evidence slot with no surrounding instructions.
4. Freeze the claim-support known-case/null self-audit + adversarial-review
   protocol.

## Smallest next action

No model/GPU/Stage-B action is authorized. The additive observer-only check is
the artifact of this round; the next decision is entirely the owner's, as above.
