# Stage-B owner release and review gate — `[OWNER-CONFIRMED / REVIEW GATE OPEN]`

**Date:** 2026-08-05
**Arm:** #4 — baseline and comparison parity
**Boundary issue:** #18 (released by durable owner decision 2026-08-05T07:23:06Z)

## Owner decision

A durable decision was recorded on issue #18 (authenticated Hermes WebUI, 2026-08-05T07:23:06Z):

- Approved #18 and delegated the Stage-B aggregator/settings/review decision to the
  autonomous research loop; this decision is **not owner-gated**.
- Released the hold. `research:hold` and `research:needs-human` labels removed
  (confirmed by label events at 07:23:03Z / 07:23:05Z).
- `research:metric-risk` **retained** until the claim-support self-audit and
  independent adversarial review complete on the finished run.
- Stage-B execution completed: 96/96 local captions (frozen first-500 subset,
  gemma3:27b, temp 0, seed 20260804, num_ctx 4096) published to
  `/mnt/nas-ai-models/research/stratum/stage-b-first500-parity-v1/`; scheduler
  lease released, model unloaded. Output remains `PENDING_INDEPENDENT_REVIEW`
  by design.
- Single-active-arm rule retained as a practical serialization convention.

## Confirmed run identity (matches on-disk provenance exactly)

- Root: `/mnt/nas-ai-models/research/stratum/stage-b-first500-parity-v1`
- Candidate manifest fingerprint: `b18843c759a8b93165a1261350ac46feea7cc62df787d44d4beb0ef9bc4b132d`
- Comparison plan fingerprint: `1e06d5c84c770b7343561f7fa1c1a164cb6dd5b078415194db0aa38c65b6ebd2`
- Generation: `gemma3:27b`, temperature 0.0, seed 20260804, context 4096,
  num_predict 384, top_k 1, top_p 1.0, local Ollama endpoint
- Records: 96 (24 frozen images × 4 conditions); 96/96 bindings verified;
  `verify-stage-b-output` → `valid`
- `run-provenance.json` / `scheduler-provenance.json`: `PENDING_INDEPENDENT_REVIEW`,
  metric self-audit `PENDING_HUMAN_SELF_AUDIT`

## What this changes

- The prior "asserted-but-undemonstrated approval" authority anomaly is **resolved**:
  the owner confirmed the approval in the durable record. The run is owner-confirmed,
  not merely structurally valid.
- The program is **no longer HELD**. The only remaining gate is `research:metric-risk`
  on #4, pending the human claim-support self-audit and independent adversarial review
  on the finished run.
- No new model/GPU execution was re-authorized. The confirmed run is the review target;
  a re-run remains a separate decision under a durable approved manifest.

## Smallest remaining decisions before metric-risk can clear

1. **Claim-support self-audit execution:** the pre-registered `metric_self_audit`
   known-case item `0yo0gxbfflugqp205k128kktigl5` is materialized (4 records), but the
   declared null-output fixture `empty-caption-null-v1` is **not** materialized
   (0 empty-caption records), so the pre-registered null/abstention step cannot execute
   as specified on this run. Owner/loop must state how that step is satisfied
   (documented remedy vs. durable-manifest re-run that materializes it).
2. **Evidence-only prompt confound:** all 24 `context-raw-geometry` records embed the
   CAPTION2 role/task instruction block inside the evidence slot
   (`evidence_prompt_clean: false`), so the declared evidence-only contrast changes
   evidence **and** embedded instructions at the model-input boundary. Decision:
   interpret with the documented caveat, or re-run with a data-only evidence renderer.
3. **Input-view-only not input-documented:** the plan declares the `input-view-only`
   contrast, but 0/96 records carry a per-image view-content digest
   (`input_view_axis_materialized: false`). Decision: interpret with the
   declared-but-not-input-documented caveat, or re-run recording per-image
   `input_view_sha256` digests.

## Verdict

`OWNER-CONFIRMED RUN / REVIEW GATE OPEN / PENDING_INDEPENDENT_REVIEW`.

No claim-support scoring, self-audit verdict, or adversarial review has run (96/96
review rows PENDING). This record does not clear metric-risk and does not authorize any
new model, GPU, scheduler, corpus, or derived-tree action.
