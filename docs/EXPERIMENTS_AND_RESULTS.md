# Experiments & Results — Stratum Contextual Specialist Research

This ledger records empirical findings and negative results permanently. A green implementation, readable artifact, or passing unit test is not an empirical PASS.

## Stage-A caption/context parity preparation — `[COMPLETED / PENDING / NON-EXECUTING]`

**Date:** 2026-08-04
**Arm:** #4 — baseline and comparison parity
**Proposal baseline:** draft PR #13 / commit `b3667ce077ff13aa86bae545a10bfa03d22edea9`

**Goal:** Materialize only the bounded, source-hashed pre-compute provenance required to judge whether a later controlled comparison could be specified. Stage A was not an inference or model-readiness exercise.

**Immutable records:**

```text
/mnt/nas-ai-models/research/stratum/stage-a-caption-context-parity/
  pilot-manifest.json
  comparison-parity-plan.json
  preparation-log.md
  review-record.md
```

**Evidence:**

- The immutable manifest records 24 selected items from six global ordinal slices, source hashes/dimensions, and selected-only availability/readability probes.
- The immutable comparison plan names the three intended one-axis contrasts (input view, prompt, evidence), but retains `stage-b-local-aggregator-pending-v1` as an intentional non-executing placeholder.
- Stage A is completed and independently audited as pre-compute evidence. The historic record set remains byte-for-byte untouched; it is not silently reissued as a first-500 or coverage-aware cohort.
- No model invocation/download, GPU or scheduler action, corpus mutation, derived-tree mutation, backfill, comparison, merge, or direct `main` push occurred.

**Verdict:** `PENDING` — structural provenance only. Draft PR #15's `caption_max_tokens` forwarding repair was independently reviewed at `db85fe9bacc55e1c444615b027a2734d63398f61`, and stacked draft PR #16 adds a mocked CLI-to-backend regression. Stage B still needs fixed local-model/generation provenance, metric self-audit, adversarial review, and separately explicit execution authority.

## First-500 core-artifact coverage audit — `[PENDING / PRE-COMPUTE]`

**Date:** 2026-08-04
**Arm:** #4 — baseline and comparison parity
**Artifact:** [`research/coverage/first-500-core-coverage-v1.json`](../research/coverage/first-500-core-coverage-v1.json)
**Design:** [`FIRST_500_CORE_COHORT_PILOT_DESIGN.md`](FIRST_500_CORE_COHORT_PILOT_DESIGN.md)

**Goal:** Test whether existing artifacts can support the declared one-axis comparison design without a backfill or new inference.

**Read-only evidence:**

- The first 500 eligible bytewise-ordered canonical filenames have readable `pose2.npy`, `seg2.npy`, `normal2.npy`, `pointmap.npy`, and `matting.npy`: **500 / 500** for every core artifact.
- Legacy caption/T5 artifacts are readable for **500 / 500**.
- Only **10 / 500** have every later-chain record: `determinations.json`, `caption2.txt`, `t52_hidden.npy`, and `t52_mask.npy`.
- The core-only cohort has 437 portrait, 23 squareish, and 40 landscape framing-proxy rows. 478 rows have one pose detection; 22 detector disagreements are quality/anomaly abstention rows, never caption content.
- The audit read no source-image bytes, decoded no image, invoked no model, and made no corpus write. It records source-membership and detail digests, not an empirical sample claim.

**Controlled-comparison assessment:**

- Input-view-only and prompt-only contrasts are designable on the 478 one-pose/core-complete rows, but neither is executable without the separately authorized fixed local aggregator and review protocol.
- The evidence-only contrast cannot use only the current materialized determinations chain for a coverage-aware 24-item design: it has 10 rows and no squareish coverage.
- A future evidence-only contrast may use an explicitly authorized deterministic computation from existing core `pose2`/`seg2` inputs, but that is new computation and must not mutate `crawlr/stratum`.
- Existing `t52_*` remains 512-token legacy output and cannot substitute for `context4k`.

**Verdict:** `PENDING` — the audit resolves the core-availability question and makes the exact later-chain gap explicit. It does not run, score, PASS, or FAIL a model.

## First-500 coverage-balanced candidate freeze — `[COMPLETED / PENDING / NON-EXECUTING]`

**Date:** 2026-08-04
**Arm:** #4 — baseline and comparison parity
**Artifact:** [`FIRST_500_COVERAGE_BALANCED_CANDIDATE_FREEZE.md`](FIRST_500_COVERAGE_BALANCED_CANDIDATE_FREEZE.md) and `/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json`

**Goal:** Bind the already designed 12 portrait / 6 squareish / 6 landscape candidate rule to source hashes only after reproducing the exact first-500 audit identities. This is additive noncanonical provenance work, not a replacement for immutable Stage A.

**Evidence:**

- The manifest is bound to the first-500 membership digest `4e9f8ca775a6e62e308afcccb1e36cce2a5d0bf1f5579631c4a76af0bc80f57c` and hidden item-detail digest `f7edebb10b42d002180f1641605babd66b2e3c159e343630ef2b769b47ea50e0` before any selected source byte was read.
- It records 24 selected source hashes/dimensions/formats after exactly 24 selected-only source reads: 12 portrait, 6 squareish, and 6 landscape. The 478-row one-pose primary pool and 22 detector-quality holdouts match the audit design.
- All 24 selected rows have readable core artifacts and legacy caption/T5 artifacts. **0 / 24** has the complete existing `determinations.json` → `caption2.txt` → `t52_*` chain.
- File SHA-256 is `8684c6e38c90b12898135235164677d780a4c897122f26a4b386f07283a9c5e0`; its content fingerprint is `b18843c759a8b93165a1261350ac46feea7cc62df787d44d4beb0ef9bc4b132d`.
- No model invocation/download, GPU or scheduler action, corpus/derived-tree mutation, backfill, legacy overwrite, comparative inference, merge, or direct `main` push occurred.

**Verdict:** `PENDING / HELD` — the frozen cohort makes a later request precise, but its zero existing later-chain coverage rules out an evidence-only comparison using only current caption-chain files. [#18](https://github.com/timlawrenz/stratum-hq/issues/18) now requires a direct owner decision on aggregator/generation provenance, metric self-audit/adversarial review, and execution authority.

## Harness initialization — `[PENDING / OWNER-REVIEWED DRAFT]`

**Date:** 2026-08-03 to 2026-08-04

**Goal:** Establish a reusable, project-neutral autonomous-research control plane grounded in the Stratum `crawlr/approved` program.

**Evidence:**

- Canonical source discovery found 11,825 flat eligible source images.
- The program keeps a 100K dossier target and a 4K compact-context target separate from legacy 512-token T5/T52 artifacts.
- Open-world specialist declarations require scope, inputs, output semantics, provenance, abstention, known failure modes, and qualification gates.
- The current GPU supervisor is observer-only; no scheduler lifecycle action is authorized.

**Verdict:** `PENDING` — the governance stack is a draft working baseline, not empirical authority.

## Comparison-plan provenance hold — `[CONCLUDED — HARNESS GATE RESOLVED]`

**Date:** 2026-08-04
**Blocked arm:** #4
**Hold issue:** #9 (closed)

**Trigger:** An adversarial synthetic audit showed that an earlier comparison-plan validator accepted escaping source paths and opaque evidence bundles.

**Remediation evidence:** Canonical relative paths, strict evidence envelopes, complete inline specialist declarations, content-bound fingerprints, canonical identities, synthetic regression coverage, and fresh live-tree validation were added and reviewed.

**Verdict:** `CONCLUDED — HARNESS GATE RESOLVED`. This is a governance result only; no empirical comparison, image inference, GPU action, corpus mutation, or backfill occurred.

## Arm 0 — Geometry-grounded captioning prototype — `[PROPOSAL — PENDING]`

**Branch / PR:** `exp/geometry-grounded-captioning`, draft PR #1.

**Goal:** Test whether Sapiens2-derived structural evidence can help an image-aware local aggregator produce more faithful contextual descriptions than the legacy single-caption path.

**Implementation evidence:**

- Additive chain: `pose2 + seg2 + optional pointmap → determinations.json → caption2.txt → t52_*`.
- Legacy `caption.txt` and `t5_*` remain untouched.
- Synthetic fixtures test geometry, determinations schema, relations, and pass isolation.

**Pre-registered gate:** A controlled evaluation must hold source-image preprocessing, prompt structure, model/generation settings, item set, and review rubric fixed.

**Known confounds / prerequisites:** Legacy captions use a bucketed/cropped image while current `caption2` opens the raw source. Existing caption output therefore cannot be interpreted as evidence-only. Draft PR #15 repairs `caption_max_tokens` forwarding and removes detector-anomaly prompt content; an independent non-executing review at `db85fe9bacc55e1c444615b027a2734d63398f61` found no implementation blocker, and stacked draft PR #16 adds a mocked CLI-to-backend regression. The unmerged stack does not authorize a controlled comparison. `t52` remains a legacy-compatible 512-token artifact rather than `context4k`.

**Verdict:** `PENDING` — preserve as a prototype; do not infer quality or downstream usefulness.
