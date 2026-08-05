# Experiments & Results — Stratum Contextual Specialist Research

This ledger records empirical findings and negative results permanently. A green implementation, readable artifact, or passing unit test is not an empirical PASS.

## Arm #32 — body-type/proportion evidence — `[EMPIRICAL RUN COMPLETE — VERDICT: BETTER]`

**Date:** 2026-08-05
**Arm:** #32 — body-type/proportions evidence specialist
**Code / PR:** `exp/stage-b-bodytype-arm32-20260805` (draft PR open), branches off the arm-#4 execution harness
**Cohort:** frozen 24-item first-500 coverage-balanced subset (12 portrait / 6 squareish / 6 landscape — same manifest as arm #4)
**Deterministic specialist:** `research_harness.proportions.compute_proportions` (Goliath-308 pose2, min confidence 0.5, continuous ratios with explicit abstention); precomputed record → `/mnt/nas-ai-models/research/stratum/stage-b-bodytype-proportions-v1.json` (23/24 subjects present, 17/24 shoulder:hip ratio measurable, 13/24 leg measures, 1 abstained, 53 low-confidence joints).
**Aggregator:** already-installed local `gemma3:27b` (digest `a418f5838eaf…`), temperature=0, seed=20260804, num_predict=384, num_ctx=4096, loopback Ollama.
**Independent reviewer:** `gemma4:e4b` (different family), temperature=0, seed=20260804, num_predict=2000, 512×512 input, same reviewer calibration as arm #4.
**Plan:** frozen `research/stage-b-plans/stage-b-bodytype-v1.json` (fingerprint `37b47cea885b5fc71e801fbd33bc902454f8a21ae52b4896aac925408a44fe1b`); conditions identical to arm #4 except the evidence condition is `context-raw-body-type` (proportions) instead of `context-raw-geometry` (full determinations).

**Scheduler lifecycle (local 4090):** request → poll/claim → launch → verify GPU activity → activate → heartbeat → release, through `registered-research-launcher` (job `stratum-stage-b-bodytype-v1`, completed 2026-08-05 ~11:23Z, `gpu_activity_seen: true`) and the independent review pass (job `stratum-stage-b-adversarial-review-bodytype-v1`, completed, 96/96). Both slots released cleanly (4090 idle).

**Evidence-only delta (cond 3 → 4):**
- supported claims **47 → 195**; unsupported **99 → 14**; omissions 11 → 28; contradictions 1 → 1; abstentions 0 → 5 (reviewer abstains where the evidence abstained).
- Support ratio **32.2% → 93.3%** (Δ +0.611).
- Paired per-item sign test on supported claims: 20 improve / 3 worsen (23 paired), one-sided binomial **p ≈ 0.000244**.
- Deterministic cross-check independent of the LLM review: 17/24 body-type captions carry ≥1 body-descriptive vocabulary trace beyond their matched baseline captions (geometric/vocabulary carry measured on the record captions directly).

**Deterministic verdict:** `autonomous-verdict --base-supported 47 --variant-supported 195 --base-unsupported 99 --variant-unsupported 14 --items 23 --p-supported 0.000244` → **BETTER** (significant p=0.000244 ≤ 0.05; support-ratio improvement; unsupported reduced, not ballooning; `inconclusive: false`).

**Boundaries respected:** local models only; outputs only under the approved noncanonical research root; no `crawlr/approved` or `crawlr/stratum` mutation; no backfill; no legacy overwrite; deterministic evidence computed in memory from existing `pose2.npy` only.

**Registry advance:** body-type dimension `proposal → validated` (0 strikes). Next selector pick: clothing (arm #29, EIG 0.7). Verdict BETTER is empirical on this 24-item frozen cohort; a formal PASS still awaits the advisory human rubric spot-check (single independent reviewer family, rubric not yet human-calibrated on known/null cases).

## First-500 coverage-balanced Stage-B comparison — `[EMPIRICAL RUN COMPLETE — PENDING_HUMAN_SPOT_CHECK]`

**Date:** 2026-08-04/05
**Arm:** #4 — baseline and comparison parity
**Code / PR:** `exp/stage-b-first500-aggregator-20260804`, draft PR #20
**Cohort:** frozen 24-item first-500 coverage-balanced subset (12 portrait / 6 squareish / 6 landscape)
**Aggregator:** already-installed local `gemma3:27b` (digest `a418f5838eaf…`), `temperature=0`, `seed=20260804`, `num_predict=384`, `num_ctx=4096`, loopback Ollama.
**Independent reviewer:** `gemma4:e4b` (different family from generator), `temperature=0`, `seed=20260804`, `num_predict=2000`, `num_ctx=8192`, 512×512 input.

**Goal:** Test, on the frozen cohort with fixed generation settings, whether declared in-memory geometry evidence (`pose2`+`seg2` only) changes claim support under a matched one-axis comparison.

**Conditions (same item, same model/settings):**
1. bucketed/cropped + legacy prompt + no evidence
2. raw + legacy prompt + no evidence
3. raw + context prompt + no evidence
4. raw + same context prompt + geometry evidence

**Empirical evidence:**
- 96/96 captions generated and published to `/mnt/nas-ai-models/research/stratum/stage-b-first500-parity-v1/` (records.jsonl, review-queue.jsonl, run-provenance.json, outputs/).
- Independent review (gemma4:e4b) scored 96/96 into claim-support buckets at `/mnt/nas-ai-models/research/stratum/stage-b-first500-parity-v1-review/`.
- **Evidence-only delta** (cond. 3 → 4): supported claims **47 → 156**; unsupported **99 → 40**; items with ≥1 supported claim **5/24 → 24/24**; omissions 11 → 27; contradictions 1 → 2. Support ratio (supported / supported+unsupported) **32% → 80%**.
- Paired per-item sign test on supported claims: 19 improve / 5 worsen, one-sided binomial p≈0.003. On unsupported: 14 decrease / 8 increase, p≈0.14 (directionally reduced, not individually significant).
- Deterministic cross-check (independent of the LLM review): all 24 geometry captions verbalize declared-evidence vocabulary; 16/24 carry ≥ half of declared traces. The supported-claim gain is traceable to evidence actually carried into the caption.

**Boundaries respected:** local models only; outputs only under the approved noncanonical research root; no `crawlr/approved` or `crawlr/stratum` mutation; no backfill; no legacy overwrite; scheduler lease claimed/activated/heartbeated/released cleanly; model unloaded after run.

**Verdict:** `EMPIRICAL RUN COMPLETE — PENDING_HUMAN_SPOT_CHECK`. Statistical improvement in supported claims from declared geometry on this 24-item frozen cohort with fixed settings. Not yet a PASS: single reviewer model, no human calibration of the rubric on known/null cases yet, cohort is 24 items, one-axis only. No corpus mutation or merge occurred.

**Deterministic verdict (2026-08-05, harness rule):** `autonomous-verdict --base-supported 47 --variant-supported 156 --base-unsupported 99 --variant-unsupported 40 --items 24 --p-supported 0.003` → **BETTER** (support ratio 0.322 → 0.796, Δ +0.474; sign-test p=0.003 ≤ 0.05; unsupported 99 → 40, not ballooning; `inconclusive: false`). The evidence-only contrast on the frozen cohort satisfies the pre-registered BETTER gate. The `PENDING_HUMAN_SPOT_CHECK` status is advisory (single independent reviewer family, rubric not yet human-calibrated on known/null cases) and does not gate the harness verdict; a formal PASS still awaits that spot-check.

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
