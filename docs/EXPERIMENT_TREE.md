# Experiment Tree — Stratum Contextual Specialist Research

This is a living map. GitHub issues are the detailed source of truth; this document provides project orientation rather than a FIFO schedule.

## Live issue tree

* **[ROOT] #2 — Open-world specialist evidence → contextual representation**
  * Program root for the canonical corpus, policy, evidence architecture, and linked arms.

* **[PENDING] #3 — Portrait evidence discovery**
  * The owner-reviewed evidence-discovery map remains preserved in draft PR #7.
  * It identifies open-world candidate evidence roles and the raw-versus-bucketed input-view confound without selecting a specialist winner.

* **[ACTIVE / METRIC-RISK / PRE-COMPUTE] #4 — Baseline and comparison parity**
  * The sole `research:active` arm.
  * Completed Stage A is immutable, independently audited, non-executing provenance work; its 24-item six-slice manifest is not the first-500 cohort. The historical request is `research/proposals/stage-a-caption-context-parity-preparation.md` / draft PR #13.
  * Read-only first-500 audit: all 500 have readable `pose2`, `seg2`, `normal2`, `pointmap`, and `matting`; only 10 have the later determinations/caption2/t52 chain.
  * [`FIRST_500_CORE_COHORT_PILOT_DESIGN.md`](FIRST_500_CORE_COHORT_PILOT_DESIGN.md) specifies the coverage-aware future selection rule and states why the current evidence-only contrast remains blocked.
  * Immediate bounded work: independently review draft PR #15, which repairs/tests `caption_max_tokens` forwarding and removes detector-anomaly prompt content; then seek a separately bounded Stage-B decision only after the fixed cohort, local aggregator, rubric, self-audit, and adversarial review are frozen.

* **[PROPOSAL / PENDING] #5 — Geometry-grounded captioning prototype** (`exp/geometry-grounded-captioning`, draft PR #1)
  * Additive chain: `pose2 + seg2 + optional pointmap → determinations → caption2 → t52`.
  * Synthetic fixture coverage exists. No controlled empirical verdict exists.
  * The arm is not production-ready and must not be merged as a result of the governance build.

## Immutable Stage-A provenance

* **[COMPLETED / PENDING / NON-EXECUTING] Caption/context parity preparation**
  * Exact noncanonical record set:
    `/mnt/nas-ai-models/research/stratum/stage-a-caption-context-parity/{pilot-manifest.json,comparison-parity-plan.json,preparation-log.md,review-record.md}`.
  * The Stage-A global ordinal selection is preserved exactly. It is not a semantic sample, first-500 cohort, Stage-B authorization, model-readiness assertion, or empirical result.

## Future candidate branches

* **[TBD] Open-world specialist qualification**
  * Candidate models, fine-tunes, deterministic measurements, embeddings, and future discoveries must each earn a role through declared scope, provenance, abstention behavior, known failure modes, and qualification gates.

* **[TBD] Downstream representation and generative utility**
  * Test how `context4k` should be consumed without truncating it into the legacy 512-token T5 path, then test controlled downstream usefulness.

## Concluded

* **[CONCLUDED — HARNESS GATE RESOLVED] #9 — Bind comparison plans to canonical paths and specialist declarations**
  * Owner-reviewed draft PR #11 remediated canonical pilot paths, closed inline evidence envelopes, required failure modes, canonical comparison/audit identities, and content-bound evidence fingerprints.
  * This is a governance result only: it does not establish caption quality, invoke a model, or authorize data/GPU work.
