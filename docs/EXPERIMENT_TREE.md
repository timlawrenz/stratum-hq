# Experiment Tree — Stratum Contextual Specialist Research

This is a living map. GitHub issues are the detailed source of truth; this document provides project orientation rather than a FIFO schedule.

## Live issue tree

* **[ROOT] #2 — Open-world specialist evidence → contextual representation**
  * Program root for the canonical corpus, policy, evidence architecture, and linked arms.

* **[PENDING] #3 — Portrait evidence discovery**
  * The owner-reviewed evidence-discovery map remains preserved in draft PR #7.
  * It identifies open-world candidate evidence roles and the raw-versus-bucketed input-view confound without selecting a specialist winner.

* **[COMPLETE / EMPIRICAL BETTER / PENDING_HUMAN_SPOT_CHECK] #4 — Baseline and comparison parity**
  * The baseline/comparison-parity arm (no longer the active arm; empirically complete, verdict BETTER, advisory human spot-check pending). The sole `research:active` arm is now **#36 dossier-context4k**.
  * Completed Stage A is immutable, independently audited, non-executing provenance work; its 24-item six-slice manifest is not the first-500 cohort. The historical request is `research/proposals/stage-a-caption-context-parity-preparation.md` / draft PR #13.
  * Read-only first-500 audit: all 500 have readable `pose2`, `seg2`, `normal2`, `pointmap`, and `matting`; only 10 have the later determinations/caption2/t52 chain.
  * [`FIRST_500_CORE_COHORT_PILOT_DESIGN.md`](FIRST_500_CORE_COHORT_PILOT_DESIGN.md) specifies the coverage-aware future selection rule and states why the current evidence-only contrast remains blocked.
  * [`FIRST_500_COVERAGE_BALANCED_CANDIDATE_FREEZE.md`](FIRST_500_COVERAGE_BALANCED_CANDIDATE_FREEZE.md) binds a new source-hashed 12/6/6 candidate subset beneath the approved noncanonical research root. It has 24/24 core + legacy coverage and 0/24 complete existing later chains.
  * Draft PR #15's `caption_max_tokens` forwarding and detector-anomaly prompt repair was independently reviewed at `db85fe9bacc55e1c444615b027a2734d63398f61`; stacked draft PR #16 adds a mocked CLI-to-backend regression. Neither draft authorizes execution.
  * **[HELD] #18** now requires an owner decision on the exact already-installed local aggregator, immutable generation settings, self-audit/adversarial review, and whether model/GPU activity is authorized for the frozen manifest.

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

* **[PROPOSAL] Evidence-dimension arms (draft PR #20, `docs/EVIDENCE_DIMENSION_ARMS.md`)**
  * Each notes a deterministic measurement from existing artifacts and a claim-support delta target reusing the measured arm #4 protocol. See `[#29 clothing/apparel]`, `[#30 hair]`, `[#31 skin-color]`, `[#32 body-type/proportions]`, `[#33 lighting]`, `[#34 setting/environment]`, `[#35 texture/material]`, `[#36 full-dossier assembly + context4k compression]`, `[#37 generative reconstruction validation (ComfyUI round-trip)]`.
  * The registry (`research/dimensions/evidence-dimension-registry-v1.json`) is the source of truth and now supports **non-stratum open-world specialists** (e.g. local Florence-2 for clothing/texture) and **reconstruction validation** (`claim-support` / `reconstruction` / `roundtrip-audit`) via local ComfyUI + CLIP scoring — the evidence space is not limited to stratum/Sapiens2 outputs.
  * **Validated (2026-08-05):** clothing #29 (BETTER, p≈0.0173), body-type #32 (ratio-corrected BETTER, p≈3e-6), **hair #30 (BETTER, p≈0.000772, draft PR #40)**, **skin-color #31 (BETTER, p≈0.000772, draft PR #41)**, and **lighting #33 (BETTER, p≈0.0013, draft PR for arm #33 — normal2+source luminance/DR/direction, seg2+normal2 evidence binding)**. **Active:** dossier-context4k #36 (sole `research:active`, selector EIG 0.19). Proposals: setting #34, texture #35, reconstruction #37.

## Concluded

* **[CONCLUDED — HARNESS GATE RESOLVED] #9 — Bind comparison plans to canonical paths and specialist declarations**
  * Owner-reviewed draft PR #11 remediated canonical pilot paths, closed inline evidence envelopes, required failure modes, canonical comparison/audit identities, and content-bound evidence fingerprints.
  * This is a governance result only: it does not establish caption quality, invoke a model, or authorize data/GPU work.
