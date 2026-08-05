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
  * [`FIRST_500_COVERAGE_BALANCED_CANDIDATE_FREEZE.md`](FIRST_500_COVERAGE_BALANCED_CANDIDATE_FREEZE.md) binds a new source-hashed 12/6/6 candidate subset beneath the approved noncanonical research root. It has 24/24 core + legacy coverage and 0/24 complete existing later chains.
  * Draft PR #15's `caption_max_tokens` forwarding and detector-anomaly prompt repair was independently reviewed at `db85fe9bacc55e1c444615b027a2734d63398f61`; stacked draft PR #16 adds a mocked CLI-to-backend regression. Neither draft authorizes execution.
  * **Owner release (2026-08-05):** [#18](https://github.com/timlawrenz/stratum-hq/issues/18) now carries a durable, authenticated owner decision (07:23:06Z) that **released the Stage-B hold** (removed `research:hold` / `research:needs-human`), delegated the Stage-B aggregator/settings/review decision to the autonomous loop (not owner-gated), confirmed the finished 96-record run, and retained `research:metric-risk` on #4 until the claim-support self-audit + independent adversarial review complete on the confirmed run. The released-hold issue is closed/resolved consistent with the #9 precedent; the remaining `research:metric-risk` gate lives on #4.
  * **[REVIEW GATE / PENDING_INDEPENDENT_REVIEW] Stage-B run confirmed:** `/mnt/nas-ai-models/research/stratum/stage-b-first500-parity-v1` (96 records, gemma3:27b, temp 0, seed 20260804, num_ctx 4096) is owner-confirmed and its provenance matches the decision exactly. It remains entirely unreviewed (96/96 rows PENDING). The sequential claim-support self-audit (known-case `0yo0gxbfflugqp205k128kktigl5` materialized; null fixture `empty-caption-null-v1` NOT materialized) and independent adversarial review are the next step, honoring the two documented contrast caveats. See [`STAGE_B_OWNER_RELEASE_AND_REVIEW_GATE.md`](STAGE_B_OWNER_RELEASE_AND_REVIEW_GATE.md).
  * **Boundary anomaly — RESOLVED by durable owner decision (2026-08-05):** a concurrent round's draft PR #20 (`exp/stage-b-first500-aggregator-20260804`) asserted an owner Stage-B approval in its GPU manifest and exercised the shared GPU scheduler. Three launches failed (21:47–22:05Z, Ollama read timeout), but the **final lifecycle COMPLETED** (request 22:08:29Z → claim → activate → `released status=completed` 22:20:22Z), producing a full 96-record output root at `/mnt/nas-ai-models/research/stratum/stage-b-first500-parity-v1`. For several rounds the durable record showed no owner decision and the authority anomaly persisted; on 2026-08-05 the owner recorded a durable, authenticated decision on #18 (07:23:06Z) that confirmed the asserted approval, released the hold, confirmed the finished run, and delegated the aggregator/settings/review decision to the loop. The run remains `PENDING_INDEPENDENT_REVIEW` with every review row unscored; no PASS/FAIL may be claimed until the human claim-support self-audit + independent adversarial review complete on the confirmed run.
  * **[METRIC-READINESS FINDING] Stage-B self-audit fixture gap:** an additive observer-only check (`research-harness check-stage-b-self-audit-readiness <root>`) shows the completed run materialized the known-case fixture `0yo0gxbfflugqp205k128kktigl5` (4 records) but **not** the declared null-output fixture `empty-caption-null-v1` (not a record_id; 0 empty-caption records), so the pre-registered null/abstention self-audit step cannot execute as specified on this run. Structural validity is unaffected (96/96 bindings verified, `verify-stage-b-output` valid). See [`STAGE_B_SELF_AUDIT_FIXTURE_READINESS.md`](STAGE_B_SELF_AUDIT_FIXTURE_READINESS.md).
  * **[METRIC-READINESS FINDING] Stage-B evidence-axis integrity:** an additive observer-only check (`research-harness check-stage-b-evidence-axis <root>`) confirms the completed run's **evidence-only contrast was actually exercised** despite the frozen cohort having 0/24 materialized later chains: `evidence_axis_ok: true` — all 24 `context-raw-geometry` record payloads are non-empty and per-image distinct (`in-memory-geometry-determinations-v1` from `pose2.npy`/`seg2.npy`, hashes bound byte-for-byte to disk) and all 72 no-evidence records carry `null` payloads. Structural, not semantic; 96/96 review rows remain PENDING. See [`STAGE_B_EVIDENCE_AXIS_INTEGRITY.md`](STAGE_B_EVIDENCE_AXIS_INTEGRITY.md).
  * **[METRIC-READINESS FINDING] Stage-B contrast divergence:** an additive observer-only check (`research-harness check-stage-b-contrast-divergence <root>`) confirms the completed run's three declared one-axis contrasts are expressed in its **outputs**: `contrast_divergence_ok: true` (20 checks, 0 failed) — 0 of 24 baseline/variant caption pairs are byte-identical on every contrast (`input-view-only`, `prompt-only`, `evidence-only`; token-Jaccard medians 0.308–0.491) and each condition keeps 24 distinct per-image captions (no boilerplate). Output-level structural only, not semantic; 96/96 review rows remain PENDING. See [`STAGE_B_CONTRAST_DIVERGENCE.md`](STAGE_B_CONTRAST_DIVERGENCE.md).
  * **[METRIC-READINESS FINDING] Stage-B evidence-prompt cleanliness (executor-level):** an additive observer-only check (`research-harness check-stage-b-evidence-prompt-clean <root>`) inspects the **rendered prompt** the completed run actually sent to the aggregator (the boundary those other checks do not cover): `evidence_prompt_clean: false` — all 24/24 `context-raw-geometry` records embed the full CAPTION2 role/task instruction block ("Your job is to VERBALIZE the geometry…", "Name the posture or activity if obvious", "Subject & Pose", "Semantics:", "Visuals:", "Background:", …) inside the evidence slot (the runner's `build_prompt(...).split("DETERMINATIONS:\n", 1)[-1]` retains the template's trailing instructions), so the evidence-only contrast changes evidence **and** embedded instructions at the model-input boundary. The 72 no-evidence records are clean. Structural only, not semantic; 96/96 review rows remain PENDING. See [`STAGE_B_EVIDENCE_PROMPT_CLEANLINESS.md`](STAGE_B_EVIDENCE_PROMPT_CLEANLINESS.md).
  * **[METRIC-READINESS FINDING] Stage-B input-view axis integrity (input-level):** an additive observer-only check (`research-harness check-stage-b-input-view-axis <root>`) inspects the **input-view side** of the completed run's own records (the boundary those other checks do not cover): `input_view_axis_declared: true` (104 checks passed — the plan declares exactly two distinct view components, `legacy-bucketed-crop-view-v1` used by exactly one condition and `raw-source-view-v1` shared by the other three, distinct fingerprints, `input-view-only` contrast paired correctly, 96/96 record bindings) but `input_view_axis_materialized: false` — **0/96 records carry a per-image view-content digest**, so the run cannot demonstrate from its own records that the bucketed and raw conditions fed different view bytes. The executor (draft PR #20) does implement `_bucketed_view` vs raw, so this is an evidentiary gap in the records, not proof the views were identical. Structural only, not semantic; 96/96 review rows remain PENDING. See [`STAGE_B_INPUT_VIEW_AXIS.md`](STAGE_B_INPUT_VIEW_AXIS.md).

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
