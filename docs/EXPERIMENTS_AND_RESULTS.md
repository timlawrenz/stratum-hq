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

## Stage-B authority boundary observation — `[COMPLETED RUN / UNREVIEWED / NEEDS-HUMAN]`

**Date:** 2026-08-04
**Arm:** #4 — baseline and comparison parity
**Blocked by:** #18 (open `research:hold` / `research:needs-human` / `research:metric-risk`)

**Trigger:** A concurrent autonomous round opened draft PR #20 (`exp/stage-b-first500-aggregator-20260804`), whose GPU manifest **asserts** an owner approval (`approved_by: timlawrenz direct #18 approval and autonomous-decision delegation in authenticated Hermes WebUI, 2026-08-04`; `manifest_state: approved`; `mode: human_reviewed`) and exercised the shared GPU scheduler.

**Read-only evidence (no corpus/model/GPU action by this round):**

- Issue #18 remains OPEN/held; its comments are agent-authored records only and state no Stage-B execution is authorized. PR #20 has no comments or reviews. No durable approval record exists anywhere in the repo. The asserted approval is therefore unverifiable from the durable record and is **not** treated as authorization.
- Scheduler events log (`/mnt/nas-ai-models/gpu-scheduler/logs/events.log`) documents multiple lifecycle attempts for `stratum-stage-b-first500-parity-v1` (GPU 4090, 22GB, 2h). The first three failed: 21:47:57Z request→21:51:19Z claim→21:52:08Z activate→21:53:22Z release-failed; 21:59:26Z→22:02:04Z claim→22:03:15Z release-failed; 22:03:33Z→22:05:41Z claim→22:05:43Z release-failed — each with `local Ollama generation failed: HTTPConnectionPool(host='127.0.0.1', port=11434): Read timed out`.
- **The fourth lifecycle COMPLETED.**
- **Correction to the earlier record:** the previous round asserted "No Stage-B output root exists … so no empirical Stage-B result was produced by any attempt." This is **disproven**. The durable scheduler log shows the run that started at 22:08:29Z (claim 22:08:40Z, activate 22:10:07Z) was released `status=completed` at **22:20:22Z**, and a complete output root exists at `/mnt/nas-ai-models/research/stratum/stage-b-first500-parity-v1` (created 22:20:21Z) — all **before** the previous round's PR #21 commit (22:23:34Z) that claimed the root was absent.
- **Evidence that the completed run is real and structurally sound (read-only verification this round):**
  - `records.jsonl` has 96 records = 24 frozen images × 4 conditions (`legacy-bucketed-no-evidence`, `legacy-raw-no-evidence`, `context-raw-no-evidence`, `context-raw-geometry`); 24 non-empty captions per condition dir (word counts 108–191, zero empty files).
  - 96/96 `source_sha256` bind to the frozen 24-item manifest; 96/96 evidence fingerprints pass canonical-JSON fingerprint validation; 96/96 prompt and input-view fingerprints match the frozen plan; 96/96 `rendered_sha256` bind `rendered_text`; the frozen plan binds its content fingerprint.
  - The four conditions isolate exactly one axis: input-view (legacy-bucketed vs legacy-raw, identical prompt), prompt (legacy-raw vs context-raw, identical view), evidence (context-raw-no-evidence vs context-raw-geometry, identical prompt+view).
  - `run-provenance.json` declares `status: PENDING_INDEPENDENT_REVIEW`, `semantic_verdict: PENDING`, and metric self-audit `PENDING_HUMAN_SELF_AUDIT`; `scheduler-provenance.json` records `status: completed` started 22:08:40Z finished 22:20:22Z.
- **The run is UNREVIEWED and carries no verdict:** all 96 `review-queue.jsonl` rows remain `unreviewed` / `PENDING`; no claim-support scoring, known-case/null self-audit, or adversarial review has been performed.

**Verdict:** `COMPLETED RUN / UNREVIEWED / NEEDS-HUMAN`. This is **not** a PASS, FAIL, or validated result. The asserted approval remains unverifiable and is not accepted as authorization; the 96-record output set is real but unreviewed. The owner must (a) confirm or deny the asserted WebUI approval for frozen manifest fingerprint `b18843c759a8b93165a1261350ac46feea7cc62df787d44d4beb0ef9bc4b132d`, record that decision durably, and (b) decide whether to accept the completed output root for the sequential claim-support self-audit plus adversarial review, or treat it as invalid and require a re-run under a durable approved manifest. This round performed no scheduler, model, corpus, merge, or main-push action.

## Stage-B self-audit fixture readiness — `[METRIC-READINESS FINDING / NON-EXECUTING]`

**Date:** 2026-08-04
**Arm:** #4 — baseline and comparison parity (held by #18)
**Artifact:** [`STAGE_B_SELF_AUDIT_FIXTURE_READINESS.md`](STAGE_B_SELF_AUDIT_FIXTURE_READINESS.md) and additive observer-only check `research_harness.stage_b_verify.check_stage_b_self_audit_readiness` (+ CLI `check-stage-b-self-audit-readiness`).

**Read-only evidence:**

- The completed output root remains structurally valid (`verify-stage-b-output` → `valid`; 96 records = 24 frozen images × 4 conditions; all bindings ok) and entirely unreviewed.
- The pre-registered `metric_self_audit` declares `known_case_item_id: 0yo0gxbfflugqp205k128kktigl5` (materialized: **yes**, 4 records) and `null_output_id: empty-caption-null-v1` (materialized: **no** — not a record_id; 0 empty-caption records).
- Therefore the pre-registered **null/abstention self-audit step cannot execute as specified** on this run's records; the known-case step can. This is a metric-precondition observation, not a PASS/FAIL or authorization.
- Full suite: **281 passed** (5 new synthetic readiness tests); `validate-program` and fresh-open-snapshot `validate-tree` remain `valid`.

**Verdict:** `PENDING / HELD`. Adds a precise, decision-relevant readiness gap to #18: the owner must decide how the declared-but-unmaterialized `empty-caption-null-v1` null/abstention self-audit step should be satisfied (e.g., accept a remedy for that step or require a re-run that materializes the fixture). No model, GPU/scheduler, corpus mutation, backfill, Stage-B execution, merge, or direct `main` push occurred.

## Stage-B evidence-axis integrity — `[METRIC-READINESS FINDING / NON-EXECUTING]`

**Date:** 2026-08-04
**Arm:** #4 — baseline and comparison parity (held by #18)
**Artifact:** [`STAGE_B_EVIDENCE_AXIS_INTEGRITY.md`](STAGE_B_EVIDENCE_AXIS_INTEGRITY.md) and additive observer-only check `research_harness.stage_b_verify.check_stage_b_evidence_axis` (+ CLI `check-stage-b-evidence-axis`).

**Read-only evidence:**

- Question answered: the frozen cohort has 0/24 materialized later chains (`determinations.json → caption2.txt → t52_*`), so — was the completed run's **evidence-only contrast actually exercised**? The run declares `context-raw-geometry` (kind `specialist_bundle`) against three `none`-kind no-evidence conditions.
- `check-stage-b-evidence-axis stage-b-first500-parity-v1` → `evidence_axis_ok: true` (197 checks, 0 failed; 24 evidence-bearing records, 72 no-evidence records).
- All 24 evidence-bearing records carry **non-empty, per-image distinct** `in-memory-geometry-determinations-v1` payloads (subject/relations/body_parts_visible/orientation/subject_extent/schema_version), declared from `stratum2.pipeline.determinations.derive_determinations` computed in memory from `pose2.npy` + `seg2.npy` only.
- 24/24 `selected_evidence_input_artifact_sha256` bind byte-for-byte to the on-disk derived files for the same source image; 72/72 no-evidence records carry `evidence_payload: null`.
- Structural only, not semantic: `run-provenance.json` still declares `PENDING_INDEPENDENT_REVIEW`, `semantic_verdict: PENDING`, metric self-audit `PENDING_HUMAN_SELF_AUDIT`; 96/96 review rows stay `unreviewed`/`PENDING`. No claim-support scoring, known-case/null self-audit, or adversarial review has run.
- Full suite: **287 passed** (6 new synthetic tests); `validate-program` and fresh-open-snapshot `validate-tree` remain `valid`.

**Verdict:** `PENDING / HELD`. The evidence axis of the completed run is structurally real and isolated, so it is decision-relevant that the owner need not treat "0/24 materialized later chains" as a blocker for *interpreting* the evidence contrast — the geometry was derived in memory from existing core artifacts. The #18 decisions are unchanged: confirm/deny the asserted approval, decide the missing `empty-caption-null-v1` null fixture, freeze the claim-support/adversarial review protocol, and confirm geometry-derivation provenance for any re-run. No model, GPU/scheduler, corpus mutation, backfill, Stage-B execution, merge, or direct `main` push occurred.

## Stage-B evidence-prompt cleanliness — `[METRIC-READINESS FINDING / NON-EXECUTING]`

**Date:** 2026-08-05
**Arm:** #4 — baseline and comparison parity (held by #18)
**Artifact:** [`STAGE_B_EVIDENCE_PROMPT_CLEANLINESS.md`](STAGE_B_EVIDENCE_PROMPT_CLEANLINESS.md) and additive observer-only check `research_harness.stage_b_verify.check_stage_b_evidence_prompt_clean` (+ CLI `check-stage-b-evidence-prompt-clean`).

**Read-only evidence:**

- Question answered: the recorded `evidence_payload` field was proven clean and the output captions were proven to diverge, but nobody had inspected the **exact rendered prompt** sent to the aggregator — the executor-level boundary where an evidence-only contrast can be silently turned into an evidence-plus-instructions contrast.
- `check-stage-b-evidence-prompt-clean stage-b-first500-parity-v1` → `evidence_prompt_clean: **false**` (99 checks passed, **24 failed**).
- All 24/24 `context-raw-geometry` records carry a readable, **per-image distinct** evidence slot (24 distinct slots → data is not boilerplate), but **every** record's evidence slot embeds the full CAPTION2 role/task instruction block: "Your job is to VERBALIZE the geometry and ADD what the determinations omit", "Name the posture or activity if obvious", "Translate the measured relations", "Describe mood, lighting quality, color palette", "Describe the setting and environment", "Subject & Pose", "Semantics:", "Visuals:", "Background:".
- Root cause (executor-level): the runner built the evidence text with `build_prompt(determinations).split("DETERMINATIONS:\n", 1)[-1].strip()`, and `build_prompt` returns the full `CAPTION2_PROMPT_TEMPLATE`, so the split retains the template's trailing role/task block alongside the determinations.
- The 72 no-evidence records carry no such instruction text in their evidence slots.
- Therefore the declared **evidence-only** contrast differs on two axes at the rendered-input boundary: the evidence payload **and** the embedded instructions. The evidence axis is not cleanly isolated in this run's rendered prompts, independent of the `evidence_payload` field being clean.
- Structural only, not semantic: `run-provenance.json` still declares `PENDING_INDEPENDENT_REVIEW`, `semantic_verdict: PENDING`, metric self-audit `PENDING_HUMAN_SELF_AUDIT`; 96/96 review rows stay `unreviewed`/`PENDING`. No claim-support scoring, known-case/null self-audit, or adversarial review has run.
- Full suite: **294 passed** (3 new synthetic tests); `validate-program` and fresh-open-snapshot `validate-tree` remain `valid`.

**Verdict:** `PENDING / HELD`. Shapes the #18 owner decision: if the completed root is considered for the claim-support self-audit + adversarial review, the evidence-only contrast must be interpreted with the documented evidence+instructions confound, or a re-run must use a data-only evidence renderer (only the determinations block, no surrounding instructions). The smallest exact decisions are unchanged (confirm/deny the asserted approval; decide the missing `empty-caption-null-v1` null fixture; freeze the self-audit/adversarial protocol; decide the evidence-rendering confound). No model, GPU/scheduler, corpus mutation, backfill, Stage-B execution, merge, or direct `main` push occurred.

## Stage-B input-view axis integrity — `[METRIC-READINESS FINDING / NON-EXECUTING]`

**Date:** 2026-08-05
**Arm:** #4 — baseline and comparison parity (held by #18)
**Artifact:** [`STAGE_B_INPUT_VIEW_AXIS.md`](STAGE_B_INPUT_VIEW_AXIS.md) and additive observer-only check `research_harness.stage_b_verify.check_stage_b_input_view_axis` (+ CLI `check-stage-b-input-view-axis`).

**Read-only evidence:**

- Question answered: the evidence-axis check proved the evidence payload was real and the contrast-divergence check proved the outputs differ, but nobody had inspected the **input-view side** — whether the run's own records demonstrate that the bucketed and raw conditions fed different view bytes to the aggregator.
- `check-stage-b-input-view-axis stage-b-first500-parity-v1` → `input_view_axis_declared: **true**` (104 checks passed), `input_view_axis_materialized: **false**` (0/96 records), `input_view_axis_ok: false`.
- The plan declares exactly two distinct view components — `legacy-bucketed-crop-view-v1` (resize-to-cover + center-crop renderer) used by exactly one condition (`legacy-bucketed-no-evidence`) and `raw-source-view-v1` (decoded-source renderer) shared by the other three conditions — with distinct fingerprints; the `input-view-only` contrast pairs them with `changed_axes: ["input_view"]`; 96/96 records bind their condition's declared view id/fingerprint.
- **No record carries a per-image view-content digest** (no `input_view_sha256`/equivalent), so the run cannot demonstrate from its own records that the bucketed and raw views differed per image — unlike the evidence axis, which stores per-image `selected_evidence_input_artifact_sha256` bound byte-for-byte to disk. The executor in draft PR #20 does implement `_bucketed_view` vs raw, so this is an evidentiary gap in the records, not proof the views were identical.
- Structural only, not semantic: `run-provenance.json` still declares `PENDING_INDEPENDENT_REVIEW`, `semantic_verdict: PENDING`, metric self-audit `PENDING_HUMAN_SELF_AUDIT`; 96/96 review rows stay `unreviewed`/`PENDING`. No claim-support scoring, known-case/null self-audit, or adversarial review has run.
- Full suite: **299 passed** (5 new synthetic tests); `validate-program` and fresh-open-snapshot `validate-tree` remain `valid`.

**Verdict:** `PENDING / HELD`. Two of the three declared one-axis contrasts now carry a documented caveat at the model-input boundary: evidence-only (evidence+instructions confound) and input-view-only (declared-and-bound but not input-documented). The #18 decisions are unchanged in kind, plus: the owner must decide whether the input-view-only contrast may be interpreted with the declared-but-not-input-documented caveat, or whether any re-run must record a per-image view-content digest (e.g. `input_view_sha256` of the exact encoded bytes) so the view axis is certifiable from the run's own records. No model, GPU/scheduler, corpus mutation, backfill, Stage-B execution, merge, or direct `main` push occurred.

## Stage-B contrast divergence — `[METRIC-READINESS FINDING / NON-EXECUTING]`

**Date:** 2026-08-05
**Arm:** #4 — baseline and comparison parity (held by #18)
**Artifact:** [`STAGE_B_CONTRAST_DIVERGENCE.md`](STAGE_B_CONTRAST_DIVERGENCE.md) and additive observer-only check `research_harness.stage_b_verify.check_stage_b_contrast_divergence` (+ CLI `check-stage-b-contrast-divergence`).

**Read-only evidence:**

- Question answered: structural binding proved the inputs were bound and the evidence axis proved the evidence payload was real, but nobody had verified the output side — did the aggregator actually produce different captions for each declared one-axis contrast, or collapse them?
- `check-stage-b-contrast-divergence stage-b-first500-parity-v1` → `contrast_divergence_ok: true` (20 checks, 0 failed): **0 of 24** baseline/variant caption pairs are byte-identical on every contrast — `input-view-only` (token-Jaccard median 0.491), `prompt-only` (median 0.308), `evidence-only` (median 0.380) — and each of the 4 conditions keeps 24 distinct per-image captions (`condition_boilerplate_ids: []`).
- The completed run therefore **expresses all three declared one-axis contrasts at the output level**; none is vacuous.
- Structural only, not semantic: `run-provenance.json` still declares `PENDING_INDEPENDENT_REVIEW`, `semantic_verdict: PENDING`, metric self-audit `PENDING_HUMAN_SELF_AUDIT`; 96/96 review rows stay `unreviewed`/`PENDING`. No claim-support scoring, known-case/null self-audit, or adversarial review has run.
- Full suite: **291 passed** (4 new synthetic tests); `validate-program` and fresh-open-snapshot `validate-tree` remain `valid`.

**Verdict:** `PENDING / HELD`. The #18 decisions are unchanged (confirm/deny the asserted approval; decide the missing `empty-caption-null-v1` null fixture; freeze the claim-support/adversarial review protocol). The run's outputs are now known to express the declared axes, sharpening whether the owner accepts the root for self-audit. No model, GPU/scheduler, corpus mutation, backfill, Stage-B execution, merge, or direct `main` push occurred.

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
