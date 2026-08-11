# Experiment Tree — Stratum Contextual Specialist Research

This is a living map. GitHub issues are the detailed source of truth; this document provides project orientation rather than a FIFO schedule.

## Live issue tree

* **[ROOT] #2 — Open-world specialist evidence → contextual representation**
  * Program root for the canonical corpus, policy, evidence architecture, and linked arms.

* **[PENDING] #3 — Portrait evidence discovery**
  * The owner-reviewed evidence-discovery map remains preserved in draft PR #7.
  * It identifies open-world candidate evidence roles and the raw-versus-bucketed input-view confound without selecting a specialist winner.

* **[COMPLETE / EMPIRICAL BETTER / PENDING_HUMAN_SPOT_CHECK] #4 — Baseline and comparison parity**
  * The baseline/comparison-parity arm (no longer the active arm; empirically complete, verdict BETTER, advisory human spot-check pending). The sole `research:active` arm is now **#73 apparent-age** (image-focus-depth-of-field #75 validated BETTER 2026-08-07).
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
  * **Post-exhaustion brainstorm-widen (2026-08-06):** new proposal arms **#58 point-map depth** (`pointmap-depth`, NEW evidence part; deterministic from source-matched `pointmap.npy`, 24/24 present), **#59 matting/alpha-fidelity** (`matting-alpha`, NEW part; `matting.npy`, 24/24), **#60 face-geometry** (`face-geometry` + NEW model class `mediapipe-facemesh-3d`, Apache-2.0 478-pt), **#61 object-relations** (`object-relations` + NEW model class `grounding-dino-open-vocab`), **#62 pose-articulation** (`pose-articulation`, deterministic from pose2 GOLIATH-308 + seg2). Registered via the gated `propose-dimensions --require-new-evidence-part` (5/5). Selector: **#62 pose-articulation** EIG 0.45 (exploit, ties broken by id) → sole `research:active` arm (selection_progress 6); **round-trip COMPLETE (2026-08-07, PR #64): BETTER** — support ratio 0.4225 → 0.8195 (Δ +0.397), supported 60→168, unsupported 82→37, p=0.002172, registry `pose-articulation → validated`, then **pointmap-depth #58 → active**. **Round-trip COMPLETE (2026-08-07): BETTER** — support ratio 0.3219 → 0.7488 (Δ +0.4269), supported 47→158, unsupported 99→53, p=0.000428, registry `pointmap-depth → validated`, **next `matting-alpha #59 → active`** (selected_via explore, ε-greedy slot, selection_progress 8). **Round-trip COMPLETE (2026-08-07): BETTER** — support ratio 0.3219 → 0.8657 (Δ +0.5438), supported 47→187, unsupported 99→29, p=0.002172, registry `matting-alpha → validated`, **next `face-geometry #60 → active`** (selected_via exploit, selection_progress 9). Remaining arm: object-relations #61 is a proposal.
  * The registry (`research/dimensions/evidence-dimension-registry-v1.json`) is the source of truth and now supports **non-stratum open-world specialists** (e.g. local Florence-2 for clothing/texture) and **reconstruction validation** (`claim-support` / `reconstruction` / `roundtrip-audit`) via local ComfyUI + CLIP scoring — the evidence space is not limited to stratum/Sapiens2 outputs.
  * **Validated (2026-08-05/06):** clothing #29 (BETTER, p≈0.0173), body-type #32 (ratio-corrected BETTER, p≈3e-6), **hair #30 (BETTER, p≈0.000772, draft PR #40)**, **skin-color #31 (BETTER, p≈0.000772, draft PR #41)**, **lighting #33 (BETTER, p≈0.0013, draft PR #42 — normal2+source luminance/DR/direction, seg2+normal2 evidence binding)**, **setting #34 (BETTER, p≈0.003305, draft PR #53 — seg2 Background coverage/dominant color/tone/vibrancy/pattern bands, seg2 evidence binding)**, and **texture #35 (BETTER, p=0.000139, draft PR #54 — per-region-class fabric/skin surface+pattern bands from seg2+source gradients, seg2 evidence binding)**, and **reconstruction #37 (BETTER, 2026-08-06 — generative reconstruction round-trip: CLIP ViT-L/14 mean delta +0.0679, 22/24 paired positives, null floor 0.595; branch `exp/stage-b-reconstruction-arm37-20260806`)**. **Goal arm: dossier-context4k #36** — ruling #46 LANDED 2026-08-06 via owner-merged PR #50 (Option A: reframe dossier objective to structural floor 4001 + 100K aspiration metadata); **#36 round-trip COMPLETE (2026-08-06, PR #52): BETTER** — plain-4K baseline 47/99 → evidence-compact 174/50 (ratio 0.322→0.777, p=0.000244), registry `dossier-context4k → validated`, next `setting → active`. Expansion-ceiling audit made program-floor-aware (2026-08-06): deterministic record 2040–3489 < 4001 but honest LM ceiling 8500–13500 **clears the structural floor** (`any_max_honest_floor_reached=true`, 24/24). After the #34 tick texture #35 was active (exploit, EIG 0.24); after the #35 tick reconstruction #37 was active (explore, EIG 0.10); **after the #37 tick reconstruction is validated and vlm-dense-description #47 is the sole active arm (exploit, EIG 0.10, tie-broken by id, selection_progress 5)**. Proposals: none remain — **vlm-dense-description #47 is ACTIVE** (pre-registered option-B evidence source, 2026-08-06 — open-world sourcing scan Molmo-72B/Qwen2.5-VL/InternVL3-78B + local capability probe of `qwen3-vl:32b`: 4090 27% CPU-offload too slow for 96-item batch; Strix 100% GPU ~9.6 tok/s = production batch host; draft PR #48). Selector tie-break fixed 2026-08-06 (id-tiebreaker regression test) — registering #47 exposed and closed the dict-comparison `TypeError`.

## Concluded

* **[CONCLUDED — HARNESS GATE RESOLVED] #9 — Bind comparison plans to canonical paths and specialist declarations**
  * Owner-reviewed draft PR #11 remediated canonical pilot paths, closed inline evidence envelopes, required failure modes, canonical comparison/audit identities, and content-bound evidence fingerprints.
  * This is a governance result only: it does not establish caption quality, invoke a model, or authorize data/GPU work.
