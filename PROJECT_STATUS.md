# Project Status — Stratum Contextual Specialist Research

**Last updated:** 2026-08-05 (arm #33 lighting concluded)
**Phase / status:** **ACTIVE — empirical Stage-B loop running autonomously.** Hold #18 is RELEASED. **#36 dossier-context4k is the sole `research:active`** arm in the issue tree (selector next pick). **Arm #33 (lighting) completed its full scheduler lifecycle run: VERDICT: BETTER** (support ratio 32.2%→95.1%, sign-test p≈0.0013); the registry advanced lighting `active → validated`, and dossier-context4k became active. Prior validated arms: #4 baseline/parity (BETTER, PENDING_HUMAN_SPOT_CHECK advisory), #32 body-type (BETTER), #29 clothing (BETTER), #30 hair (BETTER), #31 skin-color (BETTER). Stage A stays bounded and non-executing at `research/proposals/stage-a-caption-context-parity-preparation.md`.

## Current state

The canonical corpus is `crawlr/approved` (immutable); `crawlr/stratum` remains a partial derived tree (never mutated by us). Stage-A records, the first-500 coverage audit, and the frozen first-500 coverage-balanced candidate manifest remain intact and noncanonical.

- Arm #4 empirical evidence: 96/96 captions + 96/96 independent gemma4 reviews + reviewer calibration. Evidence-only delta supported 47→156, unsupported 99→40, ratio 32%→80%, sign-test p≈0.003 → **BETTER**.
- Arm #33 empirical evidence: frozen `stage-b-first500-lighting-v1` plan (evidence condition `context-raw-lighting` = deterministic `compute_lighting` on normal2+seg2+source pixels). Generation (job `stratum-stage-b-lighting-v2`) + independent review (job `stratum-stage-b-adversarial-review-lighting-v2`) both completed cleanly on the local 4090. Evidence-only delta supported 47→194, unsupported 99→10, ratio 32.2%→95.1% (largest delta on the cohort), sign-test p≈0.0013 → **BETTER**; registry `validated`.
- Arm #32 empirical evidence: frozen `stage-b-first500-bodytype-v1` plan (evidence condition `context-raw-body-type` = deterministic `compute_proportions` on pose2). Generation (job `stratum-stage-b-bodytype-v1`) + independent review (job `stratum-stage-b-adversarial-review-bodytype-v1`) both completed cleanly on the local 4090. Evidence-only delta supported 47→195, unsupported 99→14, ratio 32.2%→93.3%, sign-test p≈0.000244 → **BETTER**; registry `validated`.
- Arm #29 empirical evidence: frozen `stage-b-first500-clothing-v1` plan (evidence condition `context-raw-clothing` = deterministic `compute_clothing` on seg2 + source pixels). Generation (job `stratum-stage-b-clothing-v1`) + independent review (job `stratum-stage-b-adversarial-review-clothing-v1`) both completed cleanly on the local 4090. Evidence-only delta supported 72→151, unsupported 100→46, ratio 41.9%→76.7%, sign-test p≈0.0173 → **BETTER**; registry `validated`.
- The dimension registry (`research/dimensions/evidence-dimension-registry-v1.json`) now marks body-type, clothing, hair, skin-color, and lighting validated; **dossier-context4k (#36) is the sole active arm**; setting/texture remain proposals (reconstruction #37 is a separate proposal issue). Sweep not exhausted. `autonomous-select` next picks dossier-context4k (#36).

## Immediate next action

Run the next selector-chosen arm **dossier-context4k (#36)** — the program-level goal: assemble the per-item expanded dossier (target ≥100K tokens, claim-by-claim evidence links) from the five now-validated deterministic dimensions (geometry/body-type, clothing, hair, skin-color, lighting) plus relational determinations, then compress it into the ~4K-token context4k artifact and run the round-trip claim-support audit with independent adversarial review under the same frozen-cohort scheduler lifecycle. All runs are additive/noncanonical; no corpus mutation, no backfill, no legacy overwrite.

## Live research tree

- #2 is the sole open program root.
- #3 is the preserved PENDING portrait-evidence map.
- #4 is the active baseline/comparison-parity arm (empirically complete, verdict BETTER, human spot-check advisory).
- #5 is the preserved geometry-grounded-captioning prototype.
- #9 is closed (comparison-plan provenance gate resolved).
- #18 is CLOSED/released (owner directive 2026-08-04, confirmed by draft PR #28).
- #29–#37 are registered proposal arms; #32, #29, #30 hair, #31 skin-color, and #33 lighting validated (all BETTER); #36 dossier-context4k is the next selector pick (now `research:active`).

## Automation and authority

The `stratum-ffhq` strategist is re-engaged for autonomous research and is executing the autonomous decide→research→conclude→advance loop under the frozen-cohort protocol: deterministic selector, deterministic evidence from existing artifacts, scheduler-managed 4090 generation + independent review, noncanonical outputs under `/mnt/nas-ai-models/research`, verdict recording, and registry advancement. It may not mutate either corpus tree, backfill, install/download new image models, use external image services, merge, or push `main` directly (draft-PR-only).

## Headline result so far

**Arm #4: BETTER; Arm #32: BETTER; Arm #29: BETTER; Arm #30: BETTER; Arm #31: BETTER; Arm #33: BETTER.** Declared deterministic evidence (geometry; body-type proportions; DOME-29 clothing coverage + dominant colors; hair region + color; exposed-skin tone; lighting luma/DR/shadow/direction) each significantly improves supported claims on the frozen 24-item cohort under fixed view/prompt/model/settings. The empirical verdicts await only the advisory human rubric spot-check for a formal PASS. Next: dossier-context4k (arm #36) — assemble the expanded dossier and compress to context4k.

