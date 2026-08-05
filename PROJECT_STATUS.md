# Project Status — Stratum Contextual Specialist Research

**Last updated:** 2026-08-05
**Phase / status:** **ACTIVE — empirical Stage-B loop running autonomously.** Hold #18 is RELEASED. **#4 is the sole `research:active`** arm / `research:metric-risk` marker in the issue tree (baseline/comparison parity), empirically complete with an empirical verdict of **BETTER** (support ratio 32%→80%, sign-test p≈0.003; `PENDING_HUMAN_SPOT_CHECK` advisory, not a gate). Arm #32 (body-type) completed its full scheduler lifecycle run: **VERDICT: BETTER** (support ratio 32.2%→93.3%, sign-test p≈0.000244); the dimension registry advanced body-type `proposal → validated`. Next selector pick: **clothing (arm #29, EIG 0.7)**. Stage A stays bounded and non-executing at `research/proposals/stage-a-caption-context-parity-preparation.md`.

## Current state

The canonical corpus is `crawlr/approved` (immutable); `crawlr/stratum` remains a partial derived tree (never mutated by us). Stage-A records, the first-500 coverage audit, and the frozen first-500 coverage-balanced candidate manifest remain intact and noncanonical.

- Arm #4 empirical evidence: 96/96 captions + 96/96 independent gemma4 reviews + reviewer calibration. Evidence-only delta supported 47→156, unsupported 99→40, ratio 32%→80%, sign-test p≈0.003 → **BETTER**.
- Arm #32 empirical evidence: frozen `stage-b-first500-bodytype-v1` plan (evidence condition `context-raw-body-type` = deterministic `compute_proportions` on pose2). Generation (job `stratum-stage-b-bodytype-v1`) + independent review (job `stratum-stage-b-adversarial-review-bodytype-v1`) both completed cleanly on the local 4090. Evidence-only delta supported 47→195, unsupported 99→14, ratio 32.2%→93.3%, sign-test p≈0.000244 → **BETTER**; registry `validated`.
- The dimension registry (`research/dimensions/evidence-dimension-registry-v1.json`) now marks body-type validated; 7 proposals remain; sweep not exhausted. `autonomous-select` next picks clothing (#29).

## Immediate next action

Run the next selector-chosen arm (clothing #29) through the same frozen-cohort scheduler lifecycle once the determinism contract for DOME-29 clothing measurements is wired into the Stage-B evidence-kinds path, then record its verdict and advance the registry. All runs are additive/noncanonical; no corpus mutation, no backfill, no legacy overwrite.

## Live research tree

- #2 is the sole open program root.
- #3 is the preserved PENDING portrait-evidence map.
- #4 is the active baseline/comparison-parity arm (empirically complete, verdict BETTER, human spot-check advisory).
- #5 is the preserved geometry-grounded-captioning prototype.
- #9 is closed (comparison-plan provenance gate resolved).
- #18 is CLOSED/released (owner directive 2026-08-04, confirmed by draft PR #28).
- #29–#37 are registered proposal arms; #32 validated (BETTER); #29 clothing is the next selector pick.

## Automation and authority

The `stratum-ffhq` strategist is re-engaged for autonomous research and is executing the autonomous decide→research→conclude→advance loop under the frozen-cohort protocol: deterministic selector, deterministic evidence from existing artifacts, scheduler-managed 4090 generation + independent review, noncanonical outputs under `/mnt/nas-ai-models/research`, verdict recording, and registry advancement. It may not mutate either corpus tree, backfill, install/download new image models, use external image services, merge, or push `main` directly (draft-PR-only).

## Headline result so far

**Arm #4: BETTER; Arm #32: BETTER.** Declared deterministic evidence (geometry; then body-type proportions) each significantly improves supported claims on the frozen 24-item cohort under fixed view/prompt/model/settings. Both empirical verdicts await only the advisory human rubric spot-check for a formal PASS. Next: clothing (arm #29).

