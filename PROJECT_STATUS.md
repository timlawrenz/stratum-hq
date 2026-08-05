# Project Status — Stratum Contextual Specialist Research

**Last updated:** 2026-08-05
**Phase / status:** **ACTIVE — empirical Stage-B loop running autonomously.** Hold #18 is RELEASED (recorded via draft PR #28). Arm #4 (baseline/comparison parity) is the sole `research:active` / `research:metric-risk` arm and is empirically complete: 96 captions + independent gemma4 review + reviewer calibration. Its deterministic verdict is **BETTER** (support ratio 0.322→0.796, sign-test p=0.003); `PENDING_HUMAN_SPOT_CHECK` is advisory, not a gate. The dimension registry selects **arm #32 (body-type/proportions)** as the next active arm.

## Current state

The canonical corpus is `crawlr/approved` (immutable); `crawlr/stratum` remains a partial derived tree (never mutated by us). Stage-A records, the first-500 coverage audit, and the frozen first-500 coverage-balanced candidate manifest remain byte-for-byte intact and noncanonical.

- Arm #4 empirical evidence: 96/96 captions → `/mnt/nas-ai-models/research/stratum/stage-b-first500-parity-v1/`; 96/96 independent reviews → `stage-b-first500-parity-v1-review/`. Evidence-only contrast (cond 3→4): supported 47→156, unsupported 99→40, support ratio 32%→80%, sign-test p≈0.003, deterministic evidence-trace cross-check carried (16/24 ≥ half declared traces). Verdict per harness rule: **BETTER** (inconclusive=false).
- The dimension registry (`research/dimensions/evidence-dimension-registry-v1.json`) is the source of truth for proposal arms #29–#36 (#37 reconstruction is documented separately). `autonomous-select` scores body-type highest (EIG 0.8, direct pose2 extension of the validated arm-#4 result).
- Arm #32 deterministic measurement complete (CPU, existing `pose2.npy` only): 24/24 frozen items computed, 23 subject_present, 17/24 shoulder:hip ratio measurable, 13/24 leg measures, 1 abstained (low keypoint confidence), 53 low-confidence joints total. Record → `/mnt/nas-ai-models/research/stratum/stage-b-bodytype-proportions-v1.json`.

## Immediate next action

Run arm #32 body-type through the scheduler lifecycle (stage_b_launcher + stage_b_review_launcher on the local 4090) with the frozen stage-b-bodytype plan, then compute the evidence-only better-or-not verdict under the pre-registered claim-support rule, record it in `docs/EXPERIMENTS_AND_RESULTS.md` + issue #32, and advance the registry. The run is additive and noncanonical; no `crawlr/approved` / `crawlr/stratum` mutation, no backfill, no legacy overwrite.

## Live research tree

- #2 is the sole open program root.
- #3 is the preserved PENDING portrait-evidence map.
- #4 is the active baseline/comparison-parity arm (empirically complete, verdict BETTER, human spot-check advisory).
- #5 is the preserved geometry-grounded-captioning prototype.
- #9 is closed (comparison-plan provenance gate resolved).
- #18 is CLOSED/released (owner directive 2026-08-04, confirmed by draft PR #28).
- #29–#37 are registered proposal arms; #32 is the currently selected active-arm candidate.

## Automation and authority

The `stratum-ffhq` strategist is executing the autonomous decide→research→conclude→advance loop. Under the frozen-cohort protocol it may: run the deterministic selector, compute deterministic evidence from existing artifacts, generate captions and run the independent review through the scheduler-managed 4090, write noncanonical outputs under `/mnt/nas-ai-models/research`, record verdicts, and advance the registry. It may not mutate either corpus tree, backfill, install/download new image models, use external image services, merge, or push `main` directly (draft-PR-only).

## Headline result so far

**Arm #4: BETTER** (empirical + deterministic). The frozen-cohort evidence-only geometry contrast is a statistically significant improvement in supported claims under fixed view/prompt/model/settings. Formal PASS remains gated on the advisory human spot-check. Next: arm #32 body-type evidence-only contrast.
