# Project Status — Stratum Contextual Specialist Research

**Last updated:** 2026-08-06 (arm #35 texture round-trip claim-support audit COMPLETE → **BETTER**; registry advanced; reconstruction #37 active)
**Phase / status:** **ACTIVE — empirical Stage-B loop running autonomously, one arm open (reconstruction #37).**

**Arm #35 TEXTURE ROUND-TRIP BETTER (2026-08-06, harness-computed, PR #54).** The texture/material
evidence kind (`research_harness.texture`, per-region-class fabric/skin surface+pattern bands from
seg2 + source-pixel gradients) was built, band-calibrated on the frozen 24-item cohort
(24/24 measurable; fabric 4/3/3, pattern 5/3/2, skin 15/7/2 — no band ≥ 75%), frozen
(`stage-b-first500-texture-v1`, seg2 evidence input), generated on the 4090 (96 captions,
`stage-b-texture-v1`), independently reviewed (96 rows, `stage-b-texture-v1-review`), and
`autonomous-tick` computed **BETTER** — plain baseline 47 supported/99 unsupported → texture
167/34 (support ratio 0.3219 → 0.8308, Δ +0.5089; paired positive 21/24; sign-test p=0.000139).
Registry advanced atomically: **texture → validated**, **reconstruction #37 → active** (explore
slot, EIG 0.10, selection_progress 4).

**Arm #36 GOAL-ARM ROUND-TRIP BETTER (2026-08-06, harness-computed).** The reserved
post-ruling increment LANDED end to end: the `context4k` evidence kind (PR #52),
frozen plan `stage-b-roundtrip-context4k-v1`, 4090 generation (96 captions), independent
adversarial review (96 rows), and `autonomous-tick` which computed
**BETTER** — plain-4K baseline 47 supported/99 unsupported → evidence-linked ≤4K compact
174 supported/50 unsupported (support ratio 0.3219 → 0.7768, Δ +0.4549; paired positive
20/23; sign-test p=0.000244). Registry advanced atomically: **dossier-context4k → validated**,
**setting #34 → active** (exploit, EIG 0.30, novelty 0.15, ties_by id). Run roots:
`/mnt/nas-ai-models/research/stratum/stage-b-context4k-v1` + `-review`.

Hold #18 is RELEASED. **Ruling #46 LANDED 2026-08-06 (owner-merged PR #50, "resolves #46 stall"):**
**Option A accepted — the dossier objective is reframed from an absolute 100K blocking floor to a
structural floor (100K→4001 tokens, must exceed the 4K compact ceiling) with 100K recorded as aspiration
metadata.** `program.json` is schema v2 (validated); `validate-program` passes. Arm #36 `dossier-context4k`
was marked **unblocked** (gate was the now-resolved #46 ruling), re-activated, and after the round-trip
audit is now **validated** (`goal_unreachable: false`, floor 4001, gap 512).

Prior validated arms (all BETTER): #4 baseline/parity (PENDING_HUMAN_SPOT_CHECK advisory, non-gating),
#32 body-type (BETTER), #29 clothing (BETTER), #30 hair (BETTER), #31 skin-color (BETTER),
#33 lighting (BETTER, largest delta 32.2%→95.1%), #34 setting (BETTER, delta +0.5209). Stage A stays
bounded and non-executing at `research/proposals/stage-a-caption-context-parity-preparation.md`.

## Current state

The canonical corpus is `crawlr/approved` (immutable); `crawlr/stratum` remains a partial derived tree
(never mutated by us). All Stage-B outputs are additive and live under
`/mnt/nas-ai-models/research/stratum/` (noncanonical).

- **Arm #36 (dossier-context4k) — VALIDATED (goal arm, round-trip BETTER 2026-08-06).**
  - Deterministic dossier/context4k stage COMPLETE and honest (PR #45):
    24/24 frozen items, base deterministic dossier **387–648 tokens/item**, compact median ~298,
    all `under_budget`, `contract_ok: false` under the pre-reframe floors (the honesty gate was
    correctly refusing under-budget bundles — not weakened).
  - Honest expansion-ceiling audit (**now program-floor-aware, 2026-08-06**, run
    `/mnt/nas-ai-models/research/stratum/dossier-expansion-audit-v2/`): honest expanded prose
    1358–2525 tokens/item, total dossier record 2040–3489 tokens/item, generous honest LM elaboration
    ceiling 8500–13500 tokens/item. Under the **reframed structural floor 4001**, the deterministic
    record alone still does not clear it (`any_expanded_floor_reached=false`) but the **honest LM
    ceiling now DOES** (`any_max_honest_floor_reached=true`, all 24 items) — the scheduler-bound
    aggregator expansion stage can clear the structural floor without fabricating content.
  - **ROUND-TRIP BETTER (2026-08-06, PR #52):** `context4k` evidence kind + frozen plan
    `stage-b-roundtrip-context4k-v1`; 4090 generation (96 captions, `stage-b-context4k-v1`) +
    independent adversarial review (96 rows, `stage-b-context4k-v1-review`); `autonomous-tick`
    computed **BETTER** (baseline 47/99 supported/unsupported → evidence compact 174/50;
    ratio 0.3219 → 0.7768; paired 20/23; **p=0.000244**). Registry: #36 → validated, #34 → active.
- **Arm #33 lighting empirical evidence:** frozen `stage-b-first500-lighting-v1` plan; generation
  (`stratum-stage-b-lighting-v2`) + independent review (`stratum-stage-b-adversarial-review-lighting-v2`)
  both completed cleanly on the local 4090. Evidence-only delta supported 47→194, unsupported 99→10,
  ratio 32.2%→95.1%, sign-test p≈0.0013 → **BETTER**; registry `validated`.
- **Arm #32 body-type:** evidence-only delta supported 47→195, unsupported 99→14, ratio 32.2%→93.3%,
  p≈0.000244 → **BETTER**; registry `validated`.
- **Arm #29 clothing:** supported 72→151, unsupported 100→46, ratio 41.9%→76.7%, p≈0.0173 → **BETTER**;
  registry `validated`.
- **Arm #30 hair / #31 skin-color:** both BETTER, registry `validated`.
- **Arm #34 setting BETTER (2026-08-06, PR #53):** deterministic DOME-29 Background-class measurement
  (frame-coverage ratio, quantized dominant color, tone/vibrancy/pattern bands, scale-invariant, abstaining)
  computed in-memory from seg2 + source pixels. Frozen plan `stage-b-first500-setting-v1`; 4090 generation
  (96 captions, `stage-b-setting-v1`) + independent review (96 rows, `stage-b-setting-v1-review`);
  `autonomous-tick` computed **BETTER** (baseline 47/99 supported/unsupported → evidence 177/33;
  ratio 0.3219 → 0.8429, Δ +0.5209; paired 19/24; **p=0.003305**). Registry: #34 → validated, #35 → active.
- **Arm #35 texture BETTER (2026-08-06, PR #54):** deterministic per-region-class texture/material measurement
  (dominant fabric class surface/pattern bands + dominant skin class surface band from seg2 masks +
  source-pixel gradients; per-channel-normalized, 1-px-eroded interior so the silhouette boundary never counts
  as texture; scale-invariant, abstaining). Frozen plan `stage-b-first500-texture-v1`; 4090 generation
  (96 captions, `stage-b-texture-v1`) + independent review (96 rows, `stage-b-texture-v1-review`);
  `autonomous-tick` computed **BETTER** (baseline 47/99 supported/unsupported → evidence 167/34;
  ratio 0.3219 → 0.8308, Δ +0.5089; paired 21/24; **p=0.000139**). Fabric bands calibrated on the cohort
  (garment-only gating abstained 13/24 on the topless-half cohort — per-class dominant-region fixed this;
  pooled-class means degenerated to a fake 11/11 "busy" — per-class normalization fixed this).
  Registry: #35 → validated, **#37 reconstruction → active** (explore slot).
- **Registry** (`research/dimensions/evidence-dimension-registry-v1.json`): **8 validated** (body-type,
  clothing, hair, skin-color, lighting, **dossier-context4k #36**, **setting #34**, **texture #35**),
  **1 active (reconstruction #37 — sole `research:active`, selected via explore after the #35 tick,
  EIG 0.10, novelty 0.15, selection_progress 4)**, **1 proposal (vlm-dense-description #47)**. Sweep not
  exhausted, not stalled. The dependency frontier (vlm-dense-description) feeds the same goal.
  Selector tie-break fixed (2026-08-06, id-tiebreaker regression test).
- **Arm #47 sourcing verification** (2026-08-06, draft PR #48): open-world scan (Molmo-72B, Qwen2.5-VL,
  InternVL3-78B) + local capability probe of `qwen3-vl:32b` (already installed on 4090 + Strix): 4090 is
  27% CPU-offload / ~280s per 2048-token block — too slow for a 96-item batch; Strix (100GB usable) runs
  it 100% GPU ~9.6 tok/s, so Strix is the production batch host for any large-VLM arm.

## Immediate next action

**Arm #37 reconstruction is the sole `research:active` arm (selected via explore after the #35 tick,
EIG 0.10, novelty 0.15, selection_progress 4).** This is a reconstruction-validation arm (ComfyUI
round-trip + CLIP ViT-L/14 scoring) — not a caption-generation arm. Use the reconstruction method path
(`autonomous-tick --method reconstruction --reconstruction-delta <D> --items 96`) once the delta is
measured; the ComfyUI local instance at /mnt/fscache/essdee/ComfyUI is the reconstruction host. Run the
reconstruction arm per its frozen plan; all runs additive/noncanonical, no corpus mutation, no backfill,
no legacy overwrite. If the reconstruction delta measurement is not yet buildable (e.g. harness
`--method reconstruction` flag not yet present), measure the honest delta and record it, then route.

## Live research tree

- #2 is the sole open program root.
- #3 is the preserved PENDING portrait-evidence map.
- #4 is the baseline/comparison-parity arm (empirically complete, verdict BETTER, human spot-check advisory).
- #5 is the preserved geometry-grounded-captioning prototype.
- #9 closed (comparison-plan provenance gate resolved). #18 CLOSED/released (owner directive 2026-08-04).
- #29–#37 registered proposal arms; #32, #29, #30, #31, #33, #36, #34, **#35** validated (all BETTER);
  **#36 dossier-context4k is the validated goal arm (round-trip BETTER)**; **#37 reconstruction is the sole
  `research:active` arm**; #47 is a proposal.
- #46 is CLOSED: ruling LANDED via owner-merged PR #50 (Option A: structural floor + aspiration metadata).

## Automation and authority

The `stratum-ffhq` strategist is re-engaged for autonomous research under the frozen-cohort protocol: deterministic selector,
deterministic evidence from existing artifacts, scheduler-managed 4090 generation + independent review,
noncanonical outputs under `/mnt/nas-ai-models/research/stratum`, verdict recording, and registry
advancement. It may not mutate either corpus tree, backfill, merge, or push `main` directly
(draft-PR-only). Model SOURCING is open-world under the owner directive (2026-08-05): the loop may
discover/evaluate/download/install/qualify open-weight candidates locally; hosted third-party inference
of the sensitive canonical corpus requires a hold.

## Headline result so far

**Arm #4: BETTER; Arm #32: BETTER; Arm #29: BETTER; Arm #30: BETTER; Arm #31: BETTER; Arm #33: BETTER;
Arm #34: BETTER; Arm #35: BETTER; Arm #36 (goal): BETTER.**
Declared deterministic evidence (geometry; body-type proportions; DOME-29 clothing coverage + dominant
colors; hair region + color; exposed-skin tone; lighting luma/DR/shadow/direction; setting background
coverage/color/bands; texture fabric/skin surface+pattern bands) each significantly improves supported
claims on the frozen 24-item cohort under fixed view/prompt/model/settings. The **goal-arm round-trip is
BETTER**: captions generated FROM the evidence-linked ≤4K compact context (supported 47→174,
ratio 0.322→0.777, p=0.000244) beat the plain-4K summarization baseline. **Next: arm #37 reconstruction
(sole `research:active`, explore slot, EIG 0.10).**
