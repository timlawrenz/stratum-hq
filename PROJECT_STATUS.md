# Project Status — Stratum Contextual Specialist Research

**Last updated:** 2026-08-06 (arm #47 vlm-dense-description round-trip COMPLETE → **BETTER**; registry advanced; sweep now EXHAUSTED → brainstorm-new-data)
**Phase / status:** **ACTIVE — empirical Stage-B loop running; ALL 10 registered arms now validated; next action brainstorm-new-data (widen the menu).**

**Arm #37 RECONSTRUCTION ROUND-TRIP BETTER (2026-08-06, harness-computed, draft PR on
`exp/stage-b-reconstruction-arm37-20260806`).** The generative reconstruction arm
(pre-registered plan `stage-b-reconstruction-v1`: frozen 24-item pilot manifest, per-item
`context4k.md` compact as variant prompt, fixed item-independent degraded baseline, per-item
sha256 seeds, Juggernaut XL checkpoint sha256-pinned) generated 24×2 + 2 null images on the
4090 via the scheduler (ComfyUI, 832×1216, dpmpp_2m/28 steps, `stratum-stage-b-reconstruction-v1`),
then scored with CLIP ViT-L/14 (openai/clip-vit-large-patch14). `autonomous-tick --method
reconstruction` computed **BETTER** — **reconstruction_delta +0.067888** (mean per-item
CLIP similarity, context4k-generated vs source minus baseline-generated vs source), median
+0.075418, **22/24 paired positives**, null-case floor 0.5949 (the registered null prompt
bounds the generic-person baseline). Registry advanced atomically: **reconstruction #37 →
validated**, **vlm-dense-description #47 → active** (exploit, EIG 0.10, tie-broken by id,
selection_progress 5). Run root `/mnt/nas-ai-models/research/stratum/stage-b-reconstruction-v1`
(records.jsonl 50 rows, delta.json, run-provenance.json, outputs/ 50 PNGs).
Incident recorded honestly in the run provenance: the scheduler claim was RELEASED AS FAILED
at the CLIP step because the HF cache held only `config.json` for the pre-registered scorer
(the model+processor were downloaded to `$HF_HOME` afterwards and scoring/aggregation completed
in a separate CPU pass; the generated artifacts were complete and deterministic). Harness fix
in the same PR: latent **one-active invariant bug** — a NOT_BETTER strike below the
falsification limit left the struck arm active AND activated the next proposal (two
`research:active` dims; next tick hard-failed). The strike path now keeps the arm sole-active
(retry / brainstorm-on-stall), selection only after validate/falsify, with regression tests
(`test_run_tick_not_better_strike_keeps_one_active`, `test_run_tick_third_strike_falsifies_then_activates_next`).

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
- **Arm #47 VLM DENSE-DESCRIPTION ROUND-TRIP BETTER (2026-08-06, harness-computed, draft PR #57).**
  The option-B dossier-growth evidence part landed end to end: frozen block batch
  (qwen3-vl:32b → **gemma3:27b** after the former FAILED qualification on the real corpus — silent empty
  decode), 5-condition claim-support round-trip, independent review, and
  `autonomous-tick --baseline-condition context-raw-context4k --evidence-condition context-raw-vlm-dense`
  computed **BETTER** — VLM marginal support ratio 0.7376 → 0.9581 (Δ +0.2206), 163→206 supported,
  58→9 unsupported, **sign-test p=0.013302**, paired 21/24. Registry: **vlm-dense-description
  active → validated** (runs: `stage-b-vlm-dense-v1` blocks, `stage-b-vlm-dense-captions-v1` 120
  records, `-review` 120 rows). Cohort block abstention rate 0/578 flagged for the abstention audit.
  **Sweep now EXHAUSTED (10/10 validated) — next action brainstorm-new-data.**
- **Registry** (`research/dimensions/evidence-dimension-registry-v1.json`): **10/10 validated**
  (body-type, clothing, hair, skin-color, lighting, dossier-context4k #36, setting #34, texture #35,
  reconstruction #37, **vlm-dense-description #47**), 0 active, 0 blocked, 0 proposals.
  `dimension-sweep-status`: `exhausted: true`, `next_action: brainstorm-new-data`, `goal_unreachable:
  false` (floor 4001, gap 512; the VLM evidence part + deterministic record together clear it).
- **Arm #47 sourcing verification** (2026-08-06, draft PR #48): open-world scan (Molmo-72B, Qwen2.5-VL,
  InternVL3-78B) + local capability probe of `qwen3-vl:32b` (already installed on 4090 + Strix): 4090 is
  27% CPU-offload / ~280s per 2048-token block — too slow for a 96-item batch; Strix (100GB usable) runs
  it 100% GPU ~9.6 tok/s, so Strix is the production batch host for the newly-active VLM arm.

## Immediate next action

**The sweep is EXHAUSTED: all 10 registered arms are validated (last: vlm-dense-description #47
BETTER). `dimension-sweep-status` returns `next_action: brainstorm-new-data`.** Per the loop contract,
DO NOT re-run the same arm patterns — WIDEN: draft N genuinely-new candidate dimensions (new evidence
parts / new model classes / new data sources, e.g. relational/interaction, temporal/sequence,
generative/reconstruction extensions, new-model dense-describer candidates from a fresh literature
scan), each with the full declaration (scope/inputs/output_semantics/provenance/abstention_policy/
qualification_gate) and a NEW evidence part or model class, and register through the gated command:
`research_harness.cli propose-dimensions research/dimensions/evidence-dimension-registry-v1.json
--candidates <json> --count N --require-new-evidence-part --write`. The goal arm dossier-context4k is
validated (round-trip BETTER), goal_unreachable false; the VLM evidence part (validated) plus the
deterministic record honestly clear the 4001 structural floor.

## Live research tree

- #2 is the sole open program root.
- #3 is the preserved PENDING portrait-evidence map.
- #4 is the baseline/comparison-parity arm (empirically complete, verdict BETTER, human spot-check advisory).
- #5 is the preserved geometry-grounded-captioning prototype.
- #9 closed (comparison-plan provenance gate resolved). #18 CLOSED/released (owner directive 2026-08-04).
- #29–#37 registered proposal arms; #32, #29, #30, #31, #33, #36, #34, #35, **#37** validated (all BETTER);
  **#36 dossier-context4k is the validated goal arm (round-trip BETTER)**; **#47 vlm-dense-description is the sole
  `research:active` arm** (0 proposals remain).
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
Arm #34: BETTER; Arm #35: BETTER; Arm #36 (goal): BETTER; Arm #37 (reconstruction): BETTER.**
Declared deterministic evidence (geometry; body-type proportions; DOME-29 clothing coverage + dominant
colors; hair region + color; exposed-skin tone; lighting luma/DR/shadow/direction; setting background
coverage/color/bands; texture fabric/skin surface+pattern bands) each significantly improves supported
claims on the frozen 24-item cohort under fixed view/prompt/model/settings. The **goal-arm round-trip is
BETTER**: captions generated FROM the evidence-linked ≤4K compact context (supported 47→174,
ratio 0.322→0.777, p=0.000244) beat the plain-4K summarization baseline. **The generative reconstruction
check is BETTER too** (non-LLM validation of the same compact context): context4k-generated images score
+0.0679 mean CLIP ViT-L/14 similarity over the degraded-baseline generations (22/24 paired positives).
**Next: arm #47 vlm-dense-description (sole `research:active`, exploit slot, EIG 0.10).**
