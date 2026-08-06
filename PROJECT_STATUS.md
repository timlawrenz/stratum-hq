# Project Status — Stratum Contextual Specialist Research

**Last updated:** 2026-08-06 (arm #36 ruling LANDED via merged PR #50: structural-floor reframe + 100K aspiration; expansion audit made program-floor-aware; dossier-context4k unblocked + re-selected)
**Phase / status:** **ACTIVE — empirical Stage-B loop running autonomously, one arm open (goal arm re-selected post-ruling).**

Hold #18 is RELEASED. **Ruling #46 LANDED 2026-08-06 (owner-merged PR #50, "resolves #46 stall"):**
**Option A accepted — the dossier objective is reframed from an absolute 100K blocking floor to a
structural floor (100K→4001 tokens, must exceed the 4K compact ceiling) with 100K recorded as aspiration
metadata.** `program.json` is schema v2 (validated); `validate-program` passes. Arm #36 `dossier-context4k`
was marked **unblocked** (gate was the now-resolved #46 ruling) and re-activated as the **sole
`research:active`** arm by the harness tick (`autonomous-tick` → `next_action: activate`,
`next_arm: dossier-context4k`, `selected_via: exploit`, EIG 0.34, novelty 0.15) — the program-goal arm is
actionable again and the goal is reachable (`goal_unreachable: false`, floor 4001, gap 512).

Prior validated arms (all BETTER): #4 baseline/parity (PENDING_HUMAN_SPOT_CHECK advisory, non-gating),
#32 body-type (BETTER), #29 clothing (BETTER), #30 hair (BETTER), #31 skin-color (BETTER),
#33 lighting (BETTER, largest delta 32.2%→95.1%). Stage A stays bounded and non-executing at
`research/proposals/stage-a-caption-context-parity-preparation.md`.

## Current state

The canonical corpus is `crawlr/approved` (immutable); `crawlr/stratum` remains a partial derived tree
(never mutated by us). All Stage-B outputs are additive and live under
`/mnt/nas-ai-models/research/stratum/` (noncanonical).

- **Arm #36 (dossier-context4k) — sole `research:active` (goal arm, re-activated post-ruling).**
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
  - Ruling #46 LANDED via owner-merged PR #50 (Option A reframe). Arm `dossier-context4k` unblocked
    (`mark-unblocked` 2026-08-06) and re-activated as the sole `research:active` arm by the harness tick
    (`selected_via: exploit`, EIG 0.34; issue #36 label-synced to `research:active`).
  - No tick conclusion this cycle: no review root / no tick-ready marker (correct — nothing was awaiting conclusion).
- **Arm #33 lighting empirical evidence:** frozen `stage-b-first500-lighting-v1` plan; generation
  (`stratum-stage-b-lighting-v2`) + independent review (`stratum-stage-b-adversarial-review-lighting-v2`)
  both completed cleanly on the local 4090. Evidence-only delta supported 47→194, unsupported 99→10,
  ratio 32.2%→95.1%, sign-test p≈0.0013 → **BETTER**; registry `validated`.
- **Arm #32 body-type:** evidence-only delta supported 47→195, unsupported 99→14, ratio 32.2%→93.3%,
  p≈0.000244 → **BETTER**; registry `validated`.
- **Arm #29 clothing:** supported 72→151, unsupported 100→46, ratio 41.9%→76.7%, p≈0.0173 → **BETTER**;
  registry `validated`.
- **Arm #30 hair / #31 skin-color:** both BETTER, registry `validated`.
- **Registry** (`research/dimensions/evidence-dimension-registry-v1.json`): 5 validated (body-type,
  clothing, hair, skin-color, lighting), **1 active (dossier-context4k #36 — sole `research:active`,
  re-activated post-ruling via the harness tick)**, 4 proposals (setting #34, texture #35,
  reconstruction #37, vlm-dense-description #47). Sweep not exhausted, not stalled. The dependency
  frontier (setting/texture/vlm) feeds the same goal. Selector tie-break fixed (2026-08-06,
  id-tiebreaker regression test).
- **Arm #47 sourcing verification** (2026-08-06, draft PR #48): open-world scan (Molmo-72B, Qwen2.5-VL,
  InternVL3-78B) + local capability probe of `qwen3-vl:32b` (already installed on 4090 + Strix): 4090 is
  27% CPU-offload / ~280s per 2048-token block — too slow for a 96-item batch; Strix (100GB usable) runs
  it 100% GPU ~9.6 tok/s, so Strix is the production batch host for any large-VLM arm.

## Immediate next action

**Run the arm-36 round-trip at honest scale (post-ruling):** the structural floor 4001 is reachable via
the honest LM elaboration ceiling (8500–13500) once the scheduler-bound aggregator expansion stage runs;
then compress to context4k and run the evidence-linked ≤4K vs plain-4K summarization round-trip claim-support
audit (plan/manifest freeze → 4090 generation → independent review via the parameterized wrapper →
`autonomous-tick --review-dir-from … --write`). This is the post-ruling increment the reference reserved;
the round-trip harness surface (context4k condition kind) is the next build frontier. All runs are
additive/noncanonical; no corpus mutation, no backfill, no legacy overwrite.

## Live research tree

- #2 is the sole open program root.
- #3 is the preserved PENDING portrait-evidence map.
- #4 is the baseline/comparison-parity arm (empirically complete, verdict BETTER, human spot-check advisory).
- #5 is the preserved geometry-grounded-captioning prototype.
- #9 closed (comparison-plan provenance gate resolved). #18 CLOSED/released (owner directive 2026-08-04).
- #29–#37 registered proposal arms; #32, #29, #30, #31, #33 validated (all BETTER);
  #36 dossier-context4k is the sole `research:active` arm (re-activated post-ruling); #34/#35/#37/#47 are proposals.
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

**Arm #4: BETTER; Arm #32: BETTER; Arm #29: BETTER; Arm #30: BETTER; Arm #31: BETTER; Arm #33: BETTER.**
Declared deterministic evidence (geometry; body-type proportions; DOME-29 clothing coverage + dominant
colors; hair region + color; exposed-skin tone; lighting luma/DR/shadow/direction) each significantly
improves supported claims on the frozen 24-item cohort under fixed view/prompt/model/settings.
**Next: arm #36 dossier-context4k round-trip at honest scale (post-ruling).**
