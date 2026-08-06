# Project Status — Stratum Contextual Specialist Research

**Last updated:** 2026-08-06 (arm #36 decision-pending; expansion-ceiling audit + floor decision filed; arm #47 proposal registered + VLM sourcing verified)
**Phase / status:** **ACTIVE — empirical Stage-B loop running autonomously, one arm human-decision-gated.**

Hold #18 is RELEASED. **#36 dossier-context4k is the sole `research:active` arm and is BLOCKED pending human
decision #46** (`research:needs-human`): the 100K expanded / 4K compact contract floors are not honestly
reachable on the frozen 24-item cohort from the five validated deterministic dimensions, so the pre-registered
round-trip claim-support audit cannot run with contract-valid bundles until the owner picks option A (reframe
arm-36 validation to a matched ≤4K budget) or option B (grow evidence supply first). No scheduler round-trip
may be launched until that ruling lands.

Prior validated arms (all BETTER): #4 baseline/parity (PENDING_HUMAN_SPOT_CHECK advisory, non-gating),
#32 body-type (BETTER), #29 clothing (BETTER), #30 hair (BETTER), #31 skin-color (BETTER),
#33 lighting (BETTER, largest delta 32.2%→95.1%). Stage A stays bounded and non-executing at
`research/proposals/stage-a-caption-context-parity-preparation.md`.

## Current state

The canonical corpus is `crawlr/approved` (immutable); `crawlr/stratum` remains a partial derived tree
(never mutated by us). All Stage-B outputs are additive and live under
`/mnt/nas-ai-models/research/stratum/` (noncanonical).

- **Arm #36 (dossier-context4k) — sole active arm, human-decision-gated.**
  - Deterministic dossier/context4k stage COMPLETE and honest (PR #45):
    24/24 frozen items, base deterministic dossier **387–648 tokens/item**, compact median ~298,
    all `under_budget`, `contract_ok: false` (both 100K/4K floors unmet). `build_compression_bundle`
    correctly REFUSES under-budget bundles — the honesty gate is not weakened.
  - Honest expansion-ceiling audit (commit `2bd3292`, run
    `/mnt/nas-ai-models/research/stratum/dossier-expansion-audit-v1/`): honest expanded prose
    1358–2525 tokens/item, total dossier record 2040–3489 tokens/item, generous honest LM elaboration
    ceiling 8500–13500 tokens/item — the 100K floor is **7–50× above the honest ceiling**.
    `any_expanded_floor_reached = false` for all 24 items.
  - **Decision #46** (`research:needs-human`, open, zero comments): owner to pick
    **(A)** reframe arm-36 validation to a matched ≤4K round-trip (evidence-linked compact context vs
    plain-4K baseline), 100K floor recorded as unattained policy metadata — recommended; or
    **(B)** grow evidence supply first under new arms (open-weight VLM dense multi-view descriptions,
    fuller machine payload e.g. pose2 keypoint tables) to legitimately raise the honest dossier.
  - Arm #36 stays `active`; **no tick and no scheduler round-trip until the decision lands.**

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
  clothing, hair, skin-color, lighting), 1 active (dossier-context4k #36), 4 proposals (setting #34,
  texture #35, reconstruction #37, **vlm-dense-description #47 — the concrete option-B evidence source,
  pre-registered 2026-08-06 with a full declaration + qualification gate**). Sweep not exhausted, not
  stalled. `autonomous-select` keeps picking dossier-context4k via exploit (EIG 0.34, novelty bonus 0.15
  applied). Selector tie-break fixed (2026-08-06, id-tiebreaker regression test) — registering #47
  surfaced and closed the dict-comparison `TypeError`.
- **Arm #47 sourcing verification** (2026-08-06, draft PR #48): open-world scan (Molmo-72B, Qwen2.5-VL,
  InternVL3-78B) + local capability probe of `qwen3-vl:32b` (already installed on 4090 + Strix): 4090 is
  27% CPU-offload / ~280s per 2048-token block — too slow for a 96-item batch; Strix (100GB usable) runs
  it 100% GPU ~9.6 tok/s, so Strix is the production batch host for any large-VLM arm. Local-options
  exhausted confirmed by the #36 expansion-ceiling audit.

## Immediate next action

**Await human ruling on decision #46 (A or B).** On A: apply the ≤4K reframe and run the scheduler-bound
round-trip (plan/manifest freeze → 4090 generation → independent review via the parameterized wrapper →
`autonomous-tick --review-dir-from … --write`). On B: activate first-priority evidence-growth proposal
arm(s) (open-weight VLM dense description qualifies as a NEW evidence part / model class under the
open-world sourcing directive) then resume #36. All runs are additive/noncanonical; no corpus mutation,
no backfill, no legacy overwrite.

## Live research tree

- #2 is the sole open program root.
- #3 is the preserved PENDING portrait-evidence map.
- #4 is the baseline/comparison-parity arm (empirically complete, verdict BETTER, human spot-check advisory).
- #5 is the preserved geometry-grounded-captioning prototype.
- #9 closed (comparison-plan provenance gate resolved). #18 CLOSED/released (owner directive 2026-08-04).
- #29–#37 registered proposal arms; #32, #29, #30, #31, #33 validated (all BETTER);
  #36 dossier-context4k is the sole active arm and is gated on decision #46; #34/#35/#37 are proposals;
  #47 (vlm-dense-description) is the pre-registered option-B evidence-growth proposal (draft PR #48).
- #46 is the open `research:needs-human` decision that gates the arm-36 round-trip. Option B's concrete
execution path is pre-registered and sourcing-verified: proposal **#47 open-weight VLM dense description**
(local-first, Strix batch host).

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
**Next: arm #36 dossier-context4k round-trip, gated on decision #46 (A or B).**
