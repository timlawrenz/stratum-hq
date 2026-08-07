# Project Status — Stratum Contextual Specialist Research

**Last updated:** 2026-08-07 (arm #68 gaze/head-orientation ROUND-TRIP COMPLETE → BETTER; sweep EXHAUSTED 17/17; brainstorm-widen queued NEW candidates → next arm TBD)
**Phase / status:** **ACTIVE — empirical Stage-B loop running; 17/17 arms validated (all 15 feeders + dossier-context4k + reconstruction); gaze-head-orientation #68 validated BETTER; sweep EXHAUSTED (0 active, 0 proposals) → next action brainstorm-new-data; 0 blocked.**

**Arm #68 gaze/head-orientation ROUND-TRIP COMPLETE → BETTER (2026-08-07, harness-computed, draft PR on `exp/stage-b-gaze-head-arm68-20260807`).** NEW-EVIDENCE-PART camera-interaction axis reusing the SAME validated open-weight MediaPipe FaceLandmarker 478-point mesh as arm #60 (owned hardware, local CPU, `face_landmarker.task` sha256 64184e229b...): scale-invariant head-orientation bands — yaw (facing camera / partially turned / profile or turned away), pitch (level / tilted down / tilted up, cohort-centered calibration), and in-plane roll from the stable eye-line angle — via the canonical six-point PnP head-pose fit (classic OpenCV model, y-flip measured on this cohort to fix a degenerate out-of-plane fit; Euler via the classic OpenCV decomposition; plausibility gates |pitch|≤85° / |yaw|≤90°). Band calibration probe on the frozen 24-item cohort: 21/24 detected (union policy, 3 honest abstains matching arm #60), yaw 4/5/12 (max 57%), pitch 7/6/8 (max 38%), roll 10/11 (max 52%) — all bands under the 75% degeneracy line; two independent estimators (PnP and landmark-projection) agree the corpus genuinely has many turned heads. Support ratio 0.3219 → 0.9673 (Δ +0.6454), supported 47 → 207, unsupported 99 → 7, paired positive 20/20, sign-test p=1e-06. Registry: gaze-head-orientation → **validated** → sweep EXHAUSTED (17/17) → **next_action brainstorm-new-data**. One-active invariant holds (17 validated, 0 active, 0 proposals).

**Arm #69 scene-category ROUND-TRIP COMPLETE → BETTER (2026-08-07, harness-computed, draft PR on `exp/stage-b-scene-category-arm69-20260807`).** NEW-MODEL-CLASS open-weight CLIP ViT-L/14 zero-shot scene classifier (`openai/clip-vit-large-patch14`, MIT, local CPU on owned hardware, staged at `/mnt/nas-ai-models/research/stratum/models/scene-category`, model.safetensors sha256 a2bf730a...): semantic scene category (what-kind-of-place) over the full-frame source against a frozen closed 10-category set (indoor studio / plain wall backdrop / bedroom / living room / outdoor beach / outdoor garden / outdoor field / body of water / urban street / poolside), cohort-derived from the arm-#47 VLM scene vocabulary. Abstention floor 0.25 calibrated on the frozen cohort (probe: 24/24 classified, 8 distinct categories, max top-1 share 25%, p50 confidence 0.526, min confidence 0.270, 0 abstentions). Scale-invariant label only verbalized; similarity logits/probabilities stay in evidence_payload. Support ratio 0.3219 → 0.9310 (Δ +0.6091), supported 47 → 189, unsupported 99 → 14, paired positive 20/24, sign-test p=0.000772. Registry: scene-category → validated; **gaze-head-orientation #68 → active** (selected_via explore, ε-greedy slot, selection_progress 12). One-active invariant holds (16 validated, 0 proposals — #68 active). Note: scene-category binds NO derived evidence artifact (CLIP consumes only the decoded source RGB, evidence-input hashes honestly empty; seg2/pose2 stay validation-only reads).

**Arm #61 object-relations ROUND-TRIP COMPLETE → BETTER (2026-08-07, harness-computed, draft PR on `exp/stage-b-object-relations-arm61-20260807`).** NEW-MODEL-CLASS open-weight Grounding DINO (`IDEA-Research/grounding-dino-base`, Apache-2.0, text-grounded open-vocabulary detector, HF Transformers path, local CPU on owned hardware; model.safetensors sha256 5548f844...): scale-invariant object-presence count band (none/sparse/moderate/dense) + placement band (foreground/background/mix from seg2-subject overlap) + canonical class list over the frozen cohort-derived closed vocabulary (water/field/concrete/mirror/window + accessories — the furniture-centric first try was DEGENERATE 9/24); box_threshold 0.25 calibrated on the cohort (21/24 ≥1 detection). Bands calibrated (count none=8/sparse=7/moderate=5/dense=4, max 33%; placement fg=4/bg=4/mix=8/none=8, max 33%); subject-self guard excludes 'body'/person' boxes (exact-standalone-word, keeping 'body of water'). Support ratio 0.3219 → 0.8783 (Δ +0.5564), supported 47 → 166, unsupported 99 → 23, paired positive 18/21, sign-test p=0.000745. Registry: object-relations → validated; **the sweep is now EXHAUSTED (15/15 terminal)**.

**Arm #60 face-geometry ROUND-TRIP COMPLETE → BETTER (2026-08-07, harness-computed, draft PR #67 on `exp/stage-b-face-geometry-arm60-20260807`).** NEW-MODEL-CLASS open-weight MediaPipe FaceLandmarker (478-point mesh, Apache-2.0, local CPU via the tasks API, model `face_landmarker.task` sha256 64184e229b...): scale-invariant facial-geometry bands (eye-spacing / face-width, mouth / face-width, jaw / face-width, and plausibility-gated mid-face vertical share) over the full frame + seg2 Face_Neck crop (UNION detection policy, measured 21/24 frozen items detected, 3 honest abstains: 2 turned-head/no-face, 1 zero Face_Neck region). Bands calibrated from the measured cohort (eye 0.445/0.475 → 6/11/4, max 52%; mouth 0.333/0.400 → 6/10/5; jaw 0.783/0.830 → 7/11/3; midface 0.48/0.56 → 4/12/4). Support ratio 0.3219 → 0.8115 (Δ +0.4896), supported 47 → 155, unsupported 99 → 36, paired positive 17/21, sign-test p=0.003599. Registry: face-geometry → validated; **object-relations #61 → active** (selected_via exploit, selection_progress 10). One-active invariant holds (14 validated, 0 proposals — #61 is active). Capability-probe findings folded into the module: MediaPipe is resolution-sensitive and non-monotonic on this cohort (same face found full-frame on some items, only on the seg2 crop on others) so crop-only (20/24) or full-frame-only (12/24) both under-detect — the UNION is the honest policy; crop slices must be `ascontiguousarray` (MediaPipe silently drops non-contiguous views).

**Arm #59 matting-alpha ROUND-TRIP COMPLETE → BETTER (2026-08-07, harness-computed, draft PR on `exp/stage-b-matting-alpha-arm59-20260807`).** Deterministic matting / alpha-fidelity evidence from `matting.npy` (Sapiens2 per-pixel soft alpha matte, source-matched, present 24/24 frozen items but unbound by any validated arm) + `seg2` DOME-29 masks: subject alpha-coverage band (sparse/centered/fills-frame), boundary crispness band of the silhouette edge (soft/moderate/crisp from ring alpha-gradient), and soft-edge character (hair-dominant vs mixed vs skin-clean cutout) — all scale-invariant. Bands calibrated from the frozen 24-item probe after the first on-paper thresholds proved DEGENERATE (soft-edge 21/24 "sharp", detail 23/24 "fine-detail", silhouette 24/24 "closed"): the real discriminator is boundary sharpness + hair-edge character — final max share 70.8% (crisp 11/moderate 10/soft 3; hair-dominant 4/mixed 14/skin-clean 6; coverage sparse 5/centered 17/fills-frame 2). Silhouette closedness is honestly non-discriminating on this cohort (24/24 closed, no crops) and stays payload-only. Support ratio 0.3219 → 0.8657 (Δ +0.5438), supported 47 → 187, unsupported 99 → 29, paired positive 18/22, sign-test p=0.002172. Registry: matting-alpha → validated; **face-geometry #60 → active** (selected_via exploit, selection_progress 9). One-active invariant holds (13 validated, 1 proposal).

**Arm #58 pointmap-depth ROUND-TRIP COMPLETE → BETTER (2026-08-07, harness-computed, draft PR on `exp/stage-b-pointmap-depth-arm58-20260807`).** Deterministic point-map / 3D depth-ordering evidence from `pointmap.npy` (Sapiens2 CAM-frame per-pixel cloud, background zeroed) + `seg2` DOME-29 masks: region nearest/farthest depth ranking, left/right hand depth ordering, hand/arm held in front of the torso plane, normalized body depth-relief band — all scale-invariant. Bands calibrated on the frozen 24-item cohort (relief compact/moderate/pronounced = 6/12/6, max 50%; hand_ordering fires 5/24, hand_in_front 11/24). Support ratio 0.3219 → 0.7488 (Δ +0.4269), supported 47 → 158, unsupported 99 → 53, paired positive 19/22, sign-test p=0.000428. Registry: pointmap-depth → validated; **matting-alpha #59 → active** (selected_via explore, ε-greedy slot, selection_progress 8). One-active invariant holds (13 validated, 1 proposal).

**Arm #62 pose-articulation ROUND-TRIP COMPLETE → BETTER (2026-08-07, harness-computed, draft PR #64).** Deterministic kinematic articulation (per-joint elbow/knee flexion, torso/pelvis in-plane orientation, weight-bearing stance + contrapposto, limb-overlap/crossing, flexion asymmetry — all scale-invariant, from pose2 GOLIATH-308 + seg2 DOME-29). Support ratio 0.4225 → 0.8195 (Δ +0.397), supported 60 → 168, unsupported 82 → 37, paired positive 18/22, sign-test p=0.002172. Registry: pose-articulation → validated; **pointmap-depth #58 → active** (exploit, EIG 0.45, tie-broken by id, selection_progress 7). Calibration probe on the frozen cohort confirmed a discriminating elbow band (21 bent / 17 extended) and honest sparse signals for arm-crossing (2/24), contrapposto (4/24), legs-crossed (1/24).

**Arm #47 vlm-dense-description round-trip COMPLETE → BETTER (2026-08-06, harness-computed, draft PR #57).** VLM marginal support ratio 0.7376 → 0.9581 (Δ +0.2206), supported 163→206, unsupported 58→9, sign-test p=0.013302, paired 21/24. Registry: vlm-dense-description → validated. **The sweep then reported EXHAUSTED (10/10 terminal) → next_action brainstorm-new-data.**

**Brainstorm-widen (2026-08-06, this cycle):** the exhausted menu was widened with **5 genuinely-new candidate dimensions**, each with a NEW evidence part or NEW model class (redundant attribute-taggers over validated axes rejected by the gate), registered through the gated `propose-dimensions --require-new-evidence-part` channel and persisted:
- **#58 point-map/3D depth-ordering** (`pointmap-depth`, NEW part; deterministic from existing source-matched `pointmap.npy` present 24/24 on the frozen cohort but unbound by any validated arm).
- **#59 matting/alpha-fidelity** (`matting-alpha`, NEW part; deterministic from existing source-matched `matting.npy`, 24/24).
- **#60 facial detail/face-shape geometry** (`face-geometry` NEW part + `mediapipe-facemesh-3d` NEW model class; open-weight 478-point on-device mesh; local-first).
- **#61 object/accessory presence + spatial relations** (`object-relations` NEW part + `grounding-dino-open-vocab` NEW model class; Apache-2.0 open-vocabulary detection).
- **#62 pose-articulation/kinematic constraints** (`pose-articulation` NEW part; deterministic from existing pose2 GOLIATH-308 + seg2; per-joint angles, contrapposto/weight-bearing, limb-overlap, symmetry).
Selector (`autonomous-select`) computed all 5 with novelty bonus +0.15; **pose-articulation (EIG 0.45, exploit, ties broken by id) won**. `autonomous-tick` (no review root needed for the nothing-active activate path) advanced the registry: **pose-articulation → active** (selection_progress 6). Validation: 525 passed, `validate-program` valid, `validate-dimension-registry` valid. Label-sync applied (issue #62 → research:active). Data-source candidacy for #58/#59 verified on the frozen cohort (pointmap.npy/matting.npy present and source-dimension-matched on 24/24 items).

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
- **Registry** (`research/dimensions/evidence-dimension-registry-v1.json`): **17 validated**
  (body-type, clothing, hair, skin-color, lighting, dossier-context4k #36, setting #34, texture #35,
  reconstruction #37, **vlm-dense-description #47**, **pose-articulation #62**, **pointmap-depth #58**,
  **matting-alpha #59**, **face-geometry #60**, **object-relations #61**, **scene-category #69**,
  **gaze-head-orientation #68**),
  **0 proposals, 0 active**, 0 blocked. `dimension-sweep-status`: `exhausted: true`, `next_action:
  brainstorm-new-data` (menu fully validated 17/17, goal inputs 100%), `goal_unreachable: false`
  (floor 4001, gap 512; the VLM evidence part + deterministic record together clear it).
- **Arm #47 sourcing verification** (2026-08-06, draft PR #48): open-world scan (Molmo-72B, Qwen2.5-VL,
  InternVL3-78B) + local capability probe of `qwen3-vl:32b` (already installed on 4090 + Strix): 4090 is
  27% CPU-offload / ~280s per 2048-token block — too slow for a 96-item batch; Strix (100GB usable) runs
  it 100% GPU ~9.6 tok/s, so Strix is the production batch host for the newly-active VLM arm.

## Immediate next action

**Gaze-head-orientation (#68) is now VALIDATED BETTER and the registry menu is EXHAUSTED (17/17
validated, 0 proposals, 0 active) — `dimension-sweep-status` reports `next_action: brainstorm-new-data`.
Per the exhausted-menu recipe: WIDEN, do not re-run a familiar arm pattern.** Register genuinely-new
candidate dimensions through the gated `propose-dimensions --require-new-evidence-part` channel (each
must name a NEW evidence part or NEW model class; redundant attribute-taggers over validated axes are
rejected). Candidate lines for the next brainstorm-widen (to be validated on the frozen cohort before
freezing): (a) a compositional/framing axis (subject scale + placement in frame from seg2 — NEW
deterministic part); (b) a temporal/dynamic axis is out of scope for single stills, but a lens/
depth-of-field axis (relative background sharpness from source gradients + seg2 focus plane — NEW
detectable part); (c) an open-world NEW-model-class scan for a materially better task model or an
entirely new part (e.g. learned hair/lip/eye color fine-tune vs the current deterministic taggers), per
the sourcing directive — including literature/arXiv scan. Keep exactly one active arm and one open
program root; every new candidate needs the full declaration (scope/inputs/output semantics/
provenance/abstention/known failure modes/qualification gate) + a capability probe on the frozen
24-item cohort before any plan freeze.

## Live research tree

- #2 is the sole open program root.
- #3 is the preserved PENDING portrait-evidence map.
- #4 is the baseline/comparison-parity arm (empirically complete, verdict BETTER, human spot-check advisory).
- #5 is the preserved geometry-grounded-captioning prototype.
- #9 closed (comparison-plan provenance gate resolved). #18 CLOSED/released (owner directive 2026-08-04).
- #29–#47 registered proposal arms; #32, #29, #30, #31, #33, #36, #34, #35, #37, **#47 vlm-dense-description**,
  **#62 pose-articulation**, and **#58 pointmap-depth** are ALL validated (all BETTER);
  **#36 dossier-context4k is the validated goal arm (round-trip BETTER)**.
- **Post-exhaustion brainstorm-widen (2026-08-06/07):** proposal arms **#58 point-map depth**, **#59 matting/alpha**,
  **#60 face-geometry**, **#61 object-relations**, **#62 pose-articulation** registered via the gated
  `propose-dimensions --require-new-evidence-part` channel (all name a NEW evidence part; #60/#61 also name
  a NEW model class). **#62 pose-articulation VALIDATED (2026-08-07, PR #64); #58 pointmap-depth VALIDATED
  (2026-08-07); matting-alpha #59, face-geometry #60, object-relations #61 ALL VALIDATED (2026-08-07)**.
  **Second widen (2026-08-07): #68 gaze-head-orientation + #69 scene-category registered; BOTH now
  VALIDATED (2026-08-07).** Registry menu is **EXHAUSTED (17/17 validated, 0 proposals, 0 active)** —
  `next_action: brainstorm-new-data` (third widen pending this cycle).
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
Arm #34: BETTER; Arm #35: BETTER; Arm #36 (goal): BETTER; Arm #37 (reconstruction): BETTER; Arm #47 (VLM dense): BETTER;
Arm #62 (pose-articulation): BETTER; Arm #58 (pointmap-depth): BETTER; Arm #59 (matting-alpha): BETTER;
Arm #60 (face-geometry): BETTER; Arm #61 (object-relations): BETTER; Arm #69 (scene-category): BETTER;
Arm #68 (gaze-head-orientation): BETTER.**
Declared deterministic evidence (geometry; body-type proportions; DOME-29 clothing coverage + dominant
colors; hair region + color; exposed-skin tone; lighting luma/DR/shadow/direction; setting background
coverage/color/bands; texture fabric/skin surface+pattern bands) each significantly improves supported
claims on the frozen 24-item cohort under fixed view/prompt/model/settings. The **goal-arm round-trip is
BETTER**: captions generated FROM the evidence-linked ≤4K compact context (supported 47→174,
ratio 0.322→0.777, p=0.000244) beat the plain-4K summarization baseline. **The generative reconstruction
check is BETTER too** (non-LLM validation of the same compact context): context4k-generated images score
+0.0679 mean CLIP ViT-L/14 similarity over the degraded-baseline generations (22/24 paired positives).
The **VLM dense-description marginal is BETTER** (0.7376→0.9581 support ratio, p=0.013302). After the
10/10 sweep was exhausted, the **brainstorm-widen registered 5 new candidate arms (#58–#62)**, and
**pose-articulation (#62), pointmap-depth (#58), matting-alpha (#59), face-geometry (#60), and
object-relations (#61) ALL VALIDATED (2026-08-07)**. A **second widen registered #68 gaze-head-orientation
and #69 scene-category — ALL 17/17 VALIDATED (2026-08-07)**; the menu is **EXHAUSTED → third
brainstorm-widen (next_action: brainstorm-new-data)**. The **gaze/head-orientation round-trip is BETTER**
(support ratio 0.3219→0.9673, p=1e-06, Δ +0.6454) — the strongest single-arm delta yet.
