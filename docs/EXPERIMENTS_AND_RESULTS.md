# Experiments & Results — Stratum Contextual Specialist Research

This ledger records empirical findings and negative results permanently. A green implementation, readable artifact, or passing unit test is not an empirical PASS.

## Post-exhaustion brainstorm-widen (2026-08-06) — `[BRAINSTORM — 5 NEW PROPOSALS REGISTERED; POSE-ARTICULATION SELECTED ACTIVE]`

**Date:** 2026-08-06 (after arm #47 vlm-dense-description validated → sweep EXHAUSTED → gate returned `next_action: brainstorm-new-data`)
**Gate:** `dimension-sweep-status` → `exhausted: true` (10/10 terminal), `goal_unreachable: false` (floor 4001, VLM part + deterministic record clear it), 0 proposals.
**Brainstorm-widen (data-source + evidence-part + model-candidate sourcing):**
- Data-source candidacy **verified on the frozen cohort**: `pointmap.npy` (source-dimension-matched, float16, (H,W,3)) and `matting.npy` (alpha, (H,W)) are present on **24/24** frozen items, and are `core`-chain artifacts **NOT bound as evidence input by any validated arm** — genuinely-new deterministic surfaces (relative 3D depth/occlusion ordering; alpha-fidelity/soft-edge/silhouette).
- Model sourcing scan (2026-08-06): **MediaPipe Face Mesh** (Apache-2.0, 478-point on-device 3D face mesh — open, standard, local-first) and **Grounding DINO** (Apache-2.0, text-grounded open-vocabulary detection, HF Transformers path) as the new-model-class candidates for the two learned axes.
- Registered via the gated channel `propose-dimensions --require-new-evidence-part --write` (5/5 accepted, 0 rejected): **#58 `pointmap-depth`** (NEW part), **#59 `matting-alpha`** (NEW part), **#60 `face-geometry`** (NEW part `face-geometry` + NEW model class `mediapipe-facemesh-3d`), **#61 `object-relations`** (NEW part + NEW model class `grounding-dino-open-vocab`), **#62 `pose-articulation`** (NEW part; deterministic from pose2+seg2).
**Selector (`autonomous-select`):** all 5 scored with novelty bonus +0.15; **#62 pose-articulation** EIG **0.45**, exploit, ties broken by id (over pointmap-depth 0.45). Scores: pointmap-depth 0.45, matting-alpha 0.39, face-geometry 0.30, object-relations 0.24.
**Tick (`autonomous-tick --write`, nothing-active activate path):** `next_action: activate`, next_arm **pose-articulation**, selected_via exploit, **selection_progress 6**.
**Validation:** pytest **525 passed**; `validate-program` valid; `validate-dimension-registry` valid.
**Label-sync (`sync-issue-labels --apply`):** 2 ops applied — issue #62 +`research:active`, −`research:proposal`; #58–#61 keep `research:proposal`.
**Registry now:** 10 validated + 5 proposals + 1 active (pose-articulation), 0 blocked, `exhausted: false`, `next_action: none` (research-pending on the active arm).
**Next (measurement, not yet executed):** pose-articulation deterministic measurement (joint angles, torso/pelvis orientation, contrapposto/weight-bearing, limb-overlap, symmetry) → band-calibration probe on frozen cohort → freeze → 96-caption round-trip → review → tick. The other four proposals await the selector.

## Arm #47 — open-weight VLM dense-description round-trip — `[EMPIRICAL RUN COMPLETE — VERDICT: BETTER]`

**Date:** 2026-08-06 (harness surface landed; block batch complete; round-trip complete)
**Arm:** #47 — vlm-dense-description (option-B dossier-growth evidence part; was sole `research:active` arm, exploit, EIG 0.10, novelty 0.15, selection_progress 5)
**Code / PR:** `feat/vlm-dense-arm47-20260806` (draft PR #57, branch from the arm-#37 lineage)
**Harness surface (this cycle, additive):**
- `vlm-dense` evidence kind (5-condition plan: 3 neutral controls + `context-raw-context4k` matched baseline + `context-raw-vlm-dense` variant); per-item `vlm_blocks_sha256` pinned in the frozen plan; caption runner byte-verifies blocks and fails closed on drift.
- `autonomous-tick --baseline-condition/--evidence-condition` so the verdict isolates the **VLM marginal** (compact → compact+VLM block). Plain plan-derived derivation pairs no-evidence → combined record (a re-measurement of arm #36's direction), so explicit conditions are REQUIRED for this arm's falsified_if.
- `scripts/vlm_dense_generate.py` (frozen gemma3:27b dense-description batch; 2×2 collage of full-frame + seg2 focal crops; tagged scale-invariant prose) + `scripts/vlm_dense_scheduler.py` (owns the Strix lifecycle) + `scripts/freeze_vlm_dense_manifest.py`.
- Tests: `tests/test_vlm_dense.py` (9 new); full suite 525 passed; validators green.
**Model-qualification event (recorded):** pre-registered **qwen3-vl:32b FAILED qualification on the real corpus** (silent empty decodes on vision and plain text; server-side wedge, not restartable without root). **Evidence producer flipped to gemma3:27b** (digest a418f5838eaf, already local) and the 24-block batch was regenerated homogeneously.
**Round-trip run:** blocks `stage-b-vlm-dense-v1` (24/24, 578 tagged claims, 0 px leaks — cohort `[ABSTAIN]` rate 0/578 flagged for the abstention audit); captions `stage-b-vlm-dense-captions-v1` (120 records: 24×5 conditions, gemma3:27b, PENDING_INDEPENDENT_REVIEW); review `stage-b-vlm-dense-captions-v1-review` (120 rows, gemma4:e4b).
**Verdict (harness-computed `autonomous-tick --review-dir … --baseline-condition context-raw-context4k --evidence-condition context-raw-vlm-dense --write`):** **BETTER** — VLM marginal support ratio 0.7376 → 0.9581 (Δ +0.2206), supported 163 → 206, unsupported 58 → 9 (not ballooning), **sign-test p = 0.013302**, paired 21/24 (16 positive). `item_count 24`, `significant true`, `inconclusive false`.
**Registry advance:** vlm-dense-description `active → validated` (0 strikes); selection_progress 5. **All 10 dimensions now terminal → `dimension-sweep-status` reports `exhausted: true`, `next_action: brainstorm-new-data`.** The loop's next step is to WIDEN (new evidence parts / model candidates / data sources) via the gated `propose-dimensions`.
**Abstention-audit flag (pre-registered gate, surfaced honestly):** the cohort block set emitted **0/578 [ABSTAIN]** tags. The claim-support reviewer independently endorsed the vlm-dense captions (206/215 supported), and the blocks carry [INFERRED] tags, but a 0% abstention rate is exactly the "unchecked verbosity" risk the qualification gate names; it remains on the human spot-check + adversarial checklist, not papered over.
**Next (next cycle):** brainstorm-new-data → `propose-dimensions --require-new-evidence-part` for the next menu. The VLM evidence part (≈450–725 tokens/item) sits on top of the deterministic record (2040–3489 tokens/item), which together honestly clear the reframed structural floor 4001 for the goal-arm dossier — the option-B growth path is now empirically validated.


## Arm #62 — pose-articulation / kinematic-constraint evidence — `[EMPIRICAL RUN COMPLETE — VERDICT: BETTER]`

**Date:** 2026-08-07
**Arm:** #62 — pose-articulation (registered in the 2026-08-06 brainstorm-widen; was sole `research:active` arm, exploit, EIG 0.45, novelty +0.15)
**Code / PR:** `exp/pose-articulation-arm62-20260807` (draft PR #64, branch from the brainstorm-widen #63 lineage)
**Cohort:** frozen 24-item first-500 coverage-balanced subset (same manifest as all prior arms)
**Deterministic specialist:** `research_harness.pose_articulation.compute_pose_articulation` — scale-invariant kinematic articulation from pose2 GOLIATH-308 keypoints + seg2 DOME-29 masks: per-joint elbow/knee flexion angles (bent < 135° vs extended ≥ 135°), in-plane torso/pelvis orientation (torso twist / lean-from-vertical / pelvis tilt), weight-bearing stance class (weight-left/right/centered) + contrapposto detection (weight shift + pelvis tilt ≥ 4°), arm-over-spine crossing count, legs-crossed signal, seg2 arm-near-torso proximity fraction, flexion symmetry/asymmetry. Evidence inputs bound: `pose2.npy` + `seg2.npy`. Only scale-invariant facts are verbalized (flexion bands, stance classes, in-plane angles, crossing counts); absolute pixel positions/lengths and raw angles stay in the machine-readable `evidence_payload`. Body-type #32 measures static anthropometric RATIOS; #62 measures articulated KINEMATICS (joint angles, stance, overlap) — the falsified_if's redundancy axis is not engaged because the signals are measurement-distinct.
**Hypothesis:** Declared kinematic articulation facts reduce unsupported pose/stance/limb claims in captions vs matched no-evidence baseline.
**Calibration note (band probe on the real frozen cohort, before freeze):** elbow flexion 21 bent / 17 extended / 10 n/a across 48 measurements (mean 112.5°, p25 71.5 / p50 114.8 / p75 169.3) — no band ≥ 75%, the threshold discriminates; knee 19 measurable (mean 92.5°). Stance resolved 11/24, torso twist 17/24, arm-crossing 2/24, contrapposto 4/24, legs-crossed 1/24. The non-verbose signals (arms-crossing, contrapposto, legs-crossed) are genuinely sparse on this cohort but are precise when they fire.
**Run:** `/mnt/nas-ai-models/research/stratum/stage-b-pose-articulation-v1` (96 captions, gemma3:27b A-fingerprint, 4090 via scheduler; 4 conditions × 24) + independent adversarial review `/mnt/nas-ai-models/research/stratum/stage-b-pose-articulation-v1-review` (96 rows, gemma4:e4b).
**Verdict (harness-computed `autonomous-tick --review-dir … --write`):** **BETTER** — supported 60 → **168**, unsupported 82 → **37**; support ratio 0.4225 → 0.8195 (Δ +0.397); paired positive 18/22; sign-test p = **0.002172**. `inconclusive: false`, `significant: true`. (Note: this arm's matched baseline `context-raw-no-evidence` ran 60/82 — the same no-evidence condition as prior arms, so cross-arm ratio deltas are comparable in direction even though absolute counts moved with each new dossier dimension added.)
**Registry advance:** pose-articulation `active → validated` (0 strikes); selector next → **pointmap-depth #58** (active, exploit, EIG 0.45, tie-broken by id, selection_progress 7).
**Boundaries respected:** local models only; outputs only under the approved noncanonical research root; no `crawlr/approved` or `crawlr/stratum` mutation; no backfill; no legacy overwrite; deterministic evidence computed in memory from existing `pose2.npy`/`seg2.npy` only; scale-invariant verbalization retained (owner px→ratios rule). The pose-articulation dimension joined the dossier (`pose-articulation:v1` evidence id + render_pose_articulation + evidence_payload section) — honest evidence-density growth for the goal arm.
**Validation:** 532 pytest passed; `validate-program` valid; `validate-dimension-registry` valid; `validate-gpu-manifest` valid. Verdict BETTER is empirical on this 24-item frozen cohort; a formal PASS still awaits the advisory human rubric spot-check.


## Arm #58 — point-map / 3D depth-ordering evidence — `[EMPIRICAL RUN COMPLETE — VERDICT: BETTER]`

**Date:** 2026-08-07
**Arm:** #58 — pointmap-depth (registered in the 2026-08-06 brainstorm-widen; was sole `research:active` arm, exploit, EIG 0.45, novelty +0.15, selection_progress 7)
**Code / PR:** `exp/stage-b-pointmap-depth-arm58-20260807` (draft PR, branch from the pose-articulation #64 lineage)
**Cohort:** frozen 24-item first-500 coverage-balanced subset (same manifest as all prior arms)
**Deterministic specialist:** `research_harness.pointmap_depth.compute_pointmap_depth` — scale-invariant depth-ordering from `pointmap.npy` (Sapiens2 per-pixel CAM-frame 3D cloud, +X right +Y down +Z toward viewer, background zeroed) + seg2 DOME-29 masks: region nearest/farthest depth ranking (head/torso/arms/hands/legs median-Z ranks), left/right hand depth ordering (one hand clearly nearer), hand/arm held in front of the torso plane (per side), normalized body depth-relief ratio (robust Z spread p10–p90 / median Z) banded compact/moderate/pronounced. Evidence inputs bound: `pointmap.npy` + `seg2.npy` (pose2 stays a validation-only read). Only scale-invariant facts are verbalized (ordering relations, normalized ratios, bands); absolute metric Z values and raw spreads stay in the machine-readable `evidence_payload`. Genuinely-new evidence surface: relative depth/occlusion ordering no previously-validated arm binds (arms/pose bind pose2/seg2/normal2 only, not pointmap).
**Hypothesis:** Declared point-map depth-ordering facts reduce unsupported depth/occlusion/foreground-background claims in captions vs matched no-evidence baseline.
**Calibration note (band probe on the real frozen cohort, before freeze):** paper thresholds (relief floor 0.30 / pronounced 0.55) were DEGENERATE (24/24 compact); recalibrated to the measured distribution (range 0.051–0.241, p10/p50/p90 0.053/0.124/0.214) → relief floor 0.09 / pronounced 0.16, final split compact 6 / moderate 12 / pronounced 6 (max share 50%, no band ≥ 75%). hand_ordering fires 5/24 (left 2, right 3), hand_in_front 11/24 — sparse-but-precise, reported honestly. nearest_region: hands 12/24, then head/torso/legs/arms.
**Run:** `/mnt/nas-ai-models/research/stratum/stage-b-pointmap-depth-v1` (96 captions, gemma3:27b A-fingerprint, 4090 via scheduler; 4 conditions × 24) + independent adversarial review `/mnt/nas-ai-models/research/stratum/stage-b-pointmap-depth-v1-review` (96 rows, gemma4:e4b).
**Verdict (harness-computed `autonomous-tick --review-dir-from <marker> --write`):** **BETTER** — supported 47 → **158**, unsupported 99 → **53**; support ratio 0.3219 → 0.7488 (Δ +0.4269); paired positive 19/22; sign-test p = **0.000428**. `inconclusive: false`, `significant: true`.
**Registry advance:** pointmap-depth `active → validated` (0 strikes); selector next → **matting-alpha #59** (active, selected_via **explore** — ε-greedy slot at selection 8, selection_progress 8).
**Boundaries respected:** local models only; outputs only under the approved noncanonical research root; no `crawlr/approved` or `crawlr/stratum` mutation; no backfill; no legacy overwrite; deterministic evidence computed in memory from existing `pointmap.npy`/`seg2.npy` only; scale-invariant verbalization retained (owner px→ratios/meters→payload rule). The pointmap-depth dimension joined the dossier (`pointmap-depth:v1` evidence id + render_pointmap_depth + evidence_payload section) — honest evidence-density growth for the goal arm.
**Validation:** 539 pytest passed; `validate-program` valid; `validate-dimension-registry` valid; `validate-gpu-manifest` valid. Verdict BETTER is empirical on this 24-item frozen cohort; a formal PASS still awaits the advisory human rubric spot-check.


## Arm #37 — generative reconstruction validation — `[EMPIRICAL RUN COMPLETE — VERDICT: BETTER]`

**Date:** 2026-08-06
**Arm:** #37 — generative reconstruction validation (ComfyUI round-trip), non-LLM measure of the goal-arm context
**Code / PR:** `exp/stage-b-reconstruction-arm37-20260806` (draft PR #56, branch from the arm-#35 lineage)
**Cohort:** frozen 24-item first-500 coverage-balanced subset (same manifest as all prior arms)
**Pre-registered plan:** `research/stage-b-plans/stage-b-reconstruction-v1.json` (status `preregistered`, written BEFORE generation) — variant prompt = per-item `dossier-context4k-v2/<id>/context4k.md` compact verbatim; baseline prompt = fixed item-independent degraded text (no context), identical across items; null calibration = meaningless tokens, 2 images; per-item seed = sha256(image_id) truncated, SAME across conditions; checkpoint Juggernaut XL (`Juggernaut_XL_v1759168.safetensors`, sha256 pinned `dd08fa32…`), dpmpp_2m/karras, 28 steps, cfg 7.0, 832×1216, negative prompt empty; scorer pre-registered `openai/clip-vit-large-patch14` (ViT-L/14, CLIPProcessor 224 center-crop, cosine of [CLS]).
**Run:** `stratum-stage-b-reconstruction-v1` on the 4090 via the scheduler (request → poll claim → ComfyUI boot → VRAM verify → activate → heartbeat → generate 24×2+2 → release). 50/50 images generated and preserved under `/mnt/nas-ai-models/research/stratum/stage-b-reconstruction-v1/outputs/`.
**Verdict (harness-computed `autonomous-tick --method reconstruction`):** **BETTER** — `reconstruction_delta = +0.067888` (mean per-item CLIP ViT-L/14 similarity: context4k-generated vs source minus baseline-generated vs source), median +0.075418, **paired positive 22/24** (2 negative, 0 ties), items 24. Null-case floor (meaningless prompt) = 0.5949 — the registered null bounds the generic-person similarity. `BETTER iff delta > 0` per the pre-registered rule.
**Registry advance:** reconstruction `active → validated` (0 strikes); selector next → **vlm-dense-description #47** (active, exploit, EIG 0.10, tie-broken by id, selection_progress 5).
**Incident (recorded honestly in run-provenance.json):** the scheduler claim was released with status `failed` because the CLIP image processor could not load — the HF cache held only `config.json` for the pre-registered scorer. The generated artifacts were complete and deterministic; the model+processor were then downloaded to `$HF_HOME` (`/mnt/nas-ai-models/huggingface-cache`, authorized open-weight download to owned hardware) and scoring/aggregation were completed in a separate CPU pass (`research_harness.recon_score`), which wrote `delta.json`/`records.jsonl`/`run-provenance.json` (50 rows) into the same run root. No measured number depends on the incident; it is a lifecycle transparency note. Future runs preflight the scorer before claiming.
**Harness fix in the same PR (latent one-active invariant bug):** a NOT_BETTER verdict below the falsification limit left the struck arm `active` AND then activated the next proposal — two `research:active` dims (the next tick hard-failed with "more than one research:active arm"). `run_tick` now keeps the struck arm the sole active one (retry; brainstorm-on-stall if the sweep is stalled) and selects the next arm only after `validate`/`falsify`. Regression tests: `test_run_tick_not_better_strike_keeps_one_active`, `test_run_tick_third_strike_falsifies_then_activates_next`.
**Boundaries respected:** local models only; read-only canonical sources; outputs only under `/mnt/nas-ai-models/research/stratum`; scheduler-managed GPU; no `crawlr/approved` or `crawlr/stratum` mutation; no backfill; no legacy overwrite; scale-invariant scoring (CLIP cosine is framing-agnostic and no pixel values are quoted as evidence).
**Validation:** 516 pytest passed; `validate-program` valid; `validate-dimension-registry` valid. Verdict BETTER is empirical on this 24-item frozen cohort under the frozen checkpoint/sampler; a formal PASS still awaits the advisory human rubric + reconstruction-protocol spot-check.

## Arm #34 — setting/environment evidence — `[EMPIRICAL RUN COMPLETE — VERDICT: BETTER]`

**Date:** 2026-08-06
**Arm:** #34 — setting/environment evidence specialist
**Code / PR:** `exp/stage-b-setting-arm34-20260806` (draft PR #53, branch from the arm-#36 lineage)
**Cohort:** frozen 24-item first-500 coverage-balanced subset (12 portrait / 6 squareish / 6 landscape — same manifest as arms #4/#29/#30/#31/#32/#33/#36)
**Deterministic specialist:** `research_harness.setting.compute_setting` (DOME-29 Background (class 0) frame-coverage ratio, quantized dominant background color name + hex, tone/vibrancy/pattern bands from the non-subject source pixels; raw-pixel + frame-coverage gates with explicit abstention). Evidence inputs bound: `seg2.npy`. Only scale-invariant facts are verbalized (coverage ratio, color name, bands); absolute pixel counts and raw RGB/hex stay in the machine-readable `evidence_payload`.
**Hypothesis:** Declared background palette/non-subject statistics improve scene-environment claim support.
**Run:** `/mnt/nas-ai-models/research/stratum/stage-b-setting-v1` (96 captions, gemma3:27b A-fingerprint, 4090 via scheduler) + independent adversarial review `/mnt/nas-ai-models/research/stratum/stage-b-setting-v1-review` (96 rows, gemma4:e4b, 512px inputs).
**Verdict (harness-computed `autonomous-tick`):** **BETTER** — supported 47 → **177**, unsupported 99 → **33**; support ratio 0.3219 → 0.8429 (Δ +0.5209); paired positive 19/24; sign-test p = **0.003305**. `inconclusive: false`, `significant: true`.
**Registry advance:** setting `active → validated` (0 strikes); selector next → **texture #35** (active, exploit, EIG 0.24, novelty 0.15, selection_progress 3).
**Boundaries respected:** local models only; outputs only under the approved noncanonical research root; no `crawlr/approved` or `crawlr/stratum` mutation; no backfill; no legacy overwrite; deterministic evidence computed in memory from existing `seg2.npy`/source pixels only; scale-invariant verbalization retained (owner px→ratios rule). The setting dimension also joined the dossier (`setting-environment:v1` evidence id + render_setting + evidence_payload section) — honest evidence-density growth for the goal arm.
**Calibration note:** the pattern-band thresholds were recalibrated after an honest deviation-histogram probe on the first 8 items (solid < 0.10 deviant fraction, busy ≥ 0.35); the first bands (0.04/0.18) over-labeled plain walls as "busy".
**Validation:** 495 pytest passed; `validate-program` valid; `validate-dimension-registry` valid. Verdict BETTER is empirical on this 24-item frozen cohort; a formal PASS still awaits the advisory human rubric spot-check.

## Arm #35 — texture/material evidence — `[EMPIRICAL RUN COMPLETE — VERDICT: BETTER]`

**Date:** 2026-08-06
**Arm:** #35 — texture/material evidence specialist
**Code / PR:** `exp/stage-b-texture-arm35-20260806` (draft PR #54, branch from the arm-#34 lineage)
**Cohort:** frozen 24-item first-500 coverage-balanced subset (same manifest as all prior arms)
**Deterministic specialist:** `research_harness.texture.compute_texture` (per-region-class texture statistics: dominant measurable fabric class surface/pattern bands, dominant skin class surface band, from seg2 masks + source-pixel gradients; per-channel p99.9-normalized gradients over the 1-px-eroded region interior so the silhouette boundary never counts as texture). Evidence inputs bound: `seg2.npy`. Only scale-invariant facts are verbalized (bands/fractions); absolute gradient values and pixel counts stay in the machine-readable `evidence_payload`.
**Hypothesis:** Declared per-region texture proxies (edge/gradient statistics) reduce invented material/texture claims.
**Run:** `/mnt/nas-ai-models/research/stratum/stage-b-texture-v1` (96 captions, gemma3:27b A-fingerprint, 4090 via scheduler) + independent adversarial review `/mnt/nas-ai-models/research/stratum/stage-b-texture-v1-review` (96 rows, gemma4:e4b, 512px inputs).
**Verdict (harness-computed `autonomous-tick`):** **BETTER** — supported 47 → **167**, unsupported 99 → **34**; support ratio 0.3219 → 0.8308 (Δ +0.5089); paired positive 21/24; sign-test p = **0.000139**. `inconclusive: false`, `significant: true`.
**Registry advance:** texture `active → validated` (0 strikes); selector next → **reconstruction #37** (active, explore slot, EIG 0.10, novelty 0.15, selection_progress 4).
**Boundaries respected:** local models only; outputs only under the approved noncanonical research root; no `crawlr/approved` or `crawlr/stratum` mutation; no backfill; no legacy overwrite; deterministic evidence computed in memory from existing `seg2.npy`/source pixels only; scale-invariant verbalization retained. The texture dimension also joined the dossier (`texture-material:v1` evidence id + render_texture + evidence_payload section) — honest evidence-density growth for the goal arm.
**Calibration note:** the first garment-only design abstained on 13/24 items (the cohort is ~half topless portraits with no garment classes) and pooled per-class means degenerated into a fake 11/11 "busy". Per-class dominant-region measurement (fabric where present else skin) restored 24/24 measurability, and the erosion fix removed a pure silhouette-boundary artifact that inflated edge_fraction; bands were then recalibrated on the real cohort (fabric 4/3/3, pattern 5/3/2, skin 15/7/2 — no band ≥ 75%).
**Validation:** 506 pytest passed; `validate-program` valid; `validate-dimension-registry` valid. Verdict BETTER is empirical on this 24-item frozen cohort; a formal PASS still awaits the advisory human rubric spot-check.

## Arm #36 — post-ruling honest re-measurement under the reframed structural floor — `[MEASUREMENT — NOT A VERDICT]`

**Date:** 2026-08-06 (cycle after PR #50 merged)
**Arm:** #36 — dossier-context4k (goal arm, unblocked post-ruling)
**Code:** `research_harness/dossier_expand.py` made program-floor-aware (`floor_gap_analysis` now reads the contract's `expanded_dossier_min_tokens`; the audit runner passes both floors). Regression test added (`test_floor_gap_analysis_reframed_structural_floor_is_lm_reachable`).
**Run:** `/mnt/nas-ai-models/research/stratum/dossier-expansion-audit-v2/` (24/24 frozen items, CPU-only, same frozen manifest as v1 — only the floor the audit compares against changed).
**Measured:** expanded prose 1358–2525 (median 1841.5); total dossier record 2040–3489; generous honest LM elaboration ceiling 8500–13500 tokens/item.
**Honest verdict under the reframed contract (structural expanded floor 4001, compact 4000):**
- deterministic+payload record alone: `any_expanded_floor_reached = false` (still under 4001)
- **`any_max_honest_floor_reached = true` for all 24 items** — the honest LM elaboration ceiling clears the structural floor, so the round-trip claim-support audit is feasible at honest scale via the scheduler-bound aggregator expansion stage, without fabricating content.
**Registry:** `dossier-context4k` unblocked (`mark-unblocked` after the #46 ruling landed via owner-merged PR #50) → `proposal`; selector re-picks it via exploit (EIG 0.34). **No PASS/verdict**: this is a re-measurement, not an empirical BETTER/NOT_BETTER conclusion. The v1 "7–50× gap vs 100K" framing is superseded by the reframed 4001 structural floor.

## Arm #33 — lighting evidence — `[EMPIRICAL RUN COMPLETE — VERDICT: BETTER]`

**Date:** 2026-08-05
**Arm:** #33 — lighting evidence specialist
**Code / PR:** `exp/stage-b-lighting-arm33-20260805` (draft PR), stacked on the arm-#31 execution harness
**Cohort:** frozen 24-item first-500 coverage-balanced subset (12 portrait / 6 squareish / 6 landscape — same manifest as arms #4/#29/#30/#31/#32)
**Deterministic specialist:** `research_harness.lighting.compute_lighting` (camera-space normal2 direction statistics + source luminance/histogram/dynamic-range: luma band, dynamic-range band, shadow fraction, subject-vs-surround ratio, Lambertian least-squares key-light direction fit; gate-floor abstention). Evidence inputs bound: `seg2.npy + normal2.npy`. Only scale-invariant facts are verbalized (bands/fractions/direction name); continuous values and the fitted light vector stay in `evidence_payload`.
**Deterministic probe (CPU, pre-run):** 24/24 subject present; 24/24 lighting measurable; 24/24 light direction resolved. Luma bands 6 brightly-lit / 18 moderately-lit; shadow 3 heavy / 11 some / 10 little; DR 24/24 high contrast; direction 15 front / 5 front-left / 4 front-right. All 24 captions carry lighting evidence.
**Aggregator:** already-installed local `gemma3:27b` (digest `a418f5838eaf…`), temperature=0, seed=20260804, num_predict=384, num_ctx=4096, loopback Ollama.
**Independent reviewer:** `gemma4:e4b` (different family), temperature=0, seed=20260804, num_predict=2000, 512×512 input, same reviewer calibration as arms #4/#29/#30/#31/#32.
**Plan:** frozen `research/stage-b-plans/stage-b-lighting-v1.json` (fingerprint `87498692900653675bdd6a856d433bc9d94d5377489c1bb8c7ff5cab29bedeb3`); conditions identical to prior arms except the evidence condition is `context-raw-lighting`. All 24/24 cohort items measured and generated.
**Scheduler lifecycle (local 4090):** request → poll/claim → launch → verify GPU activity → activate → heartbeat → release. Generation job `stratum-stage-b-lighting-v2`, completed 2026-08-05 ~19:53Z, `gpu_activity_seen: true`, 96/96, plan fingerprint verified (an initial `-v1` claim was consumed as failed before publish by the evidence-hash guard and produced no metric — an infrastructure failure, not a strike). Independent review job `stratum-stage-b-adversarial-review-lighting-v2`, 96/96 reviewed by gemma4:e4b, completed ~16:08Z, GPU released. Both slots released cleanly (4090 idle).

**Evidence-only delta (cond 3 → 4):**
- supported claims **47 → 194**; unsupported **99 → 10**; omissions 11 → 9; contradictions 1 → 4; abstentions 0 → 1.
- Support ratio **32.2% → 95.1%** (Δ +0.6291) — the largest delta seen across the cohort so far.
- Paired per-item sign test on supported claims: 19 improve / 4 not (23 paired), one-sided binomial **p ≈ 0.0013**.
- Deterministic cross-check independent of the LLM review: 21/24 lighting-condition captions carry ≥1 declared lighting-vocab trace beyond their matched baseline (contrast/bright/dim/shadow/highlight/key light); the remaining 3 already described lighting in baseline. Supported-claim gain is traceable to carried lighting evidence.

**Deterministic verdict:** `autonomous-verdict --base-supported 47 --variant-supported 194 --base-unsupported 99 --variant-unsupported 10 --items 23 --p-supported 0.0013` → **BETTER** (significant p=0.0013 ≤ 0.05; support-ratio improvement 0.322→0.951; unsupported reduced 99→10, not ballooning; `inconclusive: false`). Confirmed by harness `autonomous-tick` from the review dir.

**Boundaries respected:** local models only; outputs only under the approved noncanonical research root `/mnt/nas-ai-models/research/stratum/stage-b-lighting-v2(-review)`; no `crawlr/approved` or `crawlr/stratum` mutation; no backfill; no legacy overwrite; deterministic evidence computed in memory from existing `normal2.npy`/`seg2.npy`/source pixels only; scale-invariant verbalization retained (owner px→ratios rule).

**Registry advance:** lighting dimension `active → validated` (0 strikes, confirmed by harness `autonomous-tick`). Next selector pick: **dossier-context4k (arm #36, EIG 0.19)** now `research:active`. Verdict BETTER is empirical on this 24-item frozen cohort; a formal PASS still awaits the advisory human rubric spot-check (single independent reviewer family, rubric not yet human-calibrated on known/null cases).

## Arm #31 — skin-color/tone evidence — `[EMPIRICAL RUN COMPLETE — VERDICT: BETTER]`

**Date:** 2026-08-05
**Arm:** #31 — skin-color/tone evidence specialist
**Code / PR:** `exp/stage-b-skin-color-arm31-20260805` (draft PR #41, stacked on the arm-#30 execution harness)
**Cohort:** frozen 24-item first-500 coverage-balanced subset (12 portrait / 6 squareish / 6 landscape — same manifest as arms #4/#29/#32/#30)
**Deterministic specialist:** `research_harness.skin_color.compute_skin_tone` (seg2 DOME-29 exposed-skin regions: Face_Neck/Torso/limb/hand/foot classes; aggregate exposure coverage + quantized dominant tone from source pixels + face/neck vs body agreement; gate-floor abstention); computed in memory per item during the bounded run. Only scale-invariant facts are verbalized (exposure fraction, quantized tone name); px stays in `evidence_payload`.
**Deterministic probe (CPU, pre-run):** 24/24 subject present, 24/24 exposed-skin tone measurable; tone histogram tan 9 / brown 8 / dark brown 3 / medium 2 / light medium 2; 23/24 face+body both measurable (8/23 region-agree). All 24 captions carry skin-tone evidence.
**Aggregator:** already-installed local `gemma3:27b` (digest `a418f5838eaf…`), temperature=0, seed=20260804, num_predict=384, num_ctx=4096, loopback Ollama.
**Independent reviewer:** `gemma4:e4b` (different family), temperature=0, seed=20260804, num_predict=2000, 512×512 input, same reviewer calibration as arms #4/#29/#32/#30.
**Plan:** frozen `research/stage-b-plans/stage-b-skin-color-v1.json` (fingerprint `a9a681e470424eacb14193d02c28289510b8ea3d160e8fe4ed6f920f3aa9f3b1`); conditions identical to arms #4/#29/#32/#30 except the evidence condition is `context-raw-skin-color` (DOME-29 exposed-skin tone) instead of `context-raw-geometry`/`context-raw-clothing`/`context-raw-body-type`/`context-raw-hair`. All 24/24 cohort items measured and generated.

**Scheduler lifecycle (local 4090):** request → poll/claim → launch → verify GPU activity → activate → heartbeat → release, through `registered-research-launcher` (job `stratum-stage-b-skin-color-v1`, completed 2026-08-05 ~17:34Z, `gpu_activity_seen: true`, 96/96, plan fingerprint verified) and the independent review pass (job `stratum-stage-b-adversarial-review-skin-color-v1`, 96/96 reviewed by gemma4:e4b, completed ~17:51Z, GPU released). Both slots released cleanly (4090 idle).

**Evidence-only delta (cond 3 → 4):**
- supported claims **47 → 176**; unsupported **99 → 21**; omissions 11 → 24; contradictions 1 → 8; abstentions 0 → 0.
- Support ratio **32.2% → 89.3%** (Δ +0.5715).
- Paired per-item sign test on supported claims: 20 improve / 4 worsen (24 paired), one-sided binomial **p ≈ 0.000772**.
- Deterministic cross-check independent of the LLM review: 22/24 skin-condition captions carry the exact declared tone name beyond baseline (the remaining 2 are palette-neighbor synonyms, e.g. "fair-skinned" vs "medium"); all 24 name skin tone. Baseline captions already routinely name a skin tone (the context prompt asks for "skin details"), so the supported-claim gain is the reviewer's attribution of those claims to carried evidence — matching the hair arm's carry profile.

**Deterministic verdict:** `autonomous-verdict --base-supported 47 --variant-supported 176 --base-unsupported 99 --variant-unsupported 21 --items 24 --p-supported 0.000772` → **BETTER** (significant p=0.000772 ≤ 0.05; support-ratio improvement 0.322→0.893; unsupported reduced 99→21, not ballooning; `inconclusive: false`). Confirmed by harness `autonomous-tick` from the review dir.

**Boundaries respected:** local models only; outputs only under the approved noncanonical research root `/mnt/nas-ai-models/research/stratum/stage-b-skin-color-v1(-review)`; no `crawlr/approved` or `crawlr/stratum` mutation; no backfill; no legacy overwrite; deterministic evidence computed in memory from existing `seg2.npy`/source pixels only; scale-invariant verbalization retained (owner px→ratios rule).

**Registry advance:** skin-color dimension `active → validated` (0 strikes, confirmed by harness `autonomous-tick`). Next selector pick: **lighting (arm #33)** now `research:active`. Verdict BETTER is empirical on this 24-item frozen cohort; a formal PASS still awaits the advisory human rubric spot-check (single independent reviewer family, rubric not yet human-calibrated on known/null cases).

## Arm #30 — hair evidence — `[EMPIRICAL RUN COMPLETE — VERDICT: BETTER]`

**Date:** 2026-08-05
**Arm:** #30 — hair evidence specialist
**Code / PR:** `exp/stage-b-hair-arm30-20260805` (draft PR #40, stacked on the arm-#32/#29 execution harness)
**Cohort:** frozen 24-item first-500 coverage-balanced subset (12 portrait / 6 squareish / 6 landscape — same manifest as arms #4/#29/#32)
**Deterministic specialist:** `research_harness.hair.compute_hair` (seg2 DOME-29 Hair(4) region coverage + quantized dominant color from source pixels + vertical position band + hair-to-face vertical-extent length proxy, gate-floor abstention); computed in memory per item during the bounded run. Only scale-invariant facts are verbalized (coverage, color name, band, ratio); px stays in `evidence_payload`.
**Deterministic probe (CPU, pre-run):** 24/24 subject present, 24/24 hair region cleared the gate, 23/24 length proxy measurable; palette black 2 / brown 6 / dark 5 / dark brown 8 / ginger 3; position top 14 / middle 9 / bottom 1. All 24 captions carry hair evidence; 3 ginger items are invented-color pressure tests.
**Aggregator:** already-installed local `gemma3:27b` (digest `a418f5838eaf…`), temperature=0, seed=20260804, num_predict=384, num_ctx=4096, loopback Ollama.
**Independent reviewer:** `gemma4:e4b` (different family), temperature=0, seed=20260804, num_predict=2000, 512×512 input, same reviewer calibration as arms #4/#29/#32.
**Plan:** frozen `research/stage-b-plans/stage-b-hair-v1.json` (fingerprint `2597eaf64025e440c604e13248d4c773420167195730501d8edee5b4925402ab`); conditions identical to arms #4/#29/#32 except the evidence condition is `context-raw-hair` (DOME-29 Hair region + color/length proxy) instead of `context-raw-geometry`/`context-raw-clothing`/`context-raw-body-type`. All 24/24 cohort items measured and generated.

**Scheduler lifecycle (local 4090):** request → poll/claim → launch → verify GPU activity → activate → heartbeat → release, through `registered-research-launcher` (job `stratum-stage-b-hair-v1`, 15:18→15:29Z, `gpu_activity_seen: true`, 96/96, plan fingerprint verified) and the independent review pass (job `stratum-stage-b-adversarial-review-hair-v1`, 96/96 reviewed by gemma4:e4b, completed ~15:50Z, GPU released). Both slots released cleanly (4090 idle).

**Evidence-only delta (cond 3 → 4):**
- supported claims **47 → 172**; unsupported **99 → 23**; omissions 11 → 18; contradictions 1 → 1; abstentions 0 → 0.
- Support ratio **32.2% → 88.2%** (Δ +0.560).
- Paired per-item sign test on supported claims: 20 improve / 4 worsen (24 paired), one-sided binomial **p ≈ 0.000772**.
- Deterministic cross-check independent of the LLM review: 19/24 hair-condition captions carry ≥1 hair vocabulary trace beyond their matched baseline (the remaining 5 already named hair color in baseline); both declared-ginger items now carry explicit "ginger hair", grounding the highest invented-color risk. Supported-claim gain is traceable to carried hair evidence.

**Deterministic verdict:** `autonomous-verdict --base-supported 47 --variant-supported 172 --base-unsupported 99 --variant-unsupported 23 --items 24 --p-supported 0.000772` → **BETTER** (significant p=0.000772 ≤ 0.05; support-ratio improvement 0.322→0.882; unsupported reduced 99→23, not ballooning; `inconclusive: false`).

**Boundaries respected:** local models only; outputs only under the approved noncanonical research root `/mnt/nas-ai-models/research/stratum/stage-b-hair-v1(-review)`; no `crawlr/approved` or `crawlr/stratum` mutation; no backfill; no legacy overwrite; deterministic evidence computed in memory from existing `seg2.npy`/source pixels only; scale-invariant verbalization retained (owner px→ratios rule).

**Registry advance:** hair dimension `active → validated` (0 strikes, confirmed by harness `autonomous-tick`). Next selector pick: **skin-color (arm #31, EIG 0.5)** now `research:active`. Verdict BETTER is empirical on this 24-item frozen cohort; a formal PASS still awaits the advisory human rubric spot-check (single independent reviewer family, rubric not yet human-calibrated on known/null cases).

## Arm #29 — clothing/apparel evidence — `[EMPIRICAL RUN COMPLETE — VERDICT: BETTER]`

**Date:** 2026-08-05
**Arm:** #29 — clothing/apparel evidence specialist
**Code / PR:** `exp/stage-b-clothing-arm29-20260805` (draft PR open), branches off the arm-#32 execution harness
**Cohort:** frozen 24-item first-500 coverage-balanced subset (12 portrait / 6 squareish / 6 landscape — same manifest as arms #4/#32)
**Deterministic specialist:** `research_harness.clothing.compute_clothing` (seg2 DOME-29 garment classes + per-class dominant color from source pixels, min-floor abstention); computed in memory per item during the bounded run.
**Aggregator:** already-installed local `gemma3:27b` (digest `a418f5838eaf…`), temperature=0, seed=20260804, num_predict=384, num_ctx=4096, loopback Ollama.
**Independent reviewer:** `gemma4:e4b` (different family), temperature=0, seed=20260804, num_predict=2000, 512×512 input, same reviewer calibration as arms #4/#32.
**Plan:** frozen `research/stage-b-plans/stage-b-clothing-v1.json` (fingerprint `6bfdb635459a93532cfbf7d3073991a0975c22d9c95e09d0f5ee5975dbe9b96c`); conditions identical to arms #4/#32 except the evidence condition is `context-raw-clothing` (DOME-29 garment coverage + dominant colors) instead of `context-raw-geometry`/`context-raw-body-type`. All 24/24 cohort items measured (24 subject-present, 14/24 with ≥1 garment class cleared).

**Scheduler lifecycle (local 4090):** request → poll/claim → launch → verify GPU activity → activate → heartbeat → release, through `registered-research-launcher` (job `stratum-stage-b-clothing-v1`, completed 2026-08-05 ~13:15Z, `gpu_activity_seen: true`) and the independent review pass (job `stratum-stage-b-adversarial-review-clothing-v1`, completed ~13:42Z, 96/96). Both slots released cleanly (4090 idle).

**Evidence-only delta (cond 3 → 4):**
- supported claims **72 → 151**; unsupported **100 → 46**; omissions 3 → 24; contradictions 1 → 6; abstentions 0 → 0.
- Support ratio **41.9% → 76.7%** (Δ +0.348).
- Paired per-item sign test on supported claims: 17 improve / 6 worsen (23 paired), one-sided binomial **p ≈ 0.0173**.
- Deterministic cross-check independent of the LLM review: declared-garment vocabulary carry was **low** (1/14 garment-bearing items had a new literal trace beyond its matched baseline). Baseline captions already named many garments (bikini tops, lace garments) and several cohort items are nude, so the carrier is modest; the supported-claim gain is nevertheless statistically significant on the reviewer rubric.

**Deterministic verdict:** `autonomous-verdict --base-supported 72 --variant-supported 151 --base-unsupported 100 --variant-unsupported 46 --items 23 --p-supported 0.017345` → **BETTER** (significant p=0.0173 ≤ 0.05; support-ratio improvement; unsupported reduced, not ballooning; `inconclusive: false`).

**Boundaries respected:** local models only; outputs only under the approved noncanonical research root; no `crawlr/approved` or `crawlr/stratum` mutation; no backfill; no legacy overwrite; deterministic evidence computed in memory from existing `seg2.npy`/source pixels only.

**Registry advance:** clothing dimension `active → validated` (0 strikes). Next selector pick: hair (arm #30, EIG 0.6). Verdict BETTER is empirical on this 24-item frozen cohort; a formal PASS still awaits the advisory human rubric spot-check (single independent reviewer family, rubric not yet human-calibrated on known/null cases).

## Arm #32 — body-type/proportion evidence — `[EMPIRICAL RUN COMPLETE — VERDICT: BETTER]`

**Date:** 2026-08-05
**Arm:** #32 — body-type/proportions evidence specialist
**Code / PR:** `exp/stage-b-bodytype-arm32-20260805` (draft PR open), branches off the arm-#4 execution harness
**Cohort:** frozen 24-item first-500 coverage-balanced subset (12 portrait / 6 squareish / 6 landscape — same manifest as arm #4)
**Deterministic specialist:** `research_harness.proportions.compute_proportions` (Goliath-308 pose2, min confidence 0.5, continuous ratios with explicit abstention); precomputed record → `/mnt/nas-ai-models/research/stratum/stage-b-bodytype-proportions-v1.json` (23/24 subjects present, 17/24 shoulder:hip ratio measurable, 13/24 leg measures, 1 abstained, 53 low-confidence joints).
**Aggregator:** already-installed local `gemma3:27b` (digest `a418f5838eaf…`), temperature=0, seed=20260804, num_predict=384, num_ctx=4096, loopback Ollama.
**Independent reviewer:** `gemma4:e4b` (different family), temperature=0, seed=20260804, num_predict=2000, 512×512 input, same reviewer calibration as arm #4.
**Plan:** frozen `research/stage-b-plans/stage-b-bodytype-v1.json` (fingerprint `37b47cea885b5fc71e801fbd33bc902454f8a21ae52b4896aac925408a44fe1b`); conditions identical to arm #4 except the evidence condition is `context-raw-body-type` (proportions) instead of `context-raw-geometry` (full determinations).

**Scheduler lifecycle (local 4090):** request → poll/claim → launch → verify GPU activity → activate → heartbeat → release, through `registered-research-launcher` (job `stratum-stage-b-bodytype-v1`, completed 2026-08-05 ~11:23Z, `gpu_activity_seen: true`) and the independent review pass (job `stratum-stage-b-adversarial-review-bodytype-v1`, completed, 96/96). Both slots released cleanly (4090 idle).

**Evidence-only delta (cond 3 → 4):**
- supported claims **47 → 195**; unsupported **99 → 14**; omissions 11 → 28; contradictions 1 → 1; abstentions 0 → 5 (reviewer abstains where the evidence abstained).
- Support ratio **32.2% → 93.3%** (Δ +0.611).
- Paired per-item sign test on supported claims: 20 improve / 3 worsen (23 paired), one-sided binomial **p ≈ 0.000244**.
- Deterministic cross-check independent of the LLM review: 17/24 body-type captions carry ≥1 body-descriptive vocabulary trace beyond their matched baseline captions (geometric/vocabulary carry measured on the record captions directly).

**Deterministic verdict:** `autonomous-verdict --base-supported 47 --variant-supported 195 --base-unsupported 99 --variant-unsupported 14 --items 23 --p-supported 0.000244` → **BETTER** (significant p=0.000244 ≤ 0.05; support-ratio improvement; unsupported reduced, not ballooning; `inconclusive: false`). **SUPERSEDED as the authoritative numbers by the ratio-only re-measurement below** (the first attempt verbalized absolute px; the re-measurement carries only scale-invariant ratios).

### Arm #32 addendum — ratio-only re-measurement (2026-08-05, owner px→ratios correction)

Per owner directive, absolute pixel measurements are camera-frame-dependent and not meaningful to a text-to-image model, so they must not be caption claims; only **scale-invariant ratios** (shoulder:hip, leg:torso, limb asymmetry) are verbalized, and raw px stay in the machine-readable `evidence_payload` JSON only. This is a controlled single-axis re-measurement of the same arm on the same frozen cohort / aggregator (`gemma3:27b`) / reviewer (`gemma4:e4b`).

- **Corrected run:** `stage-b-bodytype-ratios-v1` → 96 records + 96 independent reviews (`stage-b-bodytype-ratios-v1-review`), scheduler lifecycle on the 4090.
- **Evidence-only delta (corrected):** supported **47 → 188**; unsupported **99 → 22**; omissions 11 → 24; contradictions 1 → 1; abstentions 0 → 0.
- Support ratio **32.2% → 89.5%** (Δ +0.573).
- Paired sign-test: **22 improve / 1 worsen (23 paired), p ≈ 3e-6**.
- **Corrected deterministic verdict:** `autonomous-verdict --base-supported 47 --variant-supported 188 --base-unsupported 99 --variant-unsupported 22 --items 23 --p-supported 0.000003` → **BETTER** (p=3e-6 ≤ 0.05; ratio 0.322 → 0.895; unsupported reduced; `inconclusive: false`).
- **Result:** the BETTER verdict **survives the px→ratios correction** — it is not an artifact of verbalizing camera-frame absolutes; on the corrected scale-invariant-only evidence the signal is stronger (p≈3e-6 vs ≈2.4e-4) once px noise leaves both prompt and reviewer checklist. These corrected numbers are the authoritative arm-#32 record.
- **Registry:** body-type `active → validated` (0 strikes) stands. A formal PASS still awaits the advisory human rubric spot-check.

**Boundaries respected:** local models only; outputs only under the approved noncanonical research root; no `crawlr/approved` or `crawlr/stratum` mutation; no backfill; no legacy overwrite; deterministic evidence computed in memory from existing `pose2.npy` only.

**Registry advance:** body-type dimension `proposal → validated` (0 strikes) — confirmed by the corrected ratio-only re-measurement (BETTER, p≈3e-6). Next selector pick (post-clothing): hair (arm #30).

## First-500 coverage-balanced Stage-B comparison — `[EMPIRICAL RUN COMPLETE — PENDING_HUMAN_SPOT_CHECK]`

**Date:** 2026-08-04/05
**Arm:** #4 — baseline and comparison parity
**Code / PR:** `exp/stage-b-first500-aggregator-20260804`, draft PR #20
**Cohort:** frozen 24-item first-500 coverage-balanced subset (12 portrait / 6 squareish / 6 landscape)
**Aggregator:** already-installed local `gemma3:27b` (digest `a418f5838eaf…`), `temperature=0`, `seed=20260804`, `num_predict=384`, `num_ctx=4096`, loopback Ollama.
**Independent reviewer:** `gemma4:e4b` (different family from generator), `temperature=0`, `seed=20260804`, `num_predict=2000`, `num_ctx=8192`, 512×512 input.

**Goal:** Test, on the frozen cohort with fixed generation settings, whether declared in-memory geometry evidence (`pose2`+`seg2` only) changes claim support under a matched one-axis comparison.

**Conditions (same item, same model/settings):**
1. bucketed/cropped + legacy prompt + no evidence
2. raw + legacy prompt + no evidence
3. raw + context prompt + no evidence
4. raw + same context prompt + geometry evidence

**Empirical evidence:**
- 96/96 captions generated and published to `/mnt/nas-ai-models/research/stratum/stage-b-first500-parity-v1/` (records.jsonl, review-queue.jsonl, run-provenance.json, outputs/).
- Independent review (gemma4:e4b) scored 96/96 into claim-support buckets at `/mnt/nas-ai-models/research/stratum/stage-b-first500-parity-v1-review/`.
- **Evidence-only delta** (cond. 3 → 4): supported claims **47 → 156**; unsupported **99 → 40**; items with ≥1 supported claim **5/24 → 24/24**; omissions 11 → 27; contradictions 1 → 2. Support ratio (supported / supported+unsupported) **32% → 80%**.
- Paired per-item sign test on supported claims: 19 improve / 5 worsen, one-sided binomial p≈0.003. On unsupported: 14 decrease / 8 increase, p≈0.14 (directionally reduced, not individually significant).
- Deterministic cross-check (independent of the LLM review): all 24 geometry captions verbalize declared-evidence vocabulary; 16/24 carry ≥ half of declared traces. The supported-claim gain is traceable to evidence actually carried into the caption.

**Boundaries respected:** local models only; outputs only under the approved noncanonical research root; no `crawlr/approved` or `crawlr/stratum` mutation; no backfill; no legacy overwrite; scheduler lease claimed/activated/heartbeated/released cleanly; model unloaded after run.

**Verdict:** `EMPIRICAL RUN COMPLETE — PENDING_HUMAN_SPOT_CHECK`. Statistical improvement in supported claims from declared geometry on this 24-item frozen cohort with fixed settings. Not yet a PASS: single reviewer model, no human calibration of the rubric on known/null cases yet, cohort is 24 items, one-axis only. No corpus mutation or merge occurred.

**Deterministic verdict (2026-08-05, harness rule):** `autonomous-verdict --base-supported 47 --variant-supported 156 --base-unsupported 99 --variant-unsupported 40 --items 24 --p-supported 0.003` → **BETTER** (support ratio 0.322 → 0.796, Δ +0.474; sign-test p=0.003 ≤ 0.05; unsupported 99 → 40, not ballooning; `inconclusive: false`). The evidence-only contrast on the frozen cohort satisfies the pre-registered BETTER gate. The `PENDING_HUMAN_SPOT_CHECK` status is advisory (single independent reviewer family, rubric not yet human-calibrated on known/null cases) and does not gate the harness verdict; a formal PASS still awaits that spot-check.

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

## Arm #47 — open-weight VLM dense-description sourcing verification — `[SOURCING VERIFIED — NOT RUN]`

**Date:** 2026-08-06
**Arm:** #47 — open-weight VLM dense multi-view description (option-B dossier growth path; prerequisite sourcing verification only — no sensitive-corpus inference, no claim-support run)
**Code / PR:** `exp/arm47-vlm-sourcing-verified-20260806` (draft PR #48, docs-only)

**Why:** The arm-36 honest expansion-ceiling audit showed deterministic evidence tops out at 2040–3489 tokens/item (7–50× below the 100K floor), so option B of decision #46 requires a genuinely-new non-deterministic evidence part. Registered via the gated `propose-dimensions` channel as a NEW evidence part (`vlm-dense-description`) + NEW model class (`open-vlm-dense-captioner`); registering it surfaced and fixed a real selector tie-break bug (dict-comparison `TypeError` on equal EIG tuples — id tiebreaker + regression test).

**Open-world sourcing scan (2026-08-05/06 directive):**
- Molmo / Molmo-72B (AI2, fully open weights+data): strongest dense-description pretraining focus (PixMo) of the candidates; arXiv 2409.17146 / 2601.10611 (Molmo2).
- Qwen2-VL-72B / Qwen2.5-VL: strongest MMMU/OCRBench among open; better when in-scene OCR matters.
- InternVL3-78B: strongest MIT-licensed VLM.

**Local capability probe (non-sensitive synthetic image, `scripts/vlm_capability_probe.py`, 2026-08-06):** `qwen3-vl:32b` (already installed locally on 4090 + Strix, zero download) — GPU-placement measurement: **4090 (24GB) fits at only 27% CPU-offload, ~280s per 2048-token block — not viable for a 96-item batch; Strix (100GB usable) runs it 100% GPU at ~9.6 tok/s, 128K ctx — viable production batch host.** Conclusion: any large-VLM arm (option B execution) runs on Strix.

**Verdict:** sourcing capability verified as a measurement, NOT passed; the arm is a `research:proposal`. Activation + the 96-item claim-support run remain gated on decision #46 ruling (option B). No disturbance to `crawlr/approved` / `crawlr/stratum`, no backfill, no hosted inference of the sensitive corpus.

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
