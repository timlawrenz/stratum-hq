# Arm #61 object-relations round-trip — Grounding DINO (NEW model class)

**State:** VALIDATED BETTER (2026-08-07, harness-computed, PR #68 draft on
`exp/stage-b-object-relations-arm61-20260807`). Support ratio 0.3219 → 0.8783
(Δ +0.5564), supported 47 → 166, unsupported 99 → 23, paired positive 18/21,
sign-test p=0.000745. Registry: object-relations → validated; the sweep then
reported EXHAUSTED (15/15 terminal) → next_action brainstorm-new-data, which
closed this cycle by registering #68 gaze-head-orientation and #69
scene-category (CLIP) and activating scene-category (exploit, EIG 0.65,
selection_progress 11). Run roots:
`/mnt/nas-ai-models/research/stratum/stage-b-object-relations-v1` + `-review`.

## What this arm is
NEW-MODEL-CLASS evidence specialist: **Grounding DINO** (`IDEA-Research/
grounding-dino-base`, Apache-2.0, open-weight text-grounded open-vocabulary
detector, HF Transformers path) run on owned hardware over the full-frame
decoded source + seg2 subject mask. Emits scale-invariant facts:
- object-presence **count band** (none / sparse / moderate / dense) from
  detections above the calibrated box threshold;
- **placement band** (foreground / background / mix) from seg2-subject overlap;
- **canonical class list** (the detected classes mapped to a frozen
  cohort-derived closed vocabulary).

Only scale-invariant facts are verbalized; normalized boxes/scores/raw phrases
stay in `evidence_payload` JSON (measurement-semantics directive). No absolute
pixels in prose.

## Model asset + capability probe
- `grounding-dino-base` downloaded from HF into `/mnt/nas-ai-models/
  research/stratum/models/object-relations/` (model.safetensors ~933MB,
  sha256 `5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21`,
  Apache-2.0). **CIFS symlink trap surfaced live:** the HF cache on the NAS
  mount cannot create symlinks (`OSError: [Errno 95]`), so the stage used
  `HF_HUB_DISABLE_SYMLINKS=1` (copy mode) and the asset was copied into the
  stable models dir — good to remember for any future HF download on NAS.
- **transformers 5.8.1 API drift (arm-#37-style):** `post_process_grounded_
  object_detection` takes `threshold` (NOT `box_threshold`) and returns
  `text_labels` (strings) alongside the deprecated `labels` keys.
- Capability probe (qualification gate step 2): non-sensitive synthetic scene
  first (path + timing: load ~8s, CPU infer ~5s/im), then the frozen cohort.
  **The first furniture-centric vocabulary was DEGENERATE on this cohort**
  (9/24 detected — the cohort is scene-dominant: water/field/concrete/mirror/
  window, not chairs/desks). The frozen closed vocabulary is COHORT-DERIVED
  from the already-computed arm-#47 VLM dense-description blocks.

## Calibration (2026-08-07, band-degeneracy rule arm #34/#35/#58/#59)
Parameter-swept the box threshold over [0.10..0.35]: 0.25 is the sweet spot
(21/24 ≥1 detection; below it noise balloons — e.g. 0.10 gives 52–94
detections on busy items). Final bands calibrated on the frozen 24 items
(no band ≥ 75%):
- **count:** none=8, sparse(1)=7, moderate(2–4)=5, dense(5+)=4 (max share 33%);
- **placement:** foreground=4, background=4, mix=8, none=8 (max 33%);
- honest scene objects actually detected: body of water(12), tree(7),
  sneakers(4), earrings(4), curb(3), skateboard(2), grass(2), boat(2),
  blanket(2), plant(1), wall(1), hat(1), bracelet(1), door(1), window(1).

**Subject-self guard (live finding):** Grounding DINO fires `body`/`person`
on the subject herself (measured 0.28–0.45 boxes on several items). Raw
detections whose canonical class is an exact standalone subject word are
EXCLUDED from both the count and the prose (`body of water`, a real scene
object, survives via exact-standalone-word, not substring, exclusion).

## Evidence-kind surface (full touchpoint checklist — new kind "object-relations")
- New module `src/research_harness/object_relations.py`: `compute_object_
  relations(seg2, rgb, *, model_asset_dir=...)` (+ validators +
  `ObjectRelationsError` + `render_object_relations` + `canonical_class` +
  `_count_band`/`_placement_band` + lazy `_GroundingDinoRuntime` CPU singleton
  + subject-self guard). Pinned thresholds BOX_THRESHOLD 0.25 / TEXT_THRESHOLD
  0.20 in the module.
- `stage_b.py`: import; `_object_relations_evidence()` (binds module SHA AND
  model.safetensors SHA — provenance is model-bound); `_serialize_object_
  relations()`; `_EVIDENCE_INPUT_NAMES["object-relations"]=("seg2.npy",)`
  (pose2 stays validation-only); allowed tuple + `elif evidence_kind ==
  "object-relations"` branch (`context-raw-object-relations`,
  `stage-b-first500-object-relations-v1`); `_validate_frozen_execution_plan`
  rebuild kind; `_load_selected_item` computes object relations gated on
  `include_object_relations` (only this run pays the ~3 min CPU model cost);
  `_render_condition` branch. **Not-measured guard:** `render_object_relations`
  / `_serialize_object_relations` on an empty/None dict return NO claim (the
  anti-fabrication rule — without it the context4k/vlm-dense dossier paths
  would emit a fake "no objects detected" for every non-object-relations run).
- `dossier.py`: `DIMENSION_EVIDENCE_IDS` + `render_object_relations` +
  `_object_relations_payload` + `assemble_dossier`/`build_evidence_payload`
  params + dimension_factories row — the context4k round-trip test's
  `dossier_evidence_ids` list stays stable because the not-measured guard
  emits no claim for the other runs.

## Freeze + generation
- `scripts/freeze_object_relations_manifest.py`: plan
  `stage-b-first500-object-relations-v1`, manifest on 4090 (96 records,
  gemma3:27b digest a418f5838eaf, identical arm-#4 settings), seg2 evidence
  input. `validate-gpu-manifest` valid.
- Generation: `stage_b_launcher --request` then `--poll-and-launch`
  (background). Expect `records.jsonl` 96 rows, `run-provenance`
  PENDING_INDEPENDENT_REVIEW.
- Review: pre-queue `stage_b_review_launcher --request` (the parameterized
  wrapper does NOT request), then launch the wrapper with INLINE
  `export PYTHONPATH=<experiment>/src` (else it imports the main checkout's
  partial package and silently poll-forever).
- Tick: `autonomous-tick ... --review-dir-from <marker> --write` → verdict.

## Tests
`tests/test_object_relations.py` (10 tests): `_count_band` thresholds,
`_placement_band` strict-majority, `canonical_class` mapping (longest match,
case-insensitive), subject-self guard (`body`→True, `body of water`→False),
validators, misaligned shapes, not-measured-no-claim (anti-fabrication),
abstention render, band render, end-to-end synthetic-frame detector smoke
(loads the real model on a small synthetic image, CPU). Full suite 566 passed
after the DIMENSION roll.

## Pitfalls recap (add to the standing list if repeated)
- HF cache symlink error on CIFS/NAS → `HF_HUB_DISABLE_SYMLINKS=1` (copy
  mode), or stage the weights into a stable non-cache models dir.
- transformers GroundingDinoProcessor API: `threshold` not `box_threshold`;
  results carry `text_labels` (strings) — the deprecation warning is expected.
- Vocabulary matters enormously: furniture-centric closed vocab was 9/24
  degenerate on this scene-dominant cohort; cohort-derive it from the VLM
  blocks and parameter-sweep the threshold (0.25 won).
- Subject-self confusion: exclude exact-standalone `body`/`person` boxes from
  count and prose; keep `body of water`.
- The plan's `pilot_manifest.items` are MINIMAL (image_id/rel/sha/availability
  only) — use the candidate manifest items for `_load_selected_item`-style
  preflights that need `source_dimensions`.
