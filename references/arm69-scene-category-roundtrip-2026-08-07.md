# Arm #69 scene-category round-trip BETTER — CLIP ViT-L/14 zero-shot (NEW model class)

Validated 2026-08-07 on the frozen 24-item cohort. Registry: `scene-category → validated`,
`gaze-head-orientation #68 → active` (selected_via explore, ε-greedy slot, selection_progress 12).

## Verdict (harness-computed, never hand-transcribed)

- Support ratio **0.3219 → 0.9310** (Δ +0.6091): supported 47 → 189, unsupported 99 → 14,
  paired positive 20/24, **sign-test p=0.000772** (≤0.05).
- `autonomous-tick --review-dir-from ~/.hermes/profiles/stratum-ffhq/cron/state/tick-ready.json --write`
  run roots: `/mnt/nas-ai-models/research/stratum/stage-b-scene-category-v1` (96 records) +
  `stage-b-scene-category-v1-review` (96 reviews, gemma4:e4b).

## Evidence design (scale-invariant, closed set)

NEW-MODEL-CLASS open-weight zero-shot classifier: `openai/clip-vit-large-patch14` (MIT, local CPU,
model.safetensors sha256 `a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb`, staged at
`/mnt/nas-ai-models/research/stratum/models/scene-category`). Frozen closed 10-category set
**cohort-derived from the arm-#47 VLM scene vocabulary**: indoor studio / plain wall backdrop /
bedroom / living room / outdoor beach / outdoor garden / outdoor field / body of water /
urban street / poolside. Softmax over the closed set scaled by the model's learned logit_scale;
abstention floor **0.25** — below it the item abstains (never a guess).

- Only the scale-invariant semantic category label is verbalized ("the setting is a …"); similarity
  logits/probabilities stay in `evidence_payload` JSON, never prose.
- `_EVIDENCE_INPUT_NAMES["scene-category"]` is **the empty tuple** — CLIP consumes ONLY the
  already-decoded full-frame source RGB (SHA-bound via the item's `source_sha256`). seg2/pose2 stay
  validation-only reads for the exactly-one-subject invariant. The frozen plan's
  `evidence_input_artifact_sha256` is honestly empty per item (this is the first arm with no derived
  artifact evidence input — verified `_validate_frozen_execution_plan` accepts the empty set).

## Calibration probe (band-degeneracy rule arm #34/#35/#59 — BEFORE freezing)

`scripts/probe_scene_category.py` (read-only, CPU, no GPU claim): **24/24 classified, 8 distinct
categories across 10, max top-1 share 25%** (well under the 75% line), p50 confidence 0.526, min
confidence 0.270, 0 abstentions at the 0.25 floor. Category spread: poolside(6), outdoor beach(6),
plain wall backdrop(3), outdoor field(2), indoor studio(2), body of water(2), bedroom(2),
urban street(1).

## Round-trip numbers

- Generation: `stage_b_launcher --poll-and-launch` on the 4090, ~13 min, 96 records (24 × 4
  conditions), `{"status":"completed","gpu_activity_seen":true}`.
- Review: queued first with `stage_b_review_launcher --request` then the parameterized wrapper
  (`PYTHONPATH=src` inline), 96 reviews via gemma4:e4b → tick-ready marker written.
- Tick: BETTER (p=0.000772), registry advanced atomically, selection_progress 11→12.

## Full new-evidence-kind touchpoint checklist (arm #69)

1. `src/research_harness/scene_category.py` — `compute_scene_category`, `_zero_shot_probabilities`,
   `render_scene_category`, `SceneCategoryError`, `validate_rgb_array`.
2. `_EVIDENCE_INPUT_NAMES["scene-category"] = ()` (empty — only source RGB consumed).
3. `stage_b.py` imports; `_scene_category_evidence()` + `_serialize_scene_category()`.
4. `build_stage_b_plan`: allowed tuple + `elif evidence_kind == "scene-category"` branch
   (condition id `context-raw-scene-category`, comparison plan `stage-b-first500-scene-category-v1`).
5. `_validate_frozen_execution_plan`: `elif "context-raw-scene-category" in condition_ids:` →
   rebuild_kind "scene-category".
6. `_load_selected_item`: `include_scene_category` flag (gated on plan conditions) + compute +
   `prepared["scene_category"]`.
7. `_render_condition`: `if condition_id == "context-raw-scene-category":` branch returning
   `(raw.copy(), _context_prompt(evidence_text), scene_category)` — the reviewer consumes the same
   rendered evidence automatically.
8. `dossier.py`: `DIMENSION_EVIDENCE_IDS` += `scene-category:v1`, `render_scene_category` factory,
   `_scene_category_payload` helper + `build_evidence_payload` section, `assemble_dossier`
   dimension_factories row + `_rendered_context4k` passthrough.
9. `scripts/freeze_scene_category_manifest.py` (arm-#4 settings, evidence_kind scene-category) +
   `scripts/probe_scene_category.py`.
10. `tests/test_scene_category.py` — 7 TDD tests (validate, render empty/abstain/label/below-floor,
    softmax structure, synthetic-frame smoke). Full suite 573 passed (incl. new arm).

**Review/aggregation path untouched** — `_derive_conditions_from_plan` matches
`no-specialist-evidence-v1` + `"context" in cid` generically, so a new evidence kind needs zero
per-arm review code.

## Pitfalls hit live (each cost a real fix)

- **transformers 5.8.x API drift applies to `get_text_features` too**: newer transformers returns
  `BaseModelOutputWithPooling` from BOTH `get_image_features` and `get_text_features` — decode
  `.pooler_output` for both (the skill's recon.py note only covered `get_image_features`; the text
  path hit the same AttributeError on `.norm`). Fix: `if not torch.is_tensor(out): out = out.pooler_output`
  for image AND text embeddings.
- **Render-order guard**: `render_scene_category`/`_serialize_scene_category` must check
  `abstained` BEFORE the not-measured (`no category`) check — an abstention dict has no `category`
  key, so the old order returned `[]` on abstentions (one unit test caught it).
- **git_commit pin chase (recipe #6)**: every commit after the freeze makes the pin stale; the
  launcher requires `execution.git_commit` to be an ANCESTOR of HEAD (not equal), so the final
  refresh commit leaves the pin at its parent — acceptable, closes the loop.
- **Manual `gpu_scheduler.py poll` while a launcher owns the claim violates pitfall #2** — it
  timed out harmlessly here (the launcher had already claimed via its own atomic poll), but the
  correct recovery is the wrapper's `--action release --requeue`. Do NOT manual-poll a running job.

PR: `exp/stage-b-scene-category-arm69-20260807` (chain base `exp/stage-b-object-relations-arm61-20260807`).
