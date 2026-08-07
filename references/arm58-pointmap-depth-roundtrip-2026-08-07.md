# Arm #58 pointmap-depth round-trip (2026-08-07) — VALIDATED BETTER

## What

Deterministic point-map / 3D depth-ordering evidence specialist. Reads the
existing `pointmap.npy` (Sapiens2 per-pixel CAM-frame 3D cloud, +X right +Y
down +Z toward viewer, background zeroed) + `seg2.npy` (DOME-29) and emits
scale-invariant depth facts:

- region depth ranking (nearest/farthest part to the camera, from region-median
  Z ranks of head/torso/arms/hands/legs);
- left/right hand depth ordering (one hand clearly nearer than the other);
- hand/arm held in front of the torso plane (per side);
- normalized body depth-relief ratio (robust Z spread p10-p90 / median Z) banded
  compact / moderate / pronounced.

Only scale-invariant facts are verbalized (ordering relations, normalized
ratios, bands). Absolute metric Z values, raw spreads, and per-region median Zs
stay in the machine-readable `evidence_payload` (dossier / compressor input) and
are never caption claims — raw meters are camera-placement dependent and
unrenderable by a text-to-image model.

Provenance: deterministic CPU measurement from existing core artifacts; no model
invocation, no corpus write. New evidence part `pointmap-depth` (unbound by any
previously-validated arm).

## Band calibration (mandatory probe, read-only, frozen 24 cohort)

First probe after writing the module on paper thresholds produced a DEGENERATE
band: `relief_band` = {compact: 24} (100% — the floor `0.30` never fired).
Recalibrated from the measured distribution (relief range 0.051–0.241,
p10/p50/p90 = 0.053/0.124/0.214):

- `RELIEF_FLOOR = 0.09`, `RELIEF_PRONOUNCED = 0.16` → final split
  compact 6 / moderate 12 / pronounced 6 (max share 50%, no band ≥ 75%).

Live cohort signal frequencies (sparse-but-precise, reported honestly):

- hand_ordering fired 5/24 (left 2, right 3);
- hand_in_front fired 11/24;
- nearest_region: hands 12/24, then head 2 / torso 3 / legs 4 / arms 3;
- farthest_region: arms/legs dominant (arms 12, legs 7, hands 4, torso 1).

`scripts/probe_pointmap_depth.py` is the re-runnable probe.

## New-evidence-kind touchpoints (mirrored from arm #62, validated live)

1. `src/research_harness/pointmap_depth.py` — `compute_pointmap_depth` +
   `validate_pointmap_array`/`validate_seg2_array` + `PointmapDepthError`,
   DOME-29 class indices pinned from `stratum2.config.DOME_29`.
2. `_EVIDENCE_INPUT_NAMES["pointmap-depth"] = ("pointmap.npy", "seg2.npy")`
   (pose2 stays a validation-only read — the exactly-one-subject invariant).
3. Import in `stage_b.py`; `_pointmap_depth_evidence()` declaration +
   `_serialize_pointmap_depth()` renderer.
4. `build_stage_b_plan`: added `"pointmap-depth"` to the allowed tuple + the
   `elif evidence_kind == "pointmap-depth"` branch
   (`evidence_condition_id "context-raw-pointmap-depth"`, comparison_plan_id
   `stage-b-first500-pointmap-depth-v1`).
5. `_validate_frozen_execution_plan`: `elif "context-raw-pointmap-depth" in
   condition_ids: rebuild_kind = "pointmap-depth"`.
6. `_load_selected_item`: load `pointmap.npy` (SHA-bound evidence read) when it
   is in the expected evidence hashes, verify (H,W,3) + pixel-alignment with
   seg2, compute `compute_pointmap_depth`, append to `derived_reads`, and put it
   in the prepared dict.
7. `_render_condition`: `if condition_id == "context-raw-pointmap-depth"` branch
   returning `(raw.copy(), _context_prompt(evidence_text), pointmap_depth)` — the
   third element becomes `record["evidence_payload"]`, so the reviewer consumes
   the same rendered evidence automatically.
8. Dossier extension (`dossier.py`, arm feeds the dossier goal): added
   `pointmap-depth:v1` to `DIMENSION_EVIDENCE_IDS`, `render_pointmap_depth()`
   factory, `_pointmap_depth_payload()` helper, and the `assemble_dossier` /
   `build_evidence_payload` passthroughs (both `_rendered_context4k` call sites).

Review path NOT touched — `_derive_conditions_from_plan` matches on
`no-specialist-evidence-v1` + `"context" in cid` generically (zero per-arm review
code).

## Round-trip numbers (harness-computed, never hand-typed)

- Plan `stage-b-first500-pointmap-depth-v1`, manifest
  `research/gpu-manifests/stage-b-pointmap-depth-v1.json` (96 records = 24 items
  × 4 conditions; evidence inputs pointmap.npy+seg2.npy SHA-pinned per item).
- Generation `stratum-stage-b-pointmap-depth-v1` on the 4090 via the launcher:
  `{"status":"completed","gpu_activity_seen":true}`; 96 rows in
  `/mnt/nas-ai-models/research/stratum/stage-b-pointmap-depth-v1/records.jsonl`;
  run-provenance status PENDING_INDEPENDENT_REVIEW.
- Independent review via the parameterized wrapper
  (`stratum_review_poll_wrapper.py --run-root ... --review-root ... --job-id
  stratum-stage-b-adversarial-review-pointmap-depth`): 96 rows,
  `{"status":"completed","record_count":96,"gpu_activity_seen":true}`, tick-ready
  marker published.
- `autonomous-tick --review-dir-from <marker> --write`:

  - **verdict BETTER** — support-ratio base 0.3219 → variant 0.7488
    (delta +0.4269); supported 47 → 158, unsupported 99 → 53; paired
    positive 19/22; sign-test p = 0.000428 (significant).
  - registry: `pointmap-depth` → validated; `matting-alpha` → active
    (selected_via explore, ε-greedy slot — selection_progress 8).
  - one-active invariant holds (active: 1, validated: 12).

- Notable: the ε-greedy exploration slot fired at selection 8 (`every_n 4`),
  selecting the lower-prior matting-alpha over the exploit choice — reported as
  the harness computed it, not overridden.

## Real incidents / notes (this round)

1. **Degenerate relief band on first calibration** — paper thresholds (0.30 /
   0.55) put 24/24 in one band; recalibrated to the measured cohort distribution
   (0.09 / 0.16). Same rule as arm #34/35: any new band gets a histogram probe
   before the plan freezes.
2. **Review wrapper needs the job queued first.** The wrapper's launcher
   invocation runs WITHOUT `--request` (it only requests when `--request` is
   passed), so `--request` for the review job must happen before/separately.
   Sequence: `stage_b_review_launcher --request` (queues) then the wrapper's
   poll loop claims + runs to completion (~35 min) and writes the marker.
   (Not a bug — by design; the wrapper polls until the slot frees.)
3. **Test fixture hand support floor.** Unit-test hands were 2 px, below
   `MIN_HAND_PX=60`, so hand measurements abstained and the hand-ordering test
   failed; hands are now 8×8 blocks in the fixture. The module floor is correct
   (sparse hands abstain honestly).
4. New-touchpoint test you will break: `test_roundtrip_context4k.py
   ::test_render_context4k_condition_emits_evidence_linked_compact` asserts the
   exact `dossier_evidence_ids` list — adding a dossier dimension requires
   inserting `pointmap-depth:v1` in order (done).

## Validation

- `pytest tests/ -q`: **539 passed**.
- `validate-program`: valid (schema v2, structural floor 4001 still
  `goal_unreachable: false`).
- `validate-dimension-registry`: valid (active 1, validated 12, proposals 2).
- `validate-gpu-manifest`: valid.
- Label-sync applied (4 ops: #58 → research:validated, #59 → research:active).

## Chain / PR

Branch `exp/stage-b-pointmap-depth-arm58-20260807`, base =
`exp/pose-articulation-arm62-20260807` (previous arm branch), draft PR per the
chain-base convention. Run roots: `stage-b-pointmap-depth-v1` + `-review`.
