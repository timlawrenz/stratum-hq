# Arm #62 pose-articulation round-trip — recipe + calibration (validated 2026-08-07)

Deterministic kinematic-articulation evidence specialist. **VERDICT BETTER**
(support ratio 0.4225 → 0.8195, Δ +0.397, paired positive 18/22, sign-test
p = 0.002172), registry `pose-articulation → validated`, next active
`pointmap-depth #58` (exploit, EIG 0.45, selection_progress 7). PR #64, branch
`exp/pose-articulation-arm62-20260807` (chain base `exp/stage-b-brainstorm-widen-20260806`).

## New evidence kind — complete touchpoint list (this arm)

Followed the arm-#34/#35 checklist for a NEW evidence kind (`pose-articulation`):

1. `src/research_harness/pose_articulation.py` — `compute_pose_articulation` +
   `validate_pose2_array`/`validate_seg2_array` + `PoseArticulationError`.
2. `_EVIDENCE_INPUT_NAMES["pose-articulation"] = ("pose2.npy", "seg2.npy")`
   (both GOLIATH-308 keypoints + DOME-29 masks are evidence-bound).
3. `_pose_articulation_evidence()` declaration + `_serialize_pose_articulation()`
   in stage_b.py.
4. `build_stage_b_plan`: allowed-kind tuple + `elif evidence_kind ==
   "pose-articulation"` branch (`context-raw-pose-articulation`,
   `stage-b-first500-pose-articulation-v1`, hypothesis/falsified_if/coverage_notes).
5. `_validate_frozen_execution_plan`: `elif "context-raw-pose-articulation" in
   condition_ids: rebuild_kind = "pose-articulation"`.
6. `_load_selected_item`: `articulation = compute_pose_articulation(pose2, seg2)`
   + returned in the prepared dict.
7. `_render_condition`: `context-raw-pose-articulation` branch → serialized text +
   articulation dict as `evidence_payload` (reviewer sees the same rendered evidence).
8. Dossier: `pose-articulation:v1` in `DIMENSION_EVIDENCE_IDS`,
   `render_pose_articulation()`, `_pose_articulation_payload()`, factory row in
   `assemble_dossier`, payload section in `build_evidence_payload`, passthrough in
   `_rendered_context4k`.

**Broken-then-fixed test (expected):** `tests/test_roundtrip_context4k.py`
`dossier_evidence_ids` now includes `pose-articulation:v1` in order — that
exact-list assertion fails until the id is inserted (not a regression; the
context4k compact honestly now carries the articulation claims).

Also reconciled `tests/test_research_assets.py`
`test_resumption_documents_preserve_the_active_state_and_two_stage_boundary`:
the doc-consistency test still asserted the pre-widen "Sweep EXHAUSTED — no
active arm remains" state while PROJECT_STATUS.md (updated by PR #63 at
brainstorm time) correctly said pose-articulation is the sole `research:active`
arm. The test now asserts the post-widen live state (one-active invariant kept).

## Measurement design (scale-invariant only)

- **Flexion angles:** interior angle at elbow (shoulder→elbow→wrist) and knee
  (hip→knee→ankle). Straight = ~180°, visibly bent < 135° (prose band; raw
  degrees stay in evidence_payload).
- **Torso/pelvis orientation:** in-plane torso twist (shoulder-axis vs hip-axis
  angular diff), torso lean from vertical (spine midpoints), pelvis tilt from
  horizontal. All pure angle differences = scale-invariant.
- **Stance/contrapposto:** weight-bearing = ankle closest in horizontal
  projection to hip-midline; stance class weight-left/right/centered;
  contrapposto = clear weight shift AND pelvis tilt ≥ 4° (hip hike).
- **Limb-overlap structure:** arm-over-spine crossing via 2D segment
  intersection within the torso y-band; legs-crossed via left/right hip→ankle
  segment intersection; seg2 arm-near-torso proximity fraction (binary_dilation
  of torso mask, margin 12 px — proximity proxy, NOT class overlap, because a
  semantic seg2 labels one class per pixel).
- **Symmetry/asymmetry:** left-vs-right elbow/knee flexion angle differences.

Absolute pixel positions/lengths never verbalized; only the above (angles,
normalized ids, bands, crossing counts) reach the caption prompt. Abrupted
(None) on absent/low-confidence joints. Exactly-one-subject enforced upstream.

## Calibration probe (band threshold honesty gate)

`scripts/calibrate_pose_articulation_bands.py` + `scripts/probe_pose_articulation.py`
on the FROZEN 24-item cohort BEFORE freeze:

- elbow: 21 bent / 17 extended / 10 n/a — mean 112.5°, p25 71.5/p50 114.8/
  p75 169.3. No band ≥ 75% ⇒ 135° threshold discriminates (threshold KEPT).
- knee: 19 measurable, mean 92.5°.
- stance resolved 11/24 (weight-left 3, weight-right 5, centered 3);
  torso twist 17/24; arm-crossing 2/24; contrapposto 4/24; legs-crossed 1/24;
  arm-near-torso >0.5: right 2/24, left 0/24.

Sparse signals (crossing/contrapposto/legs-crossed) are precise-when-they-fire;
they do not dominate and are not degenerate. Reported honestly in docs.

## Round-trip execution (identical to arm-#36 recipe)

1. Freeze: `scripts/freeze_pose_articulation_manifest.py` → plan
   `stage-b-first500-pose-articulation-v1` + manifest
   `stage-b-pose-articulation-v1.json` (job `stratum-stage-b-pose-articulation-v1`,
   target 4090, 22GB, 96 records). Re-freeze after the code commit to refresh
   `git_commit` pin (f7b22e0).
2. Generation: `stage_b_launcher --request` then
   `stage_b_launcher --poll-and-launch` BACKGROUNDED (never foreground — a
   timeout orphans the lease). Completed 96/96 (4 conditions × 24), GPU seen,
   `run-provenance.json` PENDING_INDEPENDENT_REVIEW.
3. Review: `stage_b_review_launcher --request` then the same without `--request`
   (claims+reviews+releases in one process), backgrounded. 96 rows,
   `reviews.jsonl`, gemma4:e4b (row label `reviewer-qwen3vl-32b` is the known
   hardcoded provenance nit — check ReviewSettings, don't trust the row label).
4. Tick: `autonomous-tick <registry> --review-dir <root>-review --write` →
   BETTER numbers above; registry written; next arm selected.
5. Label-sync: `sync-issue-labels <registry> --apply` (op 1: #62
   research:proposal→validated + remove research:active; op 2: #58
   research:proposal→active).
6. Docs + PR: PROJECT_STATUS.md / EXPERIMENT_TREE.md /
   EXPERIMENTS_AND_RESULTS.md updated; `gh pr create --base
   exp/stage-b-brainstorm-widen-20260806` (chain base! head must be the current
   checkout branch) → PR #64; verify `gh pr view 64 --json
   headRefName,baseRefName` + `gh pr diff 64 --name-only`.

## Pitfalls / notes for the next deterministic kinematic arm

- **`_segments_intersect` is the cheap crossing test** — don't rebuild it.
  Use it with (a) real hip→ankle segments (my first legs-crossed version
  passed mixed `(lh[0], lk[1])` points and never fired; fixed to
  `_segments_intersect(lh, la, rh, ra)` fired 1/24).
- **Semantic seg2 cannot show arm/torso class overlap** (one label per pixel);
  the proximity-dilation proxy is honest — document that it is a proxy.
- **Band threshold before freeze, on the real cohort** — the calibration probe
  is cheap (CPU, 24 items) and caught nothing degenerate this time, but it is
  the gate (train the 135° value via the probe, not on a paper).
- **Cross-arm supported counts moved** (baseline no-evidence 47→60 across arms)
  because each new validated dossier dimension adds claims to the shared
  `context-raw-no-evidence` condition only when rendered there — it is the
  matched paired baseline within the run that matters, and the sign test is
  within-run paired.
