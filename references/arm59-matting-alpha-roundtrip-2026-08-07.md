# Arm #59 matting-alpha round-trip (2026-08-07) — VALIDATED BETTER

Deterministic matting / alpha-fidelity evidence specialist. Executive summary
for the next arm (face-geometry #60, the active arm after this cycle) and a
worked new-evidence-kind template — especially the band-calibration recovery when
the first on-paper thresholds are degenerate.

## Verdict (harness-computed)
- **BETTER** — support ratio 0.3219 → 0.8657 (Δ +0.5438); supported 47→187,
  unsupported 99→29; paired positive **18/22**; sign-test **p = 0.002172**;
  `inconclusive: false, significant: true`.
- Registry: `matting-alpha → validated`; `face-geometry #60 → active`
  (selected_via **exploit**, selection_progress 9).
- Run roots: `/mnt/nas-ai-models/research/stratum/stage-b-matting-alpha-v1`
  (+ `-review`). Branch `exp/stage-b-matting-alpha-arm59-20260807`, PR #66.

## Measurement semantics (arm #59, scale-invariant only)
From `matting.npy` (Sapiens2 per-pixel soft alpha matte, source-dimension-matched
`(H,W)` float16 in [0,1], present 24/24 frozen items, unbound by any validated
arm) + `seg2.npy` DOME-29 (Hair class):
- subject alpha-coverage band: opaque (alpha≥0.9) fraction of the frame,
  banded **sparse / centered / fills-frame**;
- boundary crispness band: median alpha-gradient magnitude over the 1-px
  silhouette ring (subject XOR eroded subject) — a scale-invariant edge-sharpness
  descriptor (alpha-change per px), banded **soft / moderate / crisp**;
- soft-edge character: share of the semi-transparent band (alpha∈[0.05,0.9))
  that lies inside the seg2 Hair class — the "soft detachable hair strands"
  axis, banded **skin-clean / mixed / hair-dominant**.

Only ratios, normalized gradient descriptors, and bands are verbalized. Absolute
px areas/band widths and silhouette structure stay in `evidence_payload`. Constraint
honored: `.get("abstained")` + `abstention_reason` on every abstain; degenerate
matte (values outside [0,1]) or subject mask below the px floor abort as abstained.

## BAND CALIBRATION — first on-paper thresholds were DEGENERATE (recovery recipe)
Probe-1 (paper thresholds: soft-edge 0.015/0.030, detail 0.05/0.15, silhouette
closedness) returned: soft_edge **21/24 "sharp"**, detail_band **23/24
"fine-detail"**, silhouette **24/24 "closed"** — three near-constant bands = weak
signals. The recovery: re-probe DISCRIMINATOR metrics on the real cohort, not
the original band definitions:
1. **boundary crispness** = median |grad alpha| on the silhouette ring
   (sharp cutout localizes the 0→1 transition to 1 px → high; feathered spreads
   it → low). Measured spread 0.064–0.328; threshold 0.16/0.24 → **11 crisp /
   10 moderate / 3 soft** (max 46%).
2. **hair-soft share** = fraction of the semi-transparent band inside the Hair
   class. Spread 0.014–0.787; threshold 0.20/0.50 → **6 skin-clean / 14 mixed /
   4 hair-dominant** (max 58%).
3. **coverage** = opaque-frame fraction. Threshold 0.20/0.55 → **5 sparse / 17
   centered / 2 fills-frame** (max 70.8%).
Final: 24/24 measurable, 0 abstentions, no band ≥75%. Silhouette closedness was
honestly **dropped from prose** (24/24 constant on a no-crops cohort) — payload-only.

## New-evidence-kind touchpoints (arm #59, mirror arm #58)
1. Module `src/research_harness/matting_alpha.py`:
   `compute_matting_alpha` + `validate_matting_array`/`validate_seg2_array` +
   `MattingAlphaError`. Uses `scipy.ndimage` (erosion, distance, label).
2. `_EVIDENCE_INPUT_NAMES["matting-alpha"] = ("matting.npy", "seg2.npy")`.
3. Import + `_matting_alpha_evidence()` declaration + `_serialize_matting_alpha()`.
4. `build_stage_b_plan`: add kind to allowed tuple + `elif evidence_kind == "matting-alpha"`
   (`context-raw-matting-alpha`, plan id `stage-b-first500-matting-alpha-v1`).
5. `_validate_frozen_execution_plan`: `elif "context-raw-matting-alpha" in condition_ids`.
6. `_load_selected_item`: load `matting.npy` (SHA-bound) when present; verify
   (H, W) + pixel-alignment with seg2; compute; append prepared["matting_alpha"].
7. `_render_condition`: `context-raw-matting-alpha` branch returns
   `(raw.copy(), _context_prompt(evidence_text), matting_alpha)`.
8. Dossier: `matting-alpha:v1` in DIMENSION_EVIDENCE_IDS, `render_matting_alpha()`
   factory, `_matting_alpha_payload()` helper, passthroughs in `_rendered_context4k`
   (both assemble_dossier + build_evidence_payload call sites).

Review path untouched (generic `no-specialist-evidence-v1` + `"context" in cid`).

## Test notes
- `test_roundtrip_context4k.py::test_render_context4k_condition_emits_evidence_linked_compact`
  asserts the EXACT `dossier_evidence_ids` list — `matting-alpha:v1` inserted in
  order (documented "test you will break"; fixed).
- Unit fixtures: the crisp fixture is a 1-px step (soft-edge character correctly
  gives None — no semi-transparent band; real mattes always carry anti-aliasing),
  the soft fixture feathers ALL FOUR sides (a one-side feather left step edges on
  the top/bottom/right dominating the ring median — crispness test would fail).

## Validation (this cycle)
547 pytest passed; validate-program / validate-dimension-registry /
validate-gpu-manifest all valid; label-sync 2 ops (#59 → research:validated,
#60 → research:active). Memory: cleared the tick-ready marker after the tick.
