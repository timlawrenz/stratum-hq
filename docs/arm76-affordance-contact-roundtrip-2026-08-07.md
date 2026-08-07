# Arm #76 affordance-contact / subject self-contact — round-trip

**Status:** running (2026-08-07). Deterministic new evidence part (no new model):
hand-to-own-body contact count, hand-elevation/gesture count, and subject
grounding from pose2 GOLIATH-308 + seg2 DOME-29.

## What the arm adds
Scale-invariant subject self-contact / affordance measurements (NEW deterministic
evidence part, no new model) from the frozen pose2 + seg2:

- **hand_contact_count** (0..2) — how many hands have their wrist within
  `TRUNK_CONTACT_NORM = 0.35` shoulder-widths of the subject's own trunk
  region (hands resting against the body, e.g. a hand on the hip, folded arms).
- **hand_elevation_count** (0..2) — how many wrists sit above the hip line by
  more than `WRIST_ABOVE_HIP_NORM = 0.30` shoulder-widths (a raised / gesturing
  hand).
- **grounded** (bool) — whether the subject silhouette reaches the bottom frame
  edge (standing full-frame / grounded in the lower frame).

Normalization uses the acromion width (falls back to shoulder width); a missing
or unreliable shoulder width disables the hand axes (abstain) while the
frame-based grounding measurement still fires.

**Honest scope boundary:** seg2 DOME-29 segments ONLY the subject (Background=0
is not an object label), so subject-to-EXTERNAL-object contact (held-in-hand /
leaning-on / sitting-on an object) is NOT measurable from seg2+pose2 alone.
That axis is the object-relations arm's validated Grounding-DINO domain. This
arm covers own-body contact + grounding only and never fabricates a
"holding"/"leaning on" claim. This was flagged in the arm's `falsified_if` and
`coverage_notes` (redundancy against pose-articulation #62 is also checked).

## Module map
- `src/research_harness/affordance_contact.py` — `compute_affordance_contact`
  (pose2 + seg2 -> bands + normalized payload), `validate_pose2_array` /
  `validate_seg2_array`, `render_affordance_contact`, `AffordanceContactError`.
- `stage_b.py` evidence kind `affordance-contact` (`_EVIDENCE_INPUT_NAMES =
  ("pose2.npy", "seg2.npy")`; `_affordance_contact_evidence()`; serializer;
  plan branch; rebuild_kind; include-gate; `_load_selected_item`;
  `_render_condition`). `dossier.py` `affordance-contact:v1` factory + payload.
- `scripts/probe_affordance_contact.py`, `scripts/freeze_affordance_contact_manifest.py`,
  `tests/test_affordance_contact.py` (TDD, 14 tests).

## Calibration (frozen 24-item cohort, 2026-08-07)
| Band | Distribution | max_share |
|---|---|---|
| hand_contact_count | {0:11, 1:6, 2:7} | 0.46 |
| hand_elevation_count | {0:14, 1:6, 2:4} | 0.58 |
| grounded | {False:10, True:14} | 0.58 |
| wrist visibility | 2=18/24, <2=6/24 | — (honest abstention) |

All bands under the 75% degeneracy line. Wrist keypoints land on body/hand/arm
seg2 classes in the cohort, confirming the wrist->trunk distance signal reads
real geometry.

## Key pitfalls (from the design probe)
1. **DOME-29 has no object classes** — seg2 is subject + Background only. Any
   "contact" claim about an external object would be fabricated; the honest
   deterministic axes are own-body contact and grounding. Do NOT add a
   "held-in-hand" prose claim to this specialist.
2. **Normalization anchor**: use the acromion width first, shoulder fallback;
   when both are unavailable the hand axes must abstain (not guess a scale).
   The grounding band is frame-based and survives that abstention.
3. **GOLIATH-308 wrist keypoints are at the wrist joint** — the seg2 class AT
   the keypoint is frequently the hand OR the lower-arm (class 7/16 or 6/15);
   use the wrist->trunk-mask Euclidean distance, not the class label at the
   point alone.
4. Contact threshold 0.35 sw and elevation 0.30 sw were chosen from the cohort
   histogram (0/1/2 distribution); verify with the shipped probe on re-freeze.

## Run layout
- Plan `stage-b-first500-affordance-contact-v1` (96 records), GPU manifest
  `stratum-stage-b-affordance-contact-v1` (4090, 22GB, 2h), run root
  `/mnt/nas-ai-models/research/stratum/stage-b-affordance-contact-v1`,
  review root `...-review` (96 rows, gemma4:e4b).
- Command sequence in `references/roundtrip-arm-execution-recipe-2026-08-06.md`.

## Verification
`pytest tests/ -q` (635 passed incl. the 14 new), `validate-program` valid,
`validate-dimension-registry` valid, `validate-comparison-plan` valid,
`validate-gpu-manifest` valid.
