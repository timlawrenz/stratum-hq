# Arm #80 eye-color / iris-hue — round-trip

**Verdict (harness-computed, 2026-08-08): BETTER.** support ratio 0.3219 →
0.9776 (Δ +0.6557), supported 47 → 218, unsupported 99 → 5, paired positive
19/22, sign-test p = 0.000428. Runs: `/mnt/nas-ai-models/research/stratum/
stage-b-eye-color-v1` (96) + `-review` (96 rows). Registry: iris-eye-color →
validated (cycle 21/selection 22 via exploit); facial-expression → active,
selection_progress 22. PR #91 (head `exp/stage-b-eye-color-arm80-20260808`,
base `exp/stage-b-environment-clearance-arm85-20260808`).

## What the arm adds
NEW deterministic evidence part (no new model): closed-set eye-color band
(brown / dark / blue / green-hazel / gray) from the frozen pose2 GOLIATH-308
iris/pupil center + border keypoints and the already-decoded source RGB:

- For each eye with reliable iris/pupil keypoints, derive iris + pupil radii
  (median border distance), sample the ANNULUS between them (avoiding the dark
  pupil and the specular glare hotspot), robust trimmed mean -> HSV.
- Sequential classification: shadow-dark (low value + low sat) -> dark; warm
  hue -> brown; cool hue with value+sat -> blue; mid hue -> green-hazel;
  light low-chroma -> gray.

Only the coarse closed-set band is verbalized; raw RGB/HSV statistics stay
payload-only.

## Classifier re-cut (the honest move this arm made)
The first classifier's broad `value < 0.30 or sat < 0.18 -> dark` rule
mislabeled LIGHT low-saturation eyes (a hue-250.7/val-0.45 eye and a
val-0.59/sat-0.08 eye both fell into dark). Re-cut on the genuinely
discriminating axes: VALUE separates shadow-dark brown/black from well-lit
brown; HUE separates warm brown from cool blue/green; a light low-chroma eye
reads gray. Same re-probe-discriminators rule as #34/#35/#59/#75/#82/#83/#84/#85.

## Calibration (frozen 24-item cohort, 2026-08-08)
| Band | Distribution | max_share |
|---|---|---|
| eye_color_band (measured 21/24) | brown 15, dark 5, blue 1 | 0.71 |
| honest abstentions (3/24) | eyes closed/cropped/heavily occluded | — |

max_share 0.71 is honest for a brown-eye-dominant portrait cohort (under the
75% degeneracy line). The rendered evidence distribution matched the probe
EXACTLY (15/5/1 + 3 abstains in the 24 eye-color-condition prompts).

## Module map
- `src/research_harness/eye_color.py` — compute_eye_color (pose2 + RGB ->
  band + payload), validators, render_eye_color, EyeColorError.
- stage_b.py: evidence kind `eye-color` (`_EVIDENCE_INPUT_NAMES =
  ("pose2.npy",)`; source RGB bound via source_sha256), declaration,
  serializer, plan branch `context-raw-eye-color`, rebuild mapping, include
  gate, `_load_selected_item`, `_render_condition`. dossier.py: `eye-color:v1`
  id + render/payload factories + assembly wiring.
- `scripts/probe_eye_color.py`, `scripts/freeze_eye_color_manifest.py`,
  `tests/test_eye_color.py` (10 tests).

## Verification
`pytest tests/ -q` 709 passed (699 → 709, +10), validate-program valid,
validate-dimension-registry valid, validate-comparison-plan valid,
validate-gpu-manifest valid. Label-sync applied (80 → research:validated,
81 → research:active). Both generation and review on the local 4090 via the
scheduler (job `stratum-stage-b-eye-color-v1` + adversarial-review).
