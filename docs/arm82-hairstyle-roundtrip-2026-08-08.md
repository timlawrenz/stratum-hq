# Arm #82 hairstyle / hair-arrangement — round-trip

**Verdict (harness-computed, 2026-08-08): BETTER.** support ratio 0.3219 →
0.8063 (Δ +0.4844), supported 47 → 154, unsupported 99 → 37, paired positive
19/24, sign-test p = 0.003305. Runs: `/mnt/nas-ai-models/research/stratum/
stage-b-hairstyle-v1` (96) + `-review` (96 rows). Registry: hairstyle →
validated (cycle 18/selection 19 via exploit); face-visibility → active,
selection_progress 19. PR #88 (head `exp/stage-b-hairstyle-arm82-20260808`,
base `exp/stage-b-body-configuration-arm83-20260808`).

## What the arm adds
NEW deterministic evidence part (no new model): scale-invariant hairstyle
measurement from the frozen seg2 DOME-29 Hair region (present 24/24) + pose2
GOLIATH-308 shoulder/neck keypoints:

- **hair_length_band** — short / shoulder-length / long, from how far the
  hair's lowest extent hangs below the shoulder-midpoint line, normalized by
  shoulder width (`hair_below_shoulder_ratio`, cuts 0.15 / 0.60).
- **hair_arrangement_band** — down / kept-up, from the fraction of hair below
  the shoulder line (`hair_below_shoulder_fraction >= 0.10` + bsr >= 0.15 →
  down; else kept-up). 'kept-up' honestly covers short crops, buns, and
  tied-backs that a Hair silhouette cannot be separated into individually.

Raw normalized fractions / pixel spans stay payload-only; only the coarse
bands are verbalized.

## Band-degeneracy recovery (the honest move this arm made)
The on-paper up/tied-back/down arrangement scheme was DEGENERATE on the
calibration probe: `up` never fired (7/7 non-down items were short crops with
span >= 0.55, mislabeled `tied-back` — calling a short crop tied-back would be
fabrication). Re-probed the discriminator and collapsed to the pair the
geometry genuinely separates: down vs kept-up. Same re-probe rule as
#34/#35/#59/#75/#83.

## Calibration (frozen 24-item cohort, 2026-08-08)
| Band | Distribution | max_share |
|---|---|---|
| hair_length_band (measured 22/24) | long 13, shoulder-length 5, short 4 | 0.59 |
| hair_arrangement_band (measured 22/24) | down 15, kept-up 7 | 0.68 |
| honest abstentions (2/24) | unreliable shoulder/neck keypoints | — |

The rendered evidence distribution matched the probe EXACTLY (long 13 / short
4 / shoulder 5 / abstain 2 in the 24 hairstyle-condition prompts), confirming
the evidence text reached the model with no drift.

## Module map
- `src/research_harness/hairstyle.py` — compute_hairstyle (seg2 + pose2 →
  bands + payload), validators, render_hairstyle, HairstyleError.
- stage_b.py: evidence kind `hairstyle` (`_EVIDENCE_INPUT_NAMES = ("pose2.npy",
  "seg2.npy")`), `_hairstyle_evidence()`, `_serialize_hairstyle()`, plan branch
  `context-raw-hairstyle`, rebuild mapping, include gate, `_load_selected_item`,
  `_render_condition`. dossier.py: `hairstyle:v1` id + render/payload factories
  + assemble/build_evidence_payload wiring.
- `scripts/probe_hairstyle.py`, `scripts/freeze_hairstyle_manifest.py`,
  `tests/test_hairstyle.py` (19 tests).

## Verification
`pytest tests/ -q` 672 passed (653 → 672, +19), validate-program valid,
validate-dimension-registry valid, validate-comparison-plan valid,
validate-gpu-manifest valid. Label-sync applied (82 → research:validated,
84 → research:active). Both generation and review on the local 4090 via the
scheduler (job `stratum-stage-b-hairstyle-v1` + adversarial-review).
