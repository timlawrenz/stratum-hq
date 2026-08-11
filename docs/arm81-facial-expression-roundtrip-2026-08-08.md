# Arm #81 facial-expression / smile — round-trip

**Verdict (harness-computed, 2026-08-08): BETTER.** support ratio 0.3219 →
0.9689 (Δ +0.6470), supported 47 → 187, unsupported 99 → 6, paired positive
19/23, sign-test p = 0.0013. Runs: `/mnt/nas-ai-models/research/stratum/
stage-b-facial-expression-v1` (96) + `-review` (96 rows). Registry:
facial-expression → validated (cycle 22/selection 23) — **the sweep is now
EXHAUSTED (27/27 dimensions validated, 0 proposals) → next_action
brainstorm-new-data.** PR #92 (head `exp/stage-b-facial-expression-arm81-
20260808`, base `exp/stage-b-eye-color-arm80-20260808`).

## What the arm adds
NEW deterministic evidence part (no new model): scale-invariant
smile/facial-expression band (neutral / slight-smile / open-smile) from the
frozen pose2 GOLIATH-308 mouth-corner keypoints:

- spread_ratio = mouth-corner width / inter-eye reference.
- openness_ratio = mouth vertical opening / reference (open laugh signature).
- corner_elevation_ratio = corner rise above the lip midline / reference
  (smile curvature).
- Band: openness ≥ 0.28 → open-smile; else corner_elev ≥ 0.05 → slight-smile;
  else neutral.

Only the coarse band is verbalized; raw normalized ratios stay payload-only.

## Band-degeneracy recovery (the honest move this arm made)
The first openness-only 3-band cut was DEGENERATE — 17/19 items collapsed into
'slight-smile' (max_share 0.89) because this portrait cohort's mouth SPREAD is
near-constant and open laughs are rare. Re-probed the discriminator: the
genuinely-discriminating axes are OPENNESS (mouth opening, two clear outliers)
+ CORNER ELEVATION (smile curvature, sign/shape around the lip midline). Also
fixed a degenerate fallback: the eye-less scale must use mouth WIDTH, not
mouth height (height would make openness ratio trivially 1.0). Same
re-probe-discriminators rule as #34/#35/#59/#75/#82/#83/#84/#85/#80.

## Calibration (frozen 24-item cohort, 2026-08-08)
| Band | Distribution | max_share |
|---|---|---|
| expression_band (measured 19/24) | neutral 9, slight-smile 8, open-smile 2 | 0.47 |
| honest abstentions (5/24) | mouth occluded / low-confidence keypoints | — |

The rendered evidence distribution matched the probe EXACTLY (9/8/2 + 5
abstains in the 24 expression-condition prompts).

## Module map
- `src/research_harness/facial_expression.py` — compute_facial_expression
  (pose2 → band + payload), validators, render_facial_expression,
  FacialExpressionError.
- stage_b.py: evidence kind `facial-expression` (`_EVIDENCE_INPUT_NAMES =
  ("pose2.npy",)`), declaration, serializer, plan branch
  `context-raw-facial-expression`, rebuild mapping, include gate,
  `_load_selected_item`, `_render_condition`. dossier.py:
  `facial-expression:v1` id + render/payload factories + assembly wiring.
- `scripts/probe_facial_expression.py`,
  `scripts/freeze_facial_expression_manifest.py`,
  `tests/test_facial_expression.py` (12 tests).

## Verification
`pytest tests/ -q` 721 passed (709 → 721, +12), validate-program valid,
validate-dimension-registry valid, validate-comparison-plan valid,
validate-gpu-manifest valid. Label-sync applied (81 → research:validated).
Both generation and review on the local 4090 via the scheduler (job
`stratum-stage-b-facial-expression-v1` + adversarial-review).
