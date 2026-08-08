# Arm #84 face-visibility / face-prominence — round-trip

**Verdict (harness-computed, 2026-08-08): BETTER.** support ratio 0.3219 →
0.9767 (Δ +0.6548), supported 47 → 210, unsupported 99 → 5, paired positive
20/23, sign-test p = 0.000244. Runs: `/mnt/nas-ai-models/research/stratum/
stage-b-face-visibility-v1` (96) + `-review` (96 rows). Registry:
face-visibility → validated (cycle 19/selection 20 via exploit);
environment-clearance → active via the ε-greedy EXPLORE slot (exploration
slot fired at selection 20, forcing the lowest-prior proposal, EIG 0.55);
selection_progress 20. PR #89 (head `exp/stage-b-face-visibility-arm84-
20260808`, base `exp/stage-b-hairstyle-arm82-20260808`).

## What the arm adds
NEW deterministic evidence part (no new model): scale-invariant
face-prominence measurement from the frozen seg2 DOME-29 Face_Neck + Hair:

- **face_share_of_head** — Face_Neck px ÷ (Face_Neck + Hair) px inside the
  Face_Neck bbox dilated to the local head window.
- **face_visibility_band** — clearly-visible (share ≥ 0.65) /
  partially-framed (0.45–0.65) / hair-dominant (< 0.45).

Only the coarse band is verbalized; the raw ratio stays payload-only.

## Band-degeneracy recovery (the honest move this arm made)
The on-paper occlusion-overlap measure was **structurally impossible** on
hard-label seg2: one class per pixel means Face_Neck can NEVER overlap an
occluding class at a pixel, so occlusion_fraction = 0.000 for 23/23 (max_share
1.00). Re-probed discriminators and re-cut to the face-to-hair prominence
ratio. Also fixed a scale-invariance violation: a fixed 20 px local-window
margin breaks the ratio under rescaling — the margin must be proportional to
the face extent (max floor 20, 0.20 × face bbox long side). Same
re-probe-discriminators rule as #34/#35/#59/#75/#82/#83.

## Calibration (frozen 24-item cohort, 2026-08-08)
| Band | Distribution | max_share |
|---|---|---|
| face_visibility_band (measured 23/24) | hair-dominant 12, partially-framed 7, clearly-visible 4 | 0.52 |
| honest abstentions (1/24) | no Face_Neck region in frame | — |

face_share_of_head: min 0.239 / p25 0.391 / median 0.449 / p75 0.590 / max
0.888. The rendered evidence distribution matched the probe EXACTLY (12/7/4
+ 1 abstain in the 24 face-visibility-condition prompts).

## Module map
- `src/research_harness/face_visibility.py` — compute_face_visibility (seg2 →
  band + payload), validators, render_face_visibility, FaceVisibilityError.
- stage_b.py: evidence kind `face-visibility` (`_EVIDENCE_INPUT_NAMES =
  ("seg2.npy",)` — the arm binds only seg2, pose2 not needed), declaration,
  serializer, plan branch `context-raw-face-visibility`, rebuild mapping,
  include gate, `_load_selected_item`, `_render_condition`. dossier.py:
  `face-visibility:v1` id + render/payload factories + assembly wiring.
- `scripts/probe_face_visibility.py`, `scripts/probe_face_visibility_discriminators.py`,
  `scripts/freeze_face_visibility_manifest.py`,
  `tests/test_face_visibility.py` (15 tests).

## Verification
`pytest tests/ -q` 687 passed (672 → 687, +15), validate-program valid,
validate-dimension-registry valid, validate-comparison-plan valid,
validate-gpu-manifest valid. Label-sync applied (84 → research:validated,
85 → research:active). Both generation and review on the local 4090 via the
scheduler (job `stratum-stage-b-face-visibility-v1` + adversarial-review).
