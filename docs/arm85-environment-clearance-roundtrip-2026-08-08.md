# Arm #85 environment-clearance / subject-to-backdrop negative space — round-trip

**Verdict (harness-computed, 2026-08-08): BETTER.** support ratio 0.3219 →
0.9158 (Δ +0.5939), supported 47 → 185, unsupported 99 → 17, paired positive
19/21, sign-test p = 0.000111. Runs: `/mnt/nas-ai-models/research/stratum/
stage-b-environment-clearance-v1` (96) + `-review` (96 rows). Registry:
environment-clearance → validated (cycle 20/selection 21 via exploit);
iris-eye-color → active, selection_progress 21. PR #90 (head
`exp/stage-b-environment-clearance-arm85-20260808`, base
`exp/stage-b-face-visibility-arm84-20260808`).

## What the arm adds
NEW deterministic evidence part (no new model): scale-invariant
subject-to-environment clearance from the frozen seg2 DOME-29 Background split:

- **normalized directional clearances** — Background gap from the subject bbox
  edge to the frame edge in each of top/bottom/left/right, divided by the
  subject bbox extent on the SAME axis (pure ratio).
- **clearance_ratio** — median of the LEFT/RIGHT normalized clearances (the
  horizontal negative space).
- **clearance_band** — tight (< 0.15) / moderate (0.15–0.60) / spacious (≥ 0.60).

Only the coarse band is verbalized; raw normalized distances stay payload-only.

## Band-degeneracy recovery (the honest move this arm made)
The on-paper median-of-all-4-axes was DEGENERATE on this portrait cohort: the
tall full-body subject bbox makes the vertical (top/bottom) gaps near-zero for
most items → 19/22 items collapsed into 'tight' (max_share 0.86). Re-probed
the discriminator: the LEFT/RIGHT horizontal negative space is the axis that
actually separates 'close to a wall/backdrop' from 'in an open space' on a
portrait. Same re-probe-discriminators rule as #34/#35/#59/#75/#82/#83/#84.

## Calibration (frozen 24-item cohort, 2026-08-08)
| Band | Distribution | max_share |
|---|---|---|
| clearance_band (measured 22/24) | tight 10, moderate 9, spacious 3 | 0.45 |
| honest abstentions (2/24) | full-bleed subjects with zero Background clearance | — |

clearance_ratio: min 0.0 / p25 0.03 / median 0.28 / p75 0.49 / max 1.40. The
rendered evidence distribution matched the probe EXACTLY (10/9/3 + 2 abstains
in the 24 clearance-condition prompts).

## Module map
- `src/research_harness/environment_clearance.py` — compute_environment_clearance
  (seg2 → bands + payload), validators, render_environment_clearance,
  EnvironmentClearanceError.
- stage_b.py: evidence kind `environment-clearance` (`_EVIDENCE_INPUT_NAMES =
  ("seg2.npy",)`), declaration, serializer, plan branch
  `context-raw-environment-clearance`, rebuild mapping, include gate,
  `_load_selected_item`, `_render_condition`. dossier.py:
  `environment-clearance:v1` id + render/payload factories + assembly wiring.
- `scripts/probe_environment_clearance.py`,
  `scripts/freeze_environment_clearance_manifest.py`,
  `tests/test_environment_clearance.py` (12 tests).

## Operational note
First launcher-bound generation attempt failed with `research-stage-b: local
Ollama generation failed: Read timed out (read timeout=300)` — a transient
cold-load race (model not yet resident when the first request fired), NOT a
code/plan fault: the direct runner passed preflight and the retry succeeded
cleanly (96 records). Cleaned the failed job file, re-requested, re-launched.

## Verification
`pytest tests/ -q` 699 passed (687 → 699, +12), validate-program valid,
validate-dimension-registry valid, validate-comparison-plan valid,
validate-gpu-manifest valid. Label-sync applied (85 → research:validated,
80 → research:active). Both generation and review on the local 4090 via the
scheduler (job `stratum-stage-b-environment-clearance-v1` + adversarial-review).
