# Sweep EXHAUSTED 27/27 → brainstorm-widen (2026-08-08): 4 new proposal arms

**Trigger:** arm #81 facial-expression VALIDATED (BETTER, support 0.3219→0.9689,
p=0.0013) left **27/27 registered dimensions validated, 0 proposals** →
`dimension-sweep-status` returned `exhausted: true` / `next_action:
brainstorm-new-data`. The harness correctly did NOT try to re-run a terminal
arm pattern; it fired the exhausted-sweep → brainstorm-widen cycle.

## What was registered (4 new arms, 0 rejected)
All four passed the `--require-new-evidence-part` seed-diversity gate (every
candidate names a NEW evidence part or NEW model class vs the 27 validated
axes):

| id | arm_issue | NEW part | NEW model class | scope |
|---|---|---|---|---|
| hair-texture | #94 | hair-texture (curl/wave state from seg2 Hair gradient orientations) | — (deterministic) | straight / wavy / curly band |
| image-quality | #95 | image-quality (perceptual NR-IQA) | CLIP-IQA / Q-Align (open-weight, local owned hardware) | sharp / moderate / degraded |
| body-volume | #96 | body-volume (3D body bulk) | HMR2.0 / 4D-Humans whole-body mesh regression (open weights, local) | slim / average / fuller |
| garment-type | #97 | garment-type (upper/lower/skin silhouette split from seg2) | — (deterministic) | garment-category band |

## Open-world sourcing scan (grounding the two NEW-MODEL-CLASS candidates)
- **image-quality:** no-reference perceptual IQA is an active open model field —
  CLIP-IQA (torchmetrics `CLIPImageQualityAssessment`, CLIP-based, open) and
  Q-Align (LMM discrete-level scoring, ICML 2024). Both run on owned CPU/GPU;
  only the decoded source RGB is consumed. No hosted third-party inference of
  the sensitive corpus (sensitive-corpus boundary kept).
- **body-volume:** HMR2.0 / 4D-Humans (`shubham-goel/4dhumans`, open weights,
  transformer-based human mesh recovery) lifts a single image to a 3D mesh and
  can emit normalized body-volume/bulk. **License note recorded:** SMPL/SMPL-X
  parameters carry a non-commercial license — the candidate's qualification
  gate must verify the license for research use BEFORE freezing (recorded in
  the arm issue #96 body and the registry `qualification_gate`).

## Conclude the brainstorm cycle
`autonomous-tick <registry> --write` (nothing-active path — NO review root, NO
verdict, NO fabricated numbers) returned `next_action: activate`, `next_arm:
image-quality` (#95, exploit, EIG 0.65, selection_progress 23). Label-sync
applied (95 → research:active, removed research:proposal). Registry now:
1 active (image-quality), 3 proposals (hair-texture, body-volume, garment-type),
27 validated — sweep un-stalled and researchable again.

## Validation
`pytest tests/ -q` 721 passed; validate-program valid; validate-dimension-registry
valid. PR base = `exp/stage-b-facial-expression-arm81-20260808`.
