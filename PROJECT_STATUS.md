# Project Status — Stratum Contextual Specialist Research

**Last updated:** 2026-08-09
**Phase / status:** Program active — option-B evidence growth (ruling #46); no open `research:hold`; exactly one active arm.

## Current state

The open-world specialist program over `crawlr/approved` (exactly-one-woman curation) is
running on the GitHub-issue tree. The compact-context floors (100K dossier / 4K compact)
were re-authorized in decision #46 to be reached by **growing evidence supply (option B)**
rather than relaxing the floors. All prior holds (#9, #18) are closed. The harness runs
PR-only; `main` currently carries **no** research-harness code (the entire stack lives in
open draft PRs — see "Stable basis" below).

- **Active arm: #97 — garment-type / silhouette-category evidence** (NEW deterministic
  part from seg2 clothing classes; no new model). Stage-B job `stratum-stage-b-garment-type-v1`
  is queued on the shared scheduler (4090 busy with prior-project jobs).
- **Validated specialist backlog:** ~20 arms validated (lighting, hair, clothing, skin,
  body-type, setting, texture, reconstruction, matting, pointmap, face-geometry,
  object-relations, scene-category, gaze/head, apparent-age, affordance/contact,
  image-focus, pose-articulation, vlm-dense, body-configuration, facial-expression,
  eye-color, face-visibility, environment-clearance, hairstyle, …). Fresh proposals
  pending: #94 hair-texture, #95 image-quality (first-ever NOT_BETTER strike), #96 3D
  body-volume.
- **Recreation / round-trip instrument (arm #37):** proven end-to-end — local ComfyUI
  (`/mnt/fscache/essdee/ComfyUI`) + SDXL Juggernaut XL + CLIP scorer, 24-item frozen
  cohort, 50 generated PNGs archived. Result +0.0679 CLIP delta (22/24 positive),
  verdict **PENDING_HUMAN_SPOT_CHECK** — no acceptable images shown to Tim yet. The
  only local FLUX1 copy is 4-bit NF4 (fidelity confound); full FLUX wiring is a
  separate manifest step (instrument decision on file).
- **Prose caption2 renderer:** drafted in PR #93 (dossier → full-prose `caption2.txt`,
  deterministic + aggregator paths, `caption2_variants` for #79's C-before/C-after
  protocol) — TDD green, unmerged.

## Headline results so far

| Metric | Value | Verdict |
|---|---|---|
| Arm #37 reconstruction round-trip | CLIP Δ +0.0679, 22/24 paired positive | BETTER (pending visual spot-check) |
| Arm #83 body-configuration | support ratio 0.322 → 0.897, sign-test p = 0.0008 | BETTER |
| Arm #95 image-quality (first strike) | NOT_BETTER | strike 1/3 |
| VLM dense (#47) | marginal Δ +0.2206, p = 0.0133 | BETTER |
| Arm #36 dossier/context4k | 24/24 items archived (dossier-context4k-v2) | validated |
| Program contract | `validate-program research/program.json` | valid |

## Immediate next actions

1. **Merge a stable first version to `main`** (recon instrument + dossier/context4k +
   prose renderer) so the ablation study has a reviewed base.
2. **Produce the first acceptable recreation images**: prose `caption2.txt` (PR #93
   renderer) → ComfyUI. SDXL Juggernaut is the proven instrument; FLUX wiring
   (unet + text encoders + VAE, full precision) is the second instrument step.
3. **Plan and run the specialist ablation** through the recreation path: fixed cohort,
   fixed generator, per-specialist C-before/C-after (leave-one-evidence-out), DINOv3
   CLS/patch distance per #79, sign tests, ranking by marginal delta.

## Governance docs

- Experiment tree: `docs/EXPERIMENT_TREE.md`
- Permanent ledger: `docs/EXPERIMENTS_AND_RESULTS.md`
- Research contract: `RESEARCH_CONTRACT.md`
- Program contract: `research/program.json`
- Recon instrument decision: `docs/dinov3-recon-sdxl-vs-flux-instrument-2026-08-08.md`