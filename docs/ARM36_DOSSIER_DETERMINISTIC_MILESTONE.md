# Arm #36 — dossier-context4k: deterministic dossier assembly milestone

**Branch:** `exp/stage-b-dossier-context4k-arm36-20260806`
**Date:** 2026-08-06
**Registry state:** `dossier-context4k` remains `active` (deterministic stage complete, no conclusion yet).

## What this milestone adds (additive, PR-only)

1. `src/research_harness/dossier.py` — deterministic, CPU-only per-asset
   dossier assembly + context4k compression:
   - `assemble_dossier` — combines the five now-validated deterministic
     dimensions (body-type proportions, clothing, hair, skin-color, lighting)
     plus relational determinations into a claim-by-claim dossier where every
     claim carries its supporting evidence IDs (scale-invariant ratios/bands/
     names only; absolute pixel measurements stay machine-readable).
   - `compress_dossier_to_context` — deterministic compression into a compact
     context with per-claim evidence links; fills to a token budget, trims at a
     natural boundary when overshooting, and NEVER pads: it reports
     `under_budget` honestly when the corpus is smaller than the budget.
   - `build_compression_bundle` — contract-shaped bundle validated by
     `validate_compression_bundle(data, program)`; refuses mis-sized bundles.
   - `build_item_context4k_artifacts` — writes the three configured artifacts
     `context4k.json`, `context4k.md`, `compression.json` under a per-item dir.
2. `scripts/run_dossier_context4k.py` — frozen-cohort batch driver (CPU, reads
   only the candidate manifest's selected pose2/seg2/normal2 + source pixels;
   writes only to an approved noncanonical research root).
3. `tests/test_dossier.py` — 12 tests (scale-invariance of verbalized claims,
   abstention behavior, deterministic token counting, evidence-linkage of every
   compact claim, contract validation honesty gate, artifact writing).

## Honest empirical finding from the frozen first-500 24-item cohort

Run: `/mnt/nas-ai-models/research/stratum/dossier-context4k-v1/dossier-run-summary.json`

- 24/24 items assembled; every dossier carries all 6 evidence IDs
  (5 dimension specialists + relational determinations).
- **Expanded dossier (deterministic only): 387–648 tokens per item.**
- **Compact context (deterministic only): median ~298 tokens, all `under_budget`.**

The program floors are 100 000 expanded-dossier tokens and 4 000 compact-context
tokens. The deterministic specialists alone therefore do NOT reach either floor —
reaching the floors requires the **aggregator expansion stage** (model-derived
prose expansion of the evidence payload under the scheduler lifecycle) and then a
**truly lossy compression** to 4K. This is the designed honesty gate, not a
failure: the deterministic corpus is the evidence base; the model can only
elaborate claims ALREADY grounded by it.

This is exactly why `build_compression_bundle` refuses to certify an under-budget
deterministic-only bundle (see `test_dossier.py::test_build_compression_bundle_validates_against_contract`).

## Next step (round-trip claim-support audit)

Frozen expanded-dossier construction + compression are the deterministic
prerequisite. The schedulable part — aggregator expansion to ~100K tokens,
compression to context4k, caption generation from context4k vs a plain-4K
summarization baseline, independent adversarial review, and `autonomous-tick`
with the roundtrip method — is the next research arm step and needs a GPU
manifest under the scheduler lifecycle (frozen-cohort protocol). No corpus
mutation; outputs remain under `/mnt/nas-ai-models/research/stratum`.
