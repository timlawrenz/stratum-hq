# Experiments & Results — Stratum Contextual Specialist Research

This ledger records empirical findings and negative results permanently. A green implementation or passing unit test is not an empirical PASS.

## Harness initialization — `[ACTIVE]`

**Date:** 2026-08-03

**Goal:** Establish a reusable, project-neutral autonomous-research control plane while grounding it in the Stratum `crawlr/approved` program.

**Evidence:**

- Canonical source inventory: 11,825 flat source images at `/mnt/nas-ai-models/training-data/crawlr/approved`.
- The current derived tree is partial: 4,901 metadata leaves, 4,845 legacy captions, 2,113 `seg2`, 11 `pose2`, 11 `pointmap`, and 10 `caption2`/`determinations`/`t52` examples at the time of inventory.
- The existing `t5_*` and `t52_*` encoders are fixed to 512 tokens, so a target ~4K compact context requires separate artifacts and a new downstream-consumption research arm.
- Existing NAS GPU scheduler: 4090 local route; Strix remote route `ssh:max395` with 10GB evergreen reservation. Its `poll` operation performs the atomic claim; harness supervisor is observer-only.

**Draft PR:** #6 (`feat: add autonomous research harness`).

**Verdict:** `PENDING` — harness implementation passed local verification and must undergo draft-PR review. No GPU work or corpus mutation is authorized by this entry.

## Arm 1 — Portrait evidence discovery — `[ACTIVE — PENDING]`

**Date:** 2026-08-04

**Goal:** Map observable, useful, non-redundant evidence roles for the curated exactly-one-woman corpus without selecting a closed taxonomy or fixed specialist roster.

**Pre-registered gate:** The active-issue gate remains `evidence-discovery-rubric-v1`: the map must distinguish deterministic, semantic, and open-ended evidence; document scope/provenance/abstention/failure behavior; name a falsifiable next arm; and receive review before any `PASS`.

**Evidence:**

- Native `gh issue list --json …` snapshot of all four open research issues validated through `research_harness`; #3 remains the only `research:active` arm, #2 the only root, and there are no closed post-mortem issues.
- A read-only bounded inventory of `crawlr/approved` found 11,825 flat source files (10,857 JPEG / 445 PNG / 523 WebP). One immediate derived-root listing found 4,901 entries; only 3 of 7 deterministic source spot checks resolved to a derived leaf, so no completion rate was inferred.
- One observed fully enriched prototype leaf made the preprocessing confound concrete: raw source 1080×1350, legacy `pixel.npy` `(3, 1216, 832)`, and `pose2`/`seg2`/`pointmap` aligned to raw spatial dimensions. It had `determinations.json`, `caption2.txt`, and 512-token `t52_*`, but no `context4k.*` or `compression.json`.
- Source review covered the baseline caption path, current stratum2 pipeline, and prototype commit `dd9807794f893f89e04758cc9b76170de3d0b36d`. The prototype's synthetic fixtures establish wiring/corroboration behavior only; they do not establish real-image factuality or comparative utility.

**Artifact:** [`PORTRAIT_EVIDENCE_DISCOVERY_MAP.md`](PORTRAIT_EVIDENCE_DISCOVERY_MAP.md), with machine-readable inventory [`assets/portrait-evidence-discovery-inventory-2026-08-04.json`](assets/portrait-evidence-discovery-inventory-2026-08-04.json).

**Verdict:** `PENDING` — this is a documentation and provenance result, not a model-selection or caption-quality verdict. Review must precede any selection of #4; #4 then needs fixed source-hashed items, matched views/settings, claim-support rubric, null/evaluator audit, and adversarial review.

## Arm 0 — Geometry-grounded captioning prototype — `[ACTIVE — PENDING]`

**Branch / PR:** `exp/geometry-grounded-captioning`, draft PR #1.

**Goal:** Test whether Sapiens2-derived structural evidence can help an image-aware local aggregator produce more faithful contextual descriptions than the legacy single-caption path.

**Implementation evidence:**

- Additive chain: `pose2 + seg2 + optional pointmap → determinations.json → caption2.txt → t52_*`.
- Legacy `caption.txt` and `t5_*` remain untouched.
- Synthetic fixtures test geometry, determinations schema, relations, and pass isolation.

**Pre-registered gate:** Not yet valid. A controlled evaluation must first hold constant the source-image preprocessing, prompt structure, model/generation settings, item set, and review rubric.

**Known confound:** The legacy caption path uses a bucketed/cropped image while current `caption2` opens the raw source image. Any apparent output difference currently combines preprocessing, prompt, and evidence changes.

**Verdict:** `PENDING` — preserve as a prototype; do not infer quality or downstream usefulness.
