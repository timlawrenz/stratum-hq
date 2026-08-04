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

## Comparison-plan provenance hold — `[ACTIVE HOLD / PENDING]`

**Date:** 2026-08-04

**Blocked arm:** #4 — baseline and comparison parity.
**Hold issue:** #9.
**Affected draft:** PR #8 (`feat(research): validate controlled comparison plans`).

**Trigger:** An adversarial synthetic audit found that the initial comparison-plan validator accepted absolute and parent-traversal `source_relative_path` values and accepted a non-null evidence bundle containing only an ID and fingerprint. That made a nominally frozen canonical pilot and provenance-bearing evidence condition unauditable before inference.

**Human decision:** Tim approved **inline specialist declarations** for non-null evidence bundles. A real comparison plan must carry the complete open-world declaration for each specialist: stable ID, scope, inputs/view policy, output semantics, provenance, abstention policy, and qualification gate. The explicit no-specialist baseline remains `kind: "none"` and must omit declarations.

**Required remediation gate:** A remediation must reject absolute, parent-traversal, redundant-segment, and backslash pilot paths; reject opaque, incomplete, duplicate, or non-explicit non-null bundles; accept a valid inline bundle and explicit no-evidence baseline; and pass independent review plus program/tree validation.

**Verdict:** `PENDING / HOLD` — no empirical comparison, image inference, GPU action, model installation, corpus mutation, or backfill occurred. The hold remains open until the reviewed remediation is accepted and revalidated.

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
