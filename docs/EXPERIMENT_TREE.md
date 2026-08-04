# Experiment Tree — Stratum Contextual Specialist Research

This is a living map. GitHub issues are the detailed source of truth; this document provides project orientation rather than a FIFO schedule.

## Live issue tree

* **[ROOT] #2 — Open-world specialist evidence → contextual representation**
  * The program root: canonical corpus, policy, hold boundary, and linked arms.

* **[ACTIVE] #3 — Portrait evidence discovery**
  * The sole `research:active` arm. It surveyed the full open tree before selection.
  * Question: what observable, useful, non-redundant dimensions characterize exactly-one-woman images across extreme crops, full-body scenes, lighting, clothing, swimsuits/nudity, props, and environment?
  * Draft output: [`PORTRAIT_EVIDENCE_DISCOVERY_MAP.md`](PORTRAIT_EVIDENCE_DISCOVERY_MAP.md) and its bounded artifact inventory. The map is `PENDING` review, selects no model winner, and keeps candidate roles open-world.
  * Output: an evidence map and pre-registered candidate qualification plan—not a closed taxonomy.

* **[PROPOSAL / METRIC-RISK] #4 — Baseline and comparison parity**
  * Isolate preprocessing/crop, prompt structure, Sapiens2 evidence, and aggregator model before judging the prototype.

* **[PROPOSAL / PENDING] #5 — Geometry-grounded captioning prototype** (`exp/geometry-grounded-captioning`, draft PR #1)
  * Additive chain: `pose2 + seg2 + optional pointmap → determinations → caption2 → t52`.
  * Synthetic fixture coverage exists. No controlled empirical verdict exists.
  * The arm is not production-ready and must not be merged as a result of the harness build.

## Harness status

* **[DRAFT-PR REVIEW] Autonomous research control plane** (draft PR #6)
  * Project-neutral GitHub issue-tree, evidence/compression contract, fail-closed GPU manifest validation, and draft-PR-only workflow.
  * GPU supervisor is observer-only until a separately reviewed host-specific launcher exists.

## Future candidate branches

* **[TBD] Open-world specialist qualification**
  * Candidate models, fine-tunes, deterministic measurements, embeddings, and future discoveries must each earn a role through a declared scope, provenance, abstention behavior, and qualification gate.

* **[TBD] Downstream representation and generative utility**
  * Test how `context4k` should be consumed without truncating it into the legacy 512-token T5 path, then test controlled downstream usefulness.

## Concluded

* None yet.
