# Experiment Tree — Stratum Contextual Specialist Research

This is a living map. GitHub issues are the detailed source of truth; this document provides project orientation rather than a FIFO schedule.

## Live issue tree

* **[ROOT] #2 — Open-world specialist evidence → contextual representation**
  * The program root: canonical corpus, policy, evidence architecture, and linked arms.

* **[PENDING] #3 — Portrait evidence discovery**
  * The owner-approved evidence-discovery map is preserved in draft PR #7.
  * It identifies open-world candidate evidence roles and the raw-versus-bucketed input-view confound without selecting a specialist winner.

* **[ACTIVE / METRIC-RISK / PRE-COMPUTE] #4 — Baseline and comparison parity**
  * The sole `research:active` arm.
  * The provenance guard is resolved, but no comparative inference is authorized yet.
  * Immediate bounded task: owner review of `research/proposals/stage-a-caption-context-parity-preparation.md`, a draft request for a deterministic maximum-24-candidate selection/read/hash preparation pass with no real item identities/hashes in the request. Only a direct Stage-A approval may permit bounded read/hash preparation; freeze/validate the exact plan, then require a fresh Stage-B execution approval before any model, GPU, or artifact action.

* **[PROPOSAL / PENDING] #5 — Geometry-grounded captioning prototype** (`exp/geometry-grounded-captioning`, draft PR #1)
  * Additive chain: `pose2 + seg2 + optional pointmap → determinations → caption2 → t52`.
  * Synthetic fixture coverage exists. No controlled empirical verdict exists.
  * The arm is not production-ready and must not be merged as a result of the governance build.

## Accepted draft governance baseline

* **[OWNER-APPROVED / DRAFT STACK] Autonomous research control plane**
  * Draft PR #6 supplies the control plane; PR #8 adds the comparison-plan guard; PR #10 adds inline declarations; PR #11 closes envelope, identity, and fingerprint gaps.
  * PR #11 at `a7cecb89f55eef9375137e7e70dafccac7427f41` is the accepted top of the unmerged stack.
  * Acceptance is a review decision, not merge authority or empirical authority.
  * GPU supervision remains observer-only until a separately reviewed host-specific launcher and manifest authority exist.

## Future candidate branches

* **[TBD] Open-world specialist qualification**
  * Candidate models, fine-tunes, deterministic measurements, embeddings, and future discoveries must each earn a role through a declared scope, provenance, abstention behavior, known failure modes, and qualification gate.

* **[TBD] Downstream representation and generative utility**
  * Test how `context4k` should be consumed without truncating it into the legacy 512-token T5 path, then test controlled downstream usefulness.

## Concluded

* **[CONCLUDED — HARNESS GATE RESOLVED] #9 — Bind comparison plans to canonical paths and specialist declarations**
  * Owner-approved draft PR #11 remediates canonical pilot paths, closed inline evidence envelopes, required failure modes, canonical comparison/audit identities, and content-bound evidence fingerprints.
  * Fresh program and live-tree validation passed before closure; hosted checks passed and the exact staged index received independent review.
  * This is a governance result only: it does not establish caption quality, invoke a model, or authorize data/GPU work.
