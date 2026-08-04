# Experiments & Results — Stratum Contextual Specialist Research

This ledger records empirical findings and negative results permanently. A green implementation or passing unit test is not an empirical PASS.

## Harness initialization — `[PENDING / OWNER-APPROVED DRAFT]`

**Date:** 2026-08-03 to 2026-08-04

**Goal:** Establish a reusable, project-neutral autonomous-research control plane while grounding it in the Stratum `crawlr/approved` program.

**Evidence:**

- Canonical source inventory: 11,825 flat source images at `/mnt/nas-ai-models/training-data/crawlr/approved`.
- The current derived tree is partial: 4,901 metadata leaves, 4,845 legacy captions, 2,113 `seg2`, 11 `pose2`, 11 `pointmap`, and 10 `caption2`/`determinations`/`t52` examples at the time of inventory.
- The existing `t5_*` and `t52_*` encoders are fixed to 512 tokens, so a target ~4K compact context requires separate artifacts and a new downstream-consumption research arm.
- Existing NAS GPU scheduler: 4090 local route; Strix remote route `ssh:max395` with 10GB evergreen reservation. Its `poll` operation performs the atomic claim; the harness supervisor is observer-only.
- The repository owner directly reviewed and accepted the unmerged draft stack: PR #6 (harness), #7 (evidence map), #8 (comparison-plan guard), #10 (inline declarations), and #11 (strict envelopes and content-bound fingerprints).

**Accepted top:** draft PR #11, commit `a7cecb89f55eef9375137e7e70dafccac7427f41`.

**Verdict:** `PENDING` — the governance stack is accepted as a draft working baseline. It does not authorize GPU work, model invocation, corpus mutation, or an empirical conclusion.

## Comparison-plan provenance hold — `[CONCLUDED — HARNESS GATE RESOLVED]`

**Date:** 2026-08-04

**Blocked arm:** #4 — baseline and comparison parity.
**Hold issue:** #9 (closed after owner acceptance and fresh validation).
**Affected drafts:** PR #8, followed by PR #10 and PR #11.

**Trigger:** An adversarial synthetic audit found that the initial comparison-plan validator accepted absolute and parent-traversal `source_relative_path` values and accepted a non-null evidence bundle containing only an ID and fingerprint. That made a nominally frozen canonical pilot and provenance-bearing evidence condition unauditable before inference.

**Human decision:** Tim approved **inline specialist declarations** for non-null evidence bundles. A real comparison plan must carry the complete open-world declaration for each specialist: stable ID, scope, inputs/view policy, output semantics, provenance, abstention policy, known failure modes, and qualification gate. The explicit no-specialist baseline remains the closed `kind: "none"` envelope with only its kind, ID, and fingerprint.

**Remediation evidence:**

- Strict canonical relative pilot paths reject absolute paths, traversal, redundant segments, backslashes, and whitespace aliases.
- `none`, `specialist_bundle`, and inline specialist declarations are closed envelopes; hidden payload cannot masquerade as baseline evidence.
- Evidence fingerprints bind canonical UTF-8 JSON for the complete evidence object excluding its asserted fingerprint.
- Program-required `known_failure_modes` and canonical comparison/audit identities are enforced.
- The exact PR #11 staged index received independent review with no Critical/High/Medium/Low publish blockers.
- PR #11 hosted `pytest` and GitGuardian checks passed; local and clean-Python-3.11 full suites each passed 229 tests.
- At accepted commit `a7cecb89f55eef9375137e7e70dafccac7427f41`, program validation and a fresh live issue-tree validation passed before #9 closed.

**Verdict:** `CONCLUDED — HARNESS GATE RESOLVED`. No empirical comparison, image inference, GPU action, model installation, corpus mutation, or backfill occurred. The closure only permits arm #4 to prepare a separate Stage-A preparation-authorization proposal; a fresh Stage-B approval is required after the exact pilot manifest and comparison plan are frozen and validated.

## Arm 0 — Geometry-grounded captioning prototype — `[PROPOSAL — PENDING]`

**Branch / PR:** `exp/geometry-grounded-captioning`, draft PR #1.

**Goal:** Test whether Sapiens2-derived structural evidence can help an image-aware local aggregator produce more faithful contextual descriptions than the legacy single-caption path.

**Implementation evidence:**

- Additive chain: `pose2 + seg2 + optional pointmap → determinations.json → caption2.txt → t52_*`.
- Legacy `caption.txt` and `t5_*` remain untouched.
- Synthetic fixtures test geometry, determinations schema, relations, and pass isolation.

**Pre-registered gate:** Not yet valid. A controlled evaluation must first hold constant the source-image preprocessing, prompt structure, model/generation settings, item set, and review rubric.

**Known confound:** The legacy caption path uses a bucketed/cropped image while current `caption2` opens the raw source image. Any apparent output difference currently combines preprocessing, prompt, and evidence changes. `caption_max_tokens` forwarding must also be fixed and tested before a controlled prototype comparison.

**Verdict:** `PENDING` — preserve as a prototype; do not infer quality or downstream usefulness.
