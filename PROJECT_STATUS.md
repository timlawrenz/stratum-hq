# Project Status — Stratum Contextual Specialist Research

**Last updated:** 2026-08-04
**Phase / status:** **METHODOLOGY PLANNING** — arm #4 is the sole `research:active` arm and remains `research:metric-risk`; provenance hold #9 is resolved. Work is draft-PR-only, pre-compute only, and the GPU supervisor remains observer-only.

## Current state

The canonical research corpus is `crawlr/approved` (11,825 source images); `crawlr/stratum` remains a partial derived-artifact tree. An early geometry-grounded captioning prototype is preserved as draft PR #1 on `exp/geometry-grounded-captioning`.

The program studies open-world specialist evidence and contextual aggregation. Its target architecture expands an image into a provenance-bearing dossier (exactly 100K tokens in the Stratum policy profile) and compresses it into a first-class 4K context representation. The existing 512-token T5 artifacts are not assumed to be the long-context consumer.

The live GitHub tree is strict and non-FIFO:

- #2 is the open program root;
- #3 preserves the `PENDING` portrait-evidence discovery map in draft PR #7;
- #4 is the sole `research:active` baseline/comparison-parity methodology arm, now unblocked only for pre-compute pilot-authorization planning;
- #5 records the preserved Arm 0 prototype; and
- #9 is closed: its comparison-plan provenance/harness gap was resolved without an empirical run.

## Accepted draft governance baseline

The repository owner directly reviewed and accepted draft PRs #6, #7, #8, #10, and #11 because GitHub cannot record a formal self-approval on owner-authored pull requests. The direct owner-decision comments are the durable review record.

The accepted stack remains unmerged and draft-only:

```text
#6 harness → #8 comparison-plan guard → #10 inline declarations → #11 strict envelopes/fingerprints
```

The current accepted top is draft PR #11, commit `a7cecb89f55eef9375137e7e70dafccac7427f41`. Acceptance does **not** authorize a merge, a `main` push, empirical inference, model download, GPU scheduling, corpus mutation, or a backfill.

## Immediate next action

Prepare a draft-only **Stage A preparation-authorization proposal** from `research/templates/pilot-authorization-proposal.md` for arm #4. It may name the canonical root, selection protocol, maximum item count, preparation output root, and future evaluation design **without selected item identities or source hashes**. Stage A must request only bounded selection/read/hash and manifest-materialization authority.

Do not select or hash pilot images before direct Stage A approval. Stage A must explicitly deny model invocation, GPU scheduling, additive artifact generation, corpus mutation, and backfill. After Stage A, freeze and validate the exact manifest and comparison plan; only a fresh Stage B owner approval tied to those immutable identities may individually authorize any model, GPU, data, or additive-artifact execution.

## Automation state

The `stratum-ffhq` strategist is paused. It must not draft a Stage-A proposal, select candidates, read or hash source images, create a manifest, invoke a model, or operate the scheduler until a separately approved resumption handoff is applied.

- strategist: paused; any future resumption is limited to Stage-A preparation-authorization **planning**, not Stage-A execution;
- GPU observer: enabled, no-agent, and observer-only;
- cadence metrics collector: enabled, recommendation-only, and never edits a schedule.

No automation may infer execution authority from the resolved provenance hold. It must hold again if pilot selection, metric definition, model, GPU, data, or architecture authority is unclear.

## Headline result so far

**PENDING.** The geometry-grounded prototype has synthetic-fixture tests but no controlled caption-quality or downstream-conditioning verdict. The comparison provenance gate is resolved; a real pilot and evaluation remain unexecuted.

## Authority boundary

Draft PRs, documentation, tests, issue-tree state, and a pilot-authorization proposal are authorized. GPU claims/launches, model installation or download, canonical-corpus reads beyond explicitly approved pilot preparation, derived-tree mutation, full backfills, merges, and direct `main` pushes require separately explicit authority.