# Project Status — Stratum Contextual Specialist Research

**Last updated:** 2026-08-03
**Phase / status:** HARNESS READY FOR DRAFT-PR REVIEW — draft-PR-only, GPU supervisor observer-only

## Current state

The canonical research corpus is `crawlr/approved` (11,825 source images); `crawlr/stratum` is a partial derived-artifact tree. An early geometry-grounded captioning prototype is preserved as draft PR #1 on `exp/geometry-grounded-captioning`.

The program studies open-world specialist evidence and contextual aggregation. Its target architecture expands an image into a provenance-bearing dossier (exactly 100K tokens in the Stratum policy profile) and compresses it into a first-class 4K context representation. The existing 512-token T5 artifacts are not assumed to be the long-context consumer.

The live GitHub tree is strict and non-FIFO:

- #2 is the open program root;
- #3, portrait-evidence discovery, is the sole `research:active` arm and records its full-tree survey/selection rationale;
- #4 is the pending baseline/comparison-parity methodology arm;
- #5 records the preserved Arm 0 prototype.

## Active automation

Tim authorized activation of both `stratum-ffhq` records on 2026-08-04:

- `1c25ada8ed0b` — strategist, `every 60m`, pinned to `openrouter` / `openai/gpt-5.6-terra`, restricted to `web`, `terminal`, `file`, and read-only skill access.
- `ae13cfe18a81` — no-agent GPU-manifest observer, `every 5m`, running only `stratum_gpu_observer.sh`.

The strategist runs from the clean isolated worktree `/home/tim/source/activity/stratum-hq-research-agent` on `research/autonomous-workspace`. Until draft PR #6 merges, downstream experiment/documentation PRs must target `feat/autonomous-research-harness`, not `main`, so the harness diff is not duplicated.

The observer emits `[SILENT]` without an approved manifest and can only emit a hold; it cannot operate the scheduler or launch a workload. Starting either record does **not** authorize GPU request/claim/launch, model download, canonical-data mutation, backfill, merge, or direct `main` push; the research contract’s hold conditions remain in force.

## Immediate next action

Active arm #3 should survey the research tree and produce the portrait-evidence discovery map or a documented hold. It may use only the contract-authorized documentation, code-reading, artifact-inventory, issue-tree, test, branch, commit, and draft-PR work. No model install, GPU action, canonical-corpus write, or backfill is authorized.

## Headline result so far

**PENDING.** The geometry-grounded prototype has synthetic-fixture tests but has no controlled caption-quality or downstream-conditioning verdict.

## Authority boundary

Draft PRs, tests, documents, issue-tree state, and harness code are authorized. GPU claims/launches, model installation, canonical-corpus writes, full backfills, merges, and direct `main` pushes require a future explicit arm decision.
