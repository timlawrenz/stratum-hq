# Stage-A Preparation Authorization Request — Caption/Context Parity Pilot

**Arm:** #4 — baseline and comparison parity
**Parent program:** #2 — open-world specialist evidence to contextual representation
**Status:** `DRAFT / STAGE A REQUEST / NO EXECUTION AUTHORITY`
**Governance baseline:** Draft PR #12, stacked on accepted PR #11; all remain unmerged draft work.
**Prepared date:** 2026-08-04

## Decision requested

Approve or decline **Stage A preparation only**. This request does not select an image now, enumerate a directory, read a source file, inspect an artifact, calculate a hash, invoke a model, create a GPU manifest, or operate a scheduler.

If approved exactly as written, Stage A may materialize the frozen pilot manifest needed to decide whether a later controlled caption/context comparison is interpretable. It does **not** authorize that comparison.

## Research question and falsification

**Hypothesis:** Holding source items, input view, prompt form, local aggregator identity, and generation settings fixed, an explicitly declared deterministic evidence condition can improve claim-supported contextual coverage over a no-specialist baseline without increasing unsupported or contradictory claims.

**Falsified if:** After a later, separately approved Stage-B controlled comparison, the evidence condition does not improve the pre-registered support/omission/contradiction rubric over the matched no-specialist baseline, or any apparent difference is attributable to a remaining preprocessing, prompt, model, or evaluation confound.

**Decision gate:** This Stage-A request is `GO` only if it yields an auditable, bounded manifest and a filled comparison plan that validates. A future Stage-B outcome is `GO`, `PIVOT`, `PARK`, or `KILL` only after metric self-audit and adversarial review. No `PASS` claim is requested here.

## Why this is the next research move

The active bottleneck is not a richer semantic taxonomy. It is an interpretable comparison: the preserved prototype currently confounds source view, prompt structure, deterministic evidence, and generation behavior. This preparation pass retires only the missing provenance/selection uncertainty necessary to construct that comparison.

It does not claim to validate the whole `~100K dossier → ~4K context → downstream utility` program. `context4k` production or consumption is out of scope for this pilot.

## Stage-A preparation design

- **Canonical root:** `/mnt/nas-ai-models/training-data/crawlr/approved` — read-only.
- **Preparation output root:** `/mnt/nas-ai-models/research/stratum/stage-a-caption-context-parity/`.
- **Maximum item count:** maximum 24 candidates. This is a feasibility/provenance pilot only; it is not powered for a final program-level conclusion.
- **Existing-derived-artifact inspection root:** `/mnt/nas-ai-models/training-data/crawlr/stratum` — read-only availability/readability facts only for the selected candidates.
- **Required Stage-A records:**
  - `pilot-manifest.json` — selected canonical relative paths, source SHA-256 values, source dimensions, declared strata, selection rationale, and derived-artifact availability/readability facts;
  - `preparation-log.md` — exact commands, timestamps, code/branch identity, excluded-candidate reasons, and limitations;
  - `review-record.md` — reviewer confirmation that the manifest is bounded, provenance-complete, and does not claim execution;
  - `comparison-parity-plan.json` — filled but still non-executing draft for `research-harness validate-comparison-plan` after all immutable IDs/fingerprints are available.

### Selection protocol after approval

Selection happens only after Stage-A approval and begins with an approved read-only canonical directory listing. Directory entry names are used only to establish a deterministic order; no entry is opened or decoded before selection.

1. Apply the project’s existing source-file discovery predicate to the directory listing without opening source content. Sort the resulting candidate entry names by bytewise POSIX order.
2. Divide that ordered candidate list into **six equal ordinal slices**. For `N` eligible entries and zero-based slice `j`, use the half-open index range `[floor(j*N/6), floor((j+1)*N/6))` for `j = 0…5`.
3. From each nonempty slice, choose up to four evenly spaced ordinal positions, resolving ties toward the lower index. The resulting union is capped at 24 candidates; no additional refill from unselected entries is allowed.
4. If a slice has fewer than four entries, retain the smaller count and record the gap as a manifest limitation. Do not substitute a semantic proxy, inspect more candidates, or expand the cap.
5. For each selected candidate only, read the source once to record dimensions and SHA-256, then read only enough existing derived artifacts to record `present`, `readable`, or `missing/unreadable` for `pose2.npy`, `seg2.npy`, `determinations.json`, and `caption2.txt`.

No source content, dimensions, or derived artifacts are read for unselected candidates. The six ordinal slices are a deterministic provenance/feasibility sampling device, not semantic strata, representativeness evidence, or claims about people in the corpus. Any observed coverage limitations are recorded after selection and must not be silently converted into a population claim.

This protocol is intentionally a **selection/preparation mechanism**, not a semantic survey or model evaluation. It creates no caption, determination, embedding, segmentation, pose, pointmap, `context4k`, or legacy artifact.

## Stage-A requested authority

The owner may approve only the checked items below. Any unchecked item remains denied.

- [ ] Select no more than 24 canonical-source candidates using the selection protocol above.
- [ ] Read and SHA-256 hash only the selected canonical-source images once each.
- [ ] Read only the selected candidates’ existing derived-artifact availability/readability facts; do not mutate `crawlr/stratum`.
- [ ] Write only the listed manifest, preparation log, review record, and non-executing comparison-plan draft beneath the preparation output root.

## Stage-A non-authorizations

This request explicitly denies model invocation, model download/installation, GPU scheduling, GPU claims, additive artifact generation, corpus mutation, derived-tree mutation, backfill, external image-model use, merge, direct `main` push, and any Stage-B execution.

In particular:

- no inference or caption generation;
- no repair or invocation of `caption2`;
- no `context4k` production or consumption;
- no new specialist qualification claim;
- no empirical result or PASS/FAIL verdict;
- no scheduler request, poll, claim, launch, activate, heartbeat, release, or kill action;
- no modification of `caption.txt`, `t5_*`, `pose.npy`, `pose2.npy`, `seg2.npy`, `determinations.json`, `caption2.txt`, `t52_*`, or any other corpus artifact.

**Stage B execution is not requested or authorized by this document.**

## Required freeze before any Stage-B request

After an approved and completed Stage A, but before any execution request:

1. bind the pilot to exact canonical relative paths, source SHA-256 values, dimensions, coverage tags, availability facts, and immutable manifest identity;
2. fill the comparison plan with matched input-view/prompt/evidence conditions and exact one-axis contrasts;
3. identify an already-installed local aggregator and fixed generation fingerprint;
4. repair and test the prototype backend forwarding of `caption_max_tokens` before any comparison using the prototype path;
5. preserve detector disagreement as a quality anomaly, not caption content;
6. pre-register the supported/unsupported/omission/contradiction/abstention rubric, known-case/null self-audit, and adversarial review;
7. validate the frozen plan with:

```bash
research-harness validate-comparison-plan research/program.json <frozen-plan.json>
```

Only then may a fresh, separately reviewed Stage-B request name an execution scope.

## Owner decision — unfilled

- [ ] **Approve Stage A as written.**
- [ ] **Approve Stage A with changes:** <record exact edits before preparation starts>.
- [ ] **Do not approve Stage A:** <record missing decision>.

**Owner:** <unfilled>
**Date / time:** <unfilled>
**Linked issue / draft PR / output root:** #4 / <unfilled> / `/mnt/nas-ai-models/research/stratum/stage-a-caption-context-parity/`

An unfilled checkbox is not authorization. Until an owner approves Stage A, this document is planning only.

## Pre-execution state

`PENDING` — no candidate has been selected, no source has been read or hashed, no derived artifact has been inspected, no model has been invoked, and no GPU/scheduler action has occurred.

## Continuation stop rule

After this proposal is reviewed, do not add further authorization semantics unless this concrete proposal reveals a new authority, measurement, or safety gap. The next step is either direct Stage-A approval, explicit requested edits, or a deliberate hold—not more policy elaboration.
