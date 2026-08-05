# First-500 Core-Covered Cohort — Coverage-Aware Pilot Design

**Date:** 2026-08-04
**Arm / parent:** #4 / #2
**Status:** `PENDING / PRE-COMPUTE / NON-EXECUTING`

## Purpose and provenance boundary

This is a feasibility and selection-design artifact for the active comparison-parity arm. It uses the **first 500 eligible canonical filenames in bytewise POSIX order** because their existing core Stratum2 coverage reduces a missing-artifact confound.

It is **not** a replacement, interpretation, or extension of the immutable Stage-A 24-item ordinal-slice manifest. The Stage-A selection remains exactly the 24 items in:

```text
/mnt/nas-ai-models/research/stratum/stage-a-caption-context-parity/
```

Its published records remain untouched. The Stage-A manifest's six global ordinal slices are not the first-500 cohort.

## Read-only audit evidence

[`research/coverage/first-500-core-coverage-v1.json`](../research/coverage/first-500-core-coverage-v1.json) is a compact, reproducible audit record.

- Audit-record SHA-256: `4135606940fac7d014df2accf1e84e0046f47ca98e8dafedfd79d8d7d9ec13a4`.
- Audit implementation: `research_harness.core_coverage`, SHA-256 `2289e1cbb9d3fd11ee4b0bee70dcbe771bbcf34e6732627aaf6ebed0a71b899a`.
- Eligible canonical filenames: **11,825**.
- Cohort: first **500** eligible flat filenames, bytewise POSIX order.
- Cohort membership SHA-256: `4e9f8ca775a6e62e308afcccb1e36cce2a5d0bf1f5579631c4a76af0bc80f57c`.
- Omitted per-item probe-detail SHA-256: `f7edebb10b42d002180f1641605babd66b2e3c159e343630ef2b769b47ea50e0`.
- Source-content reads: **0**. The audit listed and resolved names only; it did not open, decode, hash, or dimension-read a canonical image.
- Core probes were restricted to the matching derived-item directory. NPY readability means header parsing plus bounded endpoint access; it is **not** full-tensor integrity or semantic validation.

### Availability summary

| Existing artifact or status | First-500 result |
|---|---:|
| `pose2.npy` readable | 500 / 500 |
| `seg2.npy` readable | 500 / 500 |
| `normal2.npy` readable | 500 / 500 |
| `pointmap.npy` readable | 500 / 500 |
| `matting.npy` readable | 500 / 500 |
| All five core artifacts readable | **500 / 500** |
| Legacy `caption.txt` + `t5_hidden.npy` + `t5_mask.npy` readable | 500 / 500 |
| `determinations.json` + `caption2.txt` + `t52_hidden.npy` + `t52_mask.npy` readable | **10 / 500** |

The first-500 core cohort therefore eliminates missing-core-artifact availability as a confound. It does **not** make the later determinations/caption chain complete.

### Nonsemantic coverage proxies

The existing `matting.npy` shapes give a source-free framing proxy:

| Aspect proxy | Count |
|---|---:|
| Portrait (`width / height < 0.9`) | 437 |
| Squareish (`0.9 ≤ width / height ≤ 1.1`) | 23 |
| Landscape (`width / height > 1.1`) | 40 |

`pose2.npy` reports one detected pose for 478 items and more than one for 22 (15 with two, 4 with three, 2 with four, 1 with five). Those 22 are **quality-anomaly / abstention-status** rows only. The corpus invariant remains exactly one curated woman; detector output is never prompt, caption, or representation content.

The 10 complete later-chain items are all one-pose quality-status rows, but cover only 8 portrait and 2 landscape items; there are no squareish items. They are useful for chain-readability inspection, not a coverage-aware 24-item evidence comparison.

## Can current artifacts support the one-axis matrix?

| Controlled contrast | Current feasibility | Why / remaining gate |
|---|---|---|
| **Input view only** — bucketed versus raw, fixed legacy prompt/evidence/model | **Designable on 478 primary items; not executable** | The canonical sources and fixed core cohort exist, but a valid comparison must regenerate both views under one fixed, already-installed local aggregator and settings. Existing `caption.txt` outputs alone do not prove generation parity. |
| **Prompt only** — legacy versus context prompt, fixed raw view/evidence/model | **Designable on the same 478; not executable** | No new derived artifact is needed to describe the contrast, but it still requires the fixed local aggregator, rubric self-audit, and separate Stage-B authority. |
| **Evidence only using already materialized determinations** | **Only 10 / 500** | The 490 missing `determinations.json` files make this insufficient for the 24-item coverage-aware design. Existing `caption2.txt` and `t52_*` are outputs of an earlier partial chain, not a controlled replacement for this condition. |
| **Evidence only by computing determinations from existing core artifacts at run time** | **Potentially 478; not currently authorized** | The candidate geometry declaration needs `pose2.npy` and `seg2.npy` (with optional pointmap fields), all available in the primary subset. But this is a new deterministic computation, not “existing artifacts only”; it requires an explicit Stage-B authorization and must write no artifact into `crawlr/stratum`. |
| **Use `t52_*` as the compact-context path** | **Not valid** | Only 10 exist, and T52 remains a 512-token legacy encoder. It cannot silently stand in for the required 4K provenance-bearing `context4k` representation. |

Thus, the cohort supports a future controlled **view** and **prompt** comparison design immediately, and it supports an **evidence** comparison only after separately authorized, nonpersistent-or-noncanonical determinations preparation. It does not support a present empirical claim.

## Coverage-aware future candidate rule

For a future separately authorized Stage-B request, use the following *design rule*, not a frozen replacement manifest:

1. Re-run the read-only audit and require the same first-500 membership digest above, 500/500 readable core coverage, and no silent change to the report's detail digest.
2. Define the primary geometry comparison pool as the 478 rows with complete core coverage and exactly one `pose2` detection. Hold the 22 detector-anomaly rows out of caption conditions; retain them only as quality-review/abstention cases.
3. Select at most 24 primary candidates with fixed aspect quotas: **12 portrait, 6 squareish, 6 landscape**. This deliberately over-samples non-portrait framing to test view handling; it is not a semantic or population-representative claim.
4. Within each aspect quota, rank the eligible normalized `source_relative_path` values by SHA-256 of UTF-8 `stratum-first500-coverage-design-v1\0<source_relative_path>` and take the lowest digests. Do not inspect image content to break ties.
5. At a later authorized freeze, source-hash only those selected candidates, record any attrition, and never refill from outside the first 500 or silently substitute an immutable Stage-A item.

The rule is falsifiable before execution: if a future audit cannot supply all three quotas from the fixed primary pool, stop and record the shortfall rather than widening the cohort or changing selection semantics.

## Missing determinations/caption-chain prerequisites

The current partial chain is the immediate evidence limitation:

- **490 / 500** lack each of `determinations.json`, `caption2.txt`, `t52_hidden.npy`, and `t52_mask.npy`.
- No backfill, derived-tree mutation, or additive artifact creation is authorized by this report.
- Draft PR #15 repairs/tests the preserved prototype's `caption2` `caption_max_tokens` forwarding and removes detector-anomaly prompt content; independent review is still a pre-inference prerequisite.
- A Stage-B request must replace `stage-b-local-aggregator-pending-v1` with an already-installed local model identity and immutable generation fingerprint.
- The reviewer rubric remains required: supported claims, unsupported claims, omissions, contradictions, abstentions, known-case/null self-audit, and adversarial review.

## Active-arm selection record

**Parent:** #2.
**Full-tree survey:** #2 is the sole open program root; #3 remains the preserved PENDING evidence map; #4 remains the sole `research:active` / `research:metric-risk` arm; #5 remains the preserved prototype; #9 is the closed provenance-hold post-mortem. Open draft PRs include #1 and the accepted governance / Stage-A stack through #13.
**Selected action:** Read-only first-500 core-coverage audit plus this coverage-aware selection design.
**Why it beats alternatives:** The first-500 cohort yields an immediate, falsifiable answer about whether the declared axes can be controlled without a backfill. It exposes the exact 490-item evidence-chain gap while avoiding model execution, GPU scheduling, corpus mutation, and further authorization-template churn.

## Exact next boundary

No Stage-B action is authorized. After the prototype forwarding repair is independently reviewed, the smallest new decision is a **bounded Stage-B request** tied to a frozen, source-hashed selection from this design. It must individually authorize: (1) the named already-installed local aggregator and fixed settings; (2) deterministic geometry evidence computation only for the frozen items, with outputs restricted to a named noncanonical research root or held in memory; and (3) the pre-registered one-axis review protocol. It must continue to deny model download, scheduler use unless separately approved, any mutation/backfill of `crawlr/stratum`, legacy-artifact overwrite, and any empirical PASS claim absent metric self-audit and adversarial review.

No model, inference, GPU/scheduler operation, image-content read, corpus mutation, backfill, or Stage-B comparison was performed for this artifact.
