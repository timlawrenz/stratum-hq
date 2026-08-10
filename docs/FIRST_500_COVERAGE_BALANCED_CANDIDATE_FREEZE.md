# First-500 Coverage-Balanced Candidate Freeze

**Date:** 2026-08-04
**Arm / parent:** #4 / #2
**Status:** `PENDING_PRE_COMPUTE_NON_EXECUTING`
**Stage-B hold:** [#18](https://github.com/timlawrenz/stratum-hq/issues/18)

## Purpose and boundary

This record freezes the deterministic 24-item candidate set defined by
[`FIRST_500_CORE_COHORT_PILOT_DESIGN.md`](FIRST_500_CORE_COHORT_PILOT_DESIGN.md).
It is the smallest next provenance artifact after the first-500 coverage audit
and the independent review of the `caption_max_tokens` control repair.

It is **not** a replacement, reinterpretation, or extension of the immutable
Stage-A 24-item global ordinal manifest. The immutable Stage-A record set stays
exactly at:

```text
/mnt/nas-ai-models/research/stratum/stage-a-caption-context-parity/
```

This freeze does not authorize Stage-B execution, model invocation/download,
GPU or scheduler use, corpus mutation, derived-tree mutation, backfill,
legacy-artifact replacement, merge, or an empirical PASS/FAIL claim.

## Frozen noncanonical artifact

```text
/mnt/nas-ai-models/research/stratum/
  first-500-coverage-balanced-candidate-manifest-v1.json
```

| Identity | Value |
|---|---|
| File SHA-256 | `8684c6e38c90b12898135235164677d780a4c897122f26a4b386f07283a9c5e0` |
| Manifest fingerprint | `b18843c759a8b93165a1261350ac46feea7cc62df787d44d4beb0ef9bc4b132d` |
| Freeze implementation | `research_harness.coverage_freeze` / `cb8aec408c9ab4bbbc024c231c661cff6f7ad885028112a8242dda01abce4849` |
| Bound first-500 membership digest | `4e9f8ca775a6e62e308afcccb1e36cce2a5d0bf1f5579631c4a76af0bc80f57c` |
| Bound hidden item-detail digest | `f7edebb10b42d002180f1641605babd66b2e3c159e343630ef2b769b47ea50e0` |

The manifest fingerprints its canonical JSON serialization excluding only its
asserted fingerprint field. It records normalized source-relative paths,
source SHA-256 values, dimensions, formats, selection ranks, and existing
artifact availability/readability facts for the selected items.

## Deterministic selection and source-read scope

1. Re-audit the first 500 eligible canonical filenames by bytewise POSIX order.
   The observed membership and hidden-detail digests must equal the values above
   before any source image bytes are read.
2. Form the primary pool from rows with all five readable core artifacts and
   exactly one `pose2` detection.
3. Hold detector-disagreement rows out as quality/anomaly abstention rows. They
   are never prompt, caption, or representation content.
4. Within each framing-proxy quota, rank normalized source-relative paths by
   SHA-256 of UTF-8 `stratum-first500-coverage-design-v1`, followed by one NUL
   separator and the source-relative path. Take the lowest ranks.
5. Only after that selection, read each selected source once for the same
   byte-stream SHA-256 and local image dimensions/format.

The re-audit reproduced the first-500 binding and yielded a primary pool of
**478** rows plus **22** detector-anomaly holdouts. The frozen manifest has
**12 portrait, 6 squareish, and 6 landscape** candidates, with exactly **24**
source-byte reads. No excluded source image was opened by the freeze.

## Existing-artifact feasibility on the frozen subset

| Fact | Result | Interpretation |
|---|---:|---|
| Readable core `pose2`, `seg2`, `normal2`, `pointmap`, `matting` | 24 / 24 | Eliminates core-artifact availability as a selected-cohort confound. |
| Readable legacy `caption.txt`, `t5_hidden.npy`, `t5_mask.npy` | 24 / 24 | Existing legacy artifacts remain a baseline availability fact, not regenerated parity evidence. |
| Complete existing `determinations` → `caption2` → `t52` chain | **0 / 24** | The existing partial later chain cannot supply an evidence-only condition for this coverage-balanced cohort. |
| `t52` as `context4k` | invalid | T52 remains a 512-token legacy-compatible encoder, not the required provenance-bearing 4K representation. |

The zero later-chain count is a scope result, not a failed experiment. A future
evidence-only contrast can use deterministic geometry only if a separately
approved Stage-B plan permits nonpersistent or noncanonical computation from
the selected existing core artifacts. It may not backfill or mutate
`crawlr/stratum`.

## Remaining hold and exact next decision

Issue #18 is a global program hold on further autonomous selection or execution.
The owner must approve, amend, or decline a bounded request tied to the manifest
fingerprint above. It must name:

1. the exact already-installed local aggregator and immutable model/code/config
   provenance;
2. fixed generation settings, including `caption_max_tokens`;
3. the claim-support rubric, known-case/null self-audit, reviewer sequence, and
   adversarial review; and
4. separate authorization or denial for deterministic evidence computation,
   model invocation, and any GPU scheduler lifecycle action.

Until then, the active arm remains `PENDING` / metric-risk and no model or
empirical comparison may run.
