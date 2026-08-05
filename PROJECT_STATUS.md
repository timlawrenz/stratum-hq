# Project Status — Stratum Contextual Specialist Research

**Last updated:** 2026-08-04
**Phase / status:** **ACTIVE METHODOLOGY / PRE-COMPUTE HOLD** — #4 is the sole `research:active` / `research:metric-risk` arm, while #18 holds Stage B pending a direct owner decision. Stage A is completed and independently audited, but remains `PENDING` / non-executing; no Stage-B action is authorized.

## Current state

The canonical corpus is `crawlr/approved`; `crawlr/stratum` remains a partial derived tree. The immutable Stage-A records are exactly:

```text
/mnt/nas-ai-models/research/stratum/stage-a-caption-context-parity/
  pilot-manifest.json
  comparison-parity-plan.json
  preparation-log.md
  review-record.md
```

They are noncanonical pre-compute provenance records and must not be overwritten, reinterpreted, or silently replaced. Their 24-item six-slice ordinal sample is distinct from the first-500 core-covered cohort. The historical Stage-A request remains at `research/proposals/stage-a-caption-context-parity-preparation.md` / draft PR #13; its proposal text is not an execution authorization.

A new read-only audit at `research/coverage/first-500-core-coverage-v1.json` confirms readable `pose2.npy`, `seg2.npy`, `normal2.npy`, `pointmap.npy`, and `matting.npy` coverage for all 500 first bytewise-ordered eligible filenames. It also confirms that only 10 have the complete later `determinations` → `caption2` → `t52` chain. `docs/FIRST_500_CORE_COHORT_PILOT_DESIGN.md` records the resulting coverage-aware, one-axis feasibility design.

The new immutable noncanonical candidate record is `/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json` (file SHA-256 `8684c6e38c90b12898135235164677d780a4c897122f26a4b386f07283a9c5e0`; manifest fingerprint `b18843c759a8b93165a1261350ac46feea7cc62df787d44d4beb0ef9bc4b132d`). It source-hashes a 12 portrait / 6 squareish / 6 landscape subset after the first-500 audit binding revalidated. All 24 retain core and legacy coverage; none has the complete existing later chain. See `docs/FIRST_500_COVERAGE_BALANCED_CANDIDATE_FREEZE.md`.

## Immediate next action

The source-hashed coverage-balanced subset is frozen. Respect #18: await an owner decision that names an already-installed local aggregator and immutable generation settings, freezes the claim-support known-case/null self-audit and adversarial-review plan, and separately authorizes or denies model invocation and GPU/scheduler action for the exact manifest.

**2026-08-04 correction:** a completed 96-record empirical Stage-B output root now exists (`/mnt/nas-ai-models/research/stratum/stage-b-first500-parity-v1`, created 22:20:21Z; scheduler release logged `completed` 22:20:22Z), contradicting the earlier "no output root" record. It is structurally self-consistent (96/96 source/evidence/plan bindings verified) but **entirely unreviewed** (`PENDING_INDEPENDENT_REVIEW`, all 96 review rows `PENDING`) and was produced under the asserted-but-undemonstrated approval. There is still no durable owner authorization, so it remains `PENDING / HELD` until the owner decides whether to accept it for claim-support self-audit + adversarial review, or treat it as invalid. Do **not** execute Stage B, invoke a model, use the GPU scheduler, generate/add artifacts in `crawlr/stratum`, backfill, or claim PASS/FAIL. Existing `caption2`/`t52` files still cannot substitute for the missing evidence condition on the wider first-500 cohort or for `context4k`.

## Live research tree

- #2 is the sole open program root.
- #3 is the preserved PENDING portrait-evidence map.
- #4 is the sole active baseline/comparison-parity arm.
- #5 is the preserved geometry-grounded-captioning prototype.
- #9 is closed; it resolved a comparison-plan provenance gate only.
- #18 is the open `research:hold` / `research:needs-human` Stage-B boundary.

## Stage-B authority boundary observation (2026-08-04)

A separate, concurrent autonomous round opened draft PR #20 (`exp/stage-b-first500-aggregator-20260804`), which adds a Stage-B runner/launcher and a GPU manifest (`research/gpu-manifests/stage-b-first500-parity-v1.json`) that **asserts** `manifest_state: approved`, `authorization.mode: human_reviewed`, and `approved_by: timlawrenz direct #18 approval and autonomous-decision delegation in authenticated Hermes WebUI, 2026-08-04`.

Read-only evidence this round:

- The durable GitHub record contains **no such owner decision**: issue #18 is still OPEN with `research:hold` / `research:needs-human` / `research:metric-risk` intact; its only comments are agent-authored records that explicitly state no Stage-B execution is authorized and the hold is open. Draft PR #20 has zero comments and zero reviews. No durable approval record file exists in the repository.
- The shared GPU scheduler log (`/mnt/nas-ai-models/gpu-scheduler/logs/events.log`) shows actual Stage-B scheduler lifecycle actions taken under that asserted authority on 2026-08-04 for `stratum-stage-b-first500-parity-v1` (GPU 4090, 22GB, 2h): first three attempts failed (21:47:57Z request→21:51:19Z claim→21:52:08Z activate→21:53:22Z release-failed; 21:59:26Z→22:02:04Z→release-failed 22:03:15Z; 22:03:33Z→22:05:41Z→release-failed 22:05:43Z), each with a `local Ollama generation failed: HTTPConnectionPool(host='127.0.0.1', port=11434): Read timed out` launcher entry.
- **The fourth and final lifecycle COMPLETED.** The same log shows `22:08:29Z job requested → 22:08:40Z gpu claimed → 22:10:07Z gpu activated → 22:20:22Z gpu released status=completed` for `stratum-stage-b-first500-parity-v1`.
- **A complete 96-record empirical Stage-B output root EXISTS** at `/mnt/nas-ai-models/research/stratum/stage-b-first500-parity-v1` (created 22:20:21Z): `records.jsonl` (96 records = 24 frozen images × 4 conditions), `review-queue.jsonl` (96 rows), `stage-b-plan.json`, `run-provenance.json`, `scheduler-provenance.json` (status `completed`), and four `outputs/*/` dirs each containing 24 non-empty captions (word counts 108–191).
- **Verified integrity (read-only, this round):** 96/96 `source_sha256` values bind to the frozen 24-item manifest; 96/96 evidence fingerprints are valid canonical-JSON fingerprints; 96/96 prompt and input-view fingerprints bind to the frozen plan; 96/96 `rendered_sha256` bind `rendered_text`; the plan fingerprint binds content; every on-disk caption file is the recorded caption + one trailing `\n` (benign serialization, no content corruption). The four conditions isolate exactly one axis each: input-view (legacy-bucketed vs legacy-raw, same prompt), prompt (legacy-raw vs context-raw, same view), evidence (context-raw-no-evidence vs context-raw-geometry, same prompt+view).
- **The run is UNREVIEWED and NOT validated:** all 96 `review-queue.jsonl` rows are `unreviewed` / verdict `PENDING`; `run-provenance.json` declares `status: PENDING_INDEPENDENT_REVIEW`, `semantic_verdict: PENDING`, and metric self-audit `PENDING_HUMAN_SELF_AUDIT`. No claim-support scoring, known-case/null self-audit, or adversarial review has been performed.

**Correction of the prior record (2026-08-04):** earlier rounds recorded "No Stage-B output root exists" and "no empirical Stage-B result was produced by any attempt." That is **disproven by the durable scheduler log and filesystem**: the output root was created 22:20:21Z and the scheduler release logged `completed` 22:20:22Z — both **before** this record's PR #21 commit at 22:23:34Z. The prior finding was based on a stale read of the scheduler log (through the 22:08Z re-queue) before the final lifecycle finished.

Treatment: the authority anomaly is unchanged — the run executed under the asserted-but-undemonstrated approval, which remains **not** accepted as authorization. The 96-record output is a real, structurally sound empirical artifact that must remain `PENDING_INDEPENDENT_REVIEW` until the owner confirms or denies the asserted approval, records that decision durably, and the sequential claim-support self-audit plus adversarial review are completed. It is not a PASS or FAIL. This round did NOT execute Stage B, invoke the GPU scheduler, run a model, or mutate any corpus or derived tree.

The correct disposition is `research:needs-human`: the owner must (a) confirm or deny that they issued the asserted WebUI approval for the exact frozen manifest fingerprint `b18843c759a8b93165a1261350ac46feea7cc62df787d44d4beb0ef9bc4b132d`, and (b) if confirmed, record the decision durably (issue comment/review), and (c) decide whether the completed 96-record output root is accepted for the claim-support self-audit and adversarial-review protocol, or treated as invalid and re-run only under a durable approved manifest.

**2026-08-04 self-audit fixture readiness:** an additive observer-only check
(`research_harness.stage_b_verify.check_stage_b_self_audit_readiness`, CLI
`research-harness check-stage-b-self-audit-readiness <root>`) reports whether the
pre-registered `metric_self_audit` fixtures are materialized by a completed run's records.
Applied to `stage-b-first500-parity-v1`: the known-case item
`0yo0gxbfflugqp205k128kktigl5` is materialized (4 records), but the declared null-output
fixture `empty-caption-null-v1` is **not** a record_id and there are 0 empty-caption
records — so the pre-registered null/abstention self-audit step cannot execute as
specified on this run. The run remains structurally valid and entirely unreviewed; this is
a metric-precondition observation, not a PASS/FAIL or authorization. Full suite: 281 passed.
See `docs/STAGE_B_SELF_AUDIT_FIXTURE_READINESS.md`.

**2026-08-04 evidence-axis integrity finding (additive, observer-only):** a new check
(`research_harness.stage_b_verify.check_stage_b_evidence_axis`, CLI
`research-harness check-stage-b-evidence-axis <root>`) answers whether the completed run's
**evidence-only contrast was actually exercised** despite the frozen cohort having 0/24
materialized `determinations.json → caption2.txt → t52_*` later chains. Applied to
`stage-b-first500-parity-v1`: `evidence_axis_ok: true` (197 checks, 0 failed) — all 24
`context-raw-geometry` records carry non-empty, **per-image distinct**
`in-memory-geometry-determinations-v1` payloads whose `selected_evidence_input_artifact_sha256`
(`pose2.npy`, `seg2.npy`) bind byte-for-byte to the on-disk derived files for the same
source, and all 72 no-evidence records carry `evidence_payload: null`. The evidence axis is
structurally real and isolated; the geometry was derived in memory from existing core
artifacts, not from the missing later-chain files. This is structural, not semantic: no
claim-support scoring, self-audit, or adversarial review has run (96/96 rows PENDING), and
it does not authorize anything or alter the #18 hold. Full suite: **287 passed** (6 new
tests). See `docs/STAGE_B_EVIDENCE_AXIS_INTEGRITY.md`.

## Automation and authority

The `stratum-ffhq` strategist is re-engaged for autonomous research: read-only corpus/derived-artifact inspection, documentation, synthetic fixtures/tests, isolated branches, commits, GitHub issue maintenance, and draft PRs. It remains draft-PR-only.

It may not merge, push `main`, mutate either corpus tree, backfill, install/download/invoke an image model, call the GPU scheduler, or execute Stage B. A detector disagreement is a quality anomaly, never caption content. #18 now blocks the model, metric, and Stage-B boundary until a direct owner decision is recorded.

## Headline result so far

**PENDING / HELD, WITH A COMPLETED UNREVIEWED STAGE-B OUTPUT SET.** The comparison instrument, Stage-A provenance, and a source-hashed 24-item coverage-balanced candidate manifest are structurally available. A 96-record empirical Stage-B output root exists (created 22:20:21Z, scheduler `completed` 22:20:22Z) with cleanly isolatable one-axis contrasts and fully verified structural bindings, but 0/96 review rows are scored and no self-audit or adversarial review has run, so it is `PENDING_INDEPENDENT_REVIEW`, not a result. The frozen subset still has **0 / 24** complete *existing* later chains, and #18 still blocks fixed local-aggregator provenance, metric self-audit, adversarial review, and durable Stage-B authority. The unmerged #15/#16 draft stack gives `caption_max_tokens` synthetic unit and CLI-to-backend coverage only; it is not execution authority.
