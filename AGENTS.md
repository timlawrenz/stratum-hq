# Stratum Research Agent Entry Point

## Mandatory reading order

1. `PROJECT_STATUS.md` — current phase, active hold, and one immediate action.
2. `RESEARCH_CONTRACT.md` — autonomy and evidence rules.
3. `research/program.json` — machine-readable program constraints.
4. GitHub open research issues, then relevant closed/post-mortem issues.
5. `docs/EXPERIMENT_TREE.md` and `docs/EXPERIMENTS_AND_RESULTS.md`.
6. Relevant source code and tests before proposing a code or experiment change.

## Non-negotiable rules

- The canonical active source corpus is `/mnt/nas-ai-models/training-data/crawlr/approved`.
- Every source image is curated to contain exactly one woman. Detector disagreement is a quality anomaly, not caption content.
- Preserve Stratum1 compatibility. New research artifacts must be additive; never overwrite `caption.txt`, `t5_*`, `pose.npy`, or another legacy artifact.
- Treat `crawlr/stratum` as a partial derived tree. Do not launch a backfill or mutate it without an approved research arm and GPU/data authority.
- Specialists are open-world candidates, not a fixed model roster. Each needs scope, provenance, abstention behavior, known failure modes, and a qualification gate.
- A compact context representation must preserve evidence links claim-by-claim and must not be silently truncated into the current 512-token T5 path. The Stratum profile requires a 100K-token dossier and 4K-token compact context, not merely ordered budgets.
- GitHub issues are the research tree and routing state, not a FIFO queue. Survey the whole open tree before selecting an action, record the selected arm's parent, full-tree survey, and selection rationale, and maintain exactly one open program root.
- Exactly one `research:active` arm is allowed unless the program is held.
- After three valid, comparable non-improving experiments on an arm, write a post-mortem and close/downgrade it. Infrastructure failures and invalid metrics do not count as strikes.
- A PASS requires a metric self-audit, a controlled comparison, and adversarial review. Otherwise write `PENDING`.
- GPU work must use `/mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py`. Its `poll` operation performs the atomic claim; never call a separate claim after a successful poll, bypass the scheduler, kill another job, or infer availability from a transient GPU snapshot.
- The 4090 is local; Strix jobs must use `ssh:max395`. The Strix has a 10GB evergreen Crawlr labeling reservation.
- Keep sensitive-image inference on owned hardware by default (local-first execution). Model sourcing is open-world: new candidate models (open weights, fine-tunes, deterministic or learned specialists) may be discovered, downloaded, installed, and qualified when local options are exhausted or a better/new-part model exists, including literature/arXiv research. Hosted third-party inference of the sensitive canonical corpus still requires a hold and explicit review.
- Work is PR-only: create draft PRs, never merge or directly push `main`.

## Required hold behavior

Apply `research:hold` and create/update a `research:needs-human` or `research:harness-gap` issue when a metric is untrustworthy, a new policy/architecture decision is required, a GPU/data action lacks authority, or the program contract is insufficient.

## Validation commands

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m research_harness.cli validate-program research/program.json
```

## Current experimental prototype

`exp/geometry-grounded-captioning` / draft PR #1 preserves the early Sapiens2-determinations → caption2 → t52 prototype. It is not production-ready and must be evaluated with preprocessing/prompt/evidence parity before claims are made.
