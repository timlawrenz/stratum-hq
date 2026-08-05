# Research Contract — Open-World Specialist Context for Stratum

## Objective

Determine which observable, useful, non-redundant evidence dimensions make a curated image of exactly one woman useful as conditioning for generative models, and how multiple specialists plus contextual aggregators can form a faithful compact representation.

The project has two deliverables:

1. A reusable, project-neutral scaffold for autonomous research governed through GitHub issues, evidence, tests, draft PRs, and a GPU scheduler.
2. A successful or honestly concluded Stratum research program over `crawlr/approved`.

A positive result is desirable, but scientific success is an auditable **GO, PIVOT, PARK, or KILL** decision.

## Corpus contract

- **Canonical source:** `/mnt/nas-ai-models/training-data/crawlr/approved`.
- **Derived artifact tree:** `/mnt/nas-ai-models/training-data/crawlr/stratum`; currently partial and heterogeneous.
- **Subject invariant:** every source image contains exactly one curated woman.
- A detector producing zero or multiple people is a model/data-quality anomaly. It is not a semantic statement for captions or the representation.
- The corpus can include swimwear or nudity. Sensitive-image inference runs on owned hardware by default (local-first execution). Model SOURCING is open-world: the loop may discover, download, install, and qualify new candidate models (open weights, fine-tunes, deterministic or learned specialists) from papers/literature and hubs when local options are exhausted or a materially better / new-part model exists. Hosted third-party inference of the sensitive canonical corpus requires a hold and reviewed qualification. A silent refusal, vague avoidance, or unexplained omission is a data-quality failure.

## Evidence architecture

```text
image + selected views/crops
  + deterministic and learned specialist evidence
  → expanded provenance-bearing dossier (target: ~100K tokens)
  → contextual compression (target: ~4K tokens)
  → downstream-specific representations
```

The 4K context is not a longer legacy caption. It is represented by:

- `context4k.json` — structured claims with evidence IDs;
- `context4k.md` — stable human-readable serialization;
- `compression.json` — token accounting, source evidence, model/prompt provenance, and unresolved conflicts.

The current T5 contract is 512 tokens. A long context must not be silently truncated into `t5_*`/`t52_*`; a new downstream consumption mechanism needs its own research arm.

## Specialists

A specialist may be deterministic geometry, an open model, a fine-tune, a VLM, a learned embedding, an ensemble, or an approach discovered later. The project deliberately does not limit itself to an initial model list or fixed taxonomy.

Every specialist must declare:

- scope and observable domain;
- inputs and view/crop policy;
- output semantics;
- model/code/config provenance;
- confidence or abstention behavior;
- known failure modes;
- a pre-registered qualification gate.

A specialist’s output is evidence, not automatically a fact. Aggregators preserve provenance and conflicts rather than inventing consensus.

## Research tree behavior

The agent is strategist and worker. On each round it surveys all open research issues and relevant closed results, identifies signal/stalls/dead ends, selects one high-information next action, and records why it chose that branch rather than alternatives.

- Never process issues FIFO merely because they are older.
- Prefer depth on valid positive evidence.
- Preserve negative results with post-mortems.
- One active arm only.
- Three valid comparable non-improvements require a post-mortem before the arm can continue.

## Evaluation discipline

Before declaring a result positive:

1. Pre-register a falsifiable hypothesis and gate.
2. Verify the evaluator/metric itself, including an appropriate null or baseline.
3. Compare variants on the same items, labels, seeds, preprocessing, and metric version.
4. Separate evidence/prompt/model changes from preprocessing or crop changes.
5. Run an adversarial pass; incomplete checks mean `PENDING`, not `PASS`.

The first current-methodology gate is caption-comparison parity: legacy and new captions must be compared across controlled combinations of image preprocessing, prompt structure, specialist evidence, and aggregator model.

## GPU contract

GPU work uses the existing NAS scheduler only:

```text
request → poll-and-claim → launch → verify → activate → heartbeat → release
```

- 4090: local route.
- Strix: `ssh:max395` route and a 10GB evergreen Crawlr-labeling reservation.
- Scheduler state is authoritative over a transient utilization snapshot.
- No arbitrary shell command may be generated from an issue body.
- A future approved manifest must carry a human-reviewed authorization linked to its GitHub issue, a registered launcher ID, the configured scheduler project, finite duration/VRAM within the declared accelerator capacity, and an approved noncanonical NAS output root.
- A future approved launcher must validate that manifest and verify workload PID, logs, GPU usage, output artifacts, and realistic wall time before completion/release.
- The current GPU supervisor is observer-only. It must not request, claim, launch, heartbeat, release, or kill work.

## Autonomy and hold boundary

The autonomous agent may update documentation, tests, issues, labels, branches, commits, and draft PRs. It may not merge, push `main`, mutate the canonical corpus, launch a backfill, or use GPU resources without a reviewed future arm and explicit authority. It MAY autonomously discover, download, install, and qualify new candidate models (arXiv/literature/hub research included) under the open-world model-sourcing policy, as long as sensitive-corpus inference stays on owned hardware; hosted third-party inference of the sensitive canonical corpus still requires a hold.

Apply `research:hold` and stop if any of these occur:

- metric, provenance, or policy uncertainty;
- an architecture/specialist decision outside a pre-registered arm;
- a data/GPU action lacking authority;
- a newly discovered opportunity that changes the program scope;
- conflict between evidence sources that cannot be resolved under the current contract.

The hold must include a GitHub issue describing the evidence, the missing decision, and the smallest next action.
