# Pilot Authorization Proposal

**Arm:** #<active-methodology-arm>
**Parent program:** #<program-root>
**Status:** `DRAFT / TWO-STAGE AUTHORIZATION / NO EXECUTION AUTHORITY`
**Prepared from accepted governance baseline:** <approved-commit-and-PR-stack>

> This template separates protected-read preparation from comparative execution.
> A Stage A preparation proposal may be filled without selected item identities or
> source SHA-256 values. No Stage A work begins until the owner approves Stage A.
> No model, GPU, artifact-generation, or comparative execution begins until a
> **fresh Stage B owner decision** references the exact frozen manifest and
> validated comparison plan produced after Stage A.

## Research question and falsification

**Hypothesis:** <falsifiable hypothesis about one controlled evidence, prompt, or input-view contrast>.

**Falsified if:** <pre-registered outcome that would disprove the hypothesis>.

**Decision gate:** <what outcome produces GO, PIVOT, PARK, or KILL; do not call PASS without the required audit and adversarial review>.

## Stage A — preparation authorization

**Purpose:** authorize only the bounded selection/read/hash work needed to
materialize an immutable pilot manifest. It is not inference or comparative
execution.

- **Canonical source root:** `/mnt/nas-ai-models/training-data/crawlr/approved` (read-only).
- **Preparation output root:** <approved subdirectory of `/mnt/nas-ai-models/research`>.
- **Selection protocol and strata:** <coverage of crops, framing, clothing/swimwear/nudity, pose, lighting, environment, and known artifact availability; no closed specialist taxonomy>.
- **Maximum item count:** <small bounded N and rationale>.
- **Existing-artifact inspection scope:** <exact existence/readability facts allowed for selected items; no mutation of `crawlr/stratum`>.
- **Stage A manifest location:** <versioned manifest path under the preparation output root>.

### Stage A requested authority

Mark every requested authority explicitly; unchecked authority remains denied.

- [ ] Select no more than `<maximum item count>` canonical-source pilot candidates using the stated selection protocol.
- [ ] Read and SHA-256 hash only the selected canonical-source pilot images.
- [ ] Read only the selected items' existing derived-artifact availability/readability facts; do not mutate `crawlr/stratum`.
- [ ] Write only the pilot manifest, preparation log, and review record under the approved preparation output root.

**Stage A non-authorizations:** No model invocation, GPU scheduling, additive artifact generation, corpus mutation, or backfill.

### Stage A owner decision

- [ ] **Approve Stage A as written** — authorizes only the checked preparation actions above.
- [ ] **Approve Stage A with changes:** <record exact scope edits before Stage A starts>.
- [ ] **Do not approve Stage A:** <record missing decision>.

**Owner:** <name>
**Date / time:** <ISO-8601>
**Linked issue / PR / output root:** <URLs and immutable IDs>

Stage A preparation approval does not authorize Stage B execution.

## Freeze and validate after Stage A

After completed Stage A, record—not infer—the following before any execution request:

- **Frozen pilot manifest:** canonical normalized relative path, source SHA-256, source dimensions if relevant, artifact-availability facts, coverage limitations, and selection rationale for every selected item.
- **Immutable manifest identity/digest:** <versioned manifest ID and SHA-256 or other immutable content identity>.
- **Comparison-plan identity/digest:** <versioned filled plan path and immutable identity>.
- **Fixed conditions:** input-view, prompt, and evidence definitions/fingerprints; each declared contrast changes exactly one axis.
- **Evidence bundles:** explicit `kind: "none"` baseline and content-bound inline specialist bundle(s), each with scope, inputs, output semantics, provenance, abstention policy, known failure modes, and qualification gate.
- **Local aggregator:** <already-available local model ID, code/config provenance, and `local_only: true`>.
- **Fixed generation configuration:** <seed(s), temperature, token budget, system/user prompts, image preprocessing, and canonical generation fingerprint>.
- **Prototype parity repair:** <whether `caption_max_tokens` forwarding is fixed and tested before any comparison; if not, Stage B cannot authorize an inference run>. Detector disagreement remains a quality anomaly, never caption content.
- **Representation boundary:** `context4k` remains out of the legacy 512-token T5/T52 route.
- **Human review and metric self-audit:** selected input view → provenance evidence → candidate output/context → decision rubric; supported/unsupported/omission/contradiction/abstention fields; named known/simple case; named null/degenerate output; evaluator version and expected outcomes.
- **Adversarial review:** metric definition stability, fresh-process/independent second review, edge-case inspection, and `quality_anomaly_not_caption_content` handling.

Validate the exact frozen plan before requesting Stage B:

```bash
research-harness validate-comparison-plan research/program.json <frozen-plan.json>
```

Freeze/validation does not invoke a model, claim a GPU, generate artifacts, or
turn Stage A authority into execution authority.

## Stage B — execution authorization

Stage B may be requested only after the frozen manifest and comparison plan
above validate. A fresh owner decision must reference the immutable manifest
identity/digest and comparison-plan identity/digest. Stage B execution authority
is individually checked; no unchecked authority is implied by Stage A or by plan
validation.

### Stage B requested execution authority

- [ ] Read the exact frozen canonical-source pilot items for the approved `<bounded N>` comparison conditions only.
- [ ] Read the exact frozen existing derived artifacts for those pilot items only; do not mutate `crawlr/stratum`.
- [ ] Invoke `<already-installed local model>` locally for exactly `<bounded N>` pilot items under the frozen plan's fixed conditions.
- [ ] Write comparison outputs, evaluation records, and adversarial-review artifacts only under `<approved noncanonical root>` within `/mnt/nas-ai-models/research`.
- [ ] Use GPU scheduling under separate reviewed manifest `<manifest path / ID>` for the exact frozen plan. The current supervisor is `observer_only`; this requires a separately reviewed registered launcher and does not authorize the observer to claim, launch, heartbeat, release, or kill work.
- [ ] Generate specifically named additive artifacts for only the frozen pilot items: `<pass list, model size, output root>`. This requires separately checked data/GPU authority.

**Requested accelerator / resource envelope, if any:** <host route, scheduler project, maximum duration, VRAM, registered launcher, and why this cannot be CPU/local without the scheduler>.

### Stage B owner decision

- [ ] **Approve Stage B as written** — authorizes only the individually checked execution authorities and exact immutable IDs above.
- [ ] **Approve Stage B with changes:** <record exact scope edits before execution>.
- [ ] **Do not approve Stage B:** <record missing decision>.

**Owner:** <name>
**Date / time:** <ISO-8601>
**Immutable manifest identity/digest:** <copy exact value from freeze record>
**Comparison-plan identity/digest:** <copy exact value from freeze record>
**GPU-manifest reference, if requested:** <URL and immutable ID>

A Stage B approval is invalid unless it records both fields above and the individually checked execution authorities.

## Explicit non-authorizations

Unless explicitly and individually checked under Stage B:

- No merge or direct push to `main`.
- No canonical-source mutation.
- No corpus-wide or derived-tree backfill.
- No external image model.
- No model installation or download unless separately named and approved.
- No scheduler operation without the separately approved manifest and registered launcher.
- No overwrite of `caption.txt`, `t5_*`, `pose.npy`, or other Stratum1 artifacts.
- No empirical PASS claim merely because a plan validates or an inference job completes.

## Verification before Stage B execution

- [ ] Current research tree validates with exactly one open active arm and no unresolved relevant hold.
- [ ] Stage A owner approval, manifest output root, maximum item count, and preparation log are recorded.
- [ ] All frozen source paths are canonical normalized relative paths and source hashes were verified during approved Stage A.
- [ ] `research-harness validate-comparison-plan` passes against the exact frozen plan.
- [ ] Fixed local aggregator and generation fingerprints are recorded.
- [ ] Metric self-audit and adversarial-review artifacts are pre-registered.
- [ ] Any GPU manifest validates, is explicitly approved, and uses the registered launcher; no separate claim occurs after scheduler `poll`.
- [ ] Stage B's fresh owner approval references the exact immutable manifest/plan identities and individually checked authorities.
- [ ] Pre-execution status is still `PENDING`.
