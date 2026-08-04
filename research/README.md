# Research Control Plane

This directory holds durable, versioned control-plane artifacts. GitHub issues are the detailed research tree; these files define the shared contract and machine-readable validation surfaces.

## Key files

- `program.json` — project-specific contract consumed by `research-harness`.
- `labels.json` — tracked GitHub label specification.
- `gpu-manifests/` — future approved GPU job manifests. The current supervisor is observer-only and cannot launch them.
- `templates/` — project-neutral starting formats for program contracts, compact contexts, research-arm issues, comparison-parity plans, pilot-authorization proposals, and future GPU manifests.

Future GPU manifests must use a human-reviewed authorization, a registered launcher, the program's scheduler project, a bounded duration/VRAM request, and an approved noncanonical output root. A manifest is still not executable authority while `execution_mode` remains `observer_only`.

## Evidence expansion and compression

A specialist may emit structured outputs, measurements, text, masks, embeddings, or abstentions. An expanded dossier targets approximately 100K tokens of evidence and references; it is compressed into a target approximately 4K-token context only when every retained claim cites source evidence.

The current Stratum T5 paths are 512-token encoders. Do not silently substitute a `context4k` representation for `caption.txt`, `t5_*`, or `t52_*`.

## Controlled comparison plans

Before a caption/context comparison runs, freeze a source-hashed pilot manifest, exact input-view/prompt/evidence fingerprints, and one-axis-at-a-time contrasts. The plan must also pre-register the human claim-support rubric, a known-case and null-output self-audit, and an adversarial review plan:

```bash
research-harness validate-comparison-plan research/program.json <frozen-plan.json>
```

[`templates/comparison-parity-plan.template.json`](templates/comparison-parity-plan.template.json) is deliberately a non-validating fill-in template. Replace every placeholder with an immutable pilot and real fingerprints before invoking the validator. Every pilot `source_relative_path` must be a canonical normalized POSIX path relative to the declared canonical source root: it must not be absolute, contain `..`, use backslashes, rely on redundant path segments, or have leading/trailing whitespace. Identity-bearing fields—program, pilot, item, condition, input-view, prompt, aggregator-model, and contrast IDs—also use exact canonical spellings, so whitespace cannot fabricate a one-axis contrast or corrupt a cross-reference.

Evidence is explicit rather than opaque. A no-specialist baseline is exactly `{"kind": "none", "id": "…", "fingerprint": "…"}`—no `specialists` field or undeclared payload. Any non-null evidence condition is exactly `{"kind": "specialist_bundle", "id": "…", "fingerprint": "…", "specialists": [...]}` and carries inline declarations for every specialist. Each declaration has a stable ID plus the program-required open-world fields: scope, inputs/view policy, output semantics, provenance, abstention policy, known failure modes, and qualification gate. The envelope is closed so arbitrary payload cannot masquerade as a no-specialist baseline; the declaration values remain open-ended natural language.

The evidence fingerprint is binding, not merely a label: it is SHA-256 of the UTF-8 canonical JSON serialization of the complete evidence object with only its `fingerprint` member excluded. Canonical serialization uses sorted object keys, compact separators `(',', ':')`, `ensure_ascii=false`, and disallows non-finite JSON values. Changing any retained inline declaration, specialist list order, evidence ID, or kind therefore requires recomputing the fingerprint. The comparison contract keeps the local aggregator and generation settings fixed, preserves detector disagreement as a quality anomaly rather than caption content, and keeps `context4k` out of the legacy 512-token route.

## Pilot authorization proposals

A valid comparison plan is a necessary provenance boundary, not execution authority. When a frozen plan requires source hashes but policy protects source reads, use the two-stage [`templates/pilot-authorization-proposal.md`](templates/pilot-authorization-proposal.md) rather than a single approval.

**Stage A** is a bounded preparation decision that can be filled without selected item identities or source SHA-256 values. It must name the canonical root, selection protocol, maximum item count, existing-artifact inspection scope, and approved noncanonical output root. If directly approved, it permits only the specifically checked selection/read/hash and manifest-materialization work; it explicitly denies model invocation, GPU scheduling, additive artifact generation, corpus mutation, and backfill.

After Stage A, freeze the exact manifest and comparison plan—source paths/hashes, availability facts, one-axis condition matrix, fixed local model/generation identity, claim-support rubric, self-audit, and adversarial-review references—and run `research-harness validate-comparison-plan`. **Stage B** then requires a fresh owner decision tied to the immutable manifest and plan identities. It may individually authorize only the checked model, data, output, GPU-manifest, and additive-artifact actions. Stage A never rolls forward into Stage B execution authority.

## GPU manifests

A manifest is an approval record, not an executable command. It must pass:

```bash
research-harness validate-gpu-manifest research/program.json <manifest.json>
```

The program currently has `execution_mode: observer_only`; finding an approved manifest produces a hold message rather than a claim, launch, heartbeat, or release action.
