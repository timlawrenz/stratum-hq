# Research Control Plane

This directory holds durable, versioned control-plane artifacts. GitHub issues are the detailed research tree; these files define the shared contract and machine-readable validation surfaces.

## Key files

- `program.json` — project-specific contract consumed by `research-harness`.
- `labels.json` — tracked GitHub label specification.
- `gpu-manifests/` — future approved GPU job manifests. The current supervisor is observer-only and cannot launch them.
- `templates/` — project-neutral starting formats for program contracts, compact contexts, research-arm issues, comparison-parity plans, and future GPU manifests.

Future GPU manifests must use a human-reviewed authorization, a registered launcher, the program's scheduler project, a bounded duration/VRAM request, and an approved noncanonical output root. A manifest is still not executable authority while `execution_mode` remains `observer_only`.

## Evidence expansion and compression

A specialist may emit structured outputs, measurements, text, masks, embeddings, or abstentions. An expanded dossier targets approximately 100K tokens of evidence and references; it is compressed into a target approximately 4K-token context only when every retained claim cites source evidence.

The current Stratum T5 paths are 512-token encoders. Do not silently substitute a `context4k` representation for `caption.txt`, `t5_*`, or `t52_*`.

## Controlled comparison plans

Before a caption/context comparison runs, freeze a source-hashed pilot manifest, exact input-view/prompt/evidence fingerprints, and one-axis-at-a-time contrasts. The plan must also pre-register the human claim-support rubric, a known-case and null-output self-audit, and an adversarial review plan:

```bash
research-harness validate-comparison-plan research/program.json <frozen-plan.json>
```

[`templates/comparison-parity-plan.template.json`](templates/comparison-parity-plan.template.json) is deliberately a non-validating fill-in template. Replace every placeholder with an immutable pilot and real fingerprints before invoking the validator. Every pilot `source_relative_path` must be a normalized POSIX path relative to the declared canonical source root: it must not be absolute, contain `..`, use backslashes, or rely on redundant path segments.

Evidence is explicit rather than opaque. A no-specialist baseline uses `"kind": "none"`, an evidence ID, and a fingerprint, **without** a `specialists` field. Any non-null evidence condition uses `"kind": "specialist_bundle"` and carries inline declarations for every specialist: stable ID, scope, inputs/view policy, output semantics, provenance, abstention policy, and qualification gate. This keeps a frozen comparison plan self-contained and auditable while the specialist roster remains open-world. The comparison contract keeps the local aggregator and generation settings fixed, preserves detector disagreement as a quality anomaly rather than caption content, and keeps `context4k` out of the legacy 512-token route.

## GPU manifests

A manifest is an approval record, not an executable command. It must pass:

```bash
research-harness validate-gpu-manifest research/program.json <manifest.json>
```

The program currently has `execution_mode: observer_only`; finding an approved manifest produces a hold message rather than a claim, launch, heartbeat, or release action.
