# Research Control Plane

This directory holds durable, versioned control-plane artifacts. GitHub issues are the detailed research tree; these files define the shared contract and machine-readable validation surfaces.

## Key files

- `program.json` — project-specific contract consumed by `research-harness`.
- `labels.json` — tracked GitHub label specification.
- `gpu-manifests/` — future approved GPU job manifests. The current supervisor is observer-only and cannot launch them.
- `templates/` — project-neutral starting formats for program contracts, compact contexts, research-arm issues, and future GPU manifests.

Future GPU manifests must use a human-reviewed authorization, a registered launcher, the program's scheduler project, a bounded duration/VRAM request, and an approved noncanonical output root. A manifest is still not executable authority while `execution_mode` remains `observer_only`.

## Evidence expansion and compression

A specialist may emit structured outputs, measurements, text, masks, embeddings, or abstentions. An expanded dossier targets approximately 100K tokens of evidence and references; it is compressed into a target approximately 4K-token context only when every retained claim cites source evidence.

The current Stratum T5 paths are 512-token encoders. Do not silently substitute a `context4k` representation for `caption.txt`, `t5_*`, or `t52_*`.

## GPU manifests

A manifest is an approval record, not an executable command. It must pass:

```bash
research-harness validate-gpu-manifest research/program.json <manifest.json>
```

The program currently has `execution_mode: observer_only`; finding an approved manifest produces a hold message rather than a claim, launch, heartbeat, or release action.
