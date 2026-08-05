# Evidence-Dimension Arms — Expanding the 100K→4K Program

## Purpose

The first empirical result (arm #4) established a **reusable measurement pattern** on the frozen first-500 cohort:

```
deterministic evidence (existing artifacts, read-only, in-memory)
  → caption generation (fixed local model/settings)
  → independent reviewer (different model family) claim-support delta
  → paired per-item significance + deterministic evidence-trace cross-check
```

This document enumerates **new arms** that reuse that proven pattern for additional
content dimensions. Each nameable dimension (clothing, hair, makeup, skin color,
texture, mood, lighting, setting, body type, …) becomes a candidate evidence
**specialist** whose contribution to supported contextual claims can be measured
under the same one-axis, same-model, same-settings discipline.

## Grounding: what the existing artifacts already measure

From the frozen DOME-29 `seg2` + source pixels + `pose2`/`normal2`/`pointmap`, each dimension has a deterministic measurement path:

| Dimension | Deterministic signal (existing artifacts) |
|---|---|
| Clothing / apparel | `seg2` classes Apparel(1), Upper_Clothing(23), Lower_Clothing(13), Torso(22), Socks, Shoes; per-class pixel fraction + dominant color from source pixels; coverage proxy (clothed vs exposed skin) |
| Hair | `seg2` Hair(4) region: area fraction, dominant color from source pixels, length proxy from mask extent |
| Skin color / tone | `seg2` Face_Neck(3), Upper/Lower arms+legs skin regions; dominant color / brightness in CIELAB or RGB |
| Makeup (face-region) | Face_Neck mask + lips(24/25)/teeth(26/27) classes; local color contrast around eyes/lips from source pixels |
| Body type / proportions | `pose2` Goliath-308 keypoints: shoulder:hip ratio, torso:leg length ratio, limb-length ratios, bbox aspect; `seg2` region fractions |
| Texture | Per-class region texture proxies (e.g. gradient magnitude / edge density inside a mask); `normal2` gradient statistics; source-region standard deviation |
| Lighting | `normal2` direction statistics; `pointmap`-free luminance histogram, dynamic-range and brightness from source pixels |
| Setting / environment | Non-subject (Background class 0 + non-skin region) pixel stats; color palette; scene geometry hints from `normal2` |
| Mood / expression | Not deterministically measurable alone → treat as **open-world relational language** produced by the caption model, grounded only by the face/pose signals; must be validated for unsupported-claim inflation |

## Methodology per arm (reused, not re-litigated)

1. Freeze a coverage-balanced subset from the **same first-500 core-covered cohort** (already 500/500 for `pose2/normal2/pointmap/matting/seg2`).
2. Compute the dimension's deterministic measurement **in memory** from existing artifacts — read `seg2`/`pose2`/`normal2`/source pixels per selected item only; no new model, no corpus write.
3. Generate captions under the same fixed conditions (view, prompt template, model, settings) with and without the declared dimension evidence → **one-axis evidence-only delta**.
4. Score with an **independent reviewer** (different model family from the generator) into claim-support buckets.
5. Report: supported/unsupported/omissions/contradictions per condition; paired per-item sign test; and a **deterministic evidence-trace cross-check** (does the caption verbalize the declared measurement?).

## New arms (proposals, not yet active)

- **Arm A — Clothing/apparel evidence**: coverage, garment classes, and per-garment dominant colors. *Question:* does declared apparel evidence reduce unsupported clothing claims?
- **Arm B — Hair evidence**: color, coverage, and length proxy. *Question:* does declared hair evidence reduce hair-color/coverage hallucinations?
- **Arm C — Skin-color/tone evidence**: exposed-skin dominant color/tone. *Question:* does declared skin tone reduce invented skin-tone claims?
- **Arm D — Body-type/proportion evidence**: pose-derived anthropometric ratios. *Question:* does declared proportion evidence improve body/limb descriptions (strongest prior: pose already helped in arm #4).
- **Arm E — Texture/material evidence**: per-region texture proxies. *Question:* does texture evidence reduce invented material/texture details?
- **Arm F — Lighting evidence**: luminance/dynamic-range/direction statistics. *Question:* does declared lighting evidence reduce invented shadow/lighting claims?
- **Arm G — Setting/environment evidence**: background palette and non-subject stats. *Question:* does setting evidence improve scene claims?
- **Arm H — Makeup (face-region) evidence**: lip/eye-region color contrast. *Question:* does makeup evidence improve face-detail claims (hardest; likely marginal).
- **Arm I — Mood/expression (open-world)**: no deterministic specialist; validate that *undeclared* mood language does not inflate unsupported claims (negative-control arm).

## Downstream: the actual 100K→4K program

Each verified evidence dimension is a **building block of the ~100K-token dossier**. The full program arm is to:

1. Assemble per-item the expanded dossier from **all verified dimension measurements + relational determinations** (target ≥ 100K tokens, claim-by-claim evidence links, never truncated).
2. Compress that dossier into the **~4K-token compact context** (`context4k`, first-class provenance-bearing artifact, separate from legacy 512-token T5/T52) and validate that support/evidence survives compression (round-trip claim-support audit).

This mirrors the user's framing: *"~100K tokens that describe an asset → compressed to ~4K tokens. Clothing, hair, makeup, texture, skin color, mood, lighting, setting, body types — the list is endless."*

## Measurement semantics — px vs ratios (owner directive 2026-08-05)

- Absolute pixel measurements (e.g. `between_shoulders: 137.386`) are **not**
  verbalized into captions: they are camera-frame-dependent, do not survive
  cross-picture comparison, and a text-to-image model cannot interpret them.
  They remain in the machine-readable `evidence_payload` JSON as part of the
  per-asset dossier / compressor input, but are never caption claims.
- **Scale-invariant ratios are the verbalized signal**: shoulder:hip, leg:torso,
  limb-length asymmetry. These survive cross-picture comparison and are
  meaningful guidance for describing a subject (owner agreement).
- The reviewer consumes the same rendered evidence, so it can no longer reward
  or penalize px restatement — removing that confound by construction.
- Future evidence dimensions must follow the same rule: ratios/relative intent
  verbalized; raw absolute measurements stay in JSON only.
- A dedicated `waist` keypoint is not in GOLIATH-308; hip:waist / waist:shoulder
  would require a seg2-based waist estimator (narrowest torso cross-section) —
  a candidate future measurement for arm #31/#32 extension.

## Guardrails

- Every arm keeps: local models only; outputs only under the approved noncanonical research root; no `crawlr/approved`/`crawlr/stratum` mutation; no backfill; no legacy overwrite; scheduler lifecycle for any GPU action; additive artifacts only.
- Exactly one `research:active` arm at a time (routing rule; prep work may proceed concurrently on draft branches).
- An arm is opened as a `research:proposal` issue with full machine-readable metadata; activation requires its own selection rationale + parent recording.

## Convex-sweep harness (source of truth)

The dimension list is **not** maintained as hand-written issues. The source of truth is
`research/dimensions/evidence-dimension-registry-v1.json`, validated and swept by
`src/research_harness/dimension_registry.py` via the CLI:

```bash
research-harness validate-dimension-registry research/dimensions/evidence-dimension-registry-v1.json
research-harness dimension-sweep-status       research/dimensions/evidence-dimension-registry-v1.json
```

- States: `proposal` → `active` → `validated` / `falsified` / `exhausted` (terminal). Three valid
  non-improving experiments force a terminal state (`per_dimension_strike_limit: 3`).
- The sweep is a **convex space walk**: enumerate the full dimension space first; only when every
  dimension is terminal does `dimension-sweep-status` report `exhausted: true` with
  `next_action: "brainstorm-new-data"` — a harness state for proposing genuinely *new* data
  sources/dimensions, rather than inventing variants of the same space.
- Arms defined in the registry back the proposal issues already created (#29–#36) and any future
  ones; the registry and issue tree must not drift apart.

## Non-stratum (open-world) specialists

The evidence space is **not** limited to stratum/Sapiens2-derived data. A dimension may declare
one or more `specialists` — external models that measure the dimension better than the stratum
artifacts:

- **Florence-2** (base / large / Flux-Large / PromptGen, present in
  `/mnt/fscache/essdee/ComfyUI/models/LLM`) — open-set image tagging/captioning. A strong
  non-stratum specialist candidate for clothing/apparel, texture/material, and attribute lists.
- **Dedicated diffusion checkpoints** (`/mnt/nas-ai-models/checkpoints`, 195 weights: SD1.5
  blends, SDXL, flux1Schnell, sd3, pony, SUPIR, LTXV) — used for *reconstruction* validation.

Specialist declarations must carry: name, source, scope, and known failure modes (enforced by
the registry validator). Specialists must be qualified (scope + provenance + abstention +
failure modes) before they may replace or augment a stratum measurement; they do not bypass the
local-first / scheduler / no-mutation boundaries.

## Reconstruction validation (ComfyUI round-trip)

A dimension or the full dossier may use `reconstruction` (and the program milestone uses
`roundtrip-audit`):

1. Generate an image from the caption / `context4k` via local ComfyUI
   (`/mnt/fscache/essdee/ComfyUI`, `.venv` python 3.14 + torch 2.12+cu130, CUDA on the local
   4090) using an installed checkpoint (SD1.5, SDXL, or flux1Schnell).
2. Score the generated image against the original source with CLIP ViT-L/14 similarity
   (`openai/clip-vit-large-patch14` and `timm/vit_large_patch14_clip_224.openai` are already in
   the HF cache).
3. The reconstruction score is a **stronger, non-LLM test** of how much per-asset information
   survives into the caption/context than claim-support alone.

Constraints: local-first (sensitive imagery); scheduler-managed GPU; noncanonical output root;
no corpus mutation/backfill; candidate arm/gate must be recorded before reconstruction results
are treated as evidence.
