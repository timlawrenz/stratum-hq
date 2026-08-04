# Portrait Evidence Discovery Map

**Arm:** [#3 — Portrait evidence discovery](https://github.com/timlawrenz/stratum-hq/issues/3)
**Parent:** [#2 — Program root](https://github.com/timlawrenz/stratum-hq/issues/2)
**Status:** `ACTIVE — PENDING`; this is a candidate map, not a specialist ranking or an empirical PASS.
**Captured:** 2026-08-04 on `docs/portrait-evidence-discovery-map-20260804` from workspace commit `de3a2918223c8b313f5d4ceecb4a22ea61461cdf`.

## Scope and decision boundary

The canonical source is `/mnt/nas-ai-models/training-data/crawlr/approved`. Every source image is a curated image of exactly one woman. A zero- or multi-person detector outcome is a quality anomaly to preserve for review; it is never caption content, evidence that the source contains a different number of people, or a reason to change the invariant.

This map answers a narrower question than model selection: **which evidence roles are observable, potentially useful, and non-redundant enough to earn a qualification experiment?** It deliberately does not freeze an attribute taxonomy, choose a model roster, run inference, mutate either corpus tree, or turn any current artifact into a 100K dossier or a 4K compact context.

Every row below is an open-world *role*. A future implementation may be deterministic code, a local foundation model, a fine-tune, an embedding, an ensemble, or something not yet identified. It earns a place only after declaring its inputs/view policy, provenance, confidence or abstention behavior, failure modes, and a pre-registered qualification gate.

## Why this was selected over the other open branches

The full open tree surveyed on 2026-08-04 was: #2 program root; #3 sole active discovery arm; #4 baseline/comparison parity (`research:metric-risk`); and #5 preserved geometry-grounded prototype. Open draft PRs were #6 (research harness, clean with passing `pytest`) and #1 (prototype, pending). There are no closed research or post-mortem issues.

#3 was selected because it is the highest-information authorized action: it identifies what should be measured before the program commits to a model, GPU work, a fixed ontology, or a comparison metric. It beats:

1. **#4 first:** a comparison matrix cannot establish useful specialist axes until the evidence roles, their abstention paths, and their expected redundancies are explicit.
2. **#5 implementation repair first:** the known non-default token-budget propagation gap and raw-versus-bucketed-view confound are real, but repairing a preserved prototype would not answer whether its evidence dimensions are worth qualifying.
3. **GPU/data work:** no such authority exists, and the derived tree is partial. A backfill would be both unauthorized and scientifically premature.

The existing #4 arm is the registered next falsifiable arm once this map is reviewed. It must turn only selected candidate roles into a controlled, metric-audited comparison; it does not inherit a model winner from this document.

## Bounded artifact evidence

The read-only inventory is recorded in [`assets/portrait-evidence-discovery-inventory-2026-08-04.json`](assets/portrait-evidence-discovery-inventory-2026-08-04.json). It used one flat `os.scandir` of the canonical root, one immediate listing of the derived root, and seven deterministic, evenly spaced source-file spot checks. It did **not** recurse through the NAS, read semantic image content, estimate completion from the sample, invoke a model, or write corpus artifacts.

| Observed evidence | What it establishes | What it does **not** establish |
|---|---|---|
| 11,825 source files: 10,857 JPEG, 445 PNG, and 523 WebP | The canonical source root is present and heterogeneous in container format. | Any semantic distribution, quality score, or corpus-wide specialist coverage. |
| 4,901 immediate derived-tree entries; only 3 of 7 deterministic source spot checks resolved to a derived leaf | `crawlr/stratum` is partial and heterogeneous, so artifact presence must be recorded per item. | A completion percentage or a mandate to backfill. |
| One fully enriched observed leaf had raw source dimensions 1080×1350, legacy `pixel.npy` shape `(3, 1216, 832)`, and `pose2`/`seg2`/`pointmap` aligned to raw `(1350, 1080)` spatial dimensions | The raw-vs-bucketed input-view axis is concrete, not hypothetical; it is a first-class confound for #4. | That either view produces better captions. |
| That leaf contained `determinations.json`, `caption2.txt`, and `t52_*`; `t52_hidden.npy` was `(512, 1024)` and no `context4k.json`, `context4k.md`, or `compression.json` existed | The preserved prototype is additive and remains inside the legacy 512-token encoder path. | A 4K compact context, a 100K dossier, or downstream usefulness. |
| `exp/geometry-grounded-captioning` code and synthetic tests | A candidate deterministic evidence-to-language chain exists, including per-region corroboration fixtures. | Real-image factuality, calibration, robustness, or a comparative win. |

The historical counts in `docs/EXPERIMENTS_AND_RESULTS.md` remain historical ledger claims; this bounded inventory intentionally does not replace them with a costly corpus scan.

## Candidate evidence roles

| Evidence role (not a fixed model) | Scope, inputs, output semantics, and provenance | Abstention / failure behavior | Qualification gate before adoption |
|---|---|---|---|
| **View and source trace** | Records source identifier, dimensions, decoder/container, raw view, bucket/crop transform, and any selected crop. It describes the *evidence view*, not semantic image content. Provenance must include source hash, transform version, and exact consumer view. Existing sources: `metadata.json`, source image dimensions, and opt-in `pixel.npy`. | Abstain from claims tied to a view when its transform or source hash is unknown. Failure mode: comparing raw and bucketed/cropped inputs as if they were the same condition. | Synthetic transform fixtures plus a per-item view manifest; #4 must use identical declared views for every compared condition. |
| **Visible-body geometry** | Deterministic keypoints and relations from `pose2.npy` (and, separately, legacy `pose.npy` where useful). Outputs are continuous coordinates, confidence, and sparse open-set relations—not posture/activity labels. The prototype freezes a Goliath-308 table on its own branch; any future use must cite the exact table/version. | Omit a relation when required keypoints or corroborating region evidence are absent/low-confidence. Detector-count disagreement is a quality flag only. Failures include crop truncation, occlusion, mislocalized keypoints, and table-version mistakes. | Hand-built geometry fixtures for orientation and relations; table sentinel tests; then a fixed hard-case review slice where relation support is judged against the original image without converting misses into closed labels. |
| **Visible-region coverage** | `seg2.npy`, legacy `seg.npy`, and optionally matting represent observed pixels/classes, body-region extent, clothing coverage, and foreground masks. They do not name fabric, activity, anatomy beyond supported visible regions, or hidden content. Provenance includes class table and mask preprocessing. | Omit region-dependent claims under low foreground coverage, class ambiguity, or missing masks. Failures include tight-crop collapse, clothing masking skin classes, and treating a class ID as a semantic fact beyond its scope. | Synthetic class-map tests, foreground/coverage sanity checks, and a paired pose↔seg corroboration audit over the same fixed items. Incremental utility versus pose-only evidence must be measured. |
| **Relative spatial/camera geometry** | `pointmap.npy` can provide camera-frame coordinates, relative depth ordering, and body-relative spatial relations. It is not automatically an absolute-size, distance, or camera-calibration instrument. Provenance includes model/checkpoint, mask policy, coordinate convention, and units. | Abstain from absolute distance/scale claims without a calibrated reliability gate; downgrade or omit on extreme crops, sparse foreground, or inconsistent scale. Failures include monocular scale drift and sign/convention inversions. | Synthetic coordinate-convention fixtures plus same-subject/view stability checks. Compare relative-relation support against a null and against pose/seg alone before retaining it. |
| **Surface and silhouette evidence** | `normal2.npy` and `matting.npy` are candidate evidence for visible surface orientation, silhouette boundary, and opacity. They are distinct from semantic/material interpretation. Provenance must state segmentation-mask dependence and resize policy. | Omit claims when foreground is empty, masks collapse, or spatial alignment is invalid. Failures include masked-empty maps, crop distortion, and redundant restatement of segmentation. | Shape/dtype/alignment fixtures, empty-foreground adversarial cases, and an incremental-information gate relative to segmentation plus pointmap. |
| **Appearance/layout embeddings** | Global and patch embeddings such as `dinov3_cls.npy` and `dinov3_patches.npy` may preserve appearance/layout signal for a later specialist or retrieval probe. They are evidence vectors, not self-explanatory captions or identity facts. Provenance must include model/version, preprocessing, pooling, and any learned probe. | Abstain whenever no validated decoder/probe maps an embedding to a claim. Failures include background/style confounding, crop sensitivity, and a probe that appears useful only because of split leakage. | A pre-registered, same-item task with a relevant null (for example, random projection or no-embedding baseline), held-out evaluation, and an adversarial inspection of both high- and low-confidence outputs. |
| **Local semantic interpretation** | A local image-aware captioner or future specialist can contribute color, texture, lighting, expression, environment, props, style, and open-ended activity interpretation that deterministic geometry does not measure. Its text is evidence with prompt/model provenance—not ground truth. | It must mark or omit unsupported claims rather than inventing details. Failures include refusals/omissions on sensitive images, prompt drift, preprocessing sensitivity, and geometry contradiction. External image models remain out of scope. | #4's fixed-view, fixed-settings, claim-support rubric with a degenerate/null output check, matched legacy baseline, and sequential human review. No PASS without metric self-audit and adversarial review. |
| **Grounded deterministic-to-language bridge** | The preserved prototype's `determinations.json → caption2.txt → t52_*` is a candidate bridge: geometry can constrain a local semantic renderer while remaining additive. Its current branch provides per-region corroboration and synthetic fixtures. | Future program work must treat determinations as provenance-bearing evidence, not universal “ground truth.” It must preserve field-level support/confidence, uncertainty, and conflicts. Current failures: raw-view mismatch, unforwarded non-default caption budget, unsupported absolute-camera interpretation, and 512-token `t52` limit. | First satisfy #4 preprocessing/prompt/evidence/model parity. Test a non-default token budget, a known-simple/degenerate output, supported versus unsupported claims, omissions, contradictions, and abstentions on a frozen pilot. |
| **Dossier aggregation and compact context** | A future aggregator must keep all chosen evidence paths in a ~100K-token dossier and produce `context4k.json`, `context4k.md`, and `compression.json` at exactly the program's first-class 4K target. It is an evidence-preserving representation role, not a longer caption. | Omit unresolved claims or retain conflicts explicitly. Never silently truncate into `caption.txt`, `t5_*`, or `t52_*`. Failures include lost evidence links, token-accounting drift, invented consensus, and accidental 512-token routing. | Validate every compact claim against a dossier evidence ID; enforce 100K/4K accounting and conflict serialization; then test a separate downstream consumer under its own arm. |

## Complementarity rules and explicit exclusions

1. **Do not duplicate a role merely because an artifact exists.** Geometry and segmentation may corroborate each other; normals/matting must demonstrate incremental information rather than repeat a silhouette; embeddings must earn a claim decoder rather than be treated as language.
2. **Keep measurement separate from interpretation.** Deterministic roles measure visible spatial/region evidence. A semantic specialist may name an activity, prop, material, or setting only as an evidence-bearing interpretation with support and abstention behavior.
3. **Do not confuse a detector with the corpus invariant.** The curated one-woman invariant is upstream; detector disagreement is reviewable quality evidence, never caption semantics.
4. **Keep preprocessing as evidence.** The observed raw/bucketed mismatch means crop, resize, color conversion, and selected view are required provenance, not implementation details.
5. **Keep legacy artifacts intact.** `caption.txt`, `t5_*`, `pose.npy`, and other Stratum1 outputs remain baselines. New artifacts are additive only.
6. **Do not promote pointmap scale prematurely.** Relative spatial relationships may be useful; absolute physical claims require a separate reliability gate.
7. **No compact-context shortcut.** The observed 512-token `t52` artifact is a legacy-compatible branch artifact, not `context4k`.

## Stratified review design for the next arm

Before any comparative inference, #4 should freeze a pilot manifest with source hashes and artifact-availability flags. Its review cells should span, without treating them as a taxonomy: raw versus bucketed view; tight/partial/full/environmental framing; orientation and occlusion; lighting/texture/background complexity; clothing/skin coverage; hand/prop interactions; and missing/degraded artifact cases.

For each item, the required human review order is:

```text
original selected view → provenance-bearing evidence → compact rendering (when available) → decision/rubric
```

The rubric must separately record supported claims, unsupported claims, omissions, contradictions, and abstentions. A simple known case and a null/degenerate output must be scored before any candidate comparison. This is a design requirement, not a claim that the strata have already been empirically covered.

## Status and next gate

**Verdict: `PENDING`.** The map distinguishes deterministic, semantic, and open-ended evidence roles; names abstention paths and failure modes; and points to existing #4 as a falsifiable next arm. It has not received review, no candidate has passed a qualification gate, and no controlled caption/context quality claim is made.

**Smallest next action:** review this map in its draft PR, then keep #3 active only long enough to incorporate review corrections. Select #4 only after it freezes a source-hashed stratified pilot, equal-view conditions, a claim-support rubric, null/evaluator checks, and an adversarial-review plan. No GPU, model download, corpus mutation, or backfill is implied.
