# Instrument decision: SDXL Juggernaut XL now vs FLUX1 after wiring — for the DINOv3 reconstruction-fidelity gate (#79)

**Date:** 2026-08-08
**Issue:** [#79 — DINOv3 CLS/patch reconstruction-fidelity metric (C_before/C_after vs C_base)](https://github.com/timlawrenz/stratum-hq/issues/79)
**Status:** DECISION RECORD — recommendation only; no GPU work authorized by this doc.

## The question

Tim's hypothesis for #79 names "a text-to-image model like FLUX1" as the generator
whose output CLS should move toward the source's `C_base` as the caption gets more
specific. Before the arm runs, the instrument (which diffusion checkpoint feeds the
reconstruction generator) must be a deliberate, recorded choice — the whole
measurement lives or dies on holding the checkpoint fixed across `C_before`/`C_after`.

## Verified local state (2026-08-08)

| Facility | Ready? | Evidence |
|---|---|---|
| SDXL **Juggernaut XL** checkpoint | ✅ ready | `Juggernaut_XL_v1759168.safetensors` installed in ComfyUI `models/checkpoints/`; already used for arm #37's full 24-item reconstruction run (BETTER, +0.0679) |
| FLUX1 (merged 4-bit NF4) | ⚠️ not wired | only `flux1SchnellMergedWithFlux_unetBnbNf4.safetensors` on the NAS (`/mnt/nas-ai-models/checkpoints/flux/`); ComfyUI `models/text_encoders/`, `unet/`, `diffusion_models/` all empty placeholders |
| ComfyUI | ✅ ready | `/mnt/fscache/essdee/ComfyUI/` proven in arm #37 (`recon_runner`) |
| DINOv3 | ✅ local | stratum `dinov3` pass, HF cache; source `dinov3_cls.npy` already in corpus (5/24 on the frozen cohort) |
| Clip scorer (arm #37) | ✅ | `openai/clip-vit-large-patch14` cached — replaces nothing, DINOv3 is the new primary |

## The decision

**Run the first gate with the SDXL Juggernaut XL checkpoint that is already installed
and already exercised (arm #37). Add FLUX1 only after a separate wiring setup step
(ComfyUI unet + text encoders + VAE + scheduler-reviewed manifest), as a
second-point confirmation — not the primary instrument.**

Rationale:

1. **The hypothesis is model-agnostic.** "A text-to-image model generates an image
   whose CLS is closer to `C_base` when the prompt is more detailed" does not require
   FLUX specifically. SDXL Juggernaut is a competent T2I for full-body portraits on
   this corpus (arm #37 proved it decodes pose/clothing/lighting from long prompts).
   The statistical claim being tested is about *caption specificity* — that is
   prompt-content-dependent, not model-family-dependent.
2. **FLUX adds confounds before it adds signal.** The only local FLUX is the **4-bit
   NF4 quantized merged Schnell** variant. Quantization exists to fit memory; it can
   compress the very fine-grained detail the metric must detect (a small salient
   object like the skateboard). If FLUX-NF4 loses the prop regardless of prompt,
   `C_after − C_before` shrinks for a *model* reason, not an *evidence* reason —
   a false negative the protocol would misread as "specialist didn't help."
3. **Setup debt is real and would block the loop.** Wiring FLUX = unet + T5-XXL/CLIP
   text encoders + VAE into ComfyUI, plus a scheduler manifest review. The first
   verified run can happen today with Juggernaut, which is already SHA-pinned and
   used by the proven `recon_runner` lifecycle.
4. **Two instruments are better than one, in sequence.** If the gate passes on SDXL,
   a FLUX run is a strong confirmation that the effect is caption-driven, not
   checkpoint-specific. If it fails on SDXL, a FLUX re-run tells us whether the
   negative is instrument-bound. Either way the second instrument adds information —
   but only *after* the cheap, clean SDXL reading exists.

## What this does NOT authorize

This record decides the *instrument identity* for the future run. It does **not**
authorize: the aggregator caption run, ComfyUI generation, DINOv3 forward passes on
the cohort, GPU claims, model downloads, or corpus writes. Those remain gated behind
the arm's normal freeze → scheduler-manifest → execution cycle (GPU via
`gpu_scheduler.py`, additive outputs only, local-first).

## Residual risks to record in the frozen plan

- NF4 quantization of any future FLUX leg → treat as a potential fidelity loss;
  compare FLUX-NF4 against SDXL on the same 2-item null/sanity pair before trusting
  its deltas.
- SDXL Juggernaut is a *different* family from whatever the background-conditioning
  trainer uses; the gate is a *proxy* for conditioning fidelity, not the training
  pipeline itself (same caveat as arm #37's `representation_boundary`).
- The generated 832×1216 canvas vs source aspect buckets: patch-grid alignment only
  holds when the source bucket is 832×1216 (see #79 for the per-bucket rule).