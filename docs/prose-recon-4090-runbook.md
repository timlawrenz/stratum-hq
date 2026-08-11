# Prose-recon demo — 4090 night runbook (fallback for the strix ROCm path)

**Date:** 2026-08-09. The strix ComfyUI path was attempted 5× and produced 0 images:
bf16 VAE decode hangs, fp32 VAE decode hangs, sampler crawls under `PYTORCH_TUNABLEOP_TUNING`,
tiled decode hangs, `--cpu-vae` decodes but takes ~5-10 min/image single-issue (
8 images ≈ 1 h+). The **4090 is the proven instrument** (arm #37: same checkpoint +
settings, 5.4-6 s/gen). Intended for the ~04:00Z slot (after btnet + garment jobs).

## ✅ VERIFIED WORKING on the 4090 (2026-08-09 22:00Z — full run, ~2 min)

Runner bugs fixed on this branch (commit `…history-GET…`): (1) `generate()` polled
ComfyUI `/history/{id}` with POST → **405** → every run hung at gen 1 while ComfyUI
actually completed the images; the poll must be **GET**. (2) `OUTPUT_DIR` is now
env-overridable (`PROSE_OUTPUT_DIR` — the 4090's ComfyUI outputs to
`/mnt/fscache/essdee/ComfyUI/output`, not the strix path). (3) aggregator leg is
skippable via `PROSE_SKIP_AGG=1` (use it when the strix ollama is contended).

**Aggregator on the 4090 (Tim's steer):** ollama 0.32.6 with gemma3:27b is ALREADY
running on the 4090 (`http://127.0.0.1:11434`) and coexists with ComfyUI in VRAM
(~13 GB gemma + ~9 GB SDXL on 24 GB). The runner's `OLLAMA_URL` points there, so the
aggregator captions generate locally — no dependence on the (often busy) strix.

**Exact command (local 4090):**
```bash
python3 /mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py request --gpu 4090 --project stratum-contextual-specialist-research --vram 12 --duration 1 --job-id stratum-prose-recon-4090-v1
python3 /mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py poll --gpu 4090 --job-id stratum-prose-recon-4090-v1
# boot comfy (bg): /mnt/fscache/essdee/ComfyUI/.venv/bin/python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch
python3 /mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py activate --gpu 4090 --job-id stratum-prose-recon-4090-v1
env PROSE_OUTPUT_DIR=/mnt/fscache/essdee/ComfyUI/output /mnt/fscache/essdee/ComfyUI/.venv/bin/python scripts/prose_recon_demo.py
python3 /mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py release --gpu 4090 --job-id stratum-prose-recon-4090-v1 --status completed
```
Result (22:00Z): 5× `recon-prose.png` + 2× `recon-prose-agg.png` + `_null.png` +
`_recon-manifest.json` under `/mnt/nas-ai-models/research/stratum/prose-caption2-demo-v1/`.


## Prerequisites (all verified 2026-08-09)

- ComfyUI at `/mnt/fscache/essdee/ComfyUI` (4090 local) — arm-37-proven.
- Checkpoint: `Juggernaut_XL_v1759168.safetensors` inside that ComfyUI's
  `models/checkpoints/`; sha256 `dd08fa32f98d05a2443ca1419e46df1575a0811f6e3b246d9dd47ff20f5eb66a`.
- Prose captions: `/mnt/nas-ai-models/research/stratum/prose-caption2-demo-v1/<id>/caption2.txt`
  (deterministic, polished renderer) — 5 pilot items; aggregator leg uses the strix
  ollama (`gemma3:27b`) from the 4090 runner via `http://192.168.86.137:11434`.
- Runner: `scripts/prose_recon_demo.py` (this branch). It boots nothing — it expects a
  running ComfyUI at `127.0.0.1:8188`; the wrapper `scripts/prose_recon_demo.sh` (bash,
  in this doc's author's /tmp) boots ComfyUI, runs the runner, stops ComfyUI.
  For the 4090, boot ComfyUI without ROCm workarounds (`--fp32-vae` not needed there):
  `.venv/bin/python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch`

## Scheduler lifecycle (program contract)

```
python3 /mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py request --gpu 4090 --project stratum-contextual-specialist-research --vram 12 --duration 1 --job-id stratum-prose-recon-demo-v1
# when queue front (after btnet×2 + garment):
... poll --gpu 4090 --job-id stratum-prose-recon-demo-v1   # atomic claim, "claimed"
# launch IMMEDIATELY (local 4090): boot ComfyUI (background), then run the runner
... activate --gpu 4090 --job-id stratum-prose-recon-demo-v1
... heartbeat --gpu 4090 --job-id stratum-prose-recon-demo-v1 --progress 1 --vram-used 10
# monitor: outputs land under /mnt/nas-ai-models/research/stratum/prose-caption2-demo-v1/<id>/recon-*.png
# completion: <id>/_recon-manifest.json appears → release
... release --gpu 4090 --job-id stratum-prose-recon-demo-v1 --status completed
```

Expected wall time on the 4090: ~2 min boot + ~1 min model load + 8 × ~6 s ≈ 5 min total.

## Checkpoint

- 5 items × prose (deterministic) + 2 × prose-agg (gemma3:27b) + 1 null = 8 gens.
- Seeds: `sha256(image_id)[:8]` — identical to arm-37 for direct visual comparison.
- Outputs are additive under the approved noncanonical research root; nothing canonical touched.