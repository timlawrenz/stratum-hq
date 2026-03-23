# Copilot Instructions — stratum-hq

## What this project is

stratum-hq is a **dataset-agnostic** image enrichment pipeline. Given any directory of images, it produces per-image artifact directories with multi-modal embeddings (DINOv3, T5, pose keypoints, captions). The output is designed for publishing to HuggingFace and for training diffusion models.

The project originated from [prx-tg](https://github.com/timlawrenz/prx-tg) but is intentionally decoupled from any specific image source.

## Build and run

```bash
# Install (core only — no GPU deps)
pip install -e .

# Install with GPU support
pip install -e ".[gpu]"

# Install everything
pip install -e ".[all]"

# Run the CLI
stratum process ./images/ --output ./dataset/ --passes all --device cuda
stratum status ./dataset/
stratum verify ./dataset/
```

Torch must be installed separately for your platform (CUDA/ROCm) before installing `[gpu]`.

## Architecture

### Package layout

```
src/stratum/
├── cli.py              # CLI entry point (stratum command)
├── config.py           # Constants: model IDs, bucket defs, artifact filenames
├── discovery.py        # Image scanning, image_id derivation, sharding
├── verify.py           # Dataset integrity checks
├── publish.py          # HuggingFace Hub publishing (stub)
├── pipeline/
│   ├── __init__.py     # Orchestrator: loads models, runs passes over images
│   ├── bucket.py       # Aspect ratio bucketing, bucketed image loading
│   ├── caption.py      # Ollama-based captioning (pluggable backend)
│   ├── dinov3.py       # DINOv3 CLS + patch extraction
│   ├── t5.py           # T5-Large hidden state encoding
│   ├── pixel.py        # Bucketed RGB crops (opt-in)
│   └── pose.py         # DWPose whole-body keypoints
└── dwpose/
    └── detector.py     # DWPose ONNX detector (standalone, no mmpose)
```

### Per-image directory format

Output mirrors the source directory structure. No central JSONL — the filesystem is the state.

```
source/ffhq/batch1/00001.png  →  output/ffhq/batch1/00001/
                                    ├── metadata.json
                                    ├── caption.txt
                                    ├── dinov3_cls.npy        (1024,) float32
                                    ├── dinov3_patches.npy    (N, 1024) float32
                                    ├── t5_hidden.npy         (512, 1024) float16
                                    ├── t5_mask.npy           (512,) uint8
                                    ├── pose.npy              (133, 3) float16
                                    └── pixel.npy             (3, H, W) float16  [opt-in]
```

### Pipeline passes

Each pass is independent, idempotent, and parallel-safe:

| Pass | Artifacts produced | GPU needed |
|------|--------------------|------------|
| `caption` | `caption.txt` | No (Ollama API) |
| `dinov3` | `dinov3_cls.npy`, `dinov3_patches.npy` | Yes |
| `t5` | `t5_hidden.npy`, `t5_mask.npy` | Yes |
| `pose` | `pose.npy` | No (ONNX) |
| `pixel` | `pixel.npy` | No |

`--passes all` runs caption, dinov3, t5, pose. Pixel is **opt-in only** (excluded from `all`).

### Parallel execution

Sharding is per-pass. Multiple GPUs each run one pass on a subset:

```bash
stratum process ./images/ --output ./dataset/ --passes dinov3 --shard 0/4 --device cuda:0
stratum process ./images/ --output ./dataset/ --passes dinov3 --shard 1/4 --device cuda:1
```

No coordination needed — each worker writes to different image directories.

## Key conventions

### Per-image pipeline function signature

Each pipeline module exposes a `process()` function that returns `bool` (success/failure). The orchestrator in `pipeline/__init__.py` calls them and handles model loading.

### stderr for logging, stdout for data

All progress/warning/diagnostic output uses `eprint()` (stderr). Never print diagnostics to stdout.

### Graceful degradation

Pipeline `process()` functions return `False` on error (never raise). Individual image failures are logged and skipped — they don't halt the pipeline.

### Completeness = file existence

An artifact is "done" when its file exists with correct shape/dtype. No central registry. `stratum status` counts files; `stratum verify` validates shapes.

### DINOv3 specifics

- RoPE positional embeddings, variable-length sequences
- Patch count = `(H÷16) × (W÷16)`, varies per bucket (3600–5500)
- Token layout: `[CLS, patch_1, ..., patch_N, reg_1, ..., reg_4]` — extract indices 1 through N
- Single forward pass for CLS + patches via `compute_dinov3_both()`
- `do_center_crop=False` to preserve spatial alignment

### Pose normalization

Keypoints normalized to `[-1, 1]` relative to **bucket dimensions** (not original image):
```python
x_norm = (2.0 * x_pixel / bucket_w) - 1.0
```

### Constants (in `config.py`)

```python
DINO_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
T5_MODEL_ID   = "t5-large"
NUM_POSE_KEYPOINTS = 133  # COCO-WholeBody
```

### Aspect ratio buckets

7 default buckets (~1 megapixel, dims divisible by 64). Configurable. Images are resize-to-cover + center-crop to exact bucket dimensions.

## Dependencies

Defined in `pyproject.toml` with optional extras:
- Core: `numpy`, `Pillow`, `requests`
- `[gpu]`: `transformers>=4.54.0`, `accelerate`, `safetensors`
- `[pose]`: `onnxruntime>=1.16.0`, `opencv-python>=4.5.0`
- `[publish]`: `huggingface-hub`, `pyarrow`
- `[all]`: everything above

Torch is not a declared dependency — install separately for your platform.
