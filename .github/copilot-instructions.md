# Copilot Instructions — stratum-hq

## What this project is

stratum-hq builds an enriched human image dataset that complements FFHQ. It extracts and extends the dataset generation pipeline originally developed in [prx-tg](https://github.com/timlawrenz/prx-tg).

The pipeline takes approved human photos as input and produces a rich, multi-modal dataset: JSONL metadata plus pre-computed embeddings stored as `.npy` files. The output is designed for training diffusion models (the training code itself lives in prx-tg, not here).

## Architecture

### Data pipeline (multi-pass, incremental)

The core script processes images through independent passes that can be run individually or together via `--pass`:

1. **dinov3** — DINOv3-ViT-L/16 visual embeddings (CLS token + spatial patch tokens) and Ollama-based captioning (Gemma3:27b)
2. **t5** — T5-Large text encoder hidden states from captions
3. **image** — Pixel-space bucketed RGB crops
4. **pose** — DWPose whole-body keypoints (133 COCO-WholeBody joints, ONNX inference)
5. **migrate** — Format migration (Stage 1 → Stage 2)

Each pass is idempotent and resumable. The script writes to a `.tmp` file and does an atomic merge on completion. Ctrl+C triggers a graceful merge of partial progress.

### Dataset format (Stage 2)

**JSONL record** (one per image in `approved_image_dataset.jsonl`):
```json
{
  "image_path": "data/approved/IMG_001.jpg",
  "image_id": "IMG_001",
  "width": 1920, "height": 1440,
  "aspect_bucket": "1216x832",
  "caption": "A fair-skinned woman with slender build...",
  "t5_attention_mask": [1, 1, ..., 0, 0],
  "format_version": 2
}
```

**External `.npy` files** (keyed by `image_id`):

| Directory | Shape | Dtype | Contents |
|-----------|-------|-------|----------|
| `dinov3/` | `(1024,)` | float32 | CLS token (global style) |
| `dinov3_patches/` | `(num_patches, 1024)` | float32 | Spatial patch embeddings (varies by bucket) |
| `t5_hidden/` | `(512, 1024)` | float16 | T5 text encoder hidden states |
| `images/` | `(3, H, W)` | float16 | Pixel RGB in [0,1] |
| `pose/` | `(133, 3)` | float16 | Keypoints: [x_norm, y_norm, confidence] in [-1,1] |

### Aspect ratio bucketing

All images are assigned to the closest bucket (~1 megapixel each, dims divisible by 64):

```
1024×1024  1.00   Square
 832×1216  0.68   Portrait
1216×832   1.46   Landscape
 768×1280  0.60   Tall portrait
1280×768   1.67   Wide landscape
 704×1344  0.52   Very tall
1344×704   1.91   Very wide
```

Images are resize-to-cover + center-crop to exact bucket dimensions. DINOv3 patches use the same bucket dimensions but skip center-crop (`do_center_crop=False`) to preserve spatial alignment.

### Image source

Approved photos are synced from `https://crawlr.lawrenz.com/photos.json` via `sync_approved_photos.py`. Raw files go to `data/raw/`, symlinks with detected extensions go to `data/approved/`.

## Key conventions

### Embeddings live on disk, not inline

Stage 2 records reference embeddings by `image_id`. Inline embeddings in JSONL are a Stage 1 artifact — always extract to `.npy` and remove from the record during migration.

### stderr for logging, stdout for data

All progress/warning/diagnostic output goes to `stderr` via `eprint()`. Structured data output (JSONL, summaries) goes to `stdout` or files.

### Graceful degradation over hard failures

Functions return `None` on error rather than raising exceptions. The caller checks `if result is not None:` and continues. Individual image failures don't halt the pipeline — they're logged as warnings and skipped.

### DINOv3 specifics

- Uses RoPE positional embeddings with variable-length sequences
- Patch count = `(H÷16) × (W÷16)` — varies per aspect bucket (typically 3600–5500)
- Token sequence: `[CLS, patch_1, ..., patch_N, reg_1, ..., reg_4]` — exclude CLS (index 0) and 4 register tokens (tail)
- Single forward pass for both CLS + patches via `compute_dinov3_both()`

### Pose keypoint normalization

Coordinates are normalized to `[-1, 1]` relative to bucket dimensions (not original image dimensions), so the downstream model learns scale-invariant geometry.

### Constants

```python
NUM_POSE_KEYPOINTS = 133       # COCO-WholeBody
DINO_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
T5_MODEL_ID   = "t5-large"
```

Caption generation uses a local Ollama server (Gemma3:27b), not the HuggingFace API.

## Dependencies

Core dependencies (see `requirements-approved-image-embeddings.txt`):

- Python 3.13+
- `transformers>=4.54.0` (DINOv3 support)
- `torch` (install separately for ROCm/CUDA)
- `pillow`, `numpy`, `requests`
- `onnxruntime>=1.16.0`, `opencv-python>=4.5.0` (DWPose)
