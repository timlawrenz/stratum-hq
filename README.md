# stratum-hq

A dataset-agnostic image enrichment pipeline. Given any directory of images, stratum produces per-image artifact directories containing multi-modal embeddings, captions, pose keypoints, body-part segmentation, depth maps, and surface normals — ready for publishing to HuggingFace or training diffusion models.

**stratum** uses Sapiens-1B TorchScript models (v1).  
**stratum2** uses Sapiens2 specialized safetensors checkpoints (v2) — higher quality, more modalities.

## Quick start

```bash
# Install with GPU support (torch must be installed separately for your platform)
pip install -e ".[all]"

# === stratum (Sapiens v1) ===
stratum process ./images/ --output ./dataset/ --passes all --device cuda

# === stratum2 (Sapiens v2) ===
# Requires the sapiens2 repo: pip install "sapiens @ git+https://github.com/facebookresearch/sapiens2.git"
SAPIENS2_SIZE=0.4b stratum2 process ./images/ --output ./dataset/ --passes all --device cuda

# Check progress (works for both)
stratum status ./dataset/
stratum2 status ./dataset/
```

## What it produces

### stratum (Sapiens v1)

Each source image gets its own output directory, mirroring the source structure:

```
source/ffhq/00001.png  →  dataset/ffhq/00001/
                             ├── metadata.json
                             ├── caption.txt
                             ├── dinov3_cls.npy
                             ├── dinov3_patches.npy
                             ├── t5_hidden.npy
                             ├── t5_mask.npy
                             ├── pose.npy          ← DWPose (133 kp)
                             ├── seg.npy           ← Sapiens-1B (28 cls)
                             ├── depth.npy         ← Sapiens-1B
                             └── normal.npy        ← Sapiens-1B
```

| Artifact | Shape | Dtype | Description |
|----------|-------|-------|-------------|
| `metadata.json` | — | JSON | Image dimensions, aspect bucket, source path |
| `caption.txt` | — | text | Dense objective description of the image |
| `dinov3_cls.npy` | `(1024,)` | float16 | DINOv3 CLS token — global style/composition |
| `dinov3_patches.npy` | `(N, 1024)` | float16 | DINOv3 spatial patch tokens (N varies by resolution) |
| `t5_hidden.npy` | `(512, 1024)` | float16 | T5-Large text encoder hidden states |
| `t5_mask.npy` | `(512,)` | uint8 | T5 attention mask (1=valid, 0=padding) |
| `pose.npy` | `(133, 3)` | float16 | DWPose whole-body keypoints: [x, y, confidence] in [-1, 1] |
| `seg.npy` | `(H, W)` | uint8 | Sapiens-1B 28-class body-part segmentation (class IDs 0–27) |
| `depth.npy` | `(H, W)` | float16 | Sapiens-1B relative depth, foreground-masked and normalised to [0, 1] |
| `normal.npy` | `(H, W, 3)` | float16 | Sapiens-1B per-pixel surface normals (XYZ), L2-normalised, foreground-masked |
| `pixel.npy` | `(3, H, W)` | float16 | Bucketed RGB crop in [0, 1] *(opt-in only)* |

### stratum2 (Sapiens v2)

Same directory structure; Sapiens2 artifacts use `2` suffix when they overlap with v1:

```
source/ffhq/00001.png  →  dataset/ffhq/00001/
                             ├── metadata.json
                             ├── caption.txt       ← shared (same as stratum)
                             ├── dinov3_cls.npy    ← shared
                             ├── dinov3_patches.npy ← shared
                             ├── t5_hidden.npy     ← shared
                             ├── t5_mask.npy       ← shared
                             ├── pose2.npy         ← Sapiens2 pose (308 kp)
                             ├── seg2.npy          ← Sapiens2 seg (29 cls)
                             ├── normal2.npy       ← Sapiens2 normals
                             ├── pointmap.npy      ← Sapiens2 3D XYZ (replaces depth)
                             └── matting.npy       ← Sapiens2 alpha matte
```

| Artifact | Shape | Dtype | Description |
|----------|-------|-------|-------------|
| `pose2.npy` | `(N, 308, 3)` | float32 | Sapiens2 308-keypoint pose (x, y, confidence) per person |
| `seg2.npy` | `(H, W)` | uint8 | Sapiens2 29-class body-part segmentation (class IDs 0–28) |
| `normal2.npy` | `(H, W, 3)` | float16 | Sapiens2 surface normals (XYZ), L2-normalised, foreground-masked |
| `pointmap.npy` | `(H, W, 3)` | float16 | Sapiens2 per-pixel 3D points in camera frame (metric XYZ) |
| `matting.npy` | `(H, W)` | float16 | Sapiens2 alpha matte in [0, 1] |
| *(shared)* | | | `metadata.json`, `caption.txt`, `dinov3_*.npy`, `t5_*.npy`, `pixel.npy` — same as stratum |

### Coexisting v1 + v2 artifacts

stratum and stratum2 produce different filenames for overlapping modalities, so you can run both pipelines on the same dataset:

```
dataset/ffhq/00001/
├── seg.npy         ← Sapiens-1B (28 cls)
├── seg2.npy        ← Sapiens2 (29 cls)
├── depth.npy       ← Sapiens-1B depth
├── pointmap.npy    ← Sapiens2 3D XYZ
├── pose.npy        ← DWPose (133 kp, from stratum only)
├── pose2.npy       ← Sapiens2 pose (308 kp, from stratum2 only)
├── normal.npy      ← Sapiens-1B normals
├── normal2.npy     ← Sapiens2 normals
├── caption.txt     ← shared
├── dinov3_cls.npy  ← shared
└── ...
```

## Installation

stratum requires Python 3.11+ and PyTorch (installed separately for your CUDA/ROCm platform).

```bash
# Core only (no GPU features)
pip install -e .

# With GPU-accelerated passes (DINOv3, T5)
pip install -e ".[gpu]"

# With pose estimation
pip install -e ".[pose]"

# With Sapiens segmentation, depth, and normals (v1)
pip install -e ".[sapiens]"

# With HuggingFace publishing
pip install -e ".[publish]"

# Everything (stratum v1 only)
pip install -e ".[all]"
```

Install PyTorch first following <https://pytorch.org/get-started/locally/>.

**For stratum2 (Sapiens v2)** — additionally install the sapiens2 repo:

```bash
pip install "sapiens @ git+https://github.com/facebookresearch/sapiens2.git"
```

This provides `sapiens.dense.models.init_model()` and `sapiens.pose.models.init_model()` used by stratum2's loader. The `sapiens` package includes all task config files — no bundling needed.

## Pipeline passes

### stratum (v1)

Each pass runs independently and is idempotent — if the output file already exists, the image is skipped.

| Pass | What it does | Artifacts | Requires |
|------|-------------|-----------|----------|
| `caption` | Generates a dense text description via Ollama | `caption.txt` | Ollama server |
| `dinov3` | Extracts DINOv3-ViT-L/16 CLS and patch embeddings | `dinov3_cls.npy`, `dinov3_patches.npy` | GPU + `[gpu]` |
| `t5` | Encodes caption text with T5-Large | `t5_hidden.npy`, `t5_mask.npy` | GPU + `[gpu]`, caption |
| `pose` | Extracts 133 whole-body keypoints via DWPose | `pose.npy` | `[pose]` |
| `seg` | 28-class body-part segmentation via Sapiens-1B | `seg.npy` | GPU + `[sapiens]` |
| `depth` | Relative depth estimation via Sapiens-1B | `depth.npy` | GPU + `[sapiens]`, seg |
| `normal` | Surface normal prediction via Sapiens-1B | `normal.npy` | GPU + `[sapiens]`, seg |
| `pixel` | Saves bucketed RGB crop as numpy array | `pixel.npy` | — |

`--passes all` runs caption, dinov3, t5, pose, seg, depth, and normal. The pixel pass is **opt-in**. Depth and normal depend on seg (for the foreground mask), similar to how t5 depends on caption.

### stratum2 (v2)

| Pass | What it does | Artifacts | Requires |
|------|-------------|-----------|----------|
| `caption` | Same as stratum — Ollama captioning | `caption.txt` | Ollama server |
| `dinov3` | Same as stratum — DINOv3 embeddings | `dinov3_cls.npy`, `dinov3_patches.npy` | GPU + `[gpu]` |
| `t5` | Same as stratum — T5 text encoding | `t5_hidden.npy`, `t5_mask.npy` | GPU + `[gpu]`, caption |
| `seg2` | 29-class body-part segmentation via Sapiens2 | `seg2.npy` | GPU + `sapiens` package |
| `pose2` | 308-keypoint pose via Sapiens2 + DETR detector | `pose2.npy` | GPU + `sapiens` + `transformers` |
| `matting` | Alpha matte extraction via Sapiens2 | `matting.npy` | GPU + `sapiens` package |
| `normal2` | Surface normals via Sapiens2 | `normal2.npy` | GPU + `sapiens`, seg2 |
| `pointmap` | 3D point cloud (XYZ) via Sapiens2 | `pointmap.npy` | GPU + `sapiens`, seg2 |
| `pixel` | Same as stratum — bucketed RGB crop | `pixel.npy` | — |

`--passes all` runs: caption, dinov3, t5, seg2, pose2, matting, normal2, pointmap. The `pixel` pass is **opt-in**.

**Dependency ordering**: stratum2 enforces two-phase execution:
- **Phase 1** (parallel): caption, dinov3, t5, pixel, seg2, pose2, matting
- **Phase 2** (after seg2): normal2, pointmap — both need the foreground mask from `seg2.npy`

You can also run passes as separate commands — each pass independently checks for its dependencies:
```bash
# Generate seg2 masks first
stratum2 process ./images/ --output ./dataset/ --passes seg2

# Then consume them
stratum2 process ./images/ --output ./dataset/ --passes normal2,pointmap
```

## CLI reference

### `stratum process` / `stratum2 process`

Generate dataset artifacts from a directory of images.

```bash
stratum process <input_dir> --output <output_dir> [options]
stratum2 process <input_dir> --output <output_dir> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--passes` | `all` | Comma-separated passes or `all` |
| `--device` | `auto` | Compute device: `auto`, `cpu`, `cuda`, `cuda:0`, etc. |
| `--shard N/M` | — | Process every M-th image starting at offset N |
| `--image-list` | — | File with explicit image paths (one per line) |
| `--ollama-url` | `http://localhost:11434/api/generate` | Ollama endpoint |
| `--ollama-model` | `gemma3:27b` | Ollama model for captioning |
| `--caption-max-tokens` | `500` | Maximum tokens per caption |
| `--progress-every` | `100` | Print progress every N images (0 to disable) |
| `--verbose` | off | Per-image logging |

**stratum2 additional parameters** (via environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `SAPIENS2_SIZE` | `5b` | Model variant: `0.4b`, `0.8b`, `1b`, `5b` |
| `SAPIENS2_CACHE_DIR` | `/mnt/nas-ai-models/sapiens2` | Checkpoint cache directory |

**stratum2 passes**: `caption`, `dinov3`, `t5`, `pixel`, `seg2`, `pose2`, `matting`, `normal2`, `pointmap`  
**stratum passes**: `caption`, `dinov3`, `t5`, `pixel`, `pose`, `seg`, `depth`, `normal`

### `stratum status` / `stratum2 status`

Report dataset completeness — counts artifacts per type.

```bash
stratum status <dataset_dir>
stratum2 status <dataset_dir>
```

Example output:

```
Total images: 70,000
  metadata:             70,000 / 70,000 (100.0%)
  caption:              70,000 / 70,000 (100.0%)
  dinov3_cls:           45,000 / 70,000 (64.3%)
  dinov3_patches:       45,000 / 70,000 (64.3%)
  t5_hidden:            10,000 / 70,000 (14.3%)
  pose2:                30,000 / 70,000 (42.9%)
  seg2:                 50,000 / 70,000 (71.4%)
  pointmap:             20,000 / 70,000 (28.6%)
```

### `stratum verify` / `stratum2 verify`

Check data integrity — validates shapes, dtypes, and consistency.

```bash
stratum verify <dataset_dir> [--fix]
stratum2 verify <dataset_dir> [--fix]
```

With `--fix`, corrupt files are deleted so they can be regenerated on the next `process` run.

### `stratum publish`

Publish layers to a HuggingFace dataset repository.

```bash
stratum publish <dataset_dir> --hub-repo <user/repo> --layers <layers> [options]
```

Uploads are atomic — each batch is a single HuggingFace commit, so a stalled upload leaves no partial state. Automatic retry with exponential backoff handles HTTP 429 rate limits.

Supports incremental publishing — add layers (width) or more images (depth) over time:

```bash
# First: publish captions for all 70k images
stratum publish ./dataset/ --hub-repo user/stratum-ffhq --layers caption

# Next: DINOv3 for first 10k
stratum publish ./dataset/ --hub-repo user/stratum-ffhq --layers dinov3 --limit 10000

# Then: DINOv3 for next 10k
stratum publish ./dataset/ --hub-repo user/stratum-ffhq --layers dinov3 --offset 10000 --limit 10000
```

Each publish updates a `manifest.json` tracking what's available and auto-generates a dataset card.

| Option | Default | Description |
|--------|---------|-------------|
| `--hub-repo` | *(required)* | HuggingFace repo (e.g., `user/stratum-ffhq`) |
| `--layers` | *(required)* | Comma-separated layers to publish (`caption,dinov3,t5,pose,seg,depth,normal`) |
| `--limit N` | — | Max images to publish |
| `--offset N` | `0` | Skip first N images |
| `--max-tar-mb N` | — | Split layer tars when they exceed N MB |
| `--max-files-per-tar N` | — | Split layer tars when they reach N images. Can be combined with `--max-tar-mb`; whichever limit is hit first triggers the split. Example: `--max-files-per-tar 10` |
| `--tmp-dir` | `~/.cache/stratum` | Directory for staging/temp files |
| `--license` | `cc-by-nc-sa-4.0` | SPDX license for the dataset card |
| `--attribution-file` | — | Markdown file with provenance text |
| `--verbose` | off | Log upload requests, responses, and file sizes |

### `stratum reconcile`

Rebuild the HuggingFace manifest from actual repo files. Fixes count drift caused by interrupted publishes.

```bash
stratum reconcile --hub-repo <user/repo> [--dry-run]
```

Lists files in the HuggingFace repo, parses range labels from the deterministic filenames (`data/00000-09999.parquet`, `t5/00000-00999.tar`), recomputes per-layer counts, and uploads a corrected manifest and dataset card.

| Option | Default | Description |
|--------|---------|-------------|
| `--hub-repo` | *(required)* | HuggingFace repo to reconcile |
| `--dry-run` | off | Print the corrected manifest without uploading |
| `--license` | `cc-by-nc-sa-4.0` | SPDX license for the dataset card |
| `--attribution-file` | — | Markdown file with provenance text |

## Multi-GPU parallelism

Sharding is per-pass — each GPU runs one pass on a deterministic subset of images. No coordination, locks, or shared state:

```bash
# 4 GPUs processing DINOv3 in parallel
stratum process ./images/ --output ./dataset/ --passes dinov3 --shard 0/4 --device cuda:0 &
stratum process ./images/ --output ./dataset/ --passes dinov3 --shard 1/4 --device cuda:1 &
stratum process ./images/ --output ./dataset/ --passes dinov3 --shard 2/4 --device cuda:2 &
stratum process ./images/ --output ./dataset/ --passes dinov3 --shard 3/4 --device cuda:3 &
wait

# Then T5 (needs captions to exist first)
stratum process ./images/ --output ./dataset/ --passes t5 --shard 0/4 --device cuda:0 &
stratum process ./images/ --output ./dataset/ --passes t5 --shard 1/4 --device cuda:1 &
stratum process ./images/ --output ./dataset/ --passes t5 --shard 2/4 --device cuda:2 &
stratum process ./images/ --output ./dataset/ --passes t5 --shard 3/4 --device cuda:3 &
wait
```

This works because each worker writes to different per-image directories — there is no central state file.

## Aspect ratio bucketing

Images are assigned to the closest aspect ratio bucket (~1 megapixel each, dimensions divisible by 64):

| Bucket | Ratio | Description |
|--------|-------|-------------|
| 1024×1024 | 1.00 | Square |
| 832×1216 | 0.68 | Portrait |
| 1216×832 | 1.46 | Landscape |
| 768×1280 | 0.60 | Tall portrait |
| 1280×768 | 1.67 | Wide landscape |
| 704×1344 | 0.52 | Very tall |
| 1344×704 | 1.91 | Very wide |

Images are resize-to-cover + center-crop to exact bucket dimensions. For single-resolution datasets (e.g., FFHQ's 1024×1024 images), all images map to the square bucket.

## Captioning

Captions are generated via an [Ollama](https://ollama.ai) server. The prompt produces dense, objective descriptions focused on visible attributes — pose, anatomy, clothing, lighting, composition — without subjective language or preambles.

Start an Ollama server with a vision-capable model:

```bash
ollama serve
ollama pull gemma3:27b
```

Then point stratum at it:

```bash
stratum process ./images/ --output ./dataset/ --passes caption \
    --ollama-url http://localhost:11434/api/generate \
    --ollama-model gemma3:27b
```

## Technical details

### DINOv3 extraction

- **Model**: `facebook/dinov3-vitl16-pretrain-lvd1689m` (ViT-L/16, 304M parameters)
- **CLS token**: 1024-dim global representation (style, composition)
- **Patch tokens**: `(H÷16) × (W÷16)` spatial tokens, each 1024-dim. Count varies by bucket (e.g., 1024×1024 → 4096 patches)
- All embeddings stored as float16 for storage efficiency
- Uses RoPE positional embeddings for variable-resolution support
- Preprocessing uses `do_center_crop=False` to preserve spatial alignment
- Single forward pass extracts both CLS and patches (`compute_dinov3_both`)
- Token layout: `[CLS, patch_1, ..., patch_N, reg_1, ..., reg_4]` — registers are excluded

### T5 text encoding

- **Model**: `t5-large` (248M parameters)
- 512-token context window (not CLIP's 77-token limit)
- Hidden states: `(512, 1024)` float16
- Attention mask: `(512,)` uint8 — 1 for valid tokens, 0 for padding
- Requires `caption.txt` to exist (run caption pass first)

### Pose estimation

- **Model**: DWPose (YOLOX-L detector + DWPose-L keypoint estimator, ONNX)
- 133 COCO-WholeBody keypoints: 17 body + 6 feet + 68 face + 42 hands
- Coordinates normalized to `[-1, 1]` relative to bucket dimensions
- Models auto-download from HuggingFace to `~/.cache/dwpose/`
- Standalone — requires only `onnxruntime` and `opencv-python`, no mmpose

### Sapiens segmentation, depth, and surface normals (v1)

- **Models**: Sapiens-1B TorchScript checkpoints from [facebook/sapiens](https://huggingface.co/facebook/sapiens) (1.17B parameters each)
- **License**: CC-BY-NC 4.0 (non-commercial)
- **Input resolution**: 1024×768 (H×W) — Sapiens' native resolution
- **Preprocessing**: Normalised with mean=[123.5, 116.5, 103.5], std=[58.5, 57.0, 57.5]
- **Segmentation**: 28 Goliath body-part classes (background, face/neck, hands, torso, clothing, etc.)
- **Depth**: Relative depth per pixel, normalised to [0, 1] over foreground region
- **Surface normals**: Per-pixel XYZ unit vectors, L2-normalised
- **Foreground masking**: Depth and normal outputs use the segmentation mask to zero out background
- **Output resolution**: Interpolated to aspect bucket dimensions (not 1024×768) for consistency with other artifacts
- Models auto-download from HuggingFace to `~/.cache/sapiens/`

### Sapiens2 segmentation, normals, pointmap, pose, matting (v2)

- **Models**: Sapiens2 safetensors checkpoints from [facebook/sapiens2](https://huggingface.co/facebook/sapiens2)
- **Architecture**: Specialized task heads on pretrained ViT backbones (0.4B–5B parameters). Model size controlled by `SAPIENS2_SIZE` env var.
- **License**: Sapiens2 License
- **Loading**: Uses the `sapiens` package's `init_model(config, checkpoint)` — requires `pip install "sapiens @ git+https://github.com/facebookresearch/sapiens2.git"`
- **Input**: 1024×768 (H×W), BGR format (`cv2.imread`). Preprocessing handled by `model.pipeline()` and `model.data_preprocessor()` — same normalization as v1, with internal BGR→RGB conversion.
- **Segmentation**: 29 classes (28 body parts + background): Face/Neck, Hair, Hands, Feet, Upper/Lower Arms, Upper/Lower Legs, Torso, Apparel, Eyeglass, Shoes, Socks, Teeth, Tongue, Lips, etc.
- **Pose**: 308-keypoint top-down estimation (274 face + body + hands + feet). Requires DETR ResNet-101 person detector (`facebook/detr-resnet-101-dc5`). Output: `(N, 308, 3)` — keypoint (x, y) + confidence per person.
- **Pointmap**: Per-pixel 3D XYZ points in camera frame. Model returns `(pointmap, scale)` tuple; metric coordinates = pointmap / scale. Supersedes v1 depth — depth can be derived as the Z channel.
- **Normals**: Per-pixel XYZ unit vectors, L2-normalised.
- **Matting**: Alpha matte (per-pixel opacity in [0, 1]). Only available at 1B size.
- **Foreground masking**: Normal2 and pointmap use seg2's foreground mask to zero out background. Matting and pose are standalone.
- Checkpoints download to `$SAPIENS2_CACHE_DIR` (default: `/mnt/nas-ai-models/sapiens2`).
- Config files are resolved from the installed `sapiens` package — no bundling needed.

### Per-image directories

The filesystem is the database. Completeness is determined by which files exist — there is no central registry or JSONL tracking file. This makes the pipeline:

- **Parallel-safe**: Multiple workers write to different directories
- **Resumable**: Re-running skips images that already have their artifacts
- **Inspectable**: Browse the output with standard file tools
- **Diffable**: Captions are plain text files, metadata is formatted JSON

## Running tests

```bash
PYTHONPATH=src python3 -m pytest tests/ -v
```

Or without pytest:

```bash
PYTHONPATH=src python3 -c "from tests.test_core import *; [t() for t in [
    test_compute_aspect_ratio, test_assign_aspect_bucket_square,
    test_parse_bucket_dims, test_load_bucketed_image,
    test_image_id_from_path, test_shard_image_list,
    test_discover_images, test_scan_dataset_status,
    test_resolve_passes_all, test_parse_shard,
]]"
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
