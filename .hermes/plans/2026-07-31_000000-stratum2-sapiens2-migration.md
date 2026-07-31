# Stratum2 — Sapiens2 Migration Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.
> **Goal:** Evolve stratum-hq into `stratum2` by replacing Sapiens1 TorchScript models with Sapiens2 specialized safetensors checkpoints, while preserving all non-Sapiens modalities (DINOv3, T5, Caption, Pixel).
> **Architecture:** New `src/stratum2/` package alongside `src/stratum/`. Requires the `sapiens2` repo as a dependency (`pip install -e .` from github.com/facebookresearch/sapiens2) for task-specific model loading via `init_model(config, checkpoint)`. The standalone `sapiens2.py` is backbone-only — task heads need the full repo's model registry and config system.
> **Tech Stack:** PyTorch ≥2.7, safetensors, `sapiens2` repo (git dependency), OpenCV, NumPy, Pillow, DETR (`transformers`) for pose person detection

---

## Deep Review Findings

### Sapiens2 Task Architecture (from source code audit)

**Model loading**: Task checkpoints are loaded via `init_model(config_file, checkpoint_path, device)` from `sapiens.dense.models` (seg, normal, pointmap, matting) or `sapiens.pose.models` (pose). This is an OpenMMLab-style config+checkpoint system. The standalone `sapiens2.py` is **backbone-only** and cannot load task heads.

**Inference pattern** (all dense tasks: seg, normal, pointmap, matting):
```python
model = init_model(args.config, args.checkpoint, device=args.device)
data = model.pipeline(dict(img=image))       # BGR→RGB, resize+pad to 1024×768
data = model.data_preprocessor(data)          # normalize, add batch dim
inputs = data["inputs"]
with torch.no_grad():
    result = model(inputs)
# Post-process: strip padding, resize to original dimensions
```

**Normalization** (from config files): mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], `bgr_to_rgb=True`. Input images are BGR (cv2.imread format). This is essentially the same as Sapiens1's normalization, not ImageNet.

**Preprocessing** is baked into `model.pipeline()` and `model.data_preprocessor()` — we call the model's own methods, not a standalone preprocess function.

### Dependency Chain (critical for execution order)

```
seg ──┬── normal   (normal needs seg foreground mask)
      ├── pointmap (pointmap needs seg foreground mask)
      └── (no dependency for matting or pose)

pose  ─── DETR person detector (facebook/detr-resnet-101-dc5)
matting ─── no dependencies (standalone)
```

**Stratum2 execution order**: seg → {normal, pointmap} (parallel after seg). Pose and matting can run anytime, including in parallel with seg.

### Output Formats

| Task | Model output | Stratum2 artifact | Shape | dtype |
|------|-------------|-------------------|-------|-------|
| Seg | `seg_logits` (1×29×H×W) → argmax | `seg2.npy` | (H, W) | uint8, class 0-28 |
| Normal | 1×3×H×W unit vectors | `normal2.npy` | (H, W, 3) | float16, L2-normalized |
| Pointmap | `(pointmap, scale)` → pointmap/scale | `pointmap.npy` | (H, W, 3) | float16, metric XYZ |
| Pose | K×308 keypoints + scores per person | `pose2.npy` | (N, 308, 3) | float32, (x, y, confidence) |
| Matting | 1×4×H×W [fgr_rgb(3), alpha(1)] | `matting.npy` | (H, W) | float16, alpha [0,1] |

### Pose Details
- **308 keypoints** (274 face + body + hands + feet), Sociopticon format
- **Top-down**: requires person detector → DETR `facebook/detr-resnet-101-dc5`
- Uses `UDPHeatmap` codec for keypoint decoding
- Output: per-person keypoints in image coordinates + confidence scores
- Needs config file: `configs/keypoints308/{dataset}/{model}.py`

---

## Settled Decisions

1. **Artifact naming convention** — stratum2 uses new filenames for overlapping modalities:

   | Modality | stratum (Sapiens1) | stratum2 (Sapiens2) |
   |----------|-------------------|---------------------|
   | Segmentation | `seg.npy` (28 cls) | `seg2.npy` (29 cls) |
   | Normals | `normal.npy` | `normal2.npy` |
   | Depth | `depth.npy` | *(not produced)* |
   | Pointmap | *(not produced)* | `pointmap.npy` |
   | Matting | *(not produced)* | `matting.npy` |
   | Pose | `pose.npy` (DWPose) | `pose2.npy` (Sapiens2 308 kp, supersedes DWPose) |
   | DINOv3 | `dinov3_cls.npy`, `dinov3_patches.npy` | same (shared) |
   | T5 | `t5_hidden.npy`, `t5_mask.npy` | same (shared) |
   | Caption | `caption.txt` | same (shared) |
   | Pixel | `pixel.npy` | same (shared) |
   | Metadata | `metadata.json` | same (shared) |

2. **Depth stays Sapiens1-only** — stratum2 produces `pointmap.npy` (3-channel XYZ). Depth can be derived as Z-channel.

3. **Sapiens2 pose supersedes DWPose** — stratum2 produces `pose2.npy` (308 kp). DWPose `pose.npy` is Sapiens1-only.

4. **Model size**: Single `SAPIENS2_SIZE` parameter across all tasks (0.4b, 0.8b, 1b, 5b). Dev: `0.4b`, Release: `5b`.

5. **Package name**: `stratum2` — new `src/stratum2/` package. Shares non-Sapiens infrastructure by importing from `stratum.*`.

6. **Sapiens2 repo required**: `pip install` from `github.com/facebookresearch/sapiens2` for `init_model()` and config system. Minimal inference-only configs bundled in `src/stratum2/configs/`.

---

## Task Breakdown

### Task 1: Add sapiens2 repo as dependency + verify

**Objective:** Ensure the sapiens2 repo can be imported and `init_model()` works.

**Step 1:** Clone + install sapiens2
```bash
git clone https://github.com/facebookresearch/sapiens2 /tmp/sapiens2-test
cd /tmp/sapiens2-test && pip install -e .
```

**Step 2:** Verify imports
```python
from sapiens.dense.models import init_model
from sapiens.backbones.standalone.sapiens2 import Sapiens2
print("OK")
```

**Step 3:** Add to `pyproject.toml` as optional dependency:
```toml
[project.optional-dependencies]
sapiens2 = ["sapiens2 @ git+https://github.com/facebookresearch/sapiens2.git"]
```

---

### Task 2: Create stratum2 package skeleton + config system

**Objective:** Bootstrap `src/stratum2/` with config, loader, and bundled minimal inference configs.

**Files:**
- Create: `src/stratum2/__init__.py`
- Create: `src/stratum2/config.py` — Sapiens2 constants + artifact filenames
- Create: `src/stratum2/loader.py` — download + load task models via `init_model()`
- Create: `src/stratum2/configs/` — minimal inference-only config `.py` files for each task/size

**Step 1: `config.py`**
```python
import os
from pathlib import Path

SAPIENS2_SIZE = os.environ.get("SAPIENS2_SIZE", "5b")

SAPIENS2_REPOS = {
    "seg": f"facebook/sapiens2-seg-{SAPIENS2_SIZE}",
    "normal": f"facebook/sapiens2-normal-{SAPIENS2_SIZE}",
    "pointmap": f"facebook/sapiens2-pointmap-{SAPIENS2_SIZE}",
    "pose": f"facebook/sapiens2-pose-{SAPIENS2_SIZE}",
    "matting": "facebook/sapiens2-matting-1b",  # only 1B available
}

SAPIENS2_FILENAMES = {
    "seg": f"sapiens2_{SAPIENS2_SIZE}_seg.safetensors",
    "normal": f"sapiens2_{SAPIENS2_SIZE}_normal.safetensors",
    "pointmap": f"sapiens2_{SAPIENS2_SIZE}_pointmap.safetensors",
    "pose": f"sapiens2_{SAPIENS2_SIZE}_pose.safetensors",
    "matting": "sapiens2_1b_matting.safetensors",
}

SAPIENS2_CACHE_DIR = Path(
    os.environ.get("SAPIENS2_CACHE_DIR", "/mnt/nas-ai-models/sapiens2")
)

# New artifact filenames
SEG2_FILE = "seg2.npy"
NORMAL2_FILE = "normal2.npy"
POINTMAP_FILE = "pointmap.npy"
POSE2_FILE = "pose2.npy"
MATTING_FILE = "matting.npy"

# Detector for pose
POSE_DETECTOR_REPO = "facebook/detr-resnet-101-dc5"
```

**Step 2: `loader.py`** — download safetensors + load via init_model
```python
def _download_checkpoint(repo_id: str, filename: str) -> Path:
    """Download safetensors checkpoint from HuggingFace."""
    ...

def load_sapiens2_model(task: str, device: str = "cpu"):
    """Download + load a Sapiens2 task model.
    
    Returns: (model, config_path) where model has .pipeline(), .data_preprocessor(), .__call__()
    """
    repo_id = SAPIENS2_REPOS[task]
    filename = SAPIENS2_FILENAMES[task]
    ckpt_path = _download_checkpoint(repo_id, filename)
    
    config_path = get_config_path(task, SAPIENS2_SIZE)
    
    from sapiens.dense.models import init_model
    model = init_model(str(config_path), str(ckpt_path), device=device)
    model.eval()
    return model
```

**Step 3: Bundle config files** — extract model definition + data_preprocessor from upstream configs, drop training-only sections. One config per task+size combo:
- `configs/seg_0.4b.py`, `configs/seg_1b.py`, `configs/seg_5b.py`
- `configs/normal_0.4b.py`, `configs/normal_1b.py`, `configs/normal_5b.py`
- `configs/pointmap_0.4b.py`, `configs/pointmap_1b.py`, `configs/pointmap_5b.py`
- `configs/pose_0.4b.py`, `configs/pose_1b.py`, `configs/pose_5b.py`
- `configs/matting_1b.py`

---

### Task 3: Implement Sapiens2 segmentation pipeline (seg2.npy)

**Objective:** seg pass using Sapiens2 29-class segmentation. Produces `seg2.npy`.

**Files:**
- Create: `src/stratum2/pipeline/__init__.py`
- Create: `src/stratum2/pipeline/seg.py`

**Step 1:** Write `process()`:
```python
def process(image_path, output_dir, seg_model, device, aspect_bucket=None):
    # Load image (BGR for Sapiens2)
    import cv2
    image = cv2.imread(str(image_path))  # BGR
    
    # Use model's pipeline for resize+pad+normalize
    data = seg_model.pipeline(dict(img=image))
    data = seg_model.data_preprocessor(data)
    inputs = data["inputs"].to(device)
    
    with torch.no_grad():
        seg_logits = seg_model(inputs)  # 1×29×H×W
    
    # Resize to original image dimensions
    seg_logits = F.interpolate(seg_logits, size=image.shape[:2], mode="bilinear")
    pred_labels = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    
    np.save(output_dir / SEG2_FILE, pred_labels)
    return True
```

**Step 2:** Run on one image, verify output shape (H×W) and class range (0-28).

---

### Task 4: Implement Sapiens2 normal pipeline (normal2.npy)

**Objective:** normal pass using Sapiens2 surface normals. DEPENDS on seg2.npy for foreground mask.

**Files:**
- Create: `src/stratum2/pipeline/normal.py`

**Step 1:** Write `process()` — requires `seg2.npy` to exist for foreground masking:
```python
def process(image_path, output_dir, normal_model, device, aspect_bucket=None):
    # Load seg mask
    seg_path = output_dir / SEG2_FILE
    if not seg_path.exists():
        return False
    seg = np.load(seg_path)
    fg_mask = seg > 0
    
    image = cv2.imread(str(image_path))
    data = normal_model.pipeline(dict(img=image))
    data = normal_model.data_preprocessor(data)
    inputs = data["inputs"].to(device)
    
    with torch.no_grad():
        normal = normal_model(inputs)
    
    # L2 normalize
    normal = normal / torch.norm(normal, dim=1, keepdim=True).clamp(min=1e-8)
    
    # Unpad + resize to original
    pad_left, pad_right, pad_top, pad_bottom = data["data_samples"]["meta"]["padding_size"]
    normal = normal[:, :, pad_top:inputs.shape[2]-pad_bottom, pad_left:inputs.shape[3]-pad_right]
    normal = F.interpolate(normal, size=image.shape[:2], mode="bilinear", align_corners=False)
    normal_map = normal.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float16)
    
    normal_map[~fg_mask] = 0.0
    np.save(output_dir / NORMAL2_FILE, normal_map)
    return True
```

---

### Task 5: Implement Sapiens2 pointmap pipeline (pointmap.npy)

**Objective:** pointmap pass producing per-pixel 3D XYZ coordinates. DEPENDS on seg2.npy for foreground mask.

**Files:**
- Create: `src/stratum2/pipeline/pointmap.py`

**Step 1:** Write `process()` — model returns `(pointmap, scale)`:
```python
def process(image_path, output_dir, pointmap_model, device, aspect_bucket=None):
    seg_path = output_dir / SEG2_FILE
    if not seg_path.exists():
        return False
    seg = np.load(seg_path)
    fg_mask = seg > 0
    
    image = cv2.imread(str(image_path))
    data = pointmap_model.pipeline(dict(img=image))
    data = pointmap_model.data_preprocessor(data)
    inputs = data["inputs"].to(device)
    
    with torch.no_grad():
        pointmap, scale = pointmap_model(inputs)
    
    # Convert to metric
    pointmap = pointmap / scale  # 1×3×H×W
    
    # Unpad + resize
    pad_left, pad_right, pad_top, pad_bottom = data["data_samples"]["meta"]["padding_size"]
    pointmap = pointmap[:, :, pad_top:inputs.shape[2]-pad_bottom, pad_left:inputs.shape[3]-pad_right]
    pointmap = F.interpolate(pointmap, size=image.shape[:2], mode="bilinear", align_corners=False)
    pointmap_np = pointmap.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float16)
    
    pointmap_np[~fg_mask] = 0.0
    np.save(output_dir / POINTMAP_FILE, pointmap_np)
    return True
```

---

### Task 6: Implement Sapiens2 pose pipeline (pose2.npy)

**Objective:** pose pass using Sapiens2 308-keypoint pose estimation. Requires DETR person detector. No seg dependency.

**Files:**
- Create: `src/stratum2/pipeline/pose.py`

**Step 1:** DETR person detector wrapper:
```python
from transformers import DetrForObjectDetection, DetrImageProcessor

def _detect_persons(image_bgr, detector_proc, detector_model, device, bbox_thr=0.3, nms_thr=0.3):
    """Detect persons using DETR, return bboxes [x1,y1,x2,y2]."""
    ...
```

**Step 2:** Write `process()` — uses `model.pipeline()`, `model.data_preprocessor()`, forward pass, UDPHeatmap decode:
```python
def process(image_path, output_dir, pose_model, detector, device, aspect_bucket=None):
    image = cv2.imread(str(image_path))
    bboxes = _detect_persons(image, *detector)
    
    all_keypoints = []
    for bbox in bboxes:
        data_info = dict(img=image, bbox=bbox[None], bbox_score=np.ones(1))
        data = pose_model.pipeline(data_info)
        data = pose_model.data_preprocessor(data)
        inputs = data["inputs"].to(device)
        
        with torch.no_grad():
            pred = pose_model(inputs)
        
        # Decode via UDPHeatmap codec
        keypoints, scores = pose_model.codec.decode(pred[0])
        # Transform from crop coords to image coords
        keypoints = keypoints / input_size * bbox_scale + bbox_center - 0.5 * bbox_scale
        all_keypoints.append(np.stack([keypoints[0], scores[0]], axis=-1))
    
    # Save as (N, 308, 3) array: (x, y, confidence) per keypoint per person
    result = np.array(all_keypoints, dtype=np.float32) if all_keypoints else np.zeros((0, 308, 3))
    np.save(output_dir / POSE2_FILE, result)
    return True
```

---

### Task 7: Implement Sapiens2 matting pipeline (matting.npy)

**Objective:** matting pass producing alpha matte. Standalone (no seg dependency).

**Files:**
- Create: `src/stratum2/pipeline/matting.py`

**Step 1:** Write `process()` — model returns 1×4×H×W [fgr_rgb(3), alpha(1)]:
```python
def process(image_path, output_dir, matting_model, device, aspect_bucket=None):
    image = cv2.imread(str(image_path))
    data = matting_model.pipeline(dict(img=image))
    data = matting_model.data_preprocessor(data)
    inputs = data["inputs"].to(device)
    
    with torch.no_grad():
        outputs = matting_model(inputs)  # 1×4×H×W
    
    outputs = F.interpolate(outputs, size=image.shape[:2], mode="bilinear", align_corners=False)
    alpha = outputs[0, 3].clamp(0, 1).cpu().numpy().astype(np.float16)
    
    np.save(output_dir / MATTING_FILE, alpha)
    return True
```

---

### Task 8: Implement stratum2 orchestrator (run_passes)

**Objective:** `run_passes()` that loads Sapiens2 models and orchestrates all passes with correct dependency order.

**Files:**
- Create: `src/stratum2/orchestrator.py`

**Step 1:** Write `run_passes()` — enforces execution order:
1. **Phase 1**: caption, dinov3, t5, pixel (non-Sapiens, no deps), **seg2**, **pose2**, **matting** — all parallel
2. **Phase 2**: normal2, pointmap — only after seg2 exists (need foreground mask)

**Step 2:** Reuse from `stratum.*`:
- `stratum.discovery` (image_id_from_path, output_dir_for_image)
- `stratum.pipeline.bucket` (assign_aspect_bucket, etc.)
- `stratum.pipeline.caption` → `caption.txt`
- `stratum.pipeline.dinov3` → `dinov3_cls.npy`, `dinov3_patches.npy`
- `stratum.pipeline.t5` → `t5_hidden.npy`, `t5_mask.npy`
- `stratum.pipeline.pixel` → `pixel.npy`
- `stratum.config` constants for shared artifacts

**Step 3:** Drop DWPose — stratum2 does NOT produce `pose.npy`, only `pose2.npy`.

---

### Task 9: Create stratum2 CLI

**Objective:** New `stratum2` command.

**Files:**
- Create: `src/stratum2/cli.py`
- Modify: `pyproject.toml` (add `stratum2` console script)

**Step 1:** CLI with `process` subcommand. Supported passes: `caption`, `dinov3`, `t5`, `pixel`, `seg2`, `normal2`, `pointmap`, `pose2`, `matting`.

**Step 2:** Register in pyproject.toml:
```toml
[project.scripts]
stratum2 = "stratum2.cli:main"
```

---

### Task 10: Integration test — single image

**Objective:** End-to-end test of stratum2 on one example image.

**Step 1:** Download 0.4B checkpoints for fast test:
```bash
for task in seg normal pointmap pose; do
  huggingface-cli download facebook/sapiens2-${task}-0.4b \
    --local-dir ~/.cache/sapiens2/
done
# matting is 1B only
huggingface-cli download facebook/sapiens2-matting-1b --local-dir ~/.cache/sapiens2/
```

**Step 2:** Run stratum2 on an example image:
```bash
SAPIENS2_SIZE=0.4b stratum2 process example-dataset/ --passes seg2,normal2,pointmap,pose2,matting
```

**Step 3:** Verify outputs exist with correct shapes.

---

### Task 11: Verify backward compatibility

**Objective:** Ensure existing stratum (Sapiens1) CLI and tests still work.

**Step 1:** Run existing tests:
```bash
python -m pytest tests/ -v
```

**Step 2:** Verify stratum1 pipeline still loads Sapiens1 torchscript models.

---

### Task 12: Document Sapiens2 dependency + setup

**Objective:** Update README with stratum2 setup instructions.

**Files:**
- Modify: `README.md`

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `src/stratum2/__init__.py` | Create | Package init |
| `src/stratum2/config.py` | Create | Sapiens2 constants, artifact filenames |
| `src/stratum2/loader.py` | Create | Checkpoint download + init_model loading |
| `src/stratum2/configs/*.py` | Create (12 files) | Minimal inference configs per task/size |
| `src/stratum2/pipeline/__init__.py` | Create | Pipeline package |
| `src/stratum2/pipeline/seg.py` | Create | seg2.npy (29 classes) |
| `src/stratum2/pipeline/normal.py` | Create | normal2.npy (needs seg2) |
| `src/stratum2/pipeline/pointmap.py` | Create | pointmap.npy (needs seg2) |
| `src/stratum2/pipeline/pose.py` | Create | pose2.npy (needs DETR) |
| `src/stratum2/pipeline/matting.py` | Create | matting.npy (standalone) |
| `src/stratum2/orchestrator.py` | Create | run_passes() with dependency ordering |
| `src/stratum2/cli.py` | Create | CLI entry point |
| `pyproject.toml` | Modify | Add stratum2 script + sapiens2 dependency |
| `README.md` | Modify | Stratum2 setup docs |

## Dependency Graph

```
Phase 1 (parallel):
  caption ─── Ollama (no GPU)
  dinov3  ─── DINOv3 model
  t5      ─── T5 tokenizer+encoder
  pixel   ─── CPU resize
  seg2    ─── Sapiens2 seg model      ──┐
  pose2   ─── DETR + Sapiens2 pose    ──┤ (no seg dependency)
  matting ─── Sapiens2 matting model  ──┘

Phase 2 (after seg2):
  normal2  ─── Sapiens2 normal + seg2 mask
  pointmap ─── Sapiens2 pointmap + seg2 mask
```

## Risks & Mitigations

1. **Config file maintenance**: Upstream configs may change with new sapiens2 releases. Mitigation: pin sapiens2 git ref in pyproject.toml; configs are minimal (model def only), easy to diff.

2. **DETR dependency for pose**: Adds `transformers` as a transitive dependency. Mitigation: make pose optional, load DETR lazily only when pose pass is requested.

3. **Sapiens2 normalization is BGR**: Unlike DINOv3/T5 which expect RGB. Mitigation: seg/normal/pointmap/matting/pose all use `cv2.imread` (BGR) as the model's pipeline expects BGR→RGB internally.

4. **Pointmap scale parameter**: `model(inputs)` returns `(pointmap, scale)` — must divide by scale. If this changes in future sapiens2 releases, pointmap values will be wrong. Mitigation: pin version, validate against known-good output.

5. **VRAM pressure**: Running multiple 5B models sequentially requires careful memory management. Mitigation: explicit `del model; torch.cuda.empty_cache()` between phase transitions.

## Verification Checklist

- [ ] `from sapiens.dense.models import init_model` works
- [ ] Minimal configs load without error via `init_model(config, checkpoint)`
- [ ] `seg2.npy` has correct shape (H×W) and class range (0-28)
- [ ] `normal2.npy` has correct shape (H×W×3) and unit vectors
- [ ] `pointmap.npy` has correct shape (H×W×3) with valid XYZ values
- [ ] `pose2.npy` has correct shape (N, 308, 3) with valid keypoints
- [ ] `matting.npy` has correct shape (H×W) with alpha [0,1]
- [ ] Existing stratum CLI and tests still pass
- [ ] All non-Sapiens passes (DINOv3, T5, Caption, Pixel) work in stratum2
- [ ] Dependency order enforced: normal2/pointmap skip if seg2 missing
