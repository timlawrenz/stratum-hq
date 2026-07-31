# Plan: Add `auraface_lda` as a full member of the stratum-hq data family

**Author:** Hermes Agent · **Created:** 2026-07-02 · **Status:** draft

---

## 1. What is AuraFace-LDA?

AuraFace-LDA is a **64-dimensional identity vector** that compresses the 512-d AuraFace face-recognition embedding into a compact, discriminative subspace. It is the eidolon project's identity representation — `z_g` handles pose/expression, AuraFace-LDA handles *who the person is*.

Key properties:
- **64-d** float32, ~256 bytes per image
- **99.5% of full 512-d AuraFace verification power** (AUC within 0.005)
- **Cross-shoot verified**: R@1 = 0.842 on held-out photo shoots (real AuraFace → LDA → kNN retrieval)
- **Denoising effect**: LDA-64 reconstruction actually beats raw 512-d (AUC 0.9998 vs 0.9989), because the projection zeroes out ~448 dimensions of within-persona noise
- **Two-stage deterministic post-processing** (numpy-only) after a neural extraction step

---

## 2. How AuraFace-LDA is calculated (eidolon pipeline)

The pipeline has three stages — one neural, two deterministic:

### Stage 1: AuraFace extraction (neural, GPU-accelerated)

```
Raw image (face crop, 512×512)
  → InsightFace AuraFace model (ONNX, CUDAExecutionProvider)
  → faces[0].normed_embedding
  → 512-d float32, L2-normalized to unit hypersphere
```

**Model**: AuraFace via `insightface.app.FaceAnalysis(name='auraface')`. The model performs face detection + alignment + embedding extraction in a single call. Available through the InsightFace package.

**Dependencies**: `insightface`, `onnxruntime-gpu` (or `onnxruntime` for CPU), `opencv-python`

**Performance**: ~0.05s/image on GPU (NVIDIA RTX), ~0.3-0.5s/image on CPU

### Stage 2: Nuisance removal (numpy-only, deterministic)

Two directions are projected out of the raw 512-d vector:

1. **PC1 (domain axis)**: The largest PCA direction on 140,217 pooled FFHQ + proprietary vectors separates FFHQ from the proprietary dataset's capture style. R² ≈ 0 at identity level, but it's the dominant variance direction and a domain confound.

2. **Yaw direction**: A single direction that encodes ~41% of head-pose (horizontal) variance as measured by held-out ridge regression against DWPose yaw. Removing it costs zero identity discrimination (ΔAUC < 0.0001).

```python
def clean_auraface(v):
    vc = v - mu                            # center
    vc = vc - outer(vc @ pc1, pc1)        # remove domain axis
    vc = vc - outer(vc @ yaw, yaw)        # remove yaw component
    return vc / norm(vc)                   # renormalize to unit hypersphere
```

**Reference artifact**: `auraface_preprocess.npz` (~13 KB)

| Key | Shape | Description |
|-----|-------|-------------|
| `pooled_mean` | `(512,)` | Global mean of pooled vectors |
| `pc1_direction` | `(512,)` | PC1 unit vector (domain artifact) |
| `yaw_direction` | `(512,)` | Yaw direction unit vector |

### Stage 3: LDA projection (numpy-only, deterministic)

The cleaned 512-d vector is projected onto a supervised LDA basis fit on 259 train personas:

```python
def project_to_lda(v_clean):
    vc = v_clean - mu          # center using LDA training mean
    return vc @ W              # project onto 64 discriminative axes
```

**Reference artifact**: `auraface_lda.npz` (~262 KB)

| Key | Shape | Description |
|-----|-------|-------------|
| `lda_basis` | `(512, 64)` | LDA projection matrix (discriminative directions) |
| `lda_eigenvalues` | `(64,)` | Eigenvalues (discriminative power per axis) |
| `pooled_mean` | `(512,)` | Training mean (centered, PC1-removed) |
| `n_components` | scalar | = 64 |

**Fit details**:
- **Corpus**: 140,217 AuraFace vectors from FFHQ + proprietary multi-shoot dataset
- **Train personas**: 259, **held-out personas**: 64
- **Fit time**: minutes on CPU (sklearn LDA)

### Why 64 dimensions?

| LDA dims | Verification AUC | % of full 512-d |
|----------|-----------------|-----------------|
| 512 (raw, cleaned) | 0.969 | 100% |
| 80 | 0.965 | 99.6% |
| **64** | ~0.964 | **99.5%** |
| 40 | 0.956 | 98.7% |
| 20 | 0.934 | 96.4% |

64 was chosen because: near-lossless (ceiling 0.9998), compact enough for text-to-identity Prior training, power-of-2 dimension friendly to neural architectures, and matched z_g's 50-d convention for balanced downstream weights.

### Ceiling test (reconstruct → verify)

| Representation | AUC vs raw AuraFace |
|---|---|
| raw vs raw | 0.9989 |
| cleaned vs raw | 0.9989 |
| **GT LDA-64 → 512-d vs raw** | **0.9998** |
| GT LDA-32 → 512-d | 0.9765 |
| GT LDA-16 → 512-d | 0.8658 |

The LDA-64 reconstruction *beats* raw 512-d — the projection acts as a denoising step.

---

## 3. Face-only limitation (same as z_g)

**AuraFace requires a face in the image.** The model detects faces, aligns them, and extracts an embedding. On images without detectable faces (full-body shots where the face is too small, occluded, or absent), AuraFace returns zero faces and no embedding can be extracted.

This means AuraFace-LDA, like z_g, is a **face-only artifact**:
- **Works for**: FFHQ (all face crops), face-crop datasets, portrait photography
- **Fails for**: Full-body images where face detection misses, non-human images, landscapes
- **Edge cases**: ~40/70000 FFHQ images fail AuraFace detection (the `AF_MISSING` list in `fill_stratum.py`)

The same caveat that deferred z_g applies here.

---

## 4. Proposed integration into stratum-hq

### 4.1 New artifact

| Artifact | Shape | Dtype | Description |
|----------|-------|-------|-------------|
| `auraface_lda.npy` | `(64,)` | float32 | Compressed identity vector (AuraFace → clean → LDA-64) |

Optionally, the intermediate raw AuraFace could also be stored:

| Artifact | Shape | Dtype | Description |
|----------|-------|-------|-------------|
| `auraface_raw.npy` | `(512,)` | float32 | Raw AuraFace embedding (pre-cleaning, pre-LDA) — larger, but reversible |

For now, just `auraface_lda.npy` is the primary target — it's the compact identity vector used by eidolon's Prior training and identity retrieval.

### 4.2 New pass: `auraface`

Three sub-stages, but presented as a single pass:
1. **AuraFace detection + embedding** (requires insightface + onnxruntime + GPU)
2. **Clean** (numpy-only, uses `auraface_preprocess.npz`)
3. **LDA project** (numpy-only, uses `auraface_lda.npz`)

**Prerequisite**: None (standalone — doesn't depend on any existing stratum pass)

**Behavior**: If no face is detected, the artifact is not written (graceful skip). If the artifact already exists, skip (idempotent).

**CLI**: `stratum process ... --passes auraface` (opt-in, not in `all` by default — consistent with the face-only limitation)

### 4.3 New dependency: InsightFace

This is the key difference from z_g: **AuraFace-LDA requires a deep learning model** (InsightFace + ONNX Runtime), whereas z_g was pure numpy.

**Installation**:
```bash
pip install insightface onnxruntime-gpu opencv-python
```

Or for CPU-only:
```bash
pip install insightface onnxruntime opencv-python
```

The AuraFace model is auto-downloaded by InsightFace on first use (similar to how stratum's DWPose ONNX models are auto-downloaded from HuggingFace). The model cache lives at `~/.insightface/models/auraface/`.

**Model size**: ~200 MB for the AuraFace ONNX model. InsightFace downloads it automatically from its model zoo.

### 4.4 Files to create

```
src/stratum/pipeline/auraface.py              ← AuraFace extraction + clean + LDA pipeline pass
src/stratum/geometry/auraface_preprocessing.py ← clean_auraface(), project_to_lda() (numpy-only)
src/stratum/geometry/auraface_preprocess.npz   ← bundled reference (~13 KB)
src/stratum/geometry/auraface_lda.npz          ← bundled reference (~262 KB)
```

The preprocessing module can live alongside the z_g geometry module in `src/stratum/geometry/` or in its own subpackage. Since it's independent of the z_g geometry code, a separate module is cleaner.

### 4.5 Files to modify

| File | Changes |
|------|---------|
| `src/stratum/config.py` | Add `AURAFACE_LDA_FILE = "auraface_lda.npy"` |
| `src/stratum/cli.py` | Add `"auraface"` to `ALL_PASSES` (exclude from `DEFAULT_PASSES` — opt-in due to face-only limitation and heavy dependency) |
| `src/stratum/pipeline/__init__.py` | Register `auraface` pass, load InsightFace app once, load preprocess + LDA references |
| `src/stratum/discovery.py` | Add `"auraface_lda": AURAFACE_LDA_FILE` to `ARTIFACT_FILES` |
| `src/stratum/verify.py` | Add `auraface_lda` shape check `(64,)` + dtype `np.float32` |
| `src/stratum/publish.py` | Add `"auraface": [AURAFACE_LDA_FILE]` to `LAYER_ARTIFACTS` |
| `README.md` | Document `auraface_lda.npy` in artifacts table, note face-only limitation |
| `pyproject.toml` / `setup.py` | Add `[auraface]` extras require |

### 4.6 Pipeline pass implementation sketch

```python
# src/stratum/pipeline/auraface.py

import numpy as np
from pathlib import Path
from stratum.config import AURAFACE_LDA_FILE
from stratum.geometry.auraface_preprocessing import clean_auraface, project_to_lda


def process(image_path: Path, output_dir: Path, auraface_app,
            preprocess_ref: dict, lda_ref: dict) -> bool:
    """Extract AuraFace embedding, clean, and project to LDA-64."""

    out_path = output_dir / AURAFACE_LDA_FILE
    if out_path.exists():
        return False  # already done

    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    faces = auraface_app.get(img)
    if len(faces) == 0:
        return False  # no face detected

    # Stage 1: Raw 512-d AuraFace embedding
    raw_512 = faces[0].normed_embedding.astype(np.float64)

    # Stage 2: Clean (remove PC1 + yaw)
    cleaned = clean_auraface(raw_512, ref=preprocess_ref)

    # Stage 3: Project to LDA-64
    lda_64 = project_to_lda(cleaned, ref=lda_ref)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_path, lda_64.astype(np.float32))
    return True
```

The `clean_auraface` and `project_to_lda` functions need to be refactored to accept the reference dicts as parameters (currently they load from hardcoded paths via module-level globals).

### 4.7 Encoder loading in orchestrator

```python
if run_auraface:
    from insightface.app import FaceAnalysis
    from stratum.geometry.auraface_preprocessing import load_preprocess_ref, load_lda_ref

    auraface_app = FaceAnalysis(
        name='auraface',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    auraface_app.prepare(ctx_id=0, det_size=(512, 512))
    auraface_preprocess_ref = load_preprocess_ref()
    auraface_lda_ref = load_lda_ref()
```

### 4.8 Bundling reference artifacts

The two `.npz` files need to ship with the package. Options:
1. **Package data**: Include via `pyproject.toml` `[tool.setuptools.package-data]` or `MANIFEST.in`
2. **HuggingFace**: Host on HuggingFace and download on first use (like DWPose models), but overkill for ~275 KB
3. **Inline in package**: Use `importlib.resources` to find them relative to the module

Package data is simplest — they're tiny files.

---

## 5. Comparison: z_g vs AuraFace-LDA for stratum

| Property | z_g | AuraFace-LDA |
|----------|-----|--------------|
| **Dimension** | 50 | 64 |
| **Per-image size** | ~200 bytes | ~256 bytes |
| **Reference artifacts** | 35 KB (1 file) | 275 KB (2 files) |
| **What it encodes** | Pose + expression geometry | Identity (who the person is) |
| **Input** | DWPose face keypoints [23:91] | Raw face image (via AuraFace) |
| **Dependencies** | numpy only | insightface, onnxruntime, opencv |
| **GPU required** | No | Recommended (CPU works but slow) |
| **Face-only?** | Yes (68 face keypoints) | Yes (AuraFace face detection) |
| **Inference time** | ~1 ms | ~50 ms (GPU) / ~300 ms (CPU) |
| **Fails on FFHQ** | 138/70000 (DWPose) | 40/70000 (AuraFace detection) |
| **Cross-shoot validated?** | No (not identity) | Yes (R@1 = 0.842) |
| **Fit corpus** | 69,851 FFHQ faces | 140,217 FFHQ + proprietary |
| **License concern** | 3DDFA template (academic) | AuraFace model (Apache 2.0 via InsightFace), LDA fit on mix of FFHQ (CC BY-NC-SA) + proprietary data |

---

## 6. Recommendation

**Same verdict as z_g: face-only, defer until stratum has a face-crop mode or face-only dataset variant.**

AuraFace-LDA has additional concerns beyond the face-only issue:

1. **Heavy dependency**: InsightFace + ONNX Runtime is a significant addition to stratum's dependency footprint, unlike z_g's zero-dependency numpy approach.

2. **GPU requirement for practical use**: CPU extraction at ~0.3-0.5s/image is viable for small datasets but painful at FFHQ scale (70k images × 0.3s = ~6 hours CPU).

3. **Proprietary data in the LDA fit**: The LDA basis was trained on a mix of FFHQ (CC BY-NC-SA) and proprietary multi-shoot data. The basis *itself* is just a matrix of numbers (not copyrightable), but the provenance of the fit data should be documented.

4. **Model download**: AuraFace auto-downloads ~200 MB on first use — stratum's current models (DWPose, Sapiens) are already auto-downloaded from HuggingFace, so this pattern is established.

If/when stratum adds a face-crop mode, both `z_g` and `auraface_lda` should be added together as a face-modality pass group (`--passes face` = `zg,auraface`), giving users a complete 50-d pose + 64-d identity vector pair per face image.

---

## 7. If proceeding anyway (FFHQ-specific deployment)

For the specific case of **stratum-ffhq** (where all images are face crops), the pass would work reliably and could ship as:

```bash
pip install -e ".[auraface]"            # adds insightface + onnxruntime

stratum process ./ffhq/ --output ./stratum-ffhq/ --passes auraface
stratum publish ./stratum-ffhq/ --hub-repo nousr/stratum-ffhq --layers auraface
```

The reference artifacts would ship with stratum (275 KB), and InsightFace would auto-download the AuraFace model (~200 MB) on first run. The 40 detection failures in FFHQ are a known, acceptable loss rate (99.94% coverage).

### Implementation differences from full-body case

For FFHQ specifically, the AuraFace pass could be even simpler — FFHQ images are already 1024×1024 face-aligned crops, so:
- The InsightFace `det_size=(512, 512)` setting would resize to 512×512 for embedding extraction
- Face detection is nearly guaranteed (FFHQ images contain clearly visible faces)
- No need to handle full-body edge cases

But this would create a pass that only works on FFHQ-like data, not general stratum images. The current stratum design philosophy is dataset-agnostic — every pass should work on any image directory. Adding a face-only pass breaks that contract unless it's clearly documented as opt-in and face-specific.
