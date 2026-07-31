# Plan: Add `z_g` as a full member of the stratum-hq data family

**Author:** Hermes Agent · **Created:** 2026-07-02 · **Status:** draft

---

## 1. What is `z_g`?

`z_g` is a **50-dimensional whitened geometry vector** that encodes facial shape — 3D-head-pose-invariant, scale-invariant, translation-invariant. It is computed from the 68 face keypoints already present in stratum's DWPose output (`pose.npy`, indices 23–90).

Key properties:
- **Deterministic**: no neural model, purely linear algebra (PCA + Procrustes alignment + orthographic-PnP frontalization)
- **Tiny**: `(50,)` float32 ≈ 200 bytes per image; encoder fits within ~35 KB as an `.npz` file
- **Rich signal**: captures identity geometry (jaw shape, nose bridge, eye spacing, etc.) decoupled from head pose
- **Provenance**: fit on 70k FFHQ faces, validated via identity-verification benchmarks in eidolon

---

## 2. How `z_g` is calculated (eidolon pipeline)

The full encoding pipeline, implemented in `eidolon/experiments/geometry_pca/`:

```
pose.npy (133,3) 
  → slice face keypoints [23:91] → (68,2)
  → center_and_scale (centroid → origin, unit Frobenius norm)
  → 3D frontalization via orthographic-PnP against canonical 68-pt 3D template
    (removes out-of-plane yaw/pitch; nose depth "lifts" profile faces)
  → light 2D GPA alignment to frozen mean shape (removes residual rotation/scale)
  → flatten to 136-d, center with PCA mean, project onto 50 PC axes
  → whiten: (score - μ) / σ per component
  → output: z_g (50,) float32
```

### The frozen encoder (`encoder_production.npz`)

Pre-fit on ~70k FFHQ faces. Contents:

| Key | Shape | Description |
|-----|-------|-------------|
| `components` | `(50, 136)` | PCA component matrix |
| `pca_mean` | `(136,)` | Training mean vector |
| `whiten_mu` | `(50,)` | Per-component mean of scores |
| `whiten_sigma` | `(50,)` | Per-component std of scores |
| `gpa_mean` | `(68, 2)` | GPA reference mean shape |
| `canonical_template` | `(68, 3)` | Canonical 3D face (300W/iBUG layout) |

### Canonical 3D template

Hardcoded 68-point mean face (3DDFA/300W-LP derived), right-handed frame: +X = subject's left, +Y = up, +Z = forward toward camera. Encodes real facial depth — nose tip is most forward, eye sockets set back, jaw wraps around.

### Dependencies for inference

Only **numpy** required — no PyTorch, no ONNX, no deep learning. The frontalization step (`pose_normalize.py`) uses:
- `np.linalg.lstsq` for the orthographic-PnP rotation estimate
- `np.linalg.svd` for SO(3) projection of the rotation matrix
- Basic vector math (cross product, Gram-Schmidt, norms)

The GPA alignment (`gpa.py`) uses:
- SVD-based Procrustes rotation optimization
- Simple centering + Frobenius norm scaling

### Edge cases

- **DWPose failure** (all-zero keypoints): cannot extract z_g — 138/70000 FFHQ images
- **Degenerate z_g**: vectors with L2 norm > 25 are rejected (wild PCA projections from bad face detections) — the PosePrior harness uses this cutoff

---

## 3. Proposed integration into stratum-hq

### 3.1 New artifact

| Artifact | Shape | Dtype | Description |
|----------|-------|-------|-------------|
| `zg.npy` | `(50,)` | float32 | Whitened facial geometry vector (pose-invariant, scale-invariant) |

Placed in the per-image directory alongside the existing stratum artifacts:

```
source/ffhq/00001.png  →  dataset/ffhq/00001/
                             ├── metadata.json
                             ├── pose.npy       ← prerequisite
                             ├── zg.npy         ← NEW
                             └── ...
```

### 3.2 New pass: `zg`

- **Prerequisite**: `pose.npy` must exist (depends on the `pose` pass)
- **Behavior**: if `pose.npy` is missing or all-zero, `zg.npy` is skipped (not written)
- **Idempotent**: if `zg.npy` already exists, skip
- **CLI**: `stratum process ... --passes zg` or `--passes all` (included in `all`)

The encoder file will be bundled with the package as a resource file, loaded once at process start.

### 3.3 CLI changes

```bash
stratum process ./images/ --output ./dataset/ --passes all           # includes zg
stratum process ./images/ --output ./dataset/ --passes zg            # zg only
stratum publish ./dataset/ --hub-repo user/stratum-ffhq --layers zg  # publish zg
stratum status ./dataset/                                            # shows zg count
stratum verify ./dataset/                                            # validates zg shape/dtype
```

### 3.4 Files to create

```
src/stratum/geometry/                    ← new package (self-contained geometry primitives)
    __init__.py                          ← public API: encode_zg()
    canonical_face.py                    ← canonical 3D template (68,3) + canonical_template()
    gpa.py                               ← center_and_scale(), align_single(), get_rotation_matrix()
    pose_normalize.py                    ← estimate_rotation(), frontalize()
    encode.py                            ← encode_pose() / encode_zg()
    encoder_production.npz               ← bundled frozen encoder (~35 KB)

src/stratum/pipeline/zg.py              ← the zg pipeline pass (process() function)
```

These are extracted from eidolon's `geometry_pca/` package, simplified to remove eidolon-specific imports (no torch, no sklearn dependency for inference — sklearn was only needed for fitting).

### 3.5 Files to modify

| File | Changes |
|------|---------|
| `src/stratum/config.py` | Add `ZG_FILE = "zg.npy"` |
| `src/stratum/cli.py` | Add `"zg"` to `DEFAULT_PASSES` and `ALL_PASSES` |
| `src/stratum/pipeline/__init__.py` | Register the `zg` pass in `run_passes()` — load encoder once, run zg after pose check |
| `src/stratum/discovery.py` | Add `"zg": ZG_FILE` to `ARTIFACT_FILES` |
| `src/stratum/verify.py` | Add `zg` shape check `(50,)` + dtype `np.float32` to `verify_image_dir()`, `ARTIFACT_EXPECTED_DTYPE`, `ARTIFACT_FILE_MAP` |
| `src/stratum/publish.py` | Add `"zg": [ZG_FILE]` to `LAYER_ARTIFACTS` |
| `README.md` | Document `zg.npy` in the artifacts table, add `zg` row to pipeline passes table |

### 3.6 Implementation notes

#### zg pipeline pass (`src/stratum/pipeline/zg.py`)

```python
def process(image_path, output_dir, encoder, aspect_bucket=None) -> bool:
    pose_path = output_dir / POSE_FILE
    if not pose_path.exists():
        return False  # skipped — no pose dependency

    pose = np.load(pose_path)  # (133, 3) float16
    if pose.shape != (133, 3) or (pose[:, :2] == 0).all():
        return False  # DWPose failure

    face_2d = pose[23:91, :2].astype(np.float32)  # 68 face keypoints

    z_g = encode_zg(face_2d, encoder)

    np.save(output_dir / ZG_FILE, z_g.astype(np.float32))
    return True
```

The `encode_zg` function in `src/stratum/geometry/encode.py` mirrors `zg_inference.py`:

```python
def encode_zg(face_2d: np.ndarray, encoder: dict) -> np.ndarray:
    """Encode (68,2) face keypoints → (50,) whitened z_g vector."""
    tpl = encoder["canonical_template"].copy()
    tpl[:, 1] *= -1  # Y-flip: +Y up → +Y down (image coordinates)

    centered = center_and_scale(face_2d)
    frontal = frontalize(tpl, centered)
    aligned = align_single(frontal, encoder["gpa_mean"]).reshape(-1)
    raw = (aligned - encoder["pca_mean"]) @ encoder["components"].T
    return ((raw - encoder["whiten_mu"]) / encoder["whiten_sigma"]).astype(np.float32)
```

#### Encoder loading

In `run_passes()`, the encoder is loaded once if the `zg` pass is requested:

```python
if run_zg:
    from stratum.geometry import load_encoder
    zg_encoder = load_encoder()  # loads bundled encoder_production.npz
```

#### Package data

The `.npz` file is bundled via `pyproject.toml` and/or `MANIFEST.in`, available at runtime via `importlib.resources`:

```python
def load_encoder() -> dict:
    import importlib.resources
    ref = importlib.resources.files("stratum") / "geometry" / "encoder_production.npz"
    with importlib.resources.as_file(ref) as path:
        data = np.load(path)
    return {k: data[k] for k in data.files}
```

Or simply packaged relative to the module:

```python
_ENCODER_PATH = Path(__file__).parent / "encoder_production.npz"
def load_encoder() -> dict:
    data = np.load(_ENCODER_PATH)
    return {k: data[k] for k in data.files}
```

### 3.7 License & provenance

The canonical 3D template originates from 3DDFA/300W-LP (academic use), and the encoder was fit on FFHQ (CC BY-NC-SA 4.0). Both are compatible with stratum's current Apache 2.0 + CC-BY-NC component posture. The encoder is a purely mathematical transform of public data — it contains no original images.

### 3.8 Testing

New tests in `tests/` or inline:
- `test_zg_shape()`: verify output is `(50,)` float32
- `test_zg_reproducibility()`: same input → same z_g
- `test_zg_all_zero_pose()`: all-zero pose → gracefully skipped
- `test_zg_missing_pose()`: no pose.npy → gracefully skipped
- `test_zg_encoder_roundtrip()`: load encoder, verify keys present

---

## 4. Step-by-step implementation order

1. **Create `src/stratum/geometry/` package** with all modules, including the bundled `.npz` encoder file
2. **Create `src/stratum/pipeline/zg.py`** — the pipeline pass
3. **Update `config.py`** — add `ZG_FILE`
4. **Update `pipeline/__init__.py`** — register zg pass, load encoder, add dependency check
5. **Update `cli.py`** — add `zg` to pass lists
6. **Update `discovery.py`** — add to `ARTIFACT_FILES`
7. **Update `verify.py`** — add shape/dtype validation
8. **Update `publish.py`** — add to `LAYER_ARTIFACTS`
9. **Update `README.md`** — document the new artifact
10. **Write tests** — verify integration
11. **Run on a test set** — validate encoding quality against eidolon reference outputs

---

## 5. Impact summary

| Dimension | Impact |
|-----------|--------|
| **New dependency** | None (numpy only, already required) |
| **Encoder size** | ~35 KB shipped with package |
| **Per-image output** | ~200 bytes (50×float32) |
| **Runtime per image** | ~1 ms (pure numpy linear algebra) |
| **Disk for 70k images** | ~14 MB |
| **GPU required** | No |
| **Breaks existing data** | No (new file in per-image dir, additive) |
| **Backward compatible** | Yes — existing datasets unaffected |
