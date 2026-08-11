"""3D body-volume / bulk measurement via Multi-HMR + Anny whole-body mesh.

Arm #96. NEW MODEL CLASS (Anny SMPL-free body model + Multi-HMR regressor
trained on Anny-One/BEDLAM/ITW, naver/multi-hmr `multiHMR_672_L_anny`
checkpoint, released 2026-02-17; checkpoint license free — the Anny body-model
checkpoint license is CC-compatible, code Apache-2.0, MakeHuman assets CC0).
GENUINELY SMPL-FREE: the regressor outputs Anny-native meshes, so no
SMPL/SMPL-X registration assets are needed (verified locally — the Anny
branch loads no SMPL/SMPL-X files).

The model is the smoke-verified naver/multi-hmr Anny branch (staged locally on
owned hardware, `/mnt/nas-ai-models/research/stratum/body-volume/multi-hmr`,
patched to import a slim `slim_utils.py` instead of the pyrender rendering
chain — headless box, no OpenGL; tensor math is verbatim). It regresses, per
detected person, in camera space:
- v3d   : Anny mesh vertices (meters, camera space)
- j3d   : Anny bone positions
- shape : interpretable Anny phenotype coefficients in [0, 1], incl. age,
  gender, weight, height, muscle, proportions (WHO/Anny-calibrated).

Measurement semantics (owner directive — only SCALE-INVARIANT ratios verbalize):
- The verbalized body-volume band is calibrated from a SCALE-INVARIANT
  geometric proxy: total mesh volume / (body height)^3 (both in meters from
  the same mesh, so camera scale cancels). This is the standard shape
  normalisation and survives cross-picture comparison.
- The raw `shape` phenotype coefficients (weight/height/muscle/proportions)
  stay in the machine-readable `evidence_payload`, never in prose.
- Abstain (with a surfaced reason) on: no human detected, degenerate/implicit
  mesh, subject too small / too little coverage in seg2, or implausible
  output. Never fabricate a body-volume class.
- The corpus is curated to exactly one woman: the subject human is selected as
  the detected person closest to the seg2 subject-mask centroid (projected),
  so a reflection/background person never hijacks the measurement.

The exactly-one-subject invariant and honest abstention are enforced here;
detector disagreement remains a quality anomaly, never caption content.

Provenance: checkpoint sha256
9ad7497ee44c1e48ecca69dcf258d831e65edb806f4f42facf54d2f529e814d4
(pinned at staging 2026-08-10). Run on owned hardware only (local 4090 via the
GPU scheduler, or local CPU); no hosted third-party inference of the sensitive
corpus; no corpus write — computed in memory during the bounded run.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Mapping

import numpy as np

# DOME-29 background class is 0; subject is seg2 != 0.
MIN_SUBJECT_PX = 2000  # seg2 subject-union floor (mirror apparent-age body crop)

# Frozen Multi-HMR-Anny stack (staged locally 2026-08-10).
MULTI_HMR_DIR = "/mnt/nas-ai-models/research/stratum/body-volume/multi-hmr"
# Checkpoint (1.53 GB, sha256 pinned below).
CHECKPOINT = "/mnt/nas-ai-models/research/stratum/body-volume/multiHMR_672_L_anny.pt"
CHECKPOINT_SHA256 = "9ad7497ee44c1e48ecca69dcf258d831e65edb806f4f42facf54d2f529e814d4"

# Model training resolution (from checkpoint args: img_size 672).
IMG_SIZE = 672
# Detection threshold — CALIBRATED on the frozen-cohort probe 2026-08-10.
# The checkpoint/demo default is 0.3, which only recovers 14/24 frozen items
# (coverage below the pre-registered >=18/24 floor). Post-NMS subject-aligned
# sweep across the cohort: at det_thresh=0.1, 21/24 items (87.5%) have a
# subject-aligned person peak (3 honest non-detections: 13geqw0wbpj5 score
# 0.005, 07hyx5wjk5rc score 0.037, 1dkimn0kiybq score 0.070). 0.1 is a round,
# modest value above the noise floor; full score table in
# body-volume-calibration-probe.json (2026-08-10).
DET_THRESH = 0.1
# Non-max suppression kernel (checkpoint arg nms_kernel_size 3).
NMS_KERNEL = 3
# Camera field of view in degrees (demo default; K passed explicitly so the
# regressed intrinsics are overridden deterministically, matching the arm-#4
# fixed-view controls).
FOV = 60.0

# Body-volume bands — CALIBRATED from the frozen-cohort capability probe
# (2026-08-10, scripts/probe_body_volume.py). Detection at det_thresh 0.1:
# 21/24 plausible meshes (floor >=18/24 met); 3 honest non-detections; 5
# mesh-height-degenerate (partial-body close-ups) honestly abstained. On the
# 16 band-eligible items the normalized-volume distribution is bimodal (11
# items 0.014–0.024, then 0.042, 0.050, 0.062, 0.067, 0.099) — the natural cut
# at the gap gives slim < 0.03 / average < 0.055 / fuller >= 0.055:
# counts 11/2/3, max_share 0.6875 (under the 0.75 degeneracy gate).
VOLUME_BANDS: tuple[float, float] = (0.03, 0.055)
_BAND_LABELS = ("slim", "average", "fuller")

# Honest mesh plausibility gates (meters / normalized bulk). Calibrated from
# the frozen-cohort probe (2026-08-10): detected-people meshes span a
# human-plausible height and bulk; clear outliers (e.g. normalized_volume
# 0.363 on a doubled/uncropped mesh) must abstain rather than fabricate a
# body-volume class.
MIN_MESH_HEIGHT = 0.8
MAX_MESH_HEIGHT = 2.5
MIN_NORM_VOLUME = 0.005
MAX_NORM_VOLUME = 0.30


class BodyVolumeError(RuntimeError):
    pass


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise BodyVolumeError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise BodyVolumeError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise BodyVolumeError(f"seg2 must be integer class labels, got dtype {seg2.dtype}")


def validate_rgb_array(rgb: np.ndarray) -> None:
    if not isinstance(rgb, np.ndarray):
        raise BodyVolumeError("rgb must be a numpy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise BodyVolumeError(f"rgb must be (H, W, 3), got shape {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise BodyVolumeError(f"rgb must be uint8, got dtype {rgb.dtype}")


class _RunTime:
    """Lazy, process-wide Multi-HMR-Anny model (GPU if available)."""

    _model = None
    _device = None

    @classmethod
    def get(cls, checkpoint: str = CHECKPOINT):
        if cls._model is None:
            if MULTI_HMR_DIR not in sys.path:
                sys.path.insert(0, MULTI_HMR_DIR)
            import torch

            from multi_hmr_anny.multi_hmr import Multi_HMR as ModelAnny

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if not os.path.isfile(checkpoint):
                raise BodyVolumeError(f"checkpoint not found: {checkpoint}")
            ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
            args = ckpt.get("args")
            if args is None:
                raise BodyVolumeError("checkpoint missing args")
            kwargs = {k: v for k, v in vars(args).items()}
            model = ModelAnny(**kwargs).to(device)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            model.eval()
            cls._model = model
            cls._device = device
        return cls._model, cls._device

    @classmethod
    def reset(cls) -> None:
        cls._model = None
        cls._device = None


def _preprocess(rgb: np.ndarray, img_size: int = IMG_SIZE) -> tuple[np.ndarray, float]:
    """Resize contain + zero-pad to img_size x img_size; return normalized
    (1,3,S,S) float32 and the scale factor (padded_size / original_max_side)."""
    from PIL import Image, ImageOps

    img_h, img_w = rgb.shape[0], rgb.shape[1]
    pil = Image.fromarray(rgb)
    pil = ImageOps.contain(pil, (img_size, img_size))
    scale = img_size / max(img_h, img_w)
    arr = np.asarray(ImageOps.pad(pil, (img_size, img_size), color=(0, 0, 0)), dtype=np.float32)
    arr = arr / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    arr = np.transpose(arr, (2, 0, 1))
    arr = (arr - mean) / std
    return arr.astype(np.float32)[None], scale


def _camera_parameters(img_size: int, fov: float = FOV, device: str = "cpu"):
    """Identity-lens K with center principal point (demo default)."""
    import torch

    focal = (img_size / 2.0) / np.tan(np.deg2rad(fov) / 2.0)
    K = torch.eye(3)
    K[0, 0] = K[1, 1] = focal
    K[0, -1] = K[1, -1] = img_size / 2.0
    return K.unsqueeze(0).to(device)


def _subject_centroid(seg2: np.ndarray) -> tuple[float, float] | None:
    """Centroid (y, x) of the seg2 subject union, in ORIGINAL pixel space."""
    ys, xs = np.nonzero(seg2 != 0)
    if ys.size == 0:
        return None
    return (float(ys.mean()), float(xs.mean()))


def _project_subject_centroid(centroid_orig: tuple[float, float], scale: float,
                              img_h: int, img_w: int, img_size: int) -> tuple[float, float] | None:
    """Map an original-space centroid into the padded img_size square."""
    y_orig, x_orig = centroid_orig
    # contain keeps aspect; pad offset:
    contain_h = int(round(img_h * scale))
    contain_w = int(round(img_w * scale))
    pad_top = (img_size - contain_h) // 2
    pad_left = (img_size - contain_w) // 2
    y_pad = y_orig * scale + pad_top
    x_pad = x_orig * scale + pad_left
    return (y_pad, x_pad)


def _mesh_metrics(v3d: np.ndarray, faces_tri: np.ndarray) -> tuple[float | None, float | None]:
    """Return (volume / height^3, mesh height in meters) from the mesh.

    v3d: (N, 3) float32 meters. faces_tri: triangulated (M, 3) indices.
    Height = mesh extent along the body axis (95th - 5th percentile of the
    vertical coordinate), robust to a stray vertex.
    """
    if v3d.shape[0] < 100 or v3d.shape[1] != 3:
        return None, None
    try:
        import trimesh
    except Exception:  # noqa: BLE001
        return None, None
    mesh = trimesh.Trimesh(vertices=np.asarray(v3d, dtype=np.float64),
                           faces=np.asarray(faces_tri, dtype=np.int64), process=False)
    volume = float(mesh.volume)
    virts = np.asarray(v3d, dtype=np.float64)
    ys = np.sort(virts[:, 1])
    h_lo = ys[int(0.05 * len(ys))]
    h_hi = ys[int(0.95 * len(ys))]
    height = float(h_hi - h_lo)
    if not np.isfinite(volume) or volume <= 0 or not np.isfinite(height) or height <= 1e-4:
        return None, height if np.isfinite(height) else None
    norm = volume / (height ** 3)
    return float(norm), height


def compute_body_volume(
    seg2: np.ndarray,
    rgb: np.ndarray,
    *,
    checkpoint: str = CHECKPOINT,
    volume_bands: tuple[float, float] | None = VOLUME_BANDS,
) -> dict[str, Any]:
    """Compute a scale-invariant body-volume band from a Multi-HMR-Anny mesh.

    Returns a dict with ``abstained`` / ``abstention_reason`` on failure, or
    the measured facts (band + raw scale-invariant stats + payload).
    """
    validate_seg2_array(seg2)
    validate_rgb_array(rgb)
    if seg2.shape[0] != rgb.shape[0] or seg2.shape[1] != rgb.shape[1]:
        raise BodyVolumeError(f"seg2 {seg2.shape} must be pixel-aligned with rgb {rgb.shape}")

    import torch

    out: dict[str, Any] = {
        "subject_present": True,
        "abstained": False,
        "abstention_reason": None,
        "body_volume_band": None,
        "n_humans_detected": 0,
        "mesh_vertices": None,
        "normalized_volume": None,
        "shape_weight": None,
        "shape_height": None,
        "shape_muscle": None,
        "shape_proportions": None,
        "detection_thresh": DET_THRESH,
        "img_size": IMG_SIZE,
    }

    centroid = _subject_centroid(seg2)
    if centroid is None:
        out.update({"subject_present": False, "abstained": True,
                    "abstention_reason": "no foreground subject present"})
        return out
    subj_px = int((seg2 != 0).sum())
    if subj_px < MIN_SUBJECT_PX:
        out.update({"abstained": True, "abstention_reason": (
            f"seg2 subject too small (px={subj_px} < {MIN_SUBJECT_PX})")})
        return out

    model, device = _RunTime.get(checkpoint)
    # Preprocess
    x_np, scale = _preprocess(rgb, IMG_SIZE)
    x = torch.from_numpy(x_np).to(device)
    K = _camera_parameters(IMG_SIZE, FOV, device=device)

    with torch.no_grad():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                humans = model(x, is_training=False, nms_kernel_size=NMS_KERNEL,
                               det_thresh=DET_THRESH, K=K)
        else:
            humans = model(x, is_training=False, nms_kernel_size=NMS_KERNEL,
                           det_thresh=DET_THRESH, K=K)

    out["n_humans_detected"] = len(humans)
    if not humans:
        out.update({"abstained": True,
                    "abstention_reason": "no human detected by Multi-HMR"})
        return out

    # Select the subject human closest to the seg2 subject centroid (projected).
    y_pad, x_pad = _project_subject_centroid(centroid, scale, rgb.shape[0], rgb.shape[1], IMG_SIZE)
    best = None
    best_dist = float("inf")
    for h in humans:
        # Multi-HMR returns a list of per-person dicts; on zero detections it
        # can return the ("{}", []) tuple — skip anything that is not a real
        # person record (a dict carrying v3d/j2d).
        if not isinstance(h, dict) or "v3d" not in h:
            continue
        j2d = h.get("j2d")
        if j2d is None:
            primary = None
        else:
            j2d = j2d.detach().cpu().numpy()
            # NOTE: under autocast fp16 the projected keypoints are float16 —
            # squaring a >256px offset overflows fp16 (max 65504) to inf and
            # silently breaks subject alignment. Always compute in float64.
            primary = j2d[0].astype(np.float64) if j2d.shape[0] > 0 else None
        if primary is None:
            dist = 1e30
        else:
            dist = float((primary[0] - x_pad) ** 2 + (primary[1] - y_pad) ** 2)
        if dist < best_dist:
            best = (h, dist)
            best_dist = dist
    if best is None:
        out.update({"abstained": True,
                    "abstention_reason": "no detector person could be aligned to the subject"})
        return out
    human, dist = best

    v3d = human["v3d"].detach().cpu().numpy()  # (n_vert, 3) meters
    out["mesh_vertices"] = v3d.shape[0]
    if v3d.shape[0] < 1000:
        out.update({"abstained": True, "abstention_reason": (
            f"mesh too small ({v3d.shape[0]} vertices)")})
        return out

    # Faces (Anny native, quads) — triangulate deterministically.
    try:
        faces = model.body_model.faces.detach().cpu().numpy()
    except Exception:  # noqa: BLE001
        faces = None
    if faces is None or faces.shape[-1] != 4:
        out.update({"abstained": True, "abstention_reason": "no quad faces from the body model"})
        return out
    tri = np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]], axis=0).astype(np.int64)

    norm_vol, mesh_height = _mesh_metrics(v3d, tri)
    out["normalized_volume"] = None if norm_vol is None else round(norm_vol, 6)
    out["mesh_height_m"] = None if mesh_height is None else round(mesh_height, 4)
    if norm_vol is None or not np.isfinite(norm_vol) or norm_vol <= 0:
        out.update({"abstained": True, "abstention_reason": "degenerate mesh volume"})
        return out
    # Honest plausibility gates: a reconstructed whole-body mesh should span a
    # human-plausible height (meters) and a plausible normalized bulk. Outliers
    # (unposed limbs, doubled bodies, scale failures) must abstain, never
    # fabricate a body-volume class.
    if mesh_height is None or not (MIN_MESH_HEIGHT <= mesh_height <= MAX_MESH_HEIGHT):
        out.update({"abstained": True, "abstention_reason": (
            f"mesh height {mesh_height if mesh_height is not None else 'n/a'}m "
            f"outside the plausible band [{MIN_MESH_HEIGHT}, {MAX_MESH_HEIGHT}]m")})
        return out
    if not (MIN_NORM_VOLUME <= norm_vol <= MAX_NORM_VOLUME):
        out.update({"abstained": True, "abstention_reason": (
            f"normalized body volume {norm_vol:.5f} outside the plausible "
            f"band [{MIN_NORM_VOLUME}, {MAX_NORM_VOLUME}] (degenerate mesh)")})
        return out

    shape = human["shape"].detach().cpu().numpy()  # (num_betas,) in [0,1]
    phenotype_labels = list(model.body_model.phenotype_labels)
    # Multi-HMR maps the first six phenotype slots to age/gender/weight/height/muscle/proportions.
    idx_of = {lab: idx for idx, lab in enumerate(phenotype_labels)}
    for key in ("weight", "height", "muscle", "proportions"):
        if key in idx_of and idx_of[key] < shape.shape[0]:
            out[f"shape_{key}"] = round(float(shape[idx_of[key]]), 4)

    if volume_bands is None:
        # Not yet calibrated — abstain on band but expose the measured metric.
        out.update({"abstained": True,
                    "abstention_reason": "body-volume band floors not yet calibrated",
                    "normalized_volume": out["normalized_volume"]})
        return out

    slim_hi, avg_hi = volume_bands
    if norm_vol < slim_hi:
        band = "slim"
    elif norm_vol < avg_hi:
        band = "average"
    else:
        band = "fuller"
    out["body_volume_band"] = band
    return out


def render_body_volume(config: Mapping[str, Any]) -> list[str]:
    """Scale-invariant body-volume claim for the dossier (arm #96).

    Verbalizes ONLY the coarse band; raw normalized volume / shape phenotype
    coefficients stay in the machine-readable payload.
    """
    if not config:
        return []
    if config.get("abstained"):
        reason = config.get("abstention_reason") or "body volume not measurable"
        return [f"body-volume: abstain ({reason})"]
    band = config.get("body_volume_band")
    if band == "slim":
        return ["body-volume: slender build (below-average body bulk)"]
    if band == "average":
        return ["body-volume: average build (typical body bulk for frame)"]
    if band == "fuller":
        return ["body-volume: fuller build (above-average body bulk)"]
    return []
