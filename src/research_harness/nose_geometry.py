"""Deterministic nose-geometry measurement from the facemesh.

Arm #121 (registered 2026-08-11 via the gated propose-dimensions channel,
selected by the SELECTOR EXPLOIT slot, currently ACTIVE). NEW evidence part
`nose-geometry` (no new model: reuses the already-qualified open-weight
MediaPipe FaceLandmarker 478-point mesh, the SAME model as face-geometry #60,
gaze-head #68 and eyebrow-position #111 — face_landmarker.task sha256
64184e229b...).

Non-redundance (registry selection rationale): face-geometry #60 covers
eye-spacing, mouth, jaw, midface vertical share — NOT nose proportions. Nose
shape is a recurring portrait caption claim ('delicate nose', 'broad nose',
'long nose') with no validated evidence axis.

Measurement (scale-invariant, from the 2D mesh, canonical 478 indices):
- The inter-eye reference (IPD normalizer) is the straight-line distance
  between the inner eye corners (133 / 362) — the SAME scale-invariant
  reference eye-color #80 and facial-expression #81 use, so the nose axis is
  comparable to those face axes.
- `nose_width_ratio` = |alar_L (98) - alar_R (327)| / IPD. The two alar points
  are the left/right outermost nose wings; their horizontal span normalized by
  the inter-eye distance is scale-invariant (independent of camera distance /
  face-crop), roll-invariant to first order, and the classic facial-canon width
  measure (a stereotypical nose is ~one eye-width wide — values >1 read wide,
  <1 narrow).
- `nose_length_ratio` = (nose-tip y (1) - nose-bridge-top y (168)) / IPD:
  vertical nose length from the bridge root to the tip, scale-invariant.
- Frontal-plausibility gate: measure only when IPD is plausibly large
  (>= MIN_IPD_PX) and the alar + tip + bridge landmarks are present; otherwise
  abstain with a surfaced reason.

Verbalized bands (item level, coarse): nose width band narrow / average / wide
and nose length band short / average / long; the thresholds are CALIBRATED
from the frozen-cohort probe (band-degeneracy rule: no single band may take
>=75% of measured items). Raw normalized ratios stay in the machine-readable
evidence_payload and are never caption claims.

Abstention: no face detected (measured UNION policy: full frame then seg2
Face_Neck crop), degenerate IPD, or occluded nose landmarks; never fabricate a
nose read. CPU-only in memory, no corpus write, model on owned hardware.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .face_geometry import (
    FACE_NECK,
    _MIN_FN_PX,
    _detect_mesh_on,
    validate_rgb_array as _validate_rgb_array,
    validate_seg2_array as _validate_seg2_array,
)

# ---------------------------------------------------------------------------
# Canonical 478-point MediaPipe FaceMesh landmark indices (nose).
# ---------------------------------------------------------------------------
EYE_IN_L, EYE_IN_R = 133, 362   # inner eye corners -> IPD reference
ALA_L, ALA_R = 98, 327          # outermost nose wings (alar)
NOSE_TIP = 1                    # nose tip
NOSE_BRIDGE_TOP = 168           # bridge root (between the eyes)

# Minimum inter-eye (IPD) distance (px) for a stable scale-invariant ratio.
MIN_IPD_PX = 8.0

# Plausibility band for the nose width / IPD ratio (propagated from the
# face-canon: a stereotypical nose is about one inter-eye distance wide).
NOSE_WIDTH_PLAUSIBLE = (0.5, 1.8)


class NoseGeometryError(RuntimeError):
    pass


def validate_rgb_array(rgb: np.ndarray) -> None:
    """Validate a uint8 (H, W, 3) RGB array; raise NoseGeometryError."""
    try:
        _validate_rgb_array(rgb)
    except Exception as exc:  # noqa: BLE001 - re-raise with the local type
        raise NoseGeometryError(str(exc)) from exc


def validate_seg2_array(seg2: np.ndarray) -> None:
    """Validate a 2D integer seg2 label array; raise NoseGeometryError."""
    try:
        _validate_seg2_array(seg2)
    except Exception as exc:  # noqa: BLE001 - re-raise with the local type
        raise NoseGeometryError(str(exc)) from exc


def _to_pts(mesh, img_w: float, img_h: float) -> np.ndarray:
    return np.array(
        [(p.x * img_w, p.y * img_h) for p in mesh], dtype=np.float64
    )


def _measure(pts: np.ndarray) -> dict[str, Any]:
    """Per-mesh scale-invariant nose measurements (IPD-normalized)."""
    def _pt(i: int):
        return (float(pts[i, 0]), float(pts[i, 1]))

    ilx, ily = _pt(EYE_IN_L)
    irx, iry = _pt(EYE_IN_R)
    ipd = float(np.hypot(irx - ilx, iry - ily))
    if ipd < MIN_IPD_PX:
        return {"measureable": False, "reason": f"degenerate IPD ({ipd:.1f} px)"}

    alx, aly = _pt(ALA_L)
    arx, ary = _pt(ALA_R)
    nose_width_px = float(np.hypot(arx - alx, ary - aly))
    # Horizontal alar span / IPD is the classic facial-canon width measure.
    nose_width_ratio = nose_width_px / ipd

    tip_x, tip_y = _pt(NOSE_TIP)
    br_x, br_y = _pt(NOSE_BRIDGE_TOP)
    nose_length_px = float(np.hypot(tip_x - br_x, tip_y - br_y))
    nose_length_ratio = nose_length_px / ipd

    if not (NOSE_WIDTH_PLAUSIBLE[0] <= nose_width_ratio <= NOSE_WIDTH_PLAUSIBLE[1]):
        return {
            "measureable": False,
            "reason": "nose width / IPD outside the human-plausible band "
                      f"({nose_width_ratio:.3f}), likely severe pose / occlusion",
        }

    return {
        "measureable": True,
        "nose_width_ratio": round(nose_width_ratio, 4),
        "nose_length_ratio": round(nose_length_ratio, 4),
        "ipd_px": round(ipd, 3),
        "nose_width_px": round(nose_width_px, 3),
        "nose_length_px": round(nose_length_px, 3),
    }


def compute_nose_geometry(
    seg2: np.ndarray,
    rgb: np.ndarray,
    *,
    model_asset_path: str,
) -> dict[str, Any]:
    """Compute the scale-invariant nose-geometry bands with honest abstention.

    Detection uses the measured UNION policy (full frame then seg2 Face_Neck
    crop); the exactly-one-face corpus semantics make the first valid mesh the
    subject's. Only scale-invariant ratios and the coarse bands are returned
    for prose; raw landmark-derived values stay in the machine-readable
    payload.
    """
    validate_seg2_array(seg2)
    validate_rgb_array(rgb)
    if seg2.shape[0] != rgb.shape[0] or seg2.shape[1] != rgb.shape[1]:
        raise NoseGeometryError(
            f"seg2 {seg2.shape} must be pixel-aligned with rgb {rgb.shape}"
        )

    candidates: list[tuple[str, np.ndarray]] = [("full_frame", np.ascontiguousarray(rgb))]
    mask = seg2 == FACE_NECK
    fn_px = int(mask.sum())
    if fn_px >= _MIN_FN_PX:
        ys, xs = np.where(mask)
        h, w = ys.max() - ys.min(), xs.max() - xs.min()
        margin = int(max(h, w))
        cy0, cy1 = max(0, ys.min() - margin), min(seg2.shape[0] - 1, ys.max() + margin)
        cx0, cx1 = max(0, xs.min() - margin), min(seg2.shape[1] - 1, xs.max() + margin)
        crop = np.ascontiguousarray(rgb[cy0:cy1, cx0:cx1])
        candidates.append(("seg2_face_crop", crop))

    for tag, arr in candidates:
        mesh = _detect_mesh_on(arr, model_asset_path)
        if mesh is None:
            continue
        img_w = rgb.shape[1] if tag == "full_frame" else arr.shape[1]
        img_h = rgb.shape[0] if tag == "full_frame" else arr.shape[0]
        pts = _to_pts(mesh, img_w, img_h)
        m = _measure(pts)
        if not m.get("measureable"):
            return {
                "abstained": True,
                "abstention_reason": m.get("reason", "nose not measureable"),
                "via": tag,
                "seg2_face_neck_px": fn_px,
            }
        fact = {
            "abstained": False,
            "detection": "DETECTED",
            "via": tag,
            "seg2_face_neck_px": fn_px,
            "nose_width_ratio": m["nose_width_ratio"],
            "nose_length_ratio": m["nose_length_ratio"],
            "ipd_px": m["ipd_px"],
            "nose_width_px": m["nose_width_px"],
            "nose_length_px": m["nose_length_px"],
        }
        out = _apply_bands(fact)
        return out

    if fn_px < _MIN_FN_PX:
        reason = f"seg2 Face_Neck region too small (px={fn_px}) -> no measurable face"
    else:
        reason = "no face detected on full frame or the seg2 Face_Neck crop"
    return {"abstained": True, "abstention_reason": reason, "seg2_face_neck_px": fn_px}


# ---------------------------------------------------------------------------
# Band floors — CALIBRATED from the frozen-cohort probe (2026-08-11,
# /mnt/nas-ai-models/research/stratum/nose-geometry-calibration-probe.json).
# 21/24 measured, 3 honest abstains (07hyx5 no-face, 0mel7e zero Face_Neck
# region, 08v25q no-face — the SAME facemesh abstainees as face-geometry #60 /
# eyebrow-position #111). Cohort nose_width_ratio (n=21): p33 0.925 / p66
# 1.04 -> width bands 7 narrow / 7 average / 7 wide, max_share 0.3333.
# Cohort nose_length_ratio (n=21): p33 1.36 / p66 1.56 -> length bands
# 7 short / 7 average / 7 long, max_share 0.3333. Both axes NON-degenerate
# (max_share well under 0.75); cuts are disclosed cohort-calibrated cuts, not
# hidden threshold-fitting.
# ---------------------------------------------------------------------------
WIDTH_NARROW_MAX = 0.925  # nose width / IPD below this -> narrow nose
WIDTH_WIDE_MIN = 1.04     # above this -> wide nose
LENGTH_SHORT_MAX = 1.36   # nose length / IPD below this -> short nose
LENGTH_LONG_MIN = 1.56    # above this -> long nose


def set_band_floors(width_narrow_max: float, width_wide_min: float,
                    length_short_max: float, length_long_min: float) -> None:
    """Override the calibrated band cuts (probe convenience; the module's
    authoritative constants are the module-level WIDTH_*/LENGTH_* above)."""
    global WIDTH_NARROW_MAX, WIDTH_WIDE_MIN, LENGTH_SHORT_MAX, LENGTH_LONG_MIN  # noqa: PLW0603
    WIDTH_NARROW_MAX = width_narrow_max
    WIDTH_WIDE_MIN = width_wide_min
    LENGTH_SHORT_MAX = length_short_max
    LENGTH_LONG_MIN = length_long_min


def _apply_bands(fact: dict[str, Any]) -> dict[str, Any]:
    """Attach the calibrated nose-geometry bands (width + length)."""
    wr = fact.get("nose_width_ratio")
    if wr is None:
        fact["nose_width_band"] = None
        fact["banding_unavailable"] = True
    elif wr < WIDTH_NARROW_MAX:
        fact["nose_width_band"] = "narrow"
    elif wr > WIDTH_WIDE_MIN:
        fact["nose_width_band"] = "wide"
    else:
        fact["nose_width_band"] = "average"

    lr = fact.get("nose_length_ratio")
    if lr is None:
        fact["nose_length_band"] = None
        fact["banding_unavailable"] = True
    elif lr < LENGTH_SHORT_MAX:
        fact["nose_length_band"] = "short"
    elif lr > LENGTH_LONG_MIN:
        fact["nose_length_band"] = "long"
    else:
        fact["nose_length_band"] = "average"
    return fact


def render_nose_geometry(cfg: Mapping[str, Any]) -> list[str]:
    """Scale-invariant nose-geometry claims for the dossier (arm #121)."""
    if not cfg:
        return []
    if cfg.get("abstained"):
        reason = cfg.get("abstention_reason") or "nose geometry not measurable"
        return [f"nose-geometry: abstain ({reason})"]
    if cfg.get("banding_unavailable"):
        return ["nose-geometry: measured but not banded (payload)"]
    lines: list[str] = []
    band = cfg.get("nose_width_band")
    if band == "narrow":
        lines.append("nose-geometry: the nose is narrow (narrow alar width relative to the eyes)")
    elif band == "wide":
        lines.append("nose-geometry: the nose is wide (broad alar width relative to the eyes)")
    elif band == "average":
        lines.append("nose-geometry: the nose is of average width relative to the eyes")
    lband = cfg.get("nose_length_band")
    if lband == "short":
        lines.append("nose-geometry: the nose reads short (bridge-to-tip short relative to the eyes)")
    elif lband == "long":
        lines.append("nose-geometry: the nose reads long (prominent bridge-to-tip length relative to the eyes)")
    elif lband == "average":
        lines.append("nose-geometry: the nose reads of average length relative to the eyes")
    return lines
