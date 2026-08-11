"""Deterministic lip-fullness measurement from the facemesh.

Arm #122 (registered 2026-08-11 via the gated propose-dimensions channel,
selector EXPLOIT top pick, currently PROPOSAL — un-elected while hold #132
gates every Stage-B round trip). NEW evidence part `lip-fullness` (no new
model: reuses the already-qualified open-weight MediaPipe FaceLandmarker
478-point mesh, the SAME model as face-geometry #60, gaze-head #68,
eyebrow-position #111 and nose-geometry #121 — face_landmarker.task sha256
64184e229b...).

Non-redundance (registry selection rationale): facial-expression #81 covers
smile spread/openness via mouth-corner geometry; lip-color #107 is BLOCKED
(cohort homogeneous natural lips). Lip FULLNESS (vertical vermilion extent)
is an unbound axis orthogonal to both, and 'full lips' / 'thin lips' is a
recurring portrait caption claim with no validated evidence axis.

Measurement (scale-invariant, from the 2D mesh, canonical 478 indices):
- The mouth-width normalizer is the straight-line distance between the
  outer mouth corners (61 / 291) — a face-anchored scale that is
  camera-distance invariant, so the ratio survives cross-picture
  comparison (never an absolute-pixel claim).
- `lip_fullness_ratio` = |landmark 0 (upper-lip vermilion top midline) -
  landmark 17 (lower-lip vermilion bottom midline)| / mouth width — the
  total vertical vermilion extent normalized by mouth width (the classic
  facial-canon fullness measure).
- `upper_lip_ratio` = |0 - 13 (upper inner vermilion midline)| / mouth
  width and `lower_lip_ratio` = |14 - 17| / mouth width stay in the
  machine-readable payload as corroborating sub-signals (the registered
  hypothesis names the upper-lip band; the total extent is the verbalized
  primary read).
- Frontal-plausibility gate: measure only when the mouth width is
  plausibly large (>= MIN_MOUTH_PX) AND the fullness ratio is inside the
  human-plausible band; otherwise abstain with a surfaced reason.

Verbalized band (item level, coarse): thin / medium / full. The thresholds
are PROVISIONAL canon-derived floors (THIN_MAX / FULL_MIN below) pending the
frozen-cohort band-calibration probe (deferred behind hold #132: the probe
needs the full 24-item cohort and 2 sources are currently purged from
approved/); the probe will set the authoritative cohort-calibrated cuts via
set_band_floors() / a constant update, exactly as nose-geometry #121 did.
Raw normalized ratios stay in the machine-readable evidence_payload and are
never caption claims.

Abstention: no face detected (measured UNION policy: full frame then seg2
Face_Neck crop), degenerate mouth width, or implausible ratio (severe pose /
occlusion); never fabricate a lip read. CPU-only in memory, no corpus write,
model on owned hardware.
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
# Canonical 478-point MediaPipe FaceMesh landmark indices (mouth).
# ---------------------------------------------------------------------------
MOUTH_CORNER_L, MOUTH_CORNER_R = 61, 291   # outer mouth corners -> width ref
UPPER_LIP_TOP = 0                          # upper-lip vermilion top midline
LOWER_LIP_BOTTOM = 17                      # lower-lip vermilion bottom midline
UPPER_LIP_INNER = 13                       # upper inner vermilion midline
LOWER_LIP_INNER = 14                       # lower inner vermilion midline

# Minimum mouth width (px) for a stable scale-invariant ratio.
MIN_MOUTH_PX = 8.0

# Plausibility band for the vermilion-height / mouth-width ratio (propagated
# from the facial canon: typical vermilion height is roughly a third to a
# half of the mouth width; values well outside indicate severe pose /
# occlusion / landmark error).
LIP_RATIO_PLAUSIBLE = (0.15, 1.0)


class LipFullnessError(RuntimeError):
    pass


def validate_rgb_array(rgb: np.ndarray) -> None:
    """Validate a uint8 (H, W, 3) RGB array; raise LipFullnessError."""
    try:
        _validate_rgb_array(rgb)
    except Exception as exc:  # noqa: BLE001 - re-raise with the local type
        raise LipFullnessError(str(exc)) from exc


def validate_seg2_array(seg2: np.ndarray) -> None:
    """Validate a 2D integer seg2 label array; raise LipFullnessError."""
    try:
        _validate_seg2_array(seg2)
    except Exception as exc:  # noqa: BLE001 - re-raise with the local type
        raise LipFullnessError(str(exc)) from exc


def _to_pts(mesh, img_w: float, img_h: float) -> np.ndarray:
    return np.array(
        [(p.x * img_w, p.y * img_h) for p in mesh], dtype=np.float64
    )


def _measure(pts: np.ndarray) -> dict[str, Any]:
    """Per-mesh scale-invariant lip-fullness measurements (mouth-width-ref)."""
    def _pt(i: int):
        return (float(pts[i, 0]), float(pts[i, 1]))

    mlx, mly = _pt(MOUTH_CORNER_L)
    mrx, mry = _pt(MOUTH_CORNER_R)
    mouth_width_px = float(np.hypot(mrx - mlx, mry - mly))
    if mouth_width_px < MIN_MOUTH_PX:
        return {
            "measureable": False,
            "reason": f"degenerate mouth width ({mouth_width_px:.1f} px)",
        }

    ulx, uly = _pt(UPPER_LIP_TOP)
    lbx, lby = _pt(LOWER_LIP_BOTTOM)
    vermilion_height_px = float(np.hypot(lbx - ulx, lby - uly))
    # Total vertical vermilion extent / mouth width (scale-invariant).
    lip_fullness_ratio = vermilion_height_px / mouth_width_px

    uix, uiy = _pt(UPPER_LIP_INNER)
    lix, liy = _pt(LOWER_LIP_INNER)
    upper_lip_px = float(np.hypot(uix - ulx, uiy - uly))
    lower_lip_px = float(np.hypot(lbx - lix, lby - liy))
    upper_lip_ratio = upper_lip_px / mouth_width_px
    lower_lip_ratio = lower_lip_px / mouth_width_px

    if not (LIP_RATIO_PLAUSIBLE[0] <= lip_fullness_ratio <= LIP_RATIO_PLAUSIBLE[1]):
        return {
            "measureable": False,
            "reason": "vermilion-height / mouth-width outside the human-plausible "
                      f"band ({lip_fullness_ratio:.3f}), likely severe pose / occlusion",
        }

    return {
        "measureable": True,
        "lip_fullness_ratio": round(lip_fullness_ratio, 4),
        "upper_lip_ratio": round(upper_lip_ratio, 4),
        "lower_lip_ratio": round(lower_lip_ratio, 4),
        "mouth_width_px": round(mouth_width_px, 3),
        "vermilion_height_px": round(vermilion_height_px, 3),
    }


def compute_lip_fullness(
    seg2: np.ndarray,
    rgb: np.ndarray,
    *,
    model_asset_path: str,
) -> dict[str, Any]:
    """Compute the scale-invariant lip-fullness bands with honest abstention.

    Detection uses the measured UNION policy (full frame then seg2 Face_Neck
    crop); the exactly-one-face corpus semantics make the first valid mesh the
    subject's. Only scale-invariant ratios and the coarse band are returned
    for prose; raw landmark-derived values stay in the machine-readable
    payload.
    """
    validate_seg2_array(seg2)
    validate_rgb_array(rgb)
    if seg2.shape[0] != rgb.shape[0] or seg2.shape[1] != rgb.shape[1]:
        raise LipFullnessError(
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
                "abstention_reason": m.get("reason", "lips not measureable"),
                "via": tag,
                "seg2_face_neck_px": fn_px,
            }
        fact = {
            "abstained": False,
            "detection": "DETECTED",
            "via": tag,
            "seg2_face_neck_px": fn_px,
            "lip_fullness_ratio": m["lip_fullness_ratio"],
            "upper_lip_ratio": m["upper_lip_ratio"],
            "lower_lip_ratio": m["lower_lip_ratio"],
            "mouth_width_px": m["mouth_width_px"],
            "vermilion_height_px": m["vermilion_height_px"],
        }
        out = _apply_bands(fact)
        return out

    if fn_px < _MIN_FN_PX:
        reason = f"seg2 Face_Neck region too small (px={fn_px}) -> no measurable face"
    else:
        reason = "no face detected on full frame or the seg2 Face_Neck crop"
    return {"abstained": True, "abstention_reason": reason, "seg2_face_neck_px": fn_px}


# ---------------------------------------------------------------------------
# Band floors — PROVISIONAL canon-derived cuts (2026-08-11), disclosed as
# placeholder pending the frozen-cohort calibration probe (deferred behind
# hold #132: 2 of 24 frozen sources are purged from approved/, so the probe
# cannot run on the full cohort yet). The probe will set the authoritative
# cohort tercile cuts via set_band_floors() / a constant update, following
# the nose-geometry #121 pattern (measured 7/7/7 at p33/p66 cuts). Until the
# probe runs, no effectiveness claim is permitted (qualification gate
# unopened).
# ---------------------------------------------------------------------------
THIN_MAX = 0.35  # vermilion height / mouth width below this -> thin lips
FULL_MIN = 0.50  # above this -> full lips


def set_band_floors(thin_max: float, full_min: float) -> None:
    """Override the band cuts (probe convenience; the module's authoritative
    constants are the module-level THIN_MAX / FULL_MIN above)."""
    global THIN_MAX, FULL_MIN  # noqa: PLW0603
    THIN_MAX = thin_max
    FULL_MIN = full_min


def _apply_bands(fact: dict[str, Any]) -> dict[str, Any]:
    """Attach the lip-fullness band (thin / medium / full)."""
    r = fact.get("lip_fullness_ratio")
    if r is None:
        fact["lip_fullness_band"] = None
        fact["banding_unavailable"] = True
    elif r < THIN_MAX:
        fact["lip_fullness_band"] = "thin"
    elif r > FULL_MIN:
        fact["lip_fullness_band"] = "full"
    else:
        fact["lip_fullness_band"] = "medium"
    return fact


def render_lip_fullness(cfg: Mapping[str, Any]) -> list[str]:
    """Scale-invariant lip-fullness claims for the dossier (arm #122)."""
    if not cfg:
        return []
    if cfg.get("abstained"):
        reason = cfg.get("abstention_reason") or "lip fullness not measurable"
        return [f"lip-fullness: abstain ({reason})"]
    if cfg.get("banding_unavailable"):
        return ["lip-fullness: measured but not banded (payload)"]
    lines: list[str] = []
    band = cfg.get("lip_fullness_band")
    if band == "full":
        lines.append("lip-fullness: the lips are full (tall vermilion relative to mouth width)")
    elif band == "thin":
        lines.append("lip-fullness: the lips are thin (short vermilion relative to mouth width)")
    elif band == "medium":
        lines.append("lip-fullness: the lips are of medium fullness relative to mouth width")
    return lines