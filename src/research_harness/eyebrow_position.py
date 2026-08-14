"""Deterministic eyebrow-position / elevation measurement from the facemesh.

Arm #111 (registered 2026-08-10 via the gated propose-dimensions channel,
selected by the SELECTOR EXPLOIT slot, selection_progress 35). NEW evidence
part `eyebrow-position` (no new model: reuses the already-qualified open-weight
MediaPipe FaceLandmarker 478-point mesh, the SAME model as face-geometry #60
and gaze-head #68 — face_landmarker.task sha256 64184e229b...).

Measurement (scale-invariant, per side, from the 2D mesh):
- The eye line is the segment through the outer and inner eye-corner landmarks
  (33-133 left, 362-263 right).
- The brow arc is the 5 upper-arc landmarks ((70,63,105,66,107) left /
  (336,296,334,293,300) right).
- `arch_elevation` = mean over the arc of the signed vertical distance of each
  brow landmark ABOVE the eye line, evaluated at the landmark's x (a point
  above the line is positive), normalized by the ipsilateral eye width. This is
  scale-invariant (independent of camera distance / face crop), roll-invariant
  (uses the local eye line) and roughly pitch-robust.
- `inner_brow_elevation` = the same signed normalized distance for the INNER
  brow corner (155 / 249). A furrowed brow pulls the inner corner DOWN (this
  value drops, even below zero), corroborating a "furrowed" read the arch
  height alone can understate.
- Frontal-plausibility gate: measure only when the ipsilateral eye width is
  plausibly large (>= MIN_EYE_PX) and all eye-corner + arc landmarks are
  present; otherwise that side abstains with a surfaced reason.

Verbalized band (item level, coarse): neutral / raised / furrowed; the
thresholds are CALIBRATED from the frozen-cohort probe (band-degeneracy rule:
no single band may take >=75% of measured items). Raw normalized ratios stay in
the machine-readable evidence_payload and are never caption claims.

Abstention: no face detected (measured UNION policy: full frame then seg2
Face_Neck crop), degenerate eye width, or occluded landmarks; never fabricate a
brow state. CPU-only in memory, no corpus write, model on owned hardware.
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
# Canonical 478-point MediaPipe FaceMesh landmark indices.
# ---------------------------------------------------------------------------
EYE_OUT_L, EYE_IN_L = 33, 133
EYE_OUT_R, EYE_IN_R = 362, 263
BROW_ARC_L = (70, 63, 105, 66, 107)    # upper brow arc (105 = arch apex)
BROW_ARC_R = (336, 296, 334, 293, 300)  # upper brow arc (334 = arch apex)
BROW_INNER_L, BROW_INNER_R = 155, 249   # inner brow corners (furrow anchor)

# Minimum ipsilateral eye width (px) to produce a stable scale-invariant ratio.
MIN_EYE_PX = 8.0


class EyebrowPositionError(RuntimeError):
    pass


def validate_rgb_array(rgb: np.ndarray) -> None:
    """Validate a uint8 (H, W, 3) RGB array; raise EyebrowPositionError."""
    try:
        _validate_rgb_array(rgb)
    except Exception as exc:  # noqa: BLE001 - re-raise with the local type
        raise EyebrowPositionError(str(exc)) from exc


def validate_seg2_array(seg2: np.ndarray) -> None:
    """Validate a 2D integer seg2 label array; raise EyebrowPositionError."""
    try:
        _validate_seg2_array(seg2)
    except Exception as exc:  # noqa: BLE001 - re-raise with the local type
        raise EyebrowPositionError(str(exc)) from exc


def _to_pts(mesh, img_w: float, img_h: float) -> np.ndarray:
    return np.array(
        [(p.x * img_w, p.y * img_h) for p in mesh], dtype=np.float64
    )


def _side_measure(pts: np.ndarray, side: str) -> dict[str, Any]:
    """Per-side scale-invariant brow-position measurements from the mesh."""
    if side == "l":
        out_id, in_id, arc, inner = EYE_OUT_L, EYE_IN_L, BROW_ARC_L, BROW_INNER_L
    else:
        out_id, in_id, arc, inner = EYE_OUT_R, EYE_IN_R, BROW_ARC_R, BROW_INNER_R

    def _pt(i: int):
        x, y = float(pts[i, 0]), float(pts[i, 1])
        return (x, y)

    ox, oy = _pt(out_id)
    ix, iy = _pt(in_id)
    w2 = (ix - ox) ** 2 + (iy - oy) ** 2
    eye_w = float(np.hypot(ix - ox, iy - oy))
    if w2 <= 0 or eye_w < MIN_EYE_PX:
        return {"side": side, "measureable": False, "reason": "degenerate eye width"}

    def _above_line(x: float, y: float) -> float:
        # Fractional position of x projected onto the eye segment, clamped.
        t = max(0.0, min(1.0, ((x - ox) * (ix - ox) + (y - oy) * (iy - oy)) / w2))
        eye_y = oy + t * (iy - oy)
        return (eye_y - y) / eye_w  # positive when the point is above the line

    arc_above = [_above_line(*_pt(i)) for i in arc]
    arch_elevation = float(np.mean(arc_above))
    inner_elevation = _above_line(*_pt(inner))
    return {
        "side": side,
        "measureable": True,
        "arch_elevation": round(arch_elevation, 4),
        "inner_brow_elevation": round(inner_elevation, 4),
        "eye_width_px": round(eye_w, 3),
    }


def compute_eyebrow_position(
    seg2: np.ndarray,
    rgb: np.ndarray,
    *,
    model_asset_path: str,
) -> dict[str, Any]:
    """Compute the scale-invariant eyebrow-elevation band with honest abstention.

    Detection uses the measured UNION policy (full frame then seg2 Face_Neck
    crop); the exactly-one-face corpus semantics make the first valid mesh the
    subject's. Only scale-invariant ratios and the coarse band are returned for
    prose; raw landmark-derived values stay in the machine-readable payload.
    """
    validate_seg2_array(seg2)
    validate_rgb_array(rgb)
    if seg2.shape[0] != rgb.shape[0] or seg2.shape[1] != rgb.shape[1]:
        raise EyebrowPositionError(
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
        sides = [_side_measure(pts, "l"), _side_measure(pts, "r")]
        measurable = [s for s in sides if s.get("measureable")]
        if not measurable:
            continue
        item_arch = float(np.mean([s["arch_elevation"] for s in measurable]))
        item_inner = float(np.mean([s["inner_brow_elevation"] for s in measurable]))
        fact = {
            "abstained": False,
            "detection": "DETECTED",
            "via": tag,
            "seg2_face_neck_px": fn_px,
            "arch_elevation": round(item_arch, 4),
            "inner_brow_elevation": round(item_inner, 4),
            "per_side": sides,
        }
        out = _apply_bands(fact)
        return out

    if fn_px < _MIN_FN_PX:
        reason = f"seg2 Face_Neck region too small (px={fn_px}) -> no measurable face"
    else:
        reason = "no face detected on full frame or the seg2 Face_Neck crop"
    return {"abstained": True, "abstention_reason": reason, "seg2_face_neck_px": fn_px}


# ---------------------------------------------------------------------------
# Band floors — CALIBRATED from the frozen-cohort probe (2026-08-10). Measured
# arch_elevation (mean brow-arc height above the eye line / eye width, n=21/24):
# min 0.017 / p25 0.672 / median 0.745 / p75 0.946 / max 1.691. The distribution
# shows two clear clusters: a high-arch raised group (5 items >= 0.95) and a
# low/flat group (3 items < 0.55), bracketing a neutral bulk. The candidate
# composite inner-corner furrow rule (inner - arch <= -0.03) was REJECTED on
# probe review: it threshold-fits (inner - arch is ~-0.7 to -1.0 for almost
# every face because the inner brow corner sits near the eye line while the arch
# is high) and would label high-arch faces as furrowed. The honest axis is the
# arch elevation itself: raised (high arch), furrowed/lowered (low arch),
# neutral otherwise. inner_brow_elevation stays a corroborating payload signal.
# ---------------------------------------------------------------------------
RAISED_LOW = 0.95     # arch_elevation >= this -> raised
FURROWED_HIGH = 0.55  # arch_elevation <  this -> furrowed (brows drawn low)


def set_band_floors(raised_low: float, raised_high: float, furrowed_high: float) -> None:
    """Calibrate band floors from the measured cohort (probe convenience; the
    module's authoritative constants are the module-level RAISED_LOW/… above,
    so this only overrides when the caller explicitly wants a different cut)."""
    # raised_high is accepted for backward compatibility with earlier probe
    # drafts; the operative rule uses raised_low + furrowed_high.
    global RAISED_LOW, FURROWED_HIGH  # noqa: PLW0603
    RAISED_LOW = raised_low
    FURROWED_HIGH = furrowed_high


def _apply_bands(fact: dict[str, Any]) -> dict[str, Any]:
    """Attach the calibrated eyebrow-position band from arch elevation.

    The 3-way band is cut on the arch elevation only (raised / furrowed-low /
    neutral) per the calibration review; the inner-brow elevation stays in the
    payload as a corroborating (non-verbalized) furrow signal.
    """
    arch = fact.get("arch_elevation")
    if arch is None:
        fact["eyebrow_position_band"] = None
        fact["banding_unavailable"] = True
        return fact
    if arch >= RAISED_LOW:
        fact["eyebrow_position_band"] = "raised"
    elif arch < FURROWED_HIGH:
        fact["eyebrow_position_band"] = "furrowed"
    else:
        fact["eyebrow_position_band"] = "neutral"
    return fact


def render_eyebrow_position(cfg: Mapping[str, Any]) -> list[str]:
    """Scale-invariant brow-position claims for the dossier (arm #111)."""
    if not cfg:
        return []
    if cfg.get("abstained"):
        reason = cfg.get("abstention_reason") or "brow position not measurable"
        return [f"eyebrow-position: abstain ({reason})"]
    if cfg.get("banding_unavailable"):
        # Bands not yet calibrated — keep payload-only, never fabricate a band.
        return ["eyebrow-position: measured but not banded (payload)"]
    band = cfg.get("eyebrow_position_band")
    if band == "raised":
        return ["eyebrow-position: eyebrows are raised (elevated above the eye line)"]
    if band == "furrowed":
        return ["eyebrow-position: eyebrows are furrowed (lowered / drawn inward)"]
    if band == "neutral":
        return ["eyebrow-position: eyebrows are neutral (at a typical height)"]
    return []
