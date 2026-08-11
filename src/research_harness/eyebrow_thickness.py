"""Deterministic eyebrow-thickness measurement from the facemesh.

Arm #127 (registered 2026-08-11 via the gated propose-dimensions channel,
selector EXPLOIT tie-broken by id, currently PROPOSAL — un-elected while hold
#132 gates every Stage-B round trip). NEW evidence part `eyebrow-thickness`
(no new model: reuses the already-qualified open-weight MediaPipe
FaceLandmarker 478-point mesh, the SAME model as face-geometry #60,
gaze-head #68, eyebrow-position #111, nose-geometry #121, lip-fullness #122
and eye-shape #123 — face_landmarker.task sha256 64184e229b...).

Non-redundance (registry selection rationale): eyebrow-position #111 covers
brow ELEVATION (how high the brow arch sits above the eye line — validated
cycle 34); brow THICKNESS (the vertical MASS of the brow band, upper arc to
lower edge) is an orthogonal unbound axis — 'thick brows' / 'thin brows' are
recurring portrait caption claims with no validated evidence axis. The two
are independent: a thin brow can be raised, a thick brow can be low.

Measurement (scale-invariant, per side, from the 2D mesh):
- The eye line is the segment through the outer and inner eye-corner landmarks
  (33-133 left, 362-263 right) — the same face-anchored reference as
  eyebrow-position #111.
- Upper brow arc: (70, 63, 105, 66, 107) left / (336, 296, 334, 293, 300)
  right — the same 5-landmark arc as eyebrow-position #111.
- Lower brow edge: (46, 53, 52, 65, 55) left / (276, 283, 282, 295, 285)
  right — the inferior boundary of the brow band.
- `brow_thickness` = mean over the 5 arc positions of the vertical extent of
  the brow band: the signed distance of the upper-arc landmark ABOVE the eye
  line minus the signed distance of the lower-edge landmark ABOVE the eye
  line, evaluated via the same projection as eyebrow-position #111 and
  normalized by the ipsilateral eye width. Scale-invariant (independent of
  camera distance / face crop), roll-invariant (uses the local eye line),
  and roughly pitch-robust.
- Normalizer disclosure: the registry wording says 'eye height', but the
  ipsilateral eye WIDTH (the eye-line segment, the same face-anchored scale
  as eyebrow-position #111) is the normalizer here, NOT fissure height —
  fissure height collapses with eyelid state (eye-openness #113) and would
  couple the thickness read to eyelid aperture. Eye width is stable across
  eyelid states and keeps thickness orthogonal to eye-openness. Disclosed
  in the specialist bundle; no effectiveness claim until the probe.
- Per-pair plausibility: each of the 5 vertical extents must be positive
  (upper arc above lower edge) and inside the human-plausible band; a side
  needs at least 2 measurable pairs. A single measurable side is a valid
  (disclosed, asymmetry-flagged) fallback; no measurable side abstains.

Verbalized band (item level, coarse): thin / medium / thick, read from the
item-level mean brow thickness. The thresholds are COHORT-CALIBRATED tercile
cuts (THIN_MAX / THICK_MIN below, set 2026-08-11 from the frozen-cohort
calibration probe artifact
/mnt/nas-ai-models/research/stratum/eyebrow-thickness-calibration-probe.json;
20/24 measured, split 6/7/7, max_share 0.35). Bands are corpus-relative
within the frozen cohort, exactly as nose-geometry #121 and eyebrow-position
#111 calibrated. Raw normalized ratios stay in the
machine-readable evidence_payload and are never caption claims.

Abstention: no face detected (measured UNION policy: full frame then seg2
Face_Neck crop), degenerate eye width, occluded/degenerate brow landmarks
(including bangs/hair covering the brows — a bangs-forehead #110 overlap the
abstention policy surfaces honestly), or implausible per-pair extents; never
fabricate a thickness read. CPU-only in memory, no corpus write, model on
owned hardware.
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
BROW_ARC_UPPER_L = (70, 63, 105, 66, 107)     # upper brow arc (105 = arch apex)
BROW_ARC_UPPER_R = (336, 296, 334, 293, 300)  # upper brow arc (334 = arch apex)
BROW_EDGE_LOWER_L = (46, 53, 52, 65, 55)      # lower brow edge, same chain order
BROW_EDGE_LOWER_R = (276, 283, 282, 295, 285)  # lower brow edge, same chain order

# Minimum ipsilateral eye width (px) to produce a stable scale-invariant ratio.
MIN_EYE_PX = 8.0

# Human-plausible band for the per-pair brow vertical extent / eye width.
# Canonical adult brow vertical extent ~5-9mm vs eye width ~30mm -> ~0.17-0.30;
# anything below ~0.02 (flat/no brow) or above ~0.5 (severe pose / landmark
# error) is implausible and abstains.
PAIR_EXTENT_PLAUSIBLE = (0.02, 0.50)

# Minimum number of measurable pairs for a side to count.
MIN_PAIRS_PER_SIDE = 2


class EyebrowThicknessError(RuntimeError):
    pass


def validate_rgb_array(rgb: np.ndarray) -> None:
    """Validate a uint8 (H, W, 3) RGB array; raise EyebrowThicknessError."""
    try:
        _validate_rgb_array(rgb)
    except Exception as exc:  # noqa: BLE001 - re-raise with the local type
        raise EyebrowThicknessError(str(exc)) from exc


def validate_seg2_array(seg2: np.ndarray) -> None:
    """Validate a 2D integer seg2 label array; raise EyebrowThicknessError."""
    try:
        _validate_seg2_array(seg2)
    except Exception as exc:  # noqa: BLE001 - re-raise with the local type
        raise EyebrowThicknessError(str(exc)) from exc


def _to_pts(mesh, img_w: float, img_h: float) -> np.ndarray:
    return np.array(
        [(p.x * img_w, p.y * img_h) for p in mesh], dtype=np.float64
    )


def _side_measure(pts: np.ndarray, side: str) -> dict[str, Any]:
    """Per-side scale-invariant brow-thickness measurement, or None-flagged.

    Returns a dict with measureable True/False; a measureable side carries
    the mean brow thickness (upper arc minus lower edge, normalized by the
    ipsilateral eye width), the per-pair extents, and the eye width.
    """
    if side == "l":
        out_id, in_id, upper, lower = (
            EYE_OUT_L, EYE_IN_L, BROW_ARC_UPPER_L, BROW_EDGE_LOWER_L,
        )
    else:
        out_id, in_id, upper, lower = (
            EYE_OUT_R, EYE_IN_R, BROW_ARC_UPPER_R, BROW_EDGE_LOWER_R,
        )

    def _pt(i: int):
        return (float(pts[i, 0]), float(pts[i, 1]))

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
        # Distance ABOVE the eye line (positive when the point is above),
        # normalized by eye width -> scale-invariant.
        return (eye_y - y) / eye_w

    pair_extents: list[float] = []
    for u_i, l_i in zip(upper, lower):
        u_above = _above_line(*_pt(u_i))
        l_above = _above_line(*_pt(l_i))
        extent = u_above - l_above
        if (
            PAIR_EXTENT_PLAUSIBLE[0] <= extent <= PAIR_EXTENT_PLAUSIBLE[1]
            and u_above > 0.0
        ):
            pair_extents.append(extent)
    if len(pair_extents) < MIN_PAIRS_PER_SIDE:
        return {
            "side": side,
            "measureable": False,
            "reason": "degenerate brow landmarks (occluded / covered / implausible extents)",
        }
    return {
        "side": side,
        "measureable": True,
        "brow_thickness": round(float(np.mean(pair_extents)), 4),
        "pairs_measured": len(pair_extents),
        "pair_extents": [round(e, 4) for e in pair_extents],
        "eye_width_px": round(eye_w, 3),
    }


def compute_eyebrow_thickness(
    seg2: np.ndarray,
    rgb: np.ndarray,
    *,
    model_asset_path: str,
) -> dict[str, Any]:
    """Compute the scale-invariant eyebrow-thickness band with honest abstention.

    Detection uses the measured UNION policy (full frame then seg2 Face_Neck
    crop); the exactly-one-face corpus semantics make the first valid mesh the
    subject's. Only scale-invariant ratios and the coarse band are returned
    for prose; raw landmark-derived values stay in the machine-readable
    payload.
    """
    validate_seg2_array(seg2)
    validate_rgb_array(rgb)
    if seg2.shape[0] != rgb.shape[0] or seg2.shape[1] != rgb.shape[1]:
        raise EyebrowThicknessError(
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
        item_thickness = float(np.mean([s["brow_thickness"] for s in measurable]))
        fact = {
            "abstained": False,
            "detection": "DETECTED",
            "via": tag,
            "seg2_face_neck_px": fn_px,
            "brow_thickness": round(item_thickness, 4),
            "sides_measured": len(measurable),
            "per_side": sides,
        }
        if len(measurable) == 1:
            fact["asymmetry_disclosed"] = True
        return _apply_bands(fact)

    if fn_px < _MIN_FN_PX:
        reason = f"seg2 Face_Neck region too small (px={fn_px}) -> no measurable face"
    else:
        reason = "no face detected on full frame or the seg2 Face_Neck crop"
    return {"abstained": True, "abstention_reason": reason, "seg2_face_neck_px": fn_px}


# ---------------------------------------------------------------------------
# Band floors — COHORT-CALIBRATED tercile cuts (2026-08-11, frozen-cohort
# calibration probe, artifact
# /mnt/nas-ai-models/research/stratum/eyebrow-thickness-calibration-probe.json:
# 20/24 measured, 4 honest abstains, measured brow_thickness range
# 0.0360-0.3411). The probe's p33/p66 cuts split the measured cohort
# 6 thin / 7 medium / 7 thick (max_share 0.35); the provisional canon cuts
# (0.15/0.25) were non-degenerate (0.70) but left the outer bands barely
# populated (3/3). Authoritative cuts below are the measured cohort terciles,
# following the nose-geometry #121 pattern (measured 7/7/7 at p33/p66 cuts).
# Bands are corpus-relative within the frozen cohort; the qualification gate
# (no band >= 75%) passes at max_share 0.35.
# ---------------------------------------------------------------------------
THIN_MAX = 0.183  # brow thickness below this -> thin (cohort p33)
THICK_MIN = 0.217  # brow thickness above this -> thick (cohort p66)


def set_band_floors(thin_max: float, thick_min: float) -> None:
    """Override the band cuts (probe convenience; the module's authoritative
    constants are the module-level THIN_MAX / THICK_MIN above)."""
    global THIN_MAX, THICK_MIN  # noqa: PLW0603
    THIN_MAX = thin_max
    THICK_MIN = thick_min


def _apply_bands(fact: dict[str, Any]) -> dict[str, Any]:
    """Attach the eyebrow-thickness band (thin / medium / thick)."""
    t = fact.get("brow_thickness")
    if t is None:
        fact["eyebrow_thickness_band"] = None
        fact["banding_unavailable"] = True
    elif t < THIN_MAX:
        fact["eyebrow_thickness_band"] = "thin"
    elif t > THICK_MIN:
        fact["eyebrow_thickness_band"] = "thick"
    else:
        fact["eyebrow_thickness_band"] = "medium"
    return fact


def render_eyebrow_thickness(cfg: Mapping[str, Any]) -> list[str]:
    """Scale-invariant eyebrow-thickness claims for the dossier (arm #127)."""
    if not cfg:
        return []
    if cfg.get("abstained"):
        reason = cfg.get("abstention_reason") or "brow thickness not measurable"
        return [f"eyebrow-thickness: abstain ({reason})"]
    if cfg.get("banding_unavailable"):
        return ["eyebrow-thickness: measured but not banded (payload)"]
    band = cfg.get("eyebrow_thickness_band")
    if band == "thin":
        return ["eyebrow-thickness: eyebrows are thin (small vertical extent)"]
    if band == "thick":
        return ["eyebrow-thickness: eyebrows are thick (large vertical extent)"]
    if band == "medium":
        return ["eyebrow-thickness: eyebrows are of medium thickness (typical vertical extent)"]
    return []