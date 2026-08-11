"""Deterministic eye-shape measurement from the facemesh.

Arm #123 (registered 2026-08-11 via the gated propose-dimensions channel,
selector EXPLOIT tie-broken by id, currently PROPOSAL — un-elected while hold
#132 gates every Stage-B round trip). NEW evidence part `eye-shape` (no new
model: reuses the already-qualified open-weight MediaPipe FaceLandmarker
478-point mesh, the SAME model as face-geometry #60, gaze-head #68,
eyebrow-position #111, nose-geometry #121 and lip-fullness #122 —
face_landmarker.task sha256 64184e229b...).

Non-redundance (registry selection rationale): eye-openness #113 covers eyelid
aperture STATE; iris-eye-color #80 covers hue; face-geometry #60 covers eye
SPACING (eye_dist / face width). Eye SHAPE (palpebral fissure aspect +
corner orientation) is an unbound static axis — 'almond eyes', 'round
eyes' are recurring portrait caption claims with no validated evidence axis.

Measurement (scale-invariant, from the 2D mesh, canonical 478 indices):
- Per-eye fissure width = straight-line distance between the outer and inner
  eye corners (LEFT 33/133; RIGHT 362/263) — a face-anchored scale that is
  camera-distance invariant, so the ratio survives cross-picture comparison
  (never an absolute-pixel claim).
- Per-eye fissure height = distance between the upper-lid lower-edge center
  (LEFT 159; RIGHT 386) and the lower-lid upper-edge center (LEFT 145;
  RIGHT 374) — the vertical palpebral aperture at the pupil line.
- `eye_shape_ratio` = mean of the per-eye height/width aspects over the
  measurable eyes (both when both are measurable; a single eye when the
  other is degenerate, with the asymmetry disclosed in the payload).
- `canthal_tilt_deg` per eye = angle of the outer-to-inner corner line
  relative to horizontal (positive = upturned / 'cat-eye'), stays in the
  machine-readable payload as a corroborating sub-signal (never the primary
  band read).
- Frontal-plausibility gate: measure only when the eye width is plausibly
  large (>= MIN_EYE_PX) AND the aspect is inside the human-plausible band;
  otherwise abstain with a surfaced reason. A closed eye collapses the
  fissure height toward zero and is caught by the aspect floor (measured
  closed-eye items are the eye-openness #113 abstainees on the facemesh
  cohort: 0yo0gx..., 0mel7e..., 08v25q...).

Verbalized band (item level, coarse): almond / medium / round, read from the
fissure aspect. IMPORTANT band-set disclosure (honest re-scope of the
registered name): the registry proposal names the bands "almond/round/wide-set",
but the literal 'wide-set' reading is INTER-EYE SPAN relative to face width —
which is exactly face-geometry #60's already-validated eye-spacing axis
(EYE_CLOSE 0.445 / EYE_WIDE 0.475). Claiming it again here would violate the
arm's own falsified_if clause (redundancy with face-geometry #60 → degenerate).
The eye-SHAPE axis is therefore verbalized as almond / medium / round from the
fissure aspect at provisional canon-derived cuts (ALMOND_MAX / ROUND_MIN
below) pending the frozen-cohort calibration probe (deferred behind hold #132:
2 of 24 frozen sources are currently purged from approved/); the probe will
set the authoritative cohort-calibrated cuts via set_band_floors(), exactly as
nose-geometry #121 and lip-fullness #122 did. Inter-eye span stays payload-only
corroboration, never a caption claim of this arm. No effectiveness claim is
permitted (qualification gate unopened).

Abstention: no face detected (measured UNION policy: full frame then seg2
Face_Neck crop), degenerate eye width, fissure aspect outside the human
plausible band (closed eye / severe pose / occlusion), or implausible corner
geometry; never fabricate an eye-shape read. CPU-only in memory, no corpus
write, model on owned hardware.
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
# Canonical 478-point MediaPipe FaceMesh landmark indices (eyes).
# ---------------------------------------------------------------------------
L_EYE_OUT, L_EYE_IN = 33, 133            # left eye outer / inner corners
R_EYE_OUT, R_EYE_IN = 362, 263           # right eye outer / inner corners
L_LID_UPPER, L_LID_LOWER = 159, 145      # left upper-lid lower edge / lower-lid upper edge
R_LID_UPPER, R_LID_LOWER = 386, 374      # right upper-lid lower edge / lower-lid upper edge

# Minimum eye width (px) for a stable scale-invariant ratio.
MIN_EYE_PX = 8.0

# Plausibility band for the fissure height / width aspect. Typical adult
# palpebral fissure height is ~10mm with width ~28-30mm, so canonical aspect
# sits near 0.33-0.36; anything below ~0.05 is a closed/collapsed fissure and
# anything above 1.0 indicates severe pose / landmark error.
EYE_ASPECT_PLAUSIBLE = (0.05, 1.0)


class EyeShapeError(RuntimeError):
    pass


def validate_rgb_array(rgb: np.ndarray) -> None:
    """Validate a uint8 (H, W, 3) RGB array; raise EyeShapeError."""
    try:
        _validate_rgb_array(rgb)
    except Exception as exc:  # noqa: BLE001 - re-raise with the local type
        raise EyeShapeError(str(exc)) from exc


def validate_seg2_array(seg2: np.ndarray) -> None:
    """Validate a 2D integer seg2 label array; raise EyeShapeError."""
    try:
        _validate_seg2_array(seg2)
    except Exception as exc:  # noqa: BLE001 - re-raise with the local type
        raise EyeShapeError(str(exc)) from exc


def _to_pts(mesh, img_w: float, img_h: float) -> np.ndarray:
    return np.array(
        [(p.x * img_w, p.y * img_h) for p in mesh], dtype=np.float64
    )


def _eye_measure(pts: np.ndarray, out_i: int, in_i: int, up_i: int, lo_i: int):
    """Per-eye scale-invariant fissure aspect + canthal tilt, or None."""
    ox, oy = float(pts[out_i, 0]), float(pts[out_i, 1])
    ix, iy = float(pts[in_i, 0]), float(pts[in_i, 1])
    ux, uy = float(pts[up_i, 0]), float(pts[up_i, 1])
    lx, ly = float(pts[lo_i, 0]), float(pts[lo_i, 1])

    eye_width_px = float(np.hypot(ix - ox, iy - oy))
    if eye_width_px < MIN_EYE_PX:
        return None
    # Fissure height: vertical distance between the upper-lid lower edge and
    # the lower-lid upper edge at the pupil line.
    fissure_height_px = float(np.hypot(lx - ux, ly - uy))
    aspect = fissure_height_px / eye_width_px
    if not (EYE_ASPECT_PLAUSIBLE[0] <= aspect <= EYE_ASPECT_PLAUSIBLE[1]):
        return None
    # Canthal tilt (deg) of the outer-to-inner corner line vs horizontal
    # (positive = outer corner raised / upturned). Payload-only corroboration.
    tilt_deg = float(np.degrees(np.arctan2(oy - iy, ox - ix)))
    return {
        "aspect": round(aspect, 4),
        "eye_width_px": round(eye_width_px, 3),
        "fissure_height_px": round(fissure_height_px, 3),
        "canthal_tilt_deg": round(tilt_deg, 2),
    }


def _measure(pts: np.ndarray) -> dict[str, Any]:
    """Per-mesh scale-invariant eye-shape measurements (both eyes)."""
    left = _eye_measure(pts, L_EYE_OUT, L_EYE_IN, L_LID_UPPER, L_LID_LOWER)
    right = _eye_measure(pts, R_EYE_OUT, R_EYE_IN, R_LID_UPPER, R_LID_LOWER)
    measured = [m for m in (left, right) if m is not None]
    if not measured:
        return {
            "measureable": False,
            "reason": "no measurable eye (degenerate width or fissure aspect "
                      "outside the human-plausible band — closed eye / severe "
                      "pose / occlusion)",
        }
    aspect_mean = float(np.mean([m["aspect"] for m in measured]))
    out = {
        "measureable": True,
        "eye_shape_ratio": round(aspect_mean, 4),
        "eyes_measured": len(measured),
        "left_eye": left,
        "right_eye": right,
    }
    if len(measured) == 1:
        out["asymmetry_disclosed"] = True
    return out


def compute_eye_shape(
    seg2: np.ndarray,
    rgb: np.ndarray,
    *,
    model_asset_path: str,
) -> dict[str, Any]:
    """Compute the scale-invariant eye-shape bands with honest abstention.

    Detection uses the measured UNION policy (full frame then seg2 Face_Neck
    crop); the exactly-one-face corpus semantics make the first valid mesh the
    subject's. Only scale-invariant ratios and the coarse band are returned
    for prose; raw landmark-derived values stay in the machine-readable
    payload.
    """
    validate_seg2_array(seg2)
    validate_rgb_array(rgb)
    if seg2.shape[0] != rgb.shape[0] or seg2.shape[1] != rgb.shape[1]:
        raise EyeShapeError(
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
                "abstention_reason": m.get("reason", "eyes not measureable"),
                "via": tag,
                "seg2_face_neck_px": fn_px,
            }
        fact = {
            "abstained": False,
            "detection": "DETECTED",
            "via": tag,
            "seg2_face_neck_px": fn_px,
            "eye_shape_ratio": m["eye_shape_ratio"],
            "eyes_measured": m["eyes_measured"],
            "left_eye": m["left_eye"],
            "right_eye": m["right_eye"],
        }
        if m.get("asymmetry_disclosed"):
            fact["asymmetry_disclosed"] = True
        return _apply_bands(fact)

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
# unopened). Band-set note: the registered third band name "wide-set" is the
# inter-eye-span reading of face-geometry #60 (already validated) and is NOT
# claimed here (see module docstring); the verbalized set is almond / medium /
# round on the fissure aspect.
# ---------------------------------------------------------------------------
ALMOND_MAX = 0.30  # fissure aspect below this -> almond-shaped (elongated)
ROUND_MIN = 0.42   # above this -> round (tall fissure relative to width)


def set_band_floors(almond_max: float, round_min: float) -> None:
    """Override the band cuts (probe convenience; the module's authoritative
    constants are the module-level ALMOND_MAX / ROUND_MIN above)."""
    global ALMOND_MAX, ROUND_MIN  # noqa: PLW0603
    ALMOND_MAX = almond_max
    ROUND_MIN = round_min


def _apply_bands(fact: dict[str, Any]) -> dict[str, Any]:
    """Attach the eye-shape band (almond / medium / round)."""
    r = fact.get("eye_shape_ratio")
    if r is None:
        fact["eye_shape_band"] = None
        fact["banding_unavailable"] = True
    elif r < ALMOND_MAX:
        fact["eye_shape_band"] = "almond"
    elif r > ROUND_MIN:
        fact["eye_shape_band"] = "round"
    else:
        fact["eye_shape_band"] = "medium"
    return fact


def render_eye_shape(cfg: Mapping[str, Any]) -> list[str]:
    """Scale-invariant eye-shape claims for the dossier (arm #123)."""
    if not cfg:
        return []
    if cfg.get("abstained"):
        reason = cfg.get("abstention_reason") or "eye shape not measurable"
        return [f"eye-shape: abstain ({reason})"]
    if cfg.get("banding_unavailable"):
        return ["eye-shape: measured but not banded (payload)"]
    lines: list[str] = []
    band = cfg.get("eye_shape_band")
    if band == "almond":
        lines.append("eye-shape: the eyes are almond-shaped (elongated fissure relative to height)")
    elif band == "round":
        lines.append("eye-shape: the eyes are round (tall fissure relative to width)")
    elif band == "medium":
        lines.append("eye-shape: the eyes are of medium roundness (typical fissure aspect)")
    return lines