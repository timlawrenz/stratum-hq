"""Deterministic cheek-prominence measurement from the facemesh.

Arm #128 (registered 2026-08-11 via the gated propose-dimensions channel,
selector EXPLOIT tie-broken by id, currently PROPOSAL — un-elected while hold
#132 gates every Stage-B round trip). NEW evidence part `cheek-prominence`
(no new model: reuses the already-qualified open-weight MediaPipe
FaceLandmarker 478-point mesh, the SAME model as face-geometry #60,
gaze-head #68, eyebrow-position #111, nose-geometry #121, lip-fullness #122,
eye-shape #123 and eyebrow-thickness #127 — face_landmarker.task sha256
64184e229b...).

Non-redundance (registry selection rationale): face-geometry #60 covers
eye-spacing / mouth / jaw / midface proportions; facial-expression #81 covers
smile geometry. Cheekbone PROMINENCE (how strongly the zygomatic arch reads
against the jaw) is an unbound facial-mass axis — 'high cheekbones',
'sculpted cheekbones', 'soft cheekbones' are recurring portrait caption
claims with no validated evidence axis.

ANTI-REDUNDANCY MEASUREMENT CHOICE (disclosed): the naive 'cheek width' read
would reuse face-geometry #60's face-width cheek pair (CHEEK_L/R = 234/454),
making cheek/jaw the EXACT RECIPROCAL of #60's already-validated jaw/face-width
axis (a 1:1 monotone transform — degenerate redundancy that would trip this
arm's own falsified_if clause). The primary pair here is therefore the
ZYGOMATIC-LEVEL pair 116/345 (canonical 478-point mesh landmark atlas: the
cheekbone arch at the level just below the outer eye corners), which sits at a
different vertical level than 234/454 (widest mid-face points at mouth level)
and is unused by every other harness module. The ratio composes honestly with
the same jaw reference #60 uses (JAW_L/R = 172/397), so the two arms share a
jaw anchor without duplicating an axis.

Measurement (scale-invariant, from the 2D mesh, canonical 478 indices):
- `zygomatic_span` = straight-line distance between 116 and 345 (the
  zygomatic-arch cheekbone pair).
- `jaw_span` = straight-line distance between 172 and 397 (the same lower-jaw
  pair face-geometry #60 uses for its jaw axis, keeping the anchor shared).
- `cheek_bone_ratio` = zygomatic_span / jaw_span. Both spans live in the same
  imaging plane and the same frame, so the ratio is scale-invariant (camera
  distance / face-crop independent) and survives cross-picture comparison;
  absolute spans stay payload-only.
- Frontal-plausibility gates (honest abstention, not threshold-fitting):
  (a) both spans >= MIN_SPAN_PX (tiny/degenerate face);
  (b) the ratio inside the human-plausible band [0.9, 2.0] — canonical adult
      bizygomatic/bigonial breadth is ~1.2–1.5, so a collapsed ratio flags
      profile-view / zygomatic occlusion and an extreme ratio flags landmark
      failure;
  (c) mid-sagittal alignment: the mid-x of the zygomatic pair and the mid-x of
      the jaw pair within 0.30 × zygomatic_span (gross landmark failure under
      pose).

Verbalized band (item level, coarse): subtle / moderate / prominent, read
from the item-level zygomatic/jaw ratio. The thresholds are PROVISIONAL
canon-derived cuts (SUBTLE_MAX / PROMINENT_MIN below): canonical adult
bizygomatic breadth ~135–145 mm vs bigonial ~95–110 mm puts the typical ratio
near 1.25–1.45, so subtle < 1.25 and prominent > 1.45. They are disclosed as
placeholders pending the frozen-cohort calibration probe (deferred behind
hold #132: 2 of 24 frozen sources are currently purged from approved/); the
probe will set the authoritative cohort-calibrated cuts via set_band_floors(),
exactly as nose-geometry #121 and eyebrow-position #111 did. Raw ratios stay
in the machine-readable evidence_payload and are never caption claims.

Abstention: no face detected (measured UNION policy: full frame then seg2
Face_Neck crop), degenerate spans, implausible ratio (profile / zygomatic
occlusion / landmark failure), or violated midline alignment; never fabricate
a prominence read. CPU-only in memory, no corpus write, model on owned
hardware.
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
# Canonical 478-point MediaPipe FaceMesh landmark indices (cheekbone / jaw).
# ---------------------------------------------------------------------------
ZYGOMATIC_L, ZYGOMATIC_R = 116, 345  # zygomatic-arch cheekbone pair (see docstring)
JAW_L, JAW_R = 172, 397              # lower-jaw pair (shared with face-geometry #60)

# Minimum span (px) for a stable scale-invariant ratio.
MIN_SPAN_PX = 8.0

# Human-plausible band for zygomatic/jaw ratio (canonical ~1.2-1.5; outside
# this widely-spread band the read is a pose/occlusion/landmark artifact).
RATIO_PLAUSIBLE = (0.9, 2.0)

# Mid-sagittal alignment tolerance (fraction of the zygomatic span): the
# mid-x of the zygomatic pair and of the jaw pair must roughly coincide on a
# real face; a violated alignment means gross landmark failure under pose.
MIDLINE_TOL = 0.30


class CheekProminenceError(RuntimeError):
    pass


def validate_rgb_array(rgb: np.ndarray) -> None:
    """Validate a uint8 (H, W, 3) RGB array; raise CheekProminenceError."""
    try:
        _validate_rgb_array(rgb)
    except Exception as exc:  # noqa: BLE001 - re-raise with the local type
        raise CheekProminenceError(str(exc)) from exc


def validate_seg2_array(seg2: np.ndarray) -> None:
    """Validate a 2D integer seg2 label array; raise CheekProminenceError."""
    try:
        _validate_seg2_array(seg2)
    except Exception as exc:  # noqa: BLE001 - re-raise with the local type
        raise CheekProminenceError(str(exc)) from exc


def _to_pts(mesh, img_w: float, img_h: float) -> np.ndarray:
    return np.array(
        [(p.x * img_w, p.y * img_h) for p in mesh], dtype=np.float64
    )


def _measure(pts: np.ndarray) -> dict[str, Any]:
    """Per-mesh scale-invariant cheekbone measurement (zygomatic / jaw).

    Returns a dict with measureable True/False; a measureable mesh carries
    the zygomatic and jaw spans plus the scale-invariant ratio.
    """
    def _pt(i: int):
        return (float(pts[i, 0]), float(pts[i, 1]))

    zlx, zly = _pt(ZYGOMATIC_L)
    zrx, zry = _pt(ZYGOMATIC_R)
    zygomatic_span = float(np.hypot(zrx - zlx, zry - zly))

    jlx, jly = _pt(JAW_L)
    jrx, jry = _pt(JAW_R)
    jaw_span = float(np.hypot(jrx - jlx, jry - jly))

    if zygomatic_span < MIN_SPAN_PX or jaw_span < MIN_SPAN_PX:
        return {
            "measureable": False,
            "reason": (
                f"degenerate cheekbone/jaw spans "
                f"(zygomatic {zygomatic_span:.1f}px, jaw {jaw_span:.1f}px)"
            ),
        }

    ratio = zygomatic_span / jaw_span
    if not (RATIO_PLAUSIBLE[0] <= ratio <= RATIO_PLAUSIBLE[1]):
        return {
            "measureable": False,
            "reason": (
                f"zygomatic/jaw ratio {ratio:.3f} outside the human-plausible "
                f"band {RATIO_PLAUSIBLE} — profile view / zygomatic occlusion / "
                "landmark failure"
            ),
        }

    # Mid-sagittal alignment: both pairs' midpoints must roughly coincide in x.
    mid_x_z = (zlx + zrx) / 2.0
    mid_x_j = (jlx + jrx) / 2.0
    if abs(mid_x_z - mid_x_j) > MIDLINE_TOL * zygomatic_span:
        return {
            "measureable": False,
            "reason": (
                f"mid-sagittal misalignment (|{mid_x_z:.1f} - {mid_x_j:.1f}| "
                f"> {MIDLINE_TOL} * zygomatic span) — gross landmark failure "
                "under pose"
            ),
        }

    return {
        "measureable": True,
        "cheek_bone_ratio": round(ratio, 4),
        "zygomatic_span_px": round(zygomatic_span, 3),
        "jaw_span_px": round(jaw_span, 3),
    }


def compute_cheek_prominence(
    seg2: np.ndarray,
    rgb: np.ndarray,
    *,
    model_asset_path: str,
) -> dict[str, Any]:
    """Compute the scale-invariant cheek-prominence band with honest abstention.

    Detection uses the measured UNION policy (full frame then seg2 Face_Neck
    crop); the exactly-one-face corpus semantics make the first valid mesh the
    subject's. Only scale-invariant ratios and the coarse band are returned
    for prose; raw landmark-derived values stay in the machine-readable
    payload.
    """
    validate_seg2_array(seg2)
    validate_rgb_array(rgb)
    if seg2.shape[0] != rgb.shape[0] or seg2.shape[1] != rgb.shape[1]:
        raise CheekProminenceError(
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
                "abstention_reason": m.get("reason", "cheekbone not measureable"),
                "via": tag,
                "seg2_face_neck_px": fn_px,
            }
        fact = {
            "abstained": False,
            "detection": "DETECTED",
            "via": tag,
            "seg2_face_neck_px": fn_px,
            "cheek_bone_ratio": m["cheek_bone_ratio"],
            "zygomatic_span_px": m["zygomatic_span_px"],
            "jaw_span_px": m["jaw_span_px"],
        }
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
# cannot run on the full cohort yet). Canonical adult bizygomatic breadth
# ~135-145 mm vs bigonial ~95-110 mm -> typical zygomatic/jaw ratio ~1.25-1.45;
# subtle < 1.25, prominent > 1.45. The probe will set the authoritative
# cohort tercile cuts via set_band_floors() / a constant update, following
# the eyebrow-position #111 pattern. Until the probe runs, no effectiveness
# claim is permitted (qualification gate unopened).
# ---------------------------------------------------------------------------
SUBTLE_MAX = 1.25    # zygomatic/jaw ratio below this -> subtle cheekbones
PROMINENT_MIN = 1.45  # above this -> prominent cheekbones


def set_band_floors(subtle_max: float, prominent_min: float) -> None:
    """Override the band cuts (probe convenience; the module's authoritative
    constants are the module-level SUBTLE_MAX / PROMINENT_MIN above)."""
    global SUBTLE_MAX, PROMINENT_MIN  # noqa: PLW0603
    SUBTLE_MAX = subtle_max
    PROMINENT_MIN = prominent_min


def _apply_bands(fact: dict[str, Any]) -> dict[str, Any]:
    """Attach the cheek-prominence band (subtle / moderate / prominent)."""
    r = fact.get("cheek_bone_ratio")
    if r is None:
        fact["cheek_prominence_band"] = None
        fact["banding_unavailable"] = True
    elif r < SUBTLE_MAX:
        fact["cheek_prominence_band"] = "subtle"
    elif r > PROMINENT_MIN:
        fact["cheek_prominence_band"] = "prominent"
    else:
        fact["cheek_prominence_band"] = "moderate"
    return fact


def render_cheek_prominence(cfg: Mapping[str, Any]) -> list[str]:
    """Scale-invariant cheek-prominence claims for the dossier (arm #128)."""
    if not cfg:
        return []
    if cfg.get("abstained"):
        reason = cfg.get("abstention_reason") or "cheekbone prominence not measurable"
        return [f"cheek-prominence: abstain ({reason})"]
    if cfg.get("banding_unavailable"):
        return ["cheek-prominence: measured but not banded (payload)"]
    band = cfg.get("cheek_prominence_band")
    if band == "subtle":
        return ["cheek-prominence: cheekbones are subtle (zygomatic arch narrow relative to the jaw)"]
    if band == "prominent":
        return ["cheek-prominence: cheekbones are prominent (zygomatic arch wide relative to the jaw)"]
    if band == "moderate":
        return ["cheek-prominence: cheekbones are of moderate prominence (typical zygomatic-to-jaw ratio)"]
    return []