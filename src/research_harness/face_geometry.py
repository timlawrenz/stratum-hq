"""Deterministic face-geometry measurements from a local MediaPipe FaceLandmarker.

Arm #60. NEW-model-class specialist: runs the open-weight MediaPipe FaceLandmarker
(478-point 3D mesh, `face_landmarker.task`, Apache-2.0, local CPU via the tasks
API, XNNPACK) on owned hardware and derives scale-invariant facial-geometry facts:

- eye spacing (interpupillary / face-width) — banded close-set / average /
  wide-set;
- mouth width (mouth / face-width) — banded narrow / average / wide;
- jaw width (jaw / face-width) — banded narrow / average / wide;
- midface vertical share (nose-base-to-brow over nose-base-to-chin) — banded
  short / average / tall, emitted ONLY when it clears a human-plausibility
  gate (face tilt/pose can otherwise produce an implausible 2.2 share that must
  abstain, never verbalize).

Only scale-invariant ratios are verbalized (the measurement-semantics directive:
absolute pixel widths are camera-frame-dependent and a text-to-image model cannot
render them). Raw x/y/z landmark coordinates, pixel bbox, and absolute ratios
stay in the machine-readable ``evidence_payload`` JSON and are never caption
claims.

Detection policy (measured 2026-08-07 capability probe): FaceLandmarker is
resolution-sensitive and NON-monotonic on this cohort — the same face is found
on full-frame for some items and only on the seg2 Face_Neck crop for others.
The only robust policy is a UNION: try the full frame first, then the seg2
Face_Neck crop (margin-expanded), take the first valid >=15px mesh, and enforce
the exactly-one-face semantics (a single woman => her face is the detected one).
Probe result: 21/24 frozen items detected (union; 20/24 crop-only, 12/24
full-frame-only), 3 abstains (2 turned-head/no-face, 1 zero Face_Neck region) —
all with surfaced abstention reasons.

Band calibration (2026-08-07, band-calibration rule arm #34/#35/#58/#59): the
1-2 class probe bands left eye "average" at 71%, near the 75% degeneracy line,
so the eye-spacing band was re-probed from the measured cohort distribution
(0.445 / 0.475 -> 6/11/4, max share 52%). Mouth (0.333/0.400 -> 6/10/5),
jaw (0.783/0.830 -> 7/11/3), midface (0.48/0.56 -> 4/12/4) all clear the line.
A silenced axis is dropped from PROSE and kept payload-only.

Provenance: local open-weight model (face_landmarker.task, sha256
64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff) run on owned
hardware only; no hosted third-party inference of the sensitive corpus; no
corpus write. model_asset_path is dependency-injected so unit tests can point at
a fixture and the runner can point at the frozen model asset.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# ---------------------------------------------------------------------------
# Landmark index constants (canonical 478-point MediaPipe FaceMesh).
# ---------------------------------------------------------------------------
L_EYE_OUT, L_EYE_IN = 33, 133
R_EYE_OUT, R_EYE_IN = 362, 263
MOUTH_L, MOUTH_R = 61, 291
CHEEK_L, CHEEK_R = 234, 454     # widest cheek points (face width)
JAW_L, JAW_R = 172, 397         # lower jaw points (jaw width)
BROW_L, BROW_R = 105, 334       # brow anchors (midface share)
NOSE_BASE = (1, 2, 98)          # nose-base anchors
CHIN = 152

FACE_NECK = 3                   # DOME-29 class index (Face_Neck)

# Detection gates / floors.
_MIN_FACE_PX = 15               # landmark bbox side must clear this
_MIN_FN_PX = 200                # seg2 Face_Neck region floor for crop fallback

# Scale-invariant band thresholds — CALIBRATED from the frozen-cohort probe
# (2026-08-07, arm #60). No band >= 75% on the 21 detected items.
EYE_CLOSE = 0.445               # below: eyes set close together
EYE_WIDE = 0.475                # above: wide-set eyes
MOUTH_NARROW = 0.333            # below: narrow / small mouth
MOUTH_WIDE = 0.400              # above: wide / full mouth
JAW_NARROW = 0.783              # below: narrow / tapered jawline
JAW_WIDE = 0.830                # above: broad jawline
MIDFACE_SHORT = 0.48            # below: short mid-face
MIDFACE_TALL = 0.56             # above: tall mid-face
# Human-plausibility gate for the vertical midface share (face tilt / pose
# collapse otherwise yields out-of-band values e.g. 2.2 that must abstain).
MIDFACE_PLAUSIBLE = (0.35, 0.70)

# Model asset (bind the sha256 in the declaration; path injected by caller).
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


class FaceGeometryError(RuntimeError):
    pass


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise FaceGeometryError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise FaceGeometryError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise FaceGeometryError(f"seg2 must be integer class labels, got dtype {seg2.dtype}")


def validate_rgb_array(rgb: np.ndarray) -> None:
    if not isinstance(rgb, np.ndarray):
        raise FaceGeometryError("rgb must be a numpy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise FaceGeometryError(f"rgb must be (H, W, 3), got shape {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise FaceGeometryError(f"rgb must be uint8, got dtype {rgb.dtype}")


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


class _FaceLandmarkerRuntime:
    """Lazy, process-wide MediaPipe FaceLandmarker (CPU, tasks API)."""

    _landmarker = None

    @classmethod
    def get(cls, model_asset_path: str):
        if cls._landmarker is None:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                FaceLandmarker,
                FaceLandmarkerOptions,
                RunningMode,
            )
            cls._landmarker = FaceLandmarker.create_from_options(
                FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_asset_path),
                    running_mode=RunningMode.IMAGE,
                    num_faces=1,
                )
            )
        return cls._landmarker

    @classmethod
    def reset(cls) -> None:
        cls._landmarker = None


def _to_mediapipe_image(arr: np.ndarray):
    from mediapipe import Image as MpImage
    from mediapipe.tasks.python.vision.core.image import ImageFormat
    return MpImage(image_format=ImageFormat.SRGB, data=arr)


def _detect_mesh_on(arr: np.ndarray, model_asset_path: str) -> list[Any] | None:
    """Return face_landmarks[0] points for one image, or None (no face)."""
    try:
        res = _FaceLandmarkerRuntime.get(model_asset_path).detect(
            _to_mediapipe_image(arr)
        )
    except Exception as exc:  # noqa: BLE001
        raise FaceGeometryError(f"landmarker invocation failed: {exc!r}") from exc
    if not res or not getattr(res, "face_landmarks", None) or not res.face_landmarks:
        return None
    return res.face_landmarks[0]


def _mesh_to_facts(mesh, img_w: int, img_h: int) -> dict[str, Any] | None:
    """Convert a 478-point mesh to scale-invariant facial-geometry facts.

    Returns None when the mesh is degenerate (bbox too small to be a face).
    Absolute pixel values (bbox) are payload-only, never prose.
    """
    pts = np.array([(p.x, p.y, p.z) for p in mesh])
    xs = pts[:, 0] * img_w
    ys = pts[:, 1] * img_h
    if xs.max() - xs.min() < _MIN_FACE_PX or ys.max() - ys.min() < _MIN_FACE_PX:
        return None
    face_w = _euclid(pts[CHEEK_L, :2], pts[CHEEK_R, :2])
    eye_dist = _euclid(pts[L_EYE_IN, :2], pts[R_EYE_IN, :2])
    interpup = _euclid(
        (pts[L_EYE_OUT, :2] + pts[L_EYE_IN, :2]) / 2,
        (pts[R_EYE_OUT, :2] + pts[R_EYE_IN, :2]) / 2,
    )
    mouth_w = _euclid(pts[MOUTH_L, :2], pts[MOUTH_R, :2])
    jaw_w = _euclid(pts[JAW_L, :2], pts[JAW_R, :2])
    brow_y = (pts[BROW_L][1] + pts[BROW_R][1]) / 2
    nose_low = float(np.mean([pts[i][1] for i in NOSE_BASE]))
    chin_y = pts[CHIN][1]

    facts: dict[str, Any] = {
        "n_landmarks": int(len(pts)),
        "z_span_rel": float(pts[:, 2].max() - pts[:, 2].min()),
        "face_bbox_px": [
            round(float(xs.min())), round(float(ys.min())),
            round(float(xs.max())), round(float(ys.max())),
        ],
    }
    if face_w > 1e-6:
        facts["eye_spacing_face_width"] = eye_dist / face_w
        facts["interpupillary_face_width"] = interpup / face_w
        facts["mouth_face_width"] = mouth_w / face_w
        facts["jaw_face_width"] = jaw_w / face_w
    if chin_y > brow_y + 1e-9:
        mid = (nose_low - brow_y) / (chin_y - brow_y)
        facts["midface_share"] = float(mid)
    return facts


def _band(value: float | None, low: float, high: float, lo_name: str, hi_name: str, mid_name: str) -> str | None:
    if value is None:
        return None
    if value < low:
        return lo_name
    if value < high:
        return mid_name
    return hi_name


def compute_face_geometry(
    seg2: np.ndarray,
    rgb: np.ndarray,
    *,
    model_asset_path: str,
) -> dict[str, Any]:
    """Compute scale-invariant facial-geometry facts from seg2 + source pixels.

    Detection policy is the measured UNION: full frame first, then the seg2
    Face_Neck crop (margin-expanded); the first valid mesh wins. Only
    scale-invariant ratios and bands are returned for prose; absolute landmark
    coordinates / pixel bbox stay in the machine-readable payload.

    Args:
        seg2: (H, W) integer DOME-29 class labels aligned with rgb.
        rgb: (H, W, 3) uint8 decoded source pixels aligned with seg2.
        model_asset_path: absolute path to the frozen face_landmarker.task.

    Returns a dict with scale-invariant facial facts; on no-face-abstain the
    dict has ``abstained=True`` and ``abstention_reason``.
    """
    validate_seg2_array(seg2)
    validate_rgb_array(rgb)
    if seg2.shape[0] != rgb.shape[0] or seg2.shape[1] != rgb.shape[1]:
        raise FaceGeometryError(
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

    img_h, img_w = rgb.shape[0], rgb.shape[1]
    for tag, arr in candidates:
        mesh = _detect_mesh_on(arr, model_asset_path)
        if mesh is None:
            continue
        facts = _mesh_to_facts(mesh, img_w if tag == "full_frame" else arr.shape[1],
                               img_h if tag == "full_frame" else arr.shape[0])
        if facts is None:
            continue
        fact = {
            "abstained": False,
            "detection": "DETECTED",
            "via": tag,
            "seg2_face_neck_px": fn_px,
            **facts,
        }
        return _apply_bands(fact)

    if fn_px < _MIN_FN_PX:
        reason = f"seg2 Face_Neck region too small (px={fn_px}) -> no measurable face"
    else:
        reason = "no face detected on full frame or the seg2 Face_Neck crop"
    return {"abstained": True, "abstention_reason": reason, "seg2_face_neck_px": fn_px}


def _apply_bands(fact: dict[str, Any]) -> dict[str, Any]:
    """Attach calibrated scale-invariant bands; drop implausible verticals."""
    fact["eye_spacing_band"] = _band(
        fact.get("eye_spacing_face_width"), EYE_CLOSE, EYE_WIDE,
        "close-set", "wide-set", "average",
    )
    fact["mouth_band"] = _band(
        fact.get("mouth_face_width"), MOUTH_NARROW, MOUTH_WIDE,
        "narrow", "wide", "average",
    )
    fact["jaw_band"] = _band(
        fact.get("jaw_face_width"), JAW_NARROW, JAW_WIDE,
        "narrow", "wide", "average",
    )
    mid = fact.get("midface_share")
    if mid is not None and MIDFACE_PLAUSIBLE[0] <= mid <= MIDFACE_PLAUSIBLE[1]:
        fact["midface_band"] = _band(
            mid, MIDFACE_SHORT, MIDFACE_TALL, "short", "tall", "average",
        )
    elif mid is not None:
        # Plausibility gate failed (extreme pose/tilt) — keep payload-only,
        # never verbalize an implausible vertical share.
        fact["midface_band"] = None
        fact["midface_plausibility_abstained"] = True
    return fact


def render_face_geometry(face: Mapping[str, Any]) -> list[str]:
    """Scale-invariant facial-geometry claims for the dossier (arm #60)."""
    lines: list[str] = []
    if face.get("abstained"):
        reason = face.get("abstention_reason") or "face not measurable"
        return [f"face-geometry: abstain ({reason})"]
    bands = [
        ("eyes are set", face.get("eye_spacing_band"), "average"),
        ("mouth is", face.get("mouth_band"), "average"),
        ("jawline is", face.get("jaw_band"), "average"),
        ("mid-face is", face.get("midface_band"), "average"),
    ]
    for label, band, skip in bands:
        if band and band != skip:
            if face.get("eye_spacing_band") is not None and label == "eyes are set":
                lines.append(f"face-geometry: {label} {band} relative to the face")
            elif label == "mouth is":
                lines.append(f"face-geometry: {label} {band} relative to the face")
            elif label == "jawline is":
                lines.append(f"face-geometry: {label} {band} (tapered to broad)")
            else:
                lines.append(f"face-geometry: {label} {band} relative to the face")
    if not lines:
        lines.append("face-geometry: no distinctive facial ratio outside the typical range")
    return lines
