"""Scale-invariant gaze / head-orientation measurements (arm #68).

Deterministic head-orientation evidence derived from the SAME validated
open-weight MediaPipe FaceLandmarker 478-point mesh reused from arm #60
(`face_geometry`). Runs on owned hardware (local CPU, tasks API), uses the
same measured UNION detection policy (full frame first, then the seg2
Face_Neck crop), and emits only scale-invariant CAMERA-INTERACTION bands:

- head yaw band  -- facing camera / partially turned / profile or turned away;
- head pitch band -- level / tilted down / tilted up;
- head roll band  -- level / tilted (in-plane).

These are directions of the head's 3D rotation in the camera frame, normalized
via a focal-length-equals-image-width pinhole camera matrix, so they ARE
scale-invariant (camera-frame-relative directions survive cross-picture
comparison; absolute offsets and angles do not survive scale and must never be
caption claims). Raw yaw/pitch/roll angles and landmark coordinates stay in
the machine-readable ``evidence_payload`` JSON and are never prose.

Uses the canonical six-point Perspective-n-Point (PnP) head-pose solution
over the facial skeleton landmarks (nose tip, chin, left/right eye corners,
left/right mouth corners) via ``cv2.solvePnP`` against a generic facial
model, then Rodrigues + the classic OpenCV Euler decomposition into
yaw/pitch. In-plane roll is measured directly from the eye-line angle in
pixel space (deterministic and scale-invariant) because PnP roll is
gimbal-unstable at near-zero pitch. This is the standard open deterministic
head-pose estimator for the MediaPipe mesh (widely used, numerically stable
for frontal-to-3/4 views; the generic-model y-flip and cohort-centered pitch
calibration are documented next to the constants below).

Abstention: the union detection policy abstains when no face is found (arm #60
measured 21/24 detected on the frozen cohort; 3 honest abstains: turned-head/
no-face, zero Face_Neck region). A mesh is also gated on bbox size (>=15px).
Detector disagreement remains a quality anomaly, never caption content.

Provenance: local open-weight model (face_landmarker.task, sha256
64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff) run on owned
hardware only; no hosted third-party inference of the sensitive corpus; no
corpus write. model_asset_path is dependency-injected so unit tests can point
at a fixture and the runner at the frozen model asset.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .face_geometry import (
    FACE_NECK,
    L_EYE_OUT,
    R_EYE_OUT,
    _MIN_FACE_PX,
    _MIN_FN_PX,
    _detect_mesh_on,
    validate_rgb_array as _validate_rgb_array,
    validate_seg2_array as _validate_seg2_array,
)


def validate_rgb_array(rgb: "np.ndarray") -> None:
    """Validate a uint8 (H, W, 3) RGB array; raise GazeHeadError on violation."""
    try:
        _validate_rgb_array(rgb)
    except Exception as exc:  # noqa: BLE001 - re-raise with the local error type
        raise GazeHeadError(str(exc)) from exc


def validate_seg2_array(seg2: "np.ndarray") -> None:
    """Validate a 2D integer seg2 label array; raise GazeHeadError on violation."""
    try:
        _validate_seg2_array(seg2)
    except Exception as exc:  # noqa: BLE001 - re-raise with the local error type
        raise GazeHeadError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Canonical six-point facial skeleton (MediaPipe FaceMesh landmark indices).
# ---------------------------------------------------------------------------
NOSE_TIP = 1
CHIN = 152
L_EYE_CORNER = 33
R_EYE_CORNER = 263
L_MOUTH_CORNER = 61
R_MOUTH_CORNER = 291

# Generic facial model from the classic OpenCV head-pose-estimation demo
# (units arbitrary; only the ROTATION is used, so the model scale cancels in
# the angles), with the Y axis FLIPPED (chin at +y) so the fitted rotation
# matches the physical head (empirically measured 2026-08-07: the un-flipped
# model yields degenerate out-of-plane pitch on this cohort, the flipped model
# yields physically-plausible yaw/pitch on 21/21 detected meshes). The strong
# z-structure (eyes z=-135, mouth z=-125, chin z=-65) is what makes out-of-plane
# yaw/pitch estimable; a near-planar model makes the PnP fit unstable.
_MODEL_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],            # nose tip
        [0.0, -330.0, -65.0],       # chin
        [-225.0, 170.0, -135.0],    # left eye left corner
        [225.0, 170.0, -135.0],     # right eye right corner
        [-150.0, -150.0, -125.0],   # left mouth corner
        [150.0, -150.0, -125.0],    # right mouth corner
    ],
    dtype=np.float64,
) * np.array([1, -1, 1])
_LANDMARK_IDS = (NOSE_TIP, CHIN, L_EYE_CORNER, R_EYE_CORNER,
                 L_MOUTH_CORNER, R_MOUTH_CORNER)
# Plausibility gate on the fitted angles: a real head cannot be pitched beyond
# +/-85 deg or rolled beyond +/-85 deg without the face no longer being
# detectable as such; out-of-band fits are degenerate and abstain.
MAX_PITCH_DEG = 85.0
MAX_YAW_DEG = 90.0


class GazeHeadError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Head-pose estimation from the 478-point mesh.
# ---------------------------------------------------------------------------
def _solve_head_pose(mesh, img_w: int, img_h: int) -> dict[str, Any] | None:
    """Return {yaw, pitch, roll} (degrees) from a 6-point PnP head-pose fit.

    ``yaw`` and ``pitch`` come from the canonical six-point PnP fit (the
    literature-standard head-pose estimator, calibration probe 2026-08-07).
    ``roll`` is measured directly from the eye-line angle in pixel space (a
    deterministic, scale-invariant in-plane angle) because the PnP roll is
    gimbal-unstable when pitch is near zero — exactly the common frontal case.

    Returns None when a landmark is out of a sane range, solvePnP fails /
    returns a numerically degenerate rotation, or the plausibility gate
    (pitch/yaw beyond the head-direction floor) fails. The pinhole camera
    matrix uses focal = image width and principal point = image center, which
    normalizes the projection by image scale and makes the ANGLE (direction)
    invariant under picture size.
    """
    import cv2

    try:
        points = np.array(
            [(mesh[i].x * img_w, mesh[i].y * img_h) for i in _LANDMARK_IDS],
            dtype=np.float64,
        )
    except (IndexError, AttributeError) as exc:
        raise GazeHeadError(f"mesh missing required landmark: {exc!r}") from exc
    if points.shape != (6, 2):
        raise GazeHeadError(f"unexpected facial-skeleton point count {points.shape}")

    # Sanity: the six points must span a plausible face footprint (not a
    # degenerate scatter that solvePnP would happily fit).
    span_w = float(np.ptp(points[:, 0]))
    span_h = float(np.ptp(points[:, 1]))
    if span_w < _MIN_FACE_PX or span_h < _MIN_FACE_PX:
        return None

    camera = np.array(
        [
            [float(img_w), 0.0, float(img_w) / 2.0],
            [0.0, float(img_w), float(img_h) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.zeros((4, 1), dtype=np.float64)
    try:
        ok, rvec, _tvec = cv2.solvePnP(
            _MODEL_POINTS, points, camera, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
    except cv2.error as exc:
        raise GazeHeadError(f"solvePnP failed: {exc!r}") from exc
    if not ok:
        return None
    rot, _ = cv2.Rodrigues(rvec)
    yaw, pitch, _roll = _rotation_to_euler(rot)
    # Plausibility gate: a real head cannot be pitched/yawed beyond the
    # head-direction floor (at ~90 deg it becomes a non-detectable
    # profile/back-of-head). Out-of-band fits are degenerate (numerically
    # unstable near-planar tiny faces) and must abstain rather than emit an
    # implausible direction.
    if abs(pitch) > MAX_PITCH_DEG or abs(yaw) > MAX_YAW_DEG:
        return None
    # In-plane roll from the eye line (pixel space — scale-invariant angle).
    lx, ly = mesh[L_EYE_OUT].x, mesh[L_EYE_OUT].y
    rx, ry = mesh[R_EYE_OUT].x, mesh[R_EYE_OUT].y
    import math

    roll = float(math.degrees(math.atan2(ry - ly, rx - lx)))
    return {"yaw": yaw, "pitch": pitch, "roll": roll}


def _rotation_to_euler(rot: np.ndarray) -> tuple[float, float, float]:
    """Canonical OpenCV tutorial decomposition: (x=pitch, y=yaw, z=roll) deg.

    Follows the classic ``rotationMatrixToEulerAngles`` (extrinsic XYZ) that
    ships with the OpenCV head-pose demo this arm reuses. Returns
    (pitch_x, yaw_y, roll_z) in degrees. Sign conventions follow the standard
    solution; the banding is sign-agnostic so absolute sign is not a caption
    claim.
    """
    R = np.asarray(rot, dtype=np.float64)
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy < 1e-6:
        pitch = float(np.degrees(np.arctan2(-R[1, 2], R[1, 1])))
        yaw = float(np.degrees(np.arctan2(-R[2, 0], sy)))
        roll = 0.0
    else:
        pitch = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
        yaw = float(np.degrees(np.arctan2(-R[2, 0], sy)))
        roll = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return yaw, pitch, roll


# ---------------------------------------------------------------------------
# Scale-invariant orientation bands (CALIBRATED 2026-08-07 on the frozen
# 24-item cohort probe; no band >= 75% of the 21 detected meshes).
# ---------------------------------------------------------------------------
# Yaw (PnP, degrees): |yaw| < YAW_FACING -> facing camera; < YAW_TURNED ->
# partially turned; else profile / turned away. Measured |yaw| distribution
# {facing 4, partial 5, profile 12 (57%)} — the natural gap between 22 and 35
# deg splits the turned cluster; the corpus genuinely has many turned heads.
YAW_FACING = 12.0
YAW_TURNED = 35.0
# Pitch: the generic-model fit carries a systematic framing offset on this
# portrait corpus (measured median ~ -21 deg), so "level" is calibrated around
# that observed center, not naive zero (same move as the arm-#60 eye-spacing
# re-probe from the measured distribution). level = |pitch - center| < half.
PITCH_CENTER = -21.0
PITCH_LEVEL_HALF = 14.0
# Roll (in-plane eye-line angle, stable): |roll| < ROLL_LEVEL -> level.
ROLL_LEVEL = 12.0

# Model asset (bind the sha256 in the declaration; path injected by caller).
# Same frozen asset as arm #60/face-geometry.
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
GAZE_HEAD_MODEL_ASSET = (
    "/mnt/nas-ai-models/research/stratum/models/face-geometry/face_landmarker.task"
)


def _band_yaw(yaw: float | None) -> str | None:
    if yaw is None:
        return None
    ay = abs(yaw)
    if ay < YAW_FACING:
        return "facing camera"
    if ay < YAW_TURNED:
        return "partially turned"
    return "profile or turned away"


def _band_pitch(pitch: float | None) -> str | None:
    if pitch is None:
        return None
    if pitch > PITCH_CENTER + PITCH_LEVEL_HALF:
        return "tilted down"
    if pitch < PITCH_CENTER - PITCH_LEVEL_HALF:
        return "tilted up"
    return "level"


def _band_roll(roll: float | None) -> str | None:
    if roll is None:
        return None
    if abs(roll) < ROLL_LEVEL:
        return "level"
    return "tilted"


def _apply_orientation_bands(pose: Mapping[str, Any]) -> dict[str, Any]:
    fact = dict(pose)
    yaw = pose.get("yaw")
    pitch = pose.get("pitch")
    roll = pose.get("roll")
    fact["yaw_band"] = _band_yaw(yaw)
    fact["pitch_band"] = _band_pitch(pitch)
    fact["roll_band"] = _band_roll(roll)
    return fact


def compute_gaze_head(
    seg2: np.ndarray,
    rgb: np.ndarray,
    *,
    model_asset_path: str,
) -> dict[str, Any]:
    """Compute scale-invariant head-orientation bands from seg2 + source px.

    Reuses the validated UNION detection policy (full frame first, then the
    seg2 Face_Neck crop). Only scale-invariant direction bands are returned
    for prose; raw yaw/pitch/roll degrees and the pixel bbox stay in the
    machine-readable payload.

    Args:
        seg2: (H, W) integer DOME-29 class labels aligned with rgb.
        rgb: (H, W, 3) uint8 decoded source pixels aligned with seg2.
        model_asset_path: absolute path to the frozen face_landmarker.task.

    Returns a dict with yaw/pitch/roll bands; on no-face-abstain the dict has
    ``abstained=True`` and ``abstention_reason``.
    """
    validate_seg2_array(seg2)
    validate_rgb_array(rgb)
    if seg2.shape[0] != rgb.shape[0] or seg2.shape[1] != rgb.shape[1]:
        raise GazeHeadError(f"seg2 {seg2.shape} must be pixel-aligned with rgb {rgb.shape}")

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
        pose = _solve_head_pose(
            mesh, img_w if tag == "full_frame" else arr.shape[1],
            img_h if tag == "full_frame" else arr.shape[0],
        )
        if pose is None:
            continue
        fact = {
            "abstained": False,
            "detection": "DETECTED",
            "via": tag,
            "seg2_face_neck_px": fn_px,
            **pose,
        }
        return _apply_orientation_bands(fact)

    if fn_px < _MIN_FN_PX:
        reason = f"seg2 Face_Neck region too small (px={fn_px}) -> no measurable face"
    else:
        reason = "no face detected on full frame or the seg2 Face_Neck crop"
    return {"abstained": True, "abstention_reason": reason, "seg2_face_neck_px": fn_px}


def render_gaze_head(gaze: Mapping[str, Any]) -> list[str]:
    """Scale-invariant head-orientation claims for the dossier (arm #68)."""
    if gaze.get("abstained"):
        reason = gaze.get("abstention_reason") or "face not measurable"
        return [f"gaze-head-orientation: abstain ({reason})"]
    lines: list[str] = []
    yb = gaze.get("yaw_band")
    pb = gaze.get("pitch_band")
    rb = gaze.get("roll_band")
    if yb and yb != "facing camera":
        lines.append(f"gaze-head-orientation: head {yb} relative to the camera")
    elif yb:
        lines.append("gaze-head-orientation: head is facing the camera")
    if pb and pb != "level":
        lines.append(f"gaze-head-orientation: head is {pb} (pitch)")
    if rb and rb != "level":
        lines.append(f"gaze-head-orientation: head is {rb} to one side (in-plane)")
    if not lines:
        lines.append("gaze-head-orientation: head is level and facing the camera")
    return lines
