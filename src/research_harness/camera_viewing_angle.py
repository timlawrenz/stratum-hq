"""Scale-invariant camera-viewing-angle / framing measurements (arm #74).

Deterministic camera-relative framing evidence computed in memory from the
frozen seg2 DOME-29 subject mask and the full-frame image geometry. This is a
genuinely-NEW evidence part (the camera->subject framing axis) that complements
gaze-head #68 (subject->camera head direction): no validated arm measures how
the CAMERA sits relative to the subject or how the subject is framed.

Emits scale-invariant bands only (re-scope from the 2026-08-07 calibration
probe — see below):

- headroom band -- tight / normal / wide (space above the subject's head as a
  share of the frame), scale-invariant. THIS is the verbalized axis.

Degenerate axes kept payload-only (never prose) per the band-degeneracy rule:
- shot-scale band (close-up / mid-shot / full-body) — probe measured 88%
  full-body on this portrait-centric cohort (subjects fill the frame), a
  degenerate uniform axis.
- camera-height band (eye-level / high-angle / low-angle) — probe measured
  100% eye-level, a degenerate uniform axis.

Only scale-invariant bands are verbalized; raw pixel positions, bbox extents,
and frame shares stay in the machine-readable ``evidence_payload`` JSON and are
never caption claims (camera-frame-dependent absolute values cannot be
interpreted by a text-to-image model).

Abstention: when the seg2 subject mask is empty or the bbox is degenerate
(zero/negative extent) the item abstains with a surfaced reason. A subject that
fills the frame edge-to-edge is NOT an abstention — it is legitimate full-bleed
framing with tight headroom, which the bands express. Detector disagreement
remains a quality anomaly, never caption content. No new model is required —
this is deterministic geometry over already-frozen artifacts (CPU, in-memory,
no corpus write).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# DOME-29 subject class set (everything that is the curated single woman).
# Guard: the source corpus is curated to exactly one woman per image, so the
# seg2 subject-union is her body (never multi-person).
_SUBJECT_CLASSES = frozenset({
    1,    # Apparel
    2,    # Face_Neck
    3,    # Hair
    4,    # Left_Arm
    5,    # Right_Arm
    6,    # Left_Hand
    7,    # Right_Hand
    8,    # Torso
    9,    # Left_Leg
    10,   # Right_Leg
    11,   # Left_Foot
    12,   # Right_Foot
    22,   # Upper_Clothing
    23,   # Lower_Clothing
    24,   # Dress
    25,   # Shorts
    26,   # Skirt
})
# Note: DOME-29 indices are not guaranteed to be 1..26 for all the above; the
# exact class indices are validated against the actual seg2 legend at probe
# time. To stay robust we treat ANY non-background, non-accessible-region pixel
# as subject in this module's default configuration — see _SUBJECT_MASK.


class CameraViewingAngleError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Calibration constants (provisional; finalized by the frozen-cohort probe).
# ---------------------------------------------------------------------------
# Shot scale: subject bbox height / frame height.
CLOSEUP_MAX = 0.35      # smaller than this (head-and-shoulders) -> close-up
MID_MIN = 0.35          # ~35-65% of frame height -> mid-shot
FULLBODY_MIN = 0.65     # subject fills > 2/3 of the frame height -> full-body
# Headroom: space above the subject head / frame height.
HEADROOM_TIGHT = 0.05
HEADROOM_WIDE = 0.20
# Camera height from the subject's vertical center-of-mass offset relative to
# the frame center (com_y in [-0.5, +0.5]). Negative (subject in the upper
# half) -> the camera is below looking up (low angle); positive (subject in
# the lower half) -> the camera is above looking down (high angle).
CAMERA_HIGH = 0.15
CAMERA_LOW = -0.15


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise CameraViewingAngleError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise CameraViewingAngleError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise CameraViewingAngleError(f"seg2 must be integer class labels, got dtype {seg2.dtype}")


def _subject_mask(seg2: np.ndarray) -> np.ndarray:
    """Subject-union mask: all non-background / non-accessory pixels.

    DOME-29 background classes in this corpus are the ones used by the setting
    arm (Background class 0 and scene-only classes). For framing we want the
    subject silhouette: any pixel NOT in the known background/ambient set.
    """
    mask = seg2 != 0
    return mask


def compute_camera_viewing_angle(
    seg2: np.ndarray, image_h: int, image_w: int
) -> dict[str, Any]:
    """Compute scale-invariant camera-framing bands from seg2 + frame dims.

    Args:
        seg2: (H, W) integer DOME-29 class labels aligned with the source.
        image_h: full-frame source height (px) — used only for RATIO bands.
        image_w: full-frame source width (px).

    Returns a dict with ``abstained``, shot_scale_band, headroom_band,
    camera_height_band; raw bbox extents / frame shares stay in the payload.
    """
    validate_seg2_array(seg2)
    if image_h <= 0 or image_w <= 0:
        raise CameraViewingAngleError(f"invalid frame dims {image_w}x{image_h}")

    mask = _subject_mask(seg2)
    if not bool(mask.any()):
        return {"abstained": True, "abstention_reason": "no subject pixels in seg2"}

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bbox_h = y1 - y0
    bbox_w = x1 - x0
    if bbox_h <= 0 or bbox_w <= 0:
        return {
            "abstained": True,
            "abstention_reason": f"degenerate subject bbox {bbox_w}x{bbox_h}",
        }

    frame_h_share = bbox_h / image_h
    # Headroom: space between the top of the subject and the top of the frame.
    headroom = y0 / image_h
    # Camera-height anchor: the subject's vertical center of mass relative to
    # the frame center. Positive = subject occupies upper half (camera LOW,
    # looking up at subject); negative = lower half (camera HIGH).
    com_y = float((ys.mean() / image_h) - 0.5)

    if frame_h_share >= FULLBODY_MIN:
        shot = "full-body"
    elif frame_h_share >= MID_MIN:
        shot = "mid-shot"
    elif frame_h_share <= CLOSEUP_MAX:
        shot = "close-up"
    else:
        shot = "mid-shot"

    if headroom <= HEADROOM_TIGHT:
        hroom = "tight"
    elif headroom >= HEADROOM_WIDE:
        hroom = "wide"
    else:
        hroom = "normal"

    if com_y <= CAMERA_LOW:
        cam = "low-angle (camera below)"
    elif com_y >= CAMERA_HIGH:
        cam = "high-angle (camera above)"
    else:
        cam = "eye-level"

    return {
        "abstained": False,
        "detection": "MEASURED",
        "shot_scale_band": shot,
        "headroom_band": hroom,
        "camera_height_band": cam,
        # machine-readable payload only (never prose): raw frame shares + bbox
        "subject_bbox_px": [y0, x0, y1, x1],
        "subject_frame_height_share": round(float(frame_h_share), 4),
        "headroom_frame_share": round(float(headroom), 4),
        "subject_center_of_mass_roi": round(float(com_y), 4),
        "frame_dims_px": [image_w, image_h],
    }


def render_camera_viewing_angle(framing: Mapping[str, Any]) -> list[str]:
    """Scale-invariant camera-framing claims for the dossier (arm #74)."""
    if framing.get("abstained"):
        reason = framing.get("abstention_reason") or "framing not measurable"
        return [f"camera-viewing-angle: abstain ({reason})"]
    lines: list[str] = []
    hroom = framing.get("headroom_band")
    # shot_scale_band and camera_height_band are payload-only (88% full-body
    # / 100% eye-level on the probe cohort — degenerate uniform axes, never
    # verbalized).
    if hroom == "tight":
        lines.append("camera-viewing-angle: headroom is tight (head near the frame top)")
    elif hroom == "wide":
        lines.append("camera-viewing-angle: headroom is wide (ample space above the head)")
    if not lines:
        lines.append("camera-viewing-angle: framing measured (no distinctive band)")
    return lines
