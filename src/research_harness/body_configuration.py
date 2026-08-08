"""Deterministic whole-body posture-configuration classification from `pose2`.

Arm #83. NEW deterministic evidence part (no new model). Reads the existing
`pose2.npy` (GOLIATH-308, [x, y, conf]) and uses the `seg2` frame only for the
frame HEIGHT (a denominator for the normalized pelvis-height fraction — the
same convention as the camera-viewing-angle arm's frame-dims use). The
posture class itself is derived entirely from keypoint geometry:

- pelvis-height fraction: mean hip y / frame height (a normalized vertical
  position — standing subjects carry their hips low in the frame, seated
  subjects have elevated hips);
- torso-vs-leg extent ratio: torso vertical extent / hip-to-ankle vertical
  extent (standing legs dominate the frame below the hips; seated legs fold
  up and their vertical extent collapses toward the torso);
- knee flexion: interior angle (deg) at each knee (hip-knee-ankle) — a
  standing leg is near-extended (~150-180 deg), a seated leg is strongly bent;
- torso lean from vertical (shoulder-mid -> hip-mid), the reclined signature.

All of these are scale-invariant (pure ratios / angles), so they survive
cross-picture comparison and a text-to-image model can interpret them.

The arm emits ONE coarse body-configuration band per item (standing / seated /
reclined, or abstain) — the top-level posture CLASS that pose-articulation #62
(per-joint flexion / stance / contrapposto) intentionally does NOT emit.
Redundancy against #62's per-joint signals is checked in the falsified_if.

ONLY scale-invariant facts are verbalized (the coarse class); raw pixel
positions, pixel lengths, and the raw normalized fractions stay in the
machine-readable `evidence_payload` (camera-frame-dependent absolutes are
never caption claims).

Abstention: an item with fewer than ~4 of the 8 core joints (2 shoulders,
2 hips, 2 knees, 2 ankles) reliable, or a configuration whose class is
ambiguous, abstains with a surfaced reason rather than fabricating a posture
class. Detector disagreement remains a quality anomaly, never caption content.
No corpus write; CPU-only, in-memory.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from stratum2.config import DOME_29, GOLIATH_308

_GOLIATH_INDEX = {name: i for i, name in enumerate(GOLIATH_308)}

CORE_MIN_CONF = 0.5

# Calibrated on the frozen 24-item cohort (2026-08-07 probe, see the reference):

# Reclined: torso lean (shoulder-mid -> hip-mid, from vertical) at or above this.
RECLINED_TORSO_LEAN_DEG = 45.0

# Seated: median knee flexion below this AND pelvis elevated (hips above this
# normalized frame-height fraction). A standing leg is near-extended (>= 150);
# a seated leg folds below ~140 and the hips sit high in the frame.
SEATED_MEDIAN_KNEE_FLEXION_DEG = 140.0
SEATED_PELVIS_HEIGHT_MAX = 0.52

# Fallback standing require at least near-extended knees.
STANDING_MEDIAN_KNEE_FLEXION_MIN = 150.0


class BodyConfigurationError(RuntimeError):
    pass


def validate_pose2_array(pose: np.ndarray) -> None:
    if not isinstance(pose, np.ndarray):
        raise BodyConfigurationError("pose2 must be a numpy array")
    if pose.shape == (1, 308, 3):
        pose = pose[0]
    if pose.shape != (308, 3):
        raise BodyConfigurationError(
            f"pose2 must be shape (308,3) or (1,308,3), got {pose.shape}"
        )


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise BodyConfigurationError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise BodyConfigurationError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise BodyConfigurationError(f"seg2 must be integer class labels, got dtype {seg2.dtype}")


def _normalize_pose(pose: np.ndarray) -> np.ndarray:
    validate_pose2_array(pose)
    if pose.shape == (1, 308, 3):
        return pose[0]
    return pose


def _pt(pose: np.ndarray, name: str) -> tuple[float, float] | None:
    idx = _GOLIATH_INDEX[name]
    x, y, conf = float(pose[idx, 0]), float(pose[idx, 1]), float(pose[idx, 2])
    if x < 0 or y < 0 or conf < CORE_MIN_CONF:
        return None
    return (x, y)


def _mid(a: tuple[float, float] | None, b: tuple[float, float] | None) -> tuple[float, float] | None:
    if not (a and b):
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _interior_angle(a, b, c) -> float | None:
    """Interior angle (deg) at joint b of a-b-c. [0, 180]."""
    if not (a and b and c):
        return None
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return None
    cosang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosang))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(values))


def compute_body_configuration(
    pose: np.ndarray,
    seg2: np.ndarray | None = None,
    *,
    frame_h: float | None = None,
) -> dict[str, Any]:
    """Compute the deterministic whole-body configuration class.

    Args:
        pose: (308,3) or (1,308,3) GOLIATH-308 keypoint array.
        seg2: (H, W) DOME-29 labels; used ONLY for the frame height
            (denominator of the normalized pelvis-height fraction). The
            posture class itself is keypoint-only.
        frame_h: explicit frame height override (px). If both seg2 and
            frame_h are provided, frame_h wins; if neither, the frame
            denominator is unavailable and the pelvis-height axis abstains
            (knee flexion / torso lean / leg-extent ratios still fire since
            they are height-free).

    Returns a dict with scale-invariant payload raw values and the coarse
    `posture_class` band (standing / seated / reclined / None-abstain).
    """
    pose = _normalize_pose(pose)

    ls = _pt(pose, "left_shoulder") or _pt(pose, "left_acromion")
    rs = _pt(pose, "right_shoulder") or _pt(pose, "right_acromion")
    lh = _pt(pose, "left_hip")
    rh = _pt(pose, "right_hip")
    lk = _pt(pose, "left_knee")
    rk = _pt(pose, "right_knee")
    la = _pt(pose, "left_ankle")
    ra = _pt(pose, "right_ankle")

    core = [p for p in (ls, rs, lh, rh, lk, rk, la, ra) if p is not None]
    out: dict[str, Any] = {
        "subject_present": len(core) >= 4,
        "abstained": False,
        "abstention_reason": None,
        "posture_class": None,
        "pelvis_height_fraction": None,
        "torso_extent_px": None,
        "leg_extent_px": None,
        "torso_leg_extent_ratio": None,
        "knee_flexion_left_deg": None,
        "knee_flexion_right_deg": None,
        "median_knee_flexion_deg": None,
        "torso_lean_deg": None,
    }
    if len(core) < 4:
        out.update({
            "abstained": True,
            "abstention_reason": (
                "fewer than four reliable core joints -> abstain from posture-class claims"
            ),
        })
        return out

    # Frame height denominator (seg2 is validation-only for frame dims).
    if frame_h is None and seg2 is not None:
        validate_seg2_array(seg2)
        frame_h = float(seg2.shape[0])
    has_frame = frame_h is not None and frame_h > 0

    shoulder_mid = _mid(ls, rs)
    hip_mid = _mid(lh, rh)
    ankle_mid = _mid(la, ra)

    if hip_mid is not None:
        if has_frame:
            out["pelvis_height_fraction"] = round(hip_mid[1] / float(frame_h), 4)

    # Torso-vs-leg extent ratio (vertical extents; both scale together so the
    # ratio is scale-invariant).
    if shoulder_mid is not None and hip_mid is not None and ankle_mid is not None:
        torso_extent = abs(shoulder_mid[1] - hip_mid[1])
        leg_extent = abs(hip_mid[1] - ankle_mid[1])
        out["torso_extent_px"] = round(torso_extent, 1)
        out["leg_extent_px"] = round(leg_extent, 1)
        if torso_extent > 0:
            out["torso_leg_extent_ratio"] = round(leg_extent / torso_extent, 3)

    # Torso lean from vertical (0 = upright).
    if shoulder_mid is not None and hip_mid is not None:
        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        out["torso_lean_deg"] = round(float(math.degrees(math.atan2(abs(dx), abs(dy)))), 1)

    # Knee flexion (deg) per side + median.
    fl = _interior_angle(lh, lk, la) if (lh and lk and la) else None
    fr = _interior_angle(rh, rk, ra) if (rh and rk and ra) else None
    out["knee_flexion_left_deg"] = round(fl, 1) if fl is not None else None
    out["knee_flexion_right_deg"] = round(fr, 1) if fr is not None else None
    med_knee = _median([v for v in (fl, fr) if v is not None])
    out["median_knee_flexion_deg"] = round(med_knee, 1) if med_knee is not None else None

    # ---- Classify (scale-invariant, rule priority: reclined > seated > standing) ----
    # Re-cut 2026-08-07 from the frozen-cohort probe: the cohort separates
    # cleanly on torso lean (~upright <= 19, reclining >= 41, gap 19-41) and on
    # knee flexion (bent <= ~120, near-extended >= ~150). Classification fires
    # on whichever signal is PRESENT (knee-bend alone -> seated; torso-lean
    # alone -> reclined); overlapping cues resolve reclined-first. A gray-zone
    # knee (140-150) or a missing knee/lean with no discriminating cue abstains
    # honestly rather than guessing a posture class.
    lean = out["torso_lean_deg"]
    med_knee = out["median_knee_flexion_deg"]
    if lean is not None and lean >= RECLINED_TORSO_LEAN_DEG:
        out["posture_class"] = "reclined"
    elif med_knee is not None:
        if med_knee < SEATED_MEDIAN_KNEE_FLEXION_DEG:
            out["posture_class"] = "seated"
        elif med_knee >= STANDING_MEDIAN_KNEE_FLEXION_MIN:
            out["posture_class"] = "standing"
        else:
            # 140-150 gray zone: neither clearly seated nor clearly standing.
            out.update({
                "abstained": True,
                "abstention_reason": (
                    "ambiguous body configuration (knee flexion in the 140-150 gray zone; "
                    "no reclined torso signal)"
                ),
            })
    else:
        out.update({
            "abstained": True,
            "abstention_reason": (
                "ambiguous body configuration (no reliable knee or torso-signature)"
            ),
        })
    return out


def render_body_configuration(config: Mapping[str, Any]) -> list[str]:
    """Scale-invariant body-configuration claim for the dossier (arm #83)."""
    if config.get("abstained"):
        reason = config.get("abstention_reason") or "body configuration not measurable"
        return [f"body-configuration: abstain ({reason})"]
    if not config:
        # Dimension not measured for this item (e.g. non-body-configuration
        # runs) — emit no claim, never a fabricated posture statement.
        return []
    cls = config.get("posture_class")
    if cls == "standing":
        return ["body-configuration: subject is standing (upright, legs near-extended)"]
    if cls == "seated":
        return ["body-configuration: subject is seated (hips elevated, knees bent)"]
    if cls == "reclined":
        return ["body-configuration: subject is reclining (torso near-horizontal)"]
    return []
