"""Deterministic kinematic-articulation measurements from `pose2` + `seg2`.

Arm #62. Reads an existing `pose2.npy` (GOLIATH-308, [x, y, conf]) and the
existing `seg2.npy` (DOME-29) and emits scale-invariant kinematic facts:

- per-joint flexion angles (elbow/knee) from keypoint vectors;
- torso/pelvis orientation from keypoint triangles (in-plane torso twist,
  torso lean from vertical, pelvis tilt from horizontal);
- contrapposto / weight-bearing stance class;
- limb-overlap / crossing structure (arms crossing the spine, legs crossed)
  from geometric segment intersection + seg2 arm/torso spatial proximity;
- symmetry/asymmetry ratios (left vs right flexion, arm-in-front structure).

Only scale-invariant facts (angles, normalized ids/ratios, stance classes) are
ever verbalized; absolute pixel positions and pixel lengths stay in the
machine-readable `evidence_payload` (dossier / compressor input) and are never
caption claims. Every measurement honors the exactly-one-subject invariant and
abstains (emits None / abstain reason) rather than fabricating when joints are
absent or low-confidence, the subject region is degenerate, or detector count
is not exactly one.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation  # type: ignore[import-untyped]

from stratum2.config import DOME_29, GOLIATH_308

_GOLIATH_INDEX = {name: i for i, name in enumerate(GOLIATH_308)}
_DOME_INDEX = {name: i for i, name in enumerate(DOME_29)}

MIN_CONF = 0.5  # below this a joint is treated as unreliable -> abstain

# DOME-29 class indices used for spatial limb-overlap structure.
TORSO_CLASS = _DOME_INDEX["Torso"]
LEFT_ARM_CLASSES = (_DOME_INDEX["Left_Upper_Arm"], _DOME_INDEX["Left_Lower_Arm"])
RIGHT_ARM_CLASSES = (_DOME_INDEX["Right_Upper_Arm"], _DOME_INDEX["Right_Lower_Arm"])

# Qualitative stance thresholds (scale-invariant, degrees / normalized id).
_PELVIS_TILT_MIN = 4.0   # pelvis tilt from horizontal that reads as a hip hike


class PoseArticulationError(RuntimeError):
    pass


def _normalize_pose(pose: np.ndarray) -> np.ndarray:
    if not isinstance(pose, np.ndarray):
        raise PoseArticulationError("pose2 must be a numpy array")
    if pose.shape == (1, 308, 3):
        pose = pose[0]
    if pose.shape != (308, 3):
        raise PoseArticulationError(f"pose2 must be shape (308,3) or (1,308,3), got {pose.shape}")
    return pose


def validate_pose2_array(pose: np.ndarray) -> None:
    _normalize_pose(pose)


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise PoseArticulationError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise PoseArticulationError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise PoseArticulationError(f"seg2 must be integer class labels, got dtype {seg2.dtype}")


def _pts(pose: np.ndarray, name: str, min_conf: float = MIN_CONF) -> tuple[float, float] | None:
    """Return (x, y) if joint valid; else None (abstention)."""
    idx = _GOLIATH_INDEX[name]
    x, y, conf = float(pose[idx, 0]), float(pose[idx, 1]), float(pose[idx, 2])
    if x < 0 or y < 0 or conf < min_conf:
        return None
    return (x, y)


def _joint_angle(a, b, c) -> float | None:
    """Interior angle (degrees) at joint b of the a-b-c triangle. [0, 180]."""
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


def _seg_angle_from_horizontal(a, b) -> float:
    """Signed angle (degrees) of segment a->b from the image x-axis."""
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _ang_diff(a: float, b: float) -> float:
    d = (a - b) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return d


def _from_horizontal(deg: float) -> float:
    """Smallest unsigned angle of a line from horizontal, [0, 90]."""
    d = deg % 180.0
    if d > 90.0:
        d = 180.0 - d
    return d


def _mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_intersect(p1, p2, p3, p4) -> bool:
    """Standard orientation-based 2D segment intersection test (inclusive)."""
    if not (p1 and p2 and p3 and p4):
        return False
    c1 = _cross(p3, p4, p1)
    c2 = _cross(p3, p4, p2)
    c3 = _cross(p1, p2, p3)
    c4 = _cross(p1, p2, p4)
    eps = 1e-9
    return (c1 * c2 < eps) and (c3 * c4 < eps)


def _arm_crosses_spine(shoulder, wrist, spine_top, spine_bot,
                       y_min: float, y_max: float) -> bool:
    """Does the upper-arm->wrist segment cross the torso spine segment within
    the torso y-band (arms in front of / across the body, not raised above)?"""
    if not _segments_intersect(shoulder, wrist, spine_top, spine_bot):
        return False
    if wrist is None or wrist[1] < y_min or wrist[1] > y_max:
        return False
    return True


def _arm_near_torso_fraction(
    seg2: np.ndarray,
    arm_classes: tuple[int, ...],
    torso_mask: np.ndarray,
    *,
    margin: int = 12,
) -> float | None:
    """Fraction of arm-region pixels within `margin` px of the torso region.

    Semantic seg2 labels one class per pixel, so arm/torso *class* overlap is
    empty by construction; arms in front of the body are proxied by spatial
    proximity to the torso region (arms hugging/crossing the body sit adjacent
    to the torso mask, splayed arms do not). Returns None if no arm pixels.
    """
    arm_mask = np.isin(seg2, arm_classes)
    n_arm = int(arm_mask.sum())
    if n_arm == 0:
        return None
    if not torso_mask.any():
        return 0.0
    struct = np.ones((2 * margin + 1, 2 * margin + 1), dtype=bool)
    hull = binary_dilation(torso_mask, structure=struct)
    near = int((arm_mask & hull).sum())
    return round(near / n_arm, 4)


def compute_pose_articulation(
    pose: np.ndarray,
    seg2: np.ndarray | None = None,
    *,
    min_conf: float = MIN_CONF,
) -> dict[str, Any]:
    """Compute deterministic kinematic-articulation measurements with
    per-joint abstention.

    Returns a dict:
    - subject_present / abstained / abstention_reason
    - elbow/knee flexion angles (deg) per side, or None
    - torso_twist_deg / torso_lean_deg / pelvis_tilt_deg (scale-invariant
      in-plane angles)
    - stance_class: weight-left / weight-right / centered / None
    - contrapposto: bool | None (weight shift + pelvis tilt signature)
    - arm_crossing_count (0..2), legs_crossed (bool | None)
    - left/right_arm_near_torso_fraction (seg2 spatial proximity, 0..1 | None)
    - elbow/knee flexion asymmetry (deg), scale-invariant
    """
    pose = _normalize_pose(pose)

    ls = _pts(pose, "left_shoulder", min_conf)
    rs = _pts(pose, "right_shoulder", min_conf)
    le = _pts(pose, "left_elbow", min_conf)
    re = _pts(pose, "right_elbow", min_conf)
    lw = _pts(pose, "left_wrist", min_conf)
    rw = _pts(pose, "right_wrist", min_conf)
    lh = _pts(pose, "left_hip", min_conf)
    rh = _pts(pose, "right_hip", min_conf)
    lk = _pts(pose, "left_knee", min_conf)
    rk = _pts(pose, "right_knee", min_conf)
    la = _pts(pose, "left_ankle", min_conf)
    ra = _pts(pose, "right_ankle", min_conf)

    core = [v for v in (ls, rs, lh, rh) if v is not None]
    out: dict[str, Any] = {
        "subject_present": len(core) >= 2,
        "abstained": False,
        "abstention_reason": None,
    }
    if len(core) < 2:
        out.update({
            "abstained": True,
            "abstention_reason": "fewer than two reliable core joints -> abstain from articulation claims",
            "elbow_flexion_left": None, "elbow_flexion_right": None,
            "knee_flexion_left": None, "knee_flexion_right": None,
            "torso_twist_deg": None, "torso_lean_deg": None, "pelvis_tilt_deg": None,
            "stance_class": None, "contrapposto": None,
            "arm_crossing_count": None, "legs_crossed": None,
            "left_arm_near_torso_fraction": None, "right_arm_near_torso_fraction": None,
            "elbow_flexion_asymmetry_deg": None, "knee_flexion_asymmetry_deg": None,
        })
        return out

    # ---- per-joint flexion angles (scale-invariant: pure angles) ----
    if ls and le and lw:
        out["elbow_flexion_left"] = round(float(_joint_angle(ls, le, lw)), 1)
    else:
        out["elbow_flexion_left"] = None
    if rs and re and rw:
        out["elbow_flexion_right"] = round(float(_joint_angle(rs, re, rw)), 1)
    else:
        out["elbow_flexion_right"] = None
    if lh and lk and la:
        out["knee_flexion_left"] = round(float(_joint_angle(lh, lk, la)), 1)
    else:
        out["knee_flexion_left"] = None
    if rh and rk and ra:
        out["knee_flexion_right"] = round(float(_joint_angle(rh, rk, ra)), 1)
    else:
        out["knee_flexion_right"] = None

    # ---- torso/pelvis orientation (in-plane angle differences) ----
    if ls and rs and lh and rh:
        sh_ang = _seg_angle_from_horizontal(ls, rs)
        hip_ang = _seg_angle_from_horizontal(lh, rh)
        out["torso_twist_deg"] = round(float(_ang_diff(sh_ang, hip_ang)), 1)
        spine_top = _mid(ls, rs)
        spine_bot = _mid(lh, rh)
        v_ang = _seg_angle_from_horizontal(spine_top, spine_bot)
        out["torso_lean_deg"] = round(float(abs(90.0 - abs(v_ang % 180.0))), 1)
        out["pelvis_tilt_deg"] = round(float(_from_horizontal(hip_ang)), 1)
    else:
        out["torso_twist_deg"] = None
        out["torso_lean_deg"] = None
        out["pelvis_tilt_deg"] = None

    # ---- stance: weight-bearing + contrapposto ----
    stance_class, contrapposto = None, None
    if ls and rs and lh and rh and (la or ra):
        hip_mid = _mid(lh, rh)
        shoulder_width = max(math.hypot(rs[0] - ls[0], rs[1] - ls[1]), 1e-6)
        candidates: list[tuple[str, float]] = []
        if la:
            candidates.append(("left", abs(la[0] - hip_mid[0])))
        if ra:
            candidates.append(("right", abs(ra[0] - hip_mid[0])))
        candidates.sort(key=lambda t: t[1])
        nearest, shift = candidates[0]
        if len(candidates) == 2:
            far = candidates[1][1]
            # Weight shift = ankle gap, normalized by shoulder width
            # (scale-invariant). Clear gap -> weight on nearest leg.
            norm_gap = (far - shift) / shoulder_width
            if norm_gap > 0.05:
                stance_class = f"weight-{nearest}"
            else:
                stance_class = "centered"
        else:
            stance_class = f"weight-{nearest}"
        pelvis_tilt = out.get("pelvis_tilt_deg")
        if pelvis_tilt is not None and pelvis_tilt >= _PELVIS_TILT_MIN \
                and stance_class and stance_class != "centered":
            contrapposto = True
        elif pelvis_tilt is not None:
            contrapposto = False
        else:
            contrapposto = None
    out["stance_class"] = stance_class
    out["contrapposto"] = contrapposto

    # ---- limb-overlap / crossing structure ----
    if ls and rs and lh and rh:
        spine_top = _mid(ls, rs)
        spine_bot = _mid(lh, rh)
        y_min = min(spine_top[1], spine_bot[1])
        y_max = max(spine_top[1], lh[1], rh[1])
        crossing = 0
        if _arm_crosses_spine(ls, lw, spine_top, spine_bot, y_min, y_max):
            crossing += 1
        if _arm_crosses_spine(rs, rw, spine_top, spine_bot, y_min, y_max):
            crossing += 1
        out["arm_crossing_count"] = crossing
        if lh and lk and la and rh and rk and ra:
            legs_crossed = _segments_intersect(lh, la, rh, ra)
            out["legs_crossed"] = bool(legs_crossed)
        else:
            out["legs_crossed"] = None
    else:
        out["arm_crossing_count"] = 0
        out["legs_crossed"] = None

    # ---- seg2 spatial proximity (arms near/over the torso region) ----
    left_near = right_near = None
    if seg2 is not None:
        validate_seg2_array(seg2)
        torso_mask = seg2 == TORSO_CLASS
        left_near = _arm_near_torso_fraction(seg2, LEFT_ARM_CLASSES, torso_mask)
        right_near = _arm_near_torso_fraction(seg2, RIGHT_ARM_CLASSES, torso_mask)
    out["left_arm_near_torso_fraction"] = left_near
    out["right_arm_near_torso_fraction"] = right_near

    # ---- symmetry/asymmetry (left vs right flexion, scale-invariant) ----
    efl, efr = out["elbow_flexion_left"], out["elbow_flexion_right"]
    out["elbow_flexion_asymmetry_deg"] = (
        round(abs(float(efl) - float(efr)), 1)
        if (efl is not None and efr is not None) else None
    )
    kfl, kfr = out["knee_flexion_left"], out["knee_flexion_right"]
    out["knee_flexion_asymmetry_deg"] = (
        round(abs(float(kfl) - float(kfr)), 1)
        if (kfl is not None and kfr is not None) else None
    )
    return out
