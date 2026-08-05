"""Deterministic body-type / proportion measurements from `pose2` keypoints.

Arm #32. Reads an existing `pose2.npy` (GOLIATH-308, [x, y, conf]) and emits
continuous ratio measurements (never closed taxonomies). Every measurement must
honor the single-subject invariant and abstain (emit None) rather than
fabricate a value when the supporting joints are absent or low-confidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stratum2.config import GOLIATH_308

# Index lookup derived from the authoritative stratum2 GOLIATH-308 table so the
# measurement module can never drift from the real pose2 layout.
_GOLIATH_INDEX = {name: i for i, name in enumerate(GOLIATH_308)}
GOLIATH_KEYPOINTS = list(GOLIATH_308)

# Core joints used by body-type measurements (must exist in the 308 table).
_REQUIRED_JOINT_NAMES = (
    "nose",
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)

MIN_CONF = 0.5  # below this a joint is treated as unreliable -> abstain


class ProportionError(RuntimeError):
    pass


def _normalize_pose(pose: np.ndarray) -> np.ndarray:
    """Accept (308,3) or (1,308,3) pose2 arrays; return (308,3)."""
    if not isinstance(pose, np.ndarray):
        raise ProportionError("pose2 must be a numpy array")
    if pose.shape == (1, 308, 3):
        pose = pose[0]
    if pose.shape != (308, 3):
        raise ProportionError(f"pose2 must be shape (308,3) or (1,308,3), got {pose.shape}")
    return pose


def validate_pose2_array(pose: np.ndarray) -> None:
    _normalize_pose(pose)


def _pts(pose: np.ndarray, name: str, min_conf: float = MIN_CONF) -> tuple[float, float] | None:
    """Return (x, y) if joint valid; else None (abstention)."""
    idx = _GOLIATH_INDEX[name]
    x, y, conf = float(pose[idx, 0]), float(pose[idx, 1]), float(pose[idx, 2])
    if x < 0 or y < 0 or conf < min_conf:
        return None
    return (x, y)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _mid(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def compute_proportions(pose: np.ndarray, *, min_conf: float = MIN_CONF) -> dict[str, Any]:
    """Compute continuous body-type measurements with per-joint abstention.

    Returns a dict:
    - subject_present: True if >=2 reliable joints (else False, all ratios None)
    - between_shoulders / between_hips: widths in px when both sides reliable
    - shoulder_hip_ratio: width ratio (abs+1) when both available
    - torso_length: shoulder-mid -> hip-mid
    - left/right_leg_length: hip -> ankle per side
    - leg_torso_ratio: mean leg / torso when both available
    - low_confidence_joints: count of joints dropped under `min_conf`
    - asymmetric_available_both_sides: True when both left/right leg available
    Ratios are continuous floats or None (never fabricated).
    """
    validate_pose2_array(pose)
    pose = _normalize_pose(pose)
    ls, rs = _pts(pose, "left_shoulder", min_conf), _pts(pose, "right_shoulder", min_conf)
    lh, rh = _pts(pose, "left_hip", min_conf), _pts(pose, "right_hip", min_conf)
    lk, rk = _pts(pose, "left_knee", min_conf), _pts(pose, "right_knee", min_conf)
    la, ra = _pts(pose, "left_ankle", min_conf), _pts(pose, "right_ankle", min_conf)
    nose = _pts(pose, "nose", min_conf)

    reliable = [v for v in (ls, rs, lh, rh, lk, rk, la, ra, nose) if v is not None]
    # count dropped (present-with-low-conf) joints
    low_conf = 0
    for name in _REQUIRED_JOINT_NAMES:
        idx = _GOLIATH_INDEX[name]
        x, y, conf = float(pose[idx, 0]), float(pose[idx, 1]), float(pose[idx, 2])
        if x >= 0 and y >= 0 and conf < min_conf:
            low_conf += 1

    out: dict[str, Any] = {
        "subject_present": len(reliable) >= 2,
        "low_confidence_joints": low_conf,
        "asymmetric_available_both_sides": (lk is not None and rk is not None and
                                             la is not None and ra is not None),
    }
    if len(reliable) < 2:
        out.update({
            "between_shoulders": None, "between_hips": None, "shoulder_hip_ratio": None,
            "torso_length": None, "left_leg_length": None, "right_leg_length": None,
            "leg_torso_ratio": None,
        })
        return out

    bw_s = _dist(ls, rs) if (ls and rs) else None
    bw_h = _dist(lh, rh) if (lh and rh) else None
    out["between_shoulders"] = round(bw_s, 3) if bw_s is not None else None
    out["between_hips"] = round(bw_h, 3) if bw_h is not None else None
    if bw_s is not None and bw_h is not None and bw_h > 0:
        out["shoulder_hip_ratio"] = round((bw_s + 1.0) / (bw_h + 1.0), 4)
    else:
        out["shoulder_hip_ratio"] = None

    if ls and rs and lh and rh:
        torso = _dist(_mid(ls, rs), _mid(lh, rh))
        out["torso_length"] = round(torso, 3)
    elif ls and lh:
        out["torso_length"] = round(_dist(ls, lh), 3)
    elif rs and rh:
        out["torso_length"] = round(_dist(rs, rh), 3)
    else:
        out["torso_length"] = None

    left_leg = _dist(lh, la) if (lh and la) else None
    right_leg = _dist(rh, ra) if (rh and ra) else None
    out["left_leg_length"] = round(left_leg, 3) if left_leg is not None else None
    out["right_leg_length"] = round(right_leg, 3) if right_leg is not None else None

    leg_vals = [v for v in (left_leg, right_leg) if v is not None]
    if leg_vals and out["torso_length"] and out["torso_length"] > 0:
        mean_leg = float(np.mean(leg_vals))
        out["leg_torso_ratio"] = round(mean_leg / out["torso_length"], 4)
    else:
        out["leg_torso_ratio"] = None
    return out
