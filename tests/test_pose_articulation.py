"""TDD coverage for deterministic pose-articulation / kinematic measurements.

These measurements are the deterministic evidence for arm #62 (pose-
articulation). They must be scale-invariant kinematic facts (joint flexion
angles, in-plane torso/pelvis orientation, stance/contrapposto class,
limb-overlap structure, flexion asymmetry) computed from existing `pose2`
GOLIATH-308 keypoints + `seg2` DOME-29 masks, honoring exactly-one-subject
abstention — absolute pixel positions/lengths stay in the machine-readable
payload, never as caption claims.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.pose_articulation import (
    PoseArticulationError,
    compute_pose_articulation,
    validate_pose2_array,
    validate_seg2_array,
)


def _pose(keypoints: dict[str, tuple[float, float, float]]) -> np.ndarray:
    """Build a GOLIATH-308-shaped pose2 array with a subset of keypoints filled."""
    from research_harness.proportions import GOLIATH_KEYPOINTS

    a = np.full((308, 3), -1.0, dtype=np.float32)
    for name, (x, y, conf) in keypoints.items():
        idx = GOLIATH_KEYPOINTS.index(name)
        a[idx] = (x, y, conf)
    return a


def _seg(mask: dict[int, set[tuple[int, int]]], height: int = 100, width: int = 100) -> np.ndarray:
    a = np.zeros((height, width), dtype=np.uint8)
    for cls, cells in mask.items():
        for (y, x) in cells:
            a[y, x] = cls
    return a


def _standing_pose() -> np.ndarray:
    # Neutral standing subject: shoulders at y=120, hips at y=300, legs straight.
    return _pose(
        {
            "left_shoulder": (100.0, 120.0, 0.9),
            "right_shoulder": (160.0, 120.0, 0.9),
            "left_elbow": (96.0, 180.0, 0.9),
            "right_elbow": (164.0, 180.0, 0.9),
            "left_wrist": (94.0, 240.0, 0.9),
            "right_wrist": (166.0, 240.0, 0.9),
            "left_hip": (105.0, 300.0, 0.9),
            "right_hip": (155.0, 300.0, 0.9),
            "left_knee": (103.0, 430.0, 0.9),
            "right_knee": (157.0, 430.0, 0.9),
            "left_ankle": (102.0, 560.0, 0.9),
            "right_ankle": (158.0, 560.0, 0.9),
        }
    )


def test_subject_present_and_stance_centered() -> None:
    m = compute_pose_articulation(_standing_pose(), _seg({}))
    assert m["subject_present"] is True
    assert m["abstained"] is False
    # Both ankles about equally far from hip midline -> centered stance.
    assert m["stance_class"] == "centered"
    # Straight legs -> extended at the knee.
    assert m["knee_flexion_left"] is not None and m["knee_flexion_left"] > 160.0
    assert m["knee_flexion_right"] is not None and m["knee_flexion_right"] > 160.0


def test_elbow_flexion_bands_bent_vs_extended() -> None:
    from research_harness.pose_articulation import compute_pose_articulation

    bent = _pose(
        {
            "left_shoulder": (100.0, 120.0, 0.9),
            "right_shoulder": (160.0, 120.0, 0.9),
            "left_elbow": (130.0, 160.0, 0.9),
            "right_elbow": (170.0, 160.0, 0.9),
            "left_wrist": (132.0, 118.0, 0.9),
            "right_wrist": (168.0, 118.0, 0.9),
            "left_hip": (105.0, 300.0, 0.9),
            "right_hip": (155.0, 300.0, 0.9),
        }
    )
    m = compute_pose_articulation(bent, _seg({}))
    # Forearm folds back up ~parallel to the upper arm (acute interior angle):
    # clearly bent at the elbow, well below the 135-degree band threshold.
    assert m["elbow_flexion_left"] is not None
    assert m["elbow_flexion_left"] < 135.0
    assert m["elbow_flexion_right"] is not None
    assert m["elbow_flexion_right"] < 135.0


def test_weight_right_stance() -> None:
    from research_harness.pose_articulation import compute_pose_articulation

    p = _pose(
        {
            "left_shoulder": (110.0, 120.0, 0.9),
            "right_shoulder": (170.0, 120.0, 0.9),
            "left_hip": (120.0, 300.0, 0.9),
            "right_hip": (160.0, 300.0, 0.9),
            "left_ankle": (70.0, 560.0, 0.9),   # far left, off to the side
            "right_ankle": (150.0, 560.0, 0.9), # near hip midline (x=140)
        }
    )
    m = compute_pose_articulation(p, _seg({}))
    assert m["stance_class"] == "weight-right"


def test_arm_near_torso_fraction_from_seg2() -> None:
    from research_harness.pose_articulation import compute_pose_articulation

    # Torso class 22 occupies the center; left upper arm (11) sits adjacent.
    seg = _seg(
        {
            22: {(y, x) for y in range(40, 60) for x in range(45, 55)},
            11: {(y, x) for y in range(40, 60) for x in range(38, 45)},
        }
    )
    m = compute_pose_articulation(_standing_pose(), seg)
    # Arm pixels are all within 12px of the torso region -> near fraction ~1.0.
    assert m["left_arm_near_torso_fraction"] is not None
    assert m["left_arm_near_torso_fraction"] > 0.9
    assert m["right_arm_near_torso_fraction"] == 0.0 or m["right_arm_near_torso_fraction"] is None


def test_abstains_when_core_joints_absent() -> None:
    from research_harness.pose_articulation import compute_pose_articulation

    p = _pose(
        {
            "left_shoulder": (100.0, 120.0, 0.9),
            "right_shoulder": (160.0, 120.0, 0.9),
            # no hips
        }
    )
    m = compute_pose_articulation(p, _seg({}))
    assert m["subject_present"] is True
    assert m["stance_class"] is None
    assert m["contrapposto"] is None


def test_abstains_when_low_confidence() -> None:
    from research_harness.pose_articulation import compute_pose_articulation

    p = _pose(
        {
            "left_shoulder": (100.0, 120.0, 0.3),
            "right_shoulder": (160.0, 120.0, 0.3),
            "left_hip": (105.0, 300.0, 0.3),
            "right_hip": (155.0, 300.0, 0.3),
        }
    )
    m = compute_pose_articulation(p, _seg({}))
    # All four core joints below MIN_CONF -> subject_present should still be
    # True if at least two pass, but here none pass -> abstain.
    assert m["abstained"] is True
    assert m["knee_flexion_left"] is None


def test_flexion_asymmetry_scale_invariant() -> None:
    from research_harness.pose_articulation import compute_pose_articulation

    p = _pose(
        {
            "left_shoulder": (100.0, 120.0, 0.9),
            "right_shoulder": (160.0, 120.0, 0.9),
            "left_elbow": (130.0, 160.0, 0.9),
            "right_elbow": (164.0, 180.0, 0.9),
            "left_wrist": (130.0, 200.0, 0.9),
            "right_wrist": (166.0, 240.0, 0.9),
            "left_hip": (105.0, 300.0, 0.9),
            "right_hip": (155.0, 300.0, 0.9),
        }
    )
    m = compute_pose_articulation(p, _seg({}))
    assert m["elbow_flexion_asymmetry_deg"] is not None
    assert m["elbow_flexion_asymmetry_deg"] > 15.0
