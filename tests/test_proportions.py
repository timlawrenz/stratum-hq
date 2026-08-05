"""TDD coverage for deterministic body-type/proportion measurements from pose2.

These measurements are the deterministic evidence for arm #32 (body-type).
They must be continuous/proportional (never closed taxonomies) per Tim's
preferences, computed from existing `pose2` Goliath-308 keypoints, and
deliver exactly-one-subject semantics.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.proportions import (
    ProportionError,
    GOLIATH_KEYPOINTS,
    compute_proportions,
    validate_pose2_array,
)


def _pose(keypoints: dict[str, tuple[float, float, float]]) -> np.ndarray:
    """Build a GOLIATH-308-shaped pose2 array with a subset of keypoints filled.

    The real pose2 is (308, 3) [x, y, conf]. Callers name only the joints they
    care about; everything else stays at -1 (absent) with conf 0.
    """
    a = np.full((308, 3), -1.0, dtype=np.float32)
    for name, (x, y, conf) in keypoints.items():
        idx = GOLIATH_KEYPOINTS.index(name)
        a[idx] = (x, y, conf)
    return a


def test_validate_pose2_array_accepts_308x3() -> None:
    p = np.zeros((308, 3), dtype=np.float32)
    validate_pose2_array(p)  # must not raise


def test_validate_pose2_array_rejects_wrong_shape() -> None:
    with pytest.raises(ProportionError):
        validate_pose2_array(np.zeros((300, 3), dtype=np.float32))


def _normal_pose() -> np.ndarray:
    # A "normal" standing pose: shoulders wider than hips, torso taller than head.
    return _pose(
        {
            "left_shoulder": (100.0, 120.0, 0.9),
            "right_shoulder": (160.0, 120.0, 0.9),
            "left_hip": (105.0, 300.0, 0.9),
            "right_hip": (155.0, 300.0, 0.9),
            "left_knee": (103.0, 430.0, 0.9),
            "right_knee": (157.0, 430.0, 0.9),
            "left_ankle": (102.0, 560.0, 0.8),
            "right_ankle": (158.0, 560.0, 0.8),
            "nose": (130.0, 75.0, 0.8),
        }
    )


def test_compute_proportions_shoulder_hip_ratio() -> None:
    m = compute_proportions(_normal_pose())
    # shoulders 60 wide vs hips 50 wide -> ratio ~1.2
    assert m["shoulder_hip_ratio"] == pytest.approx(1.2, abs=0.02)
    assert m["between_shoulders"] == pytest.approx(60.0, abs=0.5)
    assert m["between_hips"] == pytest.approx(50.0, abs=0.5)


def test_compute_proportions_leg_torso_ratio() -> None:
    m = compute_proportions(_normal_pose())
    # leg length ~ (hip->ankle) ~260+ ; torso (shoulder->hip) ~180
    assert m["leg_torso_ratio"] == pytest.approx(1.45, rel=0.05)
    assert m["left_leg_length"] == pytest.approx(261.0, rel=0.01)
    assert m["torso_length"] == pytest.approx(180.0, rel=0.01)


def test_compute_proportions_constrained_on_tight_left_crop() -> None:
    # Right-side joints absent: must abstain, not fabricate a right-side measurement.
    m = compute_proportions(_pose({
        "left_shoulder": (100.0, 120.0, 0.9),
        "left_hip": (105.0, 300.0, 0.9),
        "left_knee": (103.0, 430.0, 0.9),
        "left_ankle": (102.0, 560.0, 0.8),
        "nose": (130.0, 75.0, 0.8),
    }))
    assert m["subject_present"] is True
    assert m["left_leg_length"] is not None
    assert m["right_leg_length"] is None  # abstention, not hallucination
    assert m["asymmetric_available_both_sides"] is False
    # shoulder_hip_ratio needs both sides -> abstain
    assert m["shoulder_hip_ratio"] is None


def test_compute_proportions_requires_confident_joints() -> None:
    p = _normal_pose()
    p[GOLIATH_KEYPOINTS.index("left_shoulder"), 2] = 0.1  # low conf
    m = compute_proportions(p)
    assert m["subject_present"] is True
    # a low-conf joint should reduce which ratios are emitted (abstention), never lie
    assert m["low_confidence_joints"] >= 1


def test_compute_proportions_no_subject_symbolizes_absent() -> None:
    m = compute_proportions(np.zeros((308, 3), dtype=np.float32))
    assert m["subject_present"] is False
    assert m["shoulder_hip_ratio"] is None
