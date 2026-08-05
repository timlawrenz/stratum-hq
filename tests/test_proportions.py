"""TDD coverage for deterministic body-type/proportion measurements from pose2.

These measurements are the deterministic evidence for arm #32 (body-type).
They must be continuous/proportional (never closed taxonomies) per Tim's
preferences, computed from existing `pose2` Goliath-308 keypoints, and
deliver exactly-one-subject semantics.
"""

from __future__ import annotations

from pathlib import Path

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


def test_compute_proportions_abstains_on_foreshortened_hip_plane() -> None:
    """Width ratios are only body measurements when both segments share an
    imaging plane. A reclining/diagonal subject with the hip pair near-vertical
    (edge-on, foreshortened) must ABSTAIN, not emit a huge garbage ratio."""
    # Replicating the poolside artifact: shoulders ~horizontal, hips near-vertical
    p = _pose(
        {
            "left_shoulder": (619.0, 437.0, 0.9),
            "right_shoulder": (548.0, 554.0, 0.9),
            "left_hip": (1057.0, 670.0, 0.9),
            "right_hip": (1050.0, 645.0, 0.9),
            "left_knee": (1200.0, 700.0, 0.8),
            "right_knee": (1195.0, 680.0, 0.8),
            "left_ankle": (1568.0, 777.0, 0.8),
            "right_ankle": (1555.0, 772.0, 0.8),
            "nose": (560.0, 497.0, 0.8),
        }
    )
    m = compute_proportions(p)
    # Shoulders and hips both present, so raw widths compute — but the hip pair
    # is near-vertical (edge-on, ~74°) vs shoulders ~-59°: different planes.
    assert m["between_shoulders"] is not None
    assert m["between_hips"] is not None
    assert m["shoulder_hip_ratio"] is None  # must abstain, not emit 5.x
    # abstention reason surfaceable for the serializer
    assert m.get("shoulder_hip_ratio_abstention_reason")


def test_compute_proportions_abstains_on_implausible_ratio() -> None:
    """Even when both segments are near-horizontal, a ratio outside the human
    plausible band (e.g. >2.4, a giraffe-level shoulder) is a projection
    artifact and must abstain rather than be verbalized."""
    p = _pose(
        {
            "left_shoulder": (100.0, 100.0, 0.9),
            "right_shoulder": (260.0, 100.0, 0.9),  # 160px shoulders
            "left_hip": (145.0, 240.0, 0.9),
            "right_hip": (155.0, 240.0, 0.9),       # 10px hips -> ratio ~14
            "left_knee": (140.0, 380.0, 0.8),
            "right_knee": (160.0, 380.0, 0.8),
            "left_ankle": (138.0, 500.0, 0.8),
            "right_ankle": (162.0, 500.0, 0.8),
            "nose": (180.0, 40.0, 0.8),
        }
    )
    m = compute_proportions(p)
    assert m["between_shoulders"] is not None
    assert m["between_hips"] is not None
    assert m["shoulder_hip_ratio"] is None
    assert m.get("shoulder_hip_ratio_abstention_reason")


def test_compute_proportions_no_subject_symbolizes_absent() -> None:
    m = compute_proportions(np.zeros((308, 3), dtype=np.float32))
    assert m["subject_present"] is False
    assert m["shoulder_hip_ratio"] is None


def test_proportions_serialization_verbalizes_ratios_only_no_px(tmp_path: Path) -> None:
    """The verbalized evidence must contain scale-invariant ratios and explicitly
    not verbalize absolute pixel values (camera-frame-dependent, not useful to a
    text-to-image model). Raw px stay in the JSON payload, never in the prompt."""
    from research_harness.stage_b import _serialize_proportions

    m = compute_proportions(_normal_pose())
    assert m["shoulder_hip_ratio"] is not None
    assert m["leg_torso_ratio"] is not None
    text = _serialize_proportions(m)
    # ratios present
    assert "shoulder:hip width ratio" in text
    assert "leg:torso length ratio" in text
    # no absolute pixel units verbalized
    assert "px" not in text.lower()
    assert "pixel" not in text.lower()
    # raw width keys are NOT verbalized
    assert "between_sh" not in text
    assert "torso_length" not in text
    # abstention is explicit, never a fabricated number
    m2 = compute_proportions(_pose({"left_shoulder": (1, 1, 0.9)}))
    t2 = _serialize_proportions(m2)
    assert "not measurable" in t2 or "no reliable body-keypoint" in t2
