"""TDD coverage for the body-configuration evidence specialist (arm #83).

Deterministic whole-body posture-class classification (standing / seated /
reclined) from pose2 GOLIATH-308 keypoints, with seg2 supplying only the
frame-height denominator. Only the scale-invariant coarse class is verbalized;
raw normalized fractions / pixel extents stay in the machine-readable payload.
Pure and tested without any model; no GPU needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.body_configuration import (
    RECLINED_TORSO_LEAN_DEG,
    SEATED_MEDIAN_KNEE_FLEXION_DEG,
    STANDING_MEDIAN_KNEE_FLEXION_MIN,
    BodyConfigurationError,
    compute_body_configuration,
    render_body_configuration,
    validate_pose2_array,
    validate_seg2_array,
)

from stratum2.config import GOLIATH_308

_G = {name: i for i, name in enumerate(GOLIATH_308)}


def _pose(
    *,
    ls=(300.0, 200.0, 0.9),
    rs=(500.0, 200.0, 0.9),
    lh=(370.0, 550.0, 0.9),
    rh=(430.0, 550.0, 0.9),
    lk=(380.0, 750.0, 0.9),
    rk=(420.0, 750.0, 0.9),
    la=(390.0, 950.0, 0.9),
    ra=(410.0, 950.0, 0.9),
) -> np.ndarray:
    """Default: upright standing figure (hips y=550, frame h=1000) with
    near-extended knees (leg straight 200px below hip) and ankles below."""
    pose = np.zeros((308, 3), dtype=float)
    pose[:, 2] = 1.0
    pose[_G["left_shoulder"]] = ls
    pose[_G["right_shoulder"]] = rs
    pose[_G["left_hip"]] = lh
    pose[_G["right_hip"]] = rh
    pose[_G["left_knee"]] = lk
    pose[_G["right_knee"]] = rk
    pose[_G["left_ankle"]] = la
    pose[_G["right_ankle"]] = ra
    return pose


_H = 1000


def _seg(frame_h: int = _H) -> np.ndarray:
    return np.zeros((frame_h, 800), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

def test_validate_pose2() -> None:
    with pytest.raises(BodyConfigurationError):
        validate_pose2_array(np.zeros((308, 2)))
    with pytest.raises(BodyConfigurationError):
        validate_pose2_array("nope")


def test_validate_seg2() -> None:
    with pytest.raises(BodyConfigurationError):
        validate_seg2_array(np.zeros((5, 5, 1), dtype=np.uint8))
    with pytest.raises(BodyConfigurationError):
        validate_seg2_array(np.zeros((5, 5), dtype=np.float32))


# ---------------------------------------------------------------------------
# Standing (default fixture)
# ---------------------------------------------------------------------------

def test_standing_default() -> None:
    r = compute_body_configuration(_pose(), _seg())
    assert r["posture_class"] == "standing"
    assert r["abstained"] is False
    # median knee flexion near-extended
    assert r["median_knee_flexion_deg"] is not None
    assert r["median_knee_flexion_deg"] >= STANDING_MEDIAN_KNEE_FLEXION_MIN
    # pelvis mid-lower frame
    assert r["pelvis_height_fraction"] == pytest.approx(0.55, abs=0.02)


def test_standing_is_scale_invariant() -> None:
    """Doubling pose + frame height must keep the same coarse class."""
    base = _pose()
    base_seg = _seg()
    r_small = compute_body_configuration(base, base_seg)
    factor = 2.0
    big = _pose(
        ls=(300.0 * factor, 200.0 * factor, 0.9),
        rs=(500.0 * factor, 200.0 * factor, 0.9),
        lh=(370.0 * factor, 550.0 * factor, 0.9),
        rh=(430.0 * factor, 550.0 * factor, 0.9),
        lk=(380.0 * factor, 750.0 * factor, 0.9),
        rk=(420.0 * factor, 750.0 * factor, 0.9),
        la=(390.0 * factor, 950.0 * factor, 0.9),
        ra=(410.0 * factor, 950.0 * factor, 0.9),
    )
    r_big = compute_body_configuration(big, _seg(_H * 2))
    assert r_small["posture_class"] == r_big["posture_class"] == "standing"
    # pelvis fraction identical (ratio, scale-invariant)
    assert r_big["pelvis_height_fraction"] == pytest.approx(
        r_small["pelvis_height_fraction"], abs=1e-3
    )


# ---------------------------------------------------------------------------
# Seated: hips elevated (y=350) + strongly bent knees (knee ~90 deg)
# ---------------------------------------------------------------------------

def _seated_pose():
    """Seated: hips elevated and thigh roughly horizontal (knee at ~hip
    height, horizontally offset), shin vertical below -> thigh-shin ~90 deg."""
    return _pose(
        lh=(370.0, 350.0, 0.9), rh=(430.0, 350.0, 0.9),   # hips elevated
        lk=(300.0, 350.0, 0.9), rk=(500.0, 350.0, 0.9),   # knees at hip height
        la=(300.0, 620.0, 0.9), ra=(500.0, 620.0, 0.9),   # shins vertical down
    )


def test_seated() -> None:
    r = compute_body_configuration(_seated_pose(), _seg())
    assert r["posture_class"] == "seated"
    assert r["abstained"] is False
    assert r["median_knee_flexion_deg"] < SEATED_MEDIAN_KNEE_FLEXION_DEG
    assert r["pelvis_height_fraction"] < 0.52


# ---------------------------------------------------------------------------
# Reclined: torso lean strongly from vertical (hips shifted horizontally far
# from the shoulder midpoint -> torso near-horizontal)
# ---------------------------------------------------------------------------

def _reclined_pose():
    return _pose(
        ls=(300.0, 200.0, 0.9), rs=(500.0, 200.0, 0.9),
        lh=(760.0, 540.0, 0.9), rh=(820.0, 540.0, 0.9),   # hips far right -> lean
    )


def test_reclined() -> None:
    r = compute_body_configuration(_reclined_pose(), _seg())
    assert r["posture_class"] == "reclined"
    assert r["torso_lean_deg"] is not None
    assert r["torso_lean_deg"] >= RECLINED_TORSO_LEAN_DEG


def test_reclined_priority_over_standing_legs() -> None:
    """A reclined torso wins even with near-extended (standing) legs."""
    r = compute_body_configuration(_reclined_pose(), _seg())
    assert r["posture_class"] == "reclined"


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------

def test_sparse_skeleton_abstains() -> None:
    pose = np.zeros((308, 3), dtype=float)
    pose[:, 2] = 0.0
    r = compute_body_configuration(pose, _seg())
    assert r["abstained"] is True
    assert r["posture_class"] is None
    assert r["abstention_reason"]


def test_gray_zone_knee_abstains() -> None:
    """A knee flexion in the 140-150 gray zone with no torso lean abstains."""
    pose = _pose(
        lh=(370.0, 550.0, 0.9), rh=(430.0, 550.0, 0.9),
        lk=(370.0, 700.0, 0.9), rk=(430.0, 700.0, 0.9),   # ~145 deg bend
        la=(510.0, 900.0, 0.9), ra=(290.0, 900.0, 0.9),
    )
    r = compute_body_configuration(pose, _seg())
    # ensure it lands in the gray zone
    assert 140.0 < r["median_knee_flexion_deg"] < 150.0
    assert r["abstained"] is True
    assert "gray zone" in r["abstention_reason"]


def test_no_frame_but_knee_signal_still_classifies() -> None:
    """Without seg2/frame_h the pelvis axis abstains but knee/torso still fire."""
    r = compute_body_configuration(_seated_pose(), None)
    assert r["posture_class"] == "seated"
    assert r["pelvis_height_fraction"] is None


def test_frame_h_override() -> None:
    r = compute_body_configuration(_pose(), None, frame_h=1000.0)
    assert r["pelvis_height_fraction"] == pytest.approx(0.55, abs=0.02)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_standing() -> None:
    r = compute_body_configuration(_pose(), _seg())
    lines = render_body_configuration(r)
    assert any("standing" in ln for ln in lines)


def test_render_seated() -> None:
    r = compute_body_configuration(_seated_pose(), _seg())
    lines = render_body_configuration(r)
    assert any("seated" in ln for ln in lines)


def test_render_reclined() -> None:
    r = compute_body_configuration(_reclined_pose(), _seg())
    lines = render_body_configuration(r)
    assert any("reclining" in ln for ln in lines)


def test_render_not_measured_empty() -> None:
    # Empty dict = dimension not measured -> no fabricated posture claim.
    assert render_body_configuration({}) == []


def test_render_abstain() -> None:
    r = {"abstained": True, "abstention_reason": "no subject"}
    lines = render_body_configuration(r)
    assert any("abstain" in ln for ln in lines)


def test_render_no_pixel_values_in_prose() -> None:
    r = compute_body_configuration(_pose(), _seg())
    joined = " ".join(render_body_configuration(r))
    import re
    numbers = re.findall(r"\d{3,}", joined)  # no 3+ digit pixel values
    assert not numbers


# ---------------------------------------------------------------------------
# Threshold sanity (must not be trivially degenerate)
# ---------------------------------------------------------------------------

def test_thresholds_are_sane() -> None:
    assert 30.0 < RECLINED_TORSO_LEAN_DEG < 60.0
    assert 100.0 < SEATED_MEDIAN_KNEE_FLEXION_DEG < STANDING_MEDIAN_KNEE_FLEXION_MIN
    assert STANDING_MEDIAN_KNEE_FLEXION_MIN <= 180.0
