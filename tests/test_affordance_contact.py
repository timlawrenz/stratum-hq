"""TDD coverage for the affordance-contact evidence specialist (arm #76).

Deterministic subject self-contact / affordance measurements (hand-own-body
contact, hand elevation/gesture, grounding) from pose2 GOLIATH-308 + seg2
DOME-29. Only scale-invariant facts are verbalized; raw normalized wrist
distances stay in the machine-readable payload. Pure and tested without any
model; no GPU needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.affordance_contact import (
    TRUNK_CONTACT_NORM,
    WRIST_ABOVE_HIP_NORM,
    AffordanceContactError,
    compute_affordance_contact,
    render_affordance_contact,
    validate_pose2_array,
    validate_seg2_array,
)

from stratum2.config import GOLIATH_308

_G = {name: i for i, name in enumerate(GOLIATH_308)}

_H = 1000
_W = 800


def _seg_with_trunk(trunk_fracs: list[tuple[int, int, int, int]] | None = None) -> np.ndarray:
    """seg2 with a trunk block (Torso class 22) + bottom-row grounding switch."""
    seg = np.zeros((_H, _W), dtype=np.uint8)
    seg[:, :] = 0  # Background
    if trunk_fracs:
        for (y0, y1, x0, x1) in trunk_fracs:
            seg[y0:y1, x0:x1] = 22  # Torso
    return seg


def _pose(
    *,
    lw=(400.0, 700.0, 0.9),
    rw=(400.0, 700.0, 0.9),
    lac=(300.0, 300.0, 0.9),
    rac=(500.0, 300.0, 0.9),
    lh=(370.0, 750.0, 0.9),
    rh=(430.0, 750.0, 0.9),
    l_raised=False,
    r_raised=False,
    lw_conf=0.9,
    rw_conf=0.9,
) -> np.ndarray:
    pose = np.zeros((308, 3), dtype=float)
    pose[:, 2] = 1.0  # all keypoints confident by default
    pose[_G["left_wrist"]] = [lw[0], lw[1], lw_conf]
    pose[_G["right_wrist"]] = [rw[0], rw[1], rw_conf]
    pose[_G["left_acromion"]] = lac
    pose[_G["right_acromion"]] = rac
    pose[_G["left_shoulder"]] = [lac[0] - 5, lac[1], 0.9]
    pose[_G["right_shoulder"]] = [rac[0] + 5, rac[1], 0.9]
    pose[_G["left_hip"]] = lh
    pose[_G["right_hip"]] = rh
    return pose


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

def test_validate_pose2() -> None:
    with pytest.raises(AffordanceContactError):
        validate_pose2_array(np.zeros((308, 2)))
    with pytest.raises(AffordanceContactError):
        validate_pose2_array("nope")


def test_validate_seg2() -> None:
    with pytest.raises(AffordanceContactError):
        validate_seg2_array(np.zeros((5, 5, 1), dtype=np.uint8))
    with pytest.raises(AffordanceContactError):
        validate_seg2_array(np.zeros((5, 5), dtype=np.float32))


# ---------------------------------------------------------------------------
# Scale invariance
# ---------------------------------------------------------------------------

def test_hand_contact_count_is_scale_invariant() -> None:
    """Doubling the entire figure scale must not change the contact bands."""
    base = _pose(
        lac=(300.0, 300.0, 0.9), rac=(500.0, 300.0, 0.9),  # sw=200
        lh=(370.0, 750.0, 0.9), rh=(430.0, 750.0, 0.9),
        lw=(400.0, 745.0, 0.9),  # at hip level near the trunk band -> contact, not raised
        rw=(440.0, 60.0, 0.9),   # far above the hip line -> raised, not contact
    )
    seg_small = _seg_with_trunk([(700, 740, 300, 500)])
    r_small = compute_affordance_contact(base, seg_small)
    assert r_small["hand_contact_count"] == 1
    assert r_small["hand_elevation_count"] == 1
    # Scale by 2x: pose AND seg both double; normalized geometry must be exact.
    factor = 2.0
    big = _pose(
        lac=(300.0 * factor, 300.0 * factor, 0.9), rac=(500.0 * factor, 300.0 * factor, 0.9),
        lh=(370.0 * factor, 750.0 * factor, 0.9), rh=(430.0 * factor, 750.0 * factor, 0.9),
        lw=(400.0 * factor, 745.0 * factor, 0.9), rw=(440.0 * factor, 60.0 * factor, 0.9),
    )
    seg_big = np.zeros((_H * 2, _W * 2), dtype=np.uint8)
    seg_big[700 * 2:740 * 2, 300 * 2:500 * 2] = 22
    r_big = compute_affordance_contact(big, seg_big)
    assert r_big["hand_contact_count"] == 1
    assert r_big["hand_elevation_count"] == 1
    # Both normalized distances equal to within rounding (3-decimal payload).
    assert r_big["left_wrist_trunk_dist_norm"] == pytest.approx(
        r_small["left_wrist_trunk_dist_norm"], abs=0.012
    )
    # Trunk band at y=700..740, wrist at y=745 -> 5 px / 200 sw = 0.025
    assert r_small["left_wrist_trunk_dist_norm"] == pytest.approx(0.025, abs=0.01)


def test_normalized_distance_metric() -> None:
    seg = _seg_with_trunk([(700, 740, 300, 500)])
    # left wrist 60px above the trunk band -> 60/200 = 0.30, right far away.
    pose = _pose(lw=(400.0, 640.0, 0.9), rw=(300.0, 200.0, 0.9))
    r = compute_affordance_contact(pose, seg)
    assert r["left_wrist_trunk_dist_norm"] == pytest.approx(0.30, abs=0.02)
    assert r["left_hand_contact"] is True
    assert r["right_hand_contact"] is False


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------

def test_both_hands_contact_and_grounded() -> None:
    seg = _seg_with_trunk([(700, 760, 300, 500)])
    # grounding on: subject reaches the bottom row
    seg[-1, :] = 22
    pose = _pose(lw=(380.0, 750.0, 0.9), rw=(420.0, 755.0, 0.9))
    r = compute_affordance_contact(pose, seg)
    assert r["hand_contact_count"] == 2
    assert r["grounded"] is True


def test_elevation_count() -> None:
    seg = _seg_with_trunk([(700, 760, 300, 500)])
    # right wrist far above hip line (750) -> raised
    pose = _pose(rw=(400.0, 200.0, 0.9), lw=(350.0, 740.0, 0.9))
    r = compute_affordance_contact(pose, seg)
    assert r["hand_elevation_count"] == 1
    assert r["right_hand_raised"] is True


def test_hand_with_low_confidence_abstains_that_hand() -> None:
    seg = _seg_with_trunk([(700, 760, 300, 500)])
    pose = _pose(lw=(400.0, 750.0, 0.9), lw_conf=0.2, rw=(420.0, 750.0, 0.9))
    r = compute_affordance_contact(pose, seg)
    # left hand invisible (low conf) -> not counted as contact; right only.
    assert r["left_hand_visible"] is False
    assert r["right_hand_visible"] is True
    assert r["hand_contact_count"] == 1


def test_no_shoulder_width_disables_hand_axes_but_grounding_fires() -> None:
    seg = _seg_with_trunk([(700, 760, 300, 500)])
    seg[-1, :] = 22
    pose = _pose(
        lac=(-1.0, -1.0, 0.0), rac=(-1.0, -1.0, 0.0),  # unreliable shoulders
        lh=(370.0, 750.0, 0.0), rh=(430.0, 750.0, 0.0),  # unreliable hips
        lw=(400.0, 750.0, 0.9), rw=(420.0, 750.0, 0.9),
    )
    r = compute_affordance_contact(pose, seg)
    assert r["shoulder_width_norm_ok"] is False
    assert r["hand_contact_count"] == 0
    assert r["hand_elevation_count"] == 0
    # Grounding is frame-based and scale-free -> still measured.
    assert r["grounded"] is True


def test_floating_not_grounded() -> None:
    seg = _seg_with_trunk([(700, 760, 300, 500)])
    # subject does not reach the bottom row
    pose = _pose()
    r = compute_affordance_contact(pose, seg)
    assert r["grounded"] is False


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_claims_scale_invariant_only() -> None:
    seg = _seg_with_trunk([(700, 760, 300, 500)])
    pose = _pose(lw=(380.0, 750.0, 0.9), rw=(420.0, 755.0, 0.9))
    r = compute_affordance_contact(pose, seg)
    lines = render_affordance_contact(r)
    assert any("both hands rest against her own body" in ln for ln in lines)


def test_render_not_measured_empty() -> None:
    # Empty dict = dimension not measured -> no fabricated claim.
    assert render_affordance_contact({}) == []


def test_render_abstain() -> None:
    r = {"abstained": True, "abstention_reason": "no subject"}
    lines = render_affordance_contact(r)
    assert any("abstain" in ln for ln in lines)


def test_render_no_pixel_values_in_prose() -> None:
    seg = _seg_with_trunk([(700, 760, 300, 500)])
    pose = _pose(lw=(380.0, 750.0, 0.9), rw=(420.0, 755.0, 0.9))
    r = compute_affordance_contact(pose, seg)
    joined = " ".join(render_affordance_contact(r))
    # No absolute pixel numbers should appear in the prose claims.
    import re
    numbers = re.findall(r"\d{2,}", joined)
    assert not numbers


# ---------------------------------------------------------------------------
# Threshold sanity (must not be trivially degenerate)
# ---------------------------------------------------------------------------

def test_thresholds_are_sane() -> None:
    assert 0.0 < TRUNK_CONTACT_NORM < 1.0
    assert 0.0 < WRIST_ABOVE_HIP_NORM < 1.0
    assert TRUNK_CONTACT_NORM > WRIST_ABOVE_HIP_NORM * 0.5
