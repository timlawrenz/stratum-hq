"""TDD coverage for deterministic point-map / depth-ordering measurements.

These measurements are the deterministic evidence for arm #58 (pointmap-depth).
They must be scale-invariant depth facts (region nearest/farthest ordering,
hand depth ordering, hand/arm in front of the torso plane, normalized depth
relief ratio) computed from existing `pointmap.npy` (CAM-frame 3D cloud) +
`seg2` DOME-29 masks, honoring exactly-one-subject abstention — absolute metric
Z values and raw spreads stay in the machine-readable payload, never as caption
claims.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.pointmap_depth import (
    PointmapDepthError,
    compute_pointmap_depth,
    validate_pointmap_array,
    validate_seg2_array,
)


def _pointmap(height: int = 100, width: int = 100, *, median_z: float = 2.0) -> np.ndarray:
    """Build a CAM-frame pointmap: background zeroed, subject at ~median_z."""
    pm = np.zeros((height, width, 3), dtype=np.float16)
    # Subject band (rows 20-80, cols 30-70) at depth ~median_z with noise.
    zz = np.full((61, 41), median_z, dtype=np.float16)
    strength = 0.003
    rng = np.random.default_rng(7)
    zz = zz + rng.normal(0.0, median_z * strength, size=(61, 41)).astype(np.float16)
    pm[20:81, 30:71, 2] = zz
    # X (lateral) and Y (vertical) in CAM frame for a roughly frontal subject.
    xs = np.linspace(-0.35, 0.35, 41, dtype=np.float16)[None, :]
    ys = np.linspace(0.9, -0.9, 61, dtype=np.float16)[:, None]
    pm[20:81, 30:71, 0] = xs + 0.0
    pm[20:81, 30:71, 1] = ys
    return pm


def _seg(
    *,
    torso_rows=(40, 70),
    torso_cols=(45, 60),
    left_hand=None,
    right_hand=None,
    head=None,
    height: int = 100,
    width: int = 100,
) -> np.ndarray:
    a = np.zeros((height, width), dtype=np.uint8)
    # Torso = class 22.
    for y in range(*torso_rows):
        for x in range(*torso_cols):
            a[y, x] = 22
    left_hand = left_hand or (30, 35)  # (y, x) top-left of an 8x8 hand block
    right_hand = right_hand or (30, 61)
    for dy in range(8):
        for dx in range(8):
            a[left_hand[0] + dy, left_hand[1] + dx] = 6
            a[right_hand[0] + dy, right_hand[1] + dx] = 15
    head = head or (25, 50)
    a[head[0], head[1]] = 3
    a[head[0], head[1] + 1] = 3
    a[head[0] + 1, head[1]] = 3
    a[head[0] + 1, head[1] + 1] = 3
    return a


def test_subject_present_and_measurable() -> None:
    seg = _seg()
    m = compute_pointmap_depth(_pointmap(), seg)
    assert m["subject_present"] is True
    assert m["abstained"] is False
    assert m["depth_measurable"] is True
    assert m["median_z"] is not None and 1.5 < m["median_z"] < 2.5
    assert m["relief_band"] in ("compact", "moderate", "pronounced")
    assert m["nearest_region"] is not None
    assert m["farthest_region"] is not None
    assert "torso" in m["depth_ordering"]


def test_hand_in_front_of_torso_true_and_false() -> None:
    # Left hand placed well in front (smaller Z) of the torso plane.
    pm = _pointmap()
    seg = _seg(left_hand=(40, 33), right_hand=(42, 61))
    # Hands at the same plane as torso -> not in front.
    m = compute_pointmap_depth(pm, seg)
    # With hands at torso plane, both should be False.
    assert m["left_hand_in_front"] is False or m["left_hand_in_front"] is None
    assert m["right_hand_in_front"] is False or m["right_hand_in_front"] is None

    # Now push left hand forward (median_z 1.6 vs torso 2.0) -> clearly in front.
    pm_forward = _pointmap().copy()
    for dy in range(8):
        for dx in range(8):
            pm_forward[40 + dy, 33 + dx, 2] = np.float16(1.6)
    m2 = compute_pointmap_depth(pm_forward, seg)
    assert m2["left_hand_in_front"] is True


def test_hand_ordering_detects_nearer_hand() -> None:
    pm = _pointmap()
    seg = _seg(left_hand=(40, 33), right_hand=(42, 61))
    # Push left hand closer to camera than right hand: left median Z smaller.
    pm2 = pm.copy()
    for dy in range(8):
        for dx in range(8):
            pm2[40 + dy, 33 + dx, 2] = np.float16(1.6)
    m = compute_pointmap_depth(pm2, seg)
    assert m["hand_ordering"] == "left"


def test_abstains_when_no_valid_depth() -> None:
    pm = np.zeros((100, 100, 3), dtype=np.float16)
    seg = _seg()
    m = compute_pointmap_depth(pm, seg)
    assert m["abstained"] is True
    assert m["depth_measurable"] is False
    assert "too few valid depth pixels" in m["abstention_reason"]


def test_abstains_when_scale_degenerate() -> None:
    # Subject distance implausibly far -> scale degenerate.
    pm = _pointmap(median_z=20.0)
    seg = _seg()
    m = compute_pointmap_depth(pm, seg)
    assert m["abstained"] is True
    assert "outside the human-plausible band" in m["abstention_reason"]


def test_validate_array_errors() -> None:
    with pytest.raises(PointmapDepthError):
        validate_pointmap_array(np.zeros((10, 10), dtype=np.float16))
    with pytest.raises(PointmapDepthError):
        validate_pointmap_array(np.zeros((10, 10, 3), dtype=np.int16))
    with pytest.raises(PointmapDepthError):
        validate_seg2_array(np.zeros((10, 10, 1), dtype=np.uint8))


def test_misaligned_shapes_raise() -> None:
    pm = _pointmap(100, 100)
    seg = np.zeros((50, 50), dtype=np.uint8)
    with pytest.raises(PointmapDepthError):
        compute_pointmap_depth(pm, seg)
