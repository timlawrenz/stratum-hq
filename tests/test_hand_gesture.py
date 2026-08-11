"""TDD coverage for the hand-gesture evidence specialist (arm #109).

NEW model class: open-weight MediaPipe HandLandmarker (21-point-per-hand
mesh, local CPU via the tasks API) + NEW evidence part hand-gesture:
per-visible-hand gesture class (open-palm / fist / pointing / relaxed-curl)
from scale-invariant within-hand finger-extension segment ratios, plus a
hand-raised flag gated on the pose2 GOLIATH-308 wrist/shoulder reference.

Scale-invariant; only the coarse categorical facts are verbalized; raw
landmark geometry / extension vectors stay payload-only. Pure geometry is
tested without any model; the single model-touching test uses a blank
non-sensitive frame (same pattern as arm #60).
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.hand_gesture import (
    _finger_extended,
    _gesture_class,
    _hand_raised,
    _match_sides,
    _thumb_extended,
    compute_hand_gesture,
    render_hand_gesture,
    validate_pose2_array,
    validate_rgb_array,
)
from research_harness.hand_gesture import HandGestureError

from stratum2.config import GOLIATH_308

_G = {name: i for i, name in enumerate(GOLIATH_308)}


def _hand(extended: dict[str, bool], scale: float = 1.0) -> np.ndarray:
    """Synthetic 21-point hand mesh with the requested finger extension.

    Segment chain length ~10px per joint pair (scaled). Extended fingers
    place TIP beyond the PIP by ~2 chain segments; folded fingers collapse
    TIP near the PIP. The mesh lies in a horizontal strip around y=100 so
    bbox >= the 24px floor at any scale >= 0.5.
    """
    base = 100.0 * scale
    seg = 10.0 * scale
    pts = np.zeros((21, 3), dtype=float)
    for i in range(21):
        pts[i] = (base + i * 15.0 * scale, 100.0 * scale, 0.0)
    # x offsets per chain so fingers fan out
    fan = {"thumb": -30.0, "index": 0.0, "middle": 15.0, "ring": 30.0, "pinky": 45.0}

    def chain(mcp: int, pip: int, dip: int, tip: int, name: str, thumb: bool = False) -> None:
        fx = fan[name] * scale
        if thumb:
            # MediaPipe thumb chain: MCP(2) - IP(3) - TIP(4)
            pts[mcp] = (base + fx, 100.0 * scale, 0.0)
            pts[pip] = (base + fx + 8.0 * scale, 95.0 * scale, 0.0)
            pts[tip] = (base + fx + (26.0 if extended[name] else 12.0) * scale,
                        92.0 * scale, 0.0)
            return
        pts[mcp] = (base + fx, 100.0 * scale, 0.0)
        pts[pip] = (base + fx, 90.0 * scale, 0.0)
        pts[dip] = (base + fx, 82.0 * scale, 0.0)
        pts[tip] = (base + fx, 64.0 if extended[name] else 84.0 * scale, 0.0)

    chain(2, 3, 3, 4, "thumb", thumb=True)
    chain(5, 6, 7, 8, "index")
    chain(9, 10, 11, 12, "middle")
    chain(13, 14, 15, 16, "ring")
    chain(17, 18, 19, 20, "pinky")
    return pts


_OPEN = {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True}
_FIST = {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}
_POINT = {"thumb": False, "index": True, "middle": False, "ring": False, "pinky": False}
_PEACE = {"thumb": False, "index": True, "middle": True, "ring": False, "pinky": False}


def _pose(wrist_l=(140.0, 90.0), wrist_r=(260.0, 90.0),
          shoulder_l=(100.0, 200.0), shoulder_r=(230.0, 200.0)) -> np.ndarray:
    pose = np.zeros((308, 3), dtype=float)
    for name, pt in (("left_wrist", wrist_l), ("right_wrist", wrist_r),
                     ("left_shoulder", shoulder_l), ("right_shoulder", shoulder_r)):
        if pt is not None:
            pose[_G[name]] = (*pt, 0.95)
    return pose


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

def test_validate_arrays() -> None:
    with pytest.raises(HandGestureError):
        validate_rgb_array(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(HandGestureError):
        validate_rgb_array(np.zeros((10, 10, 3), dtype=float))
    with pytest.raises(HandGestureError):
        validate_pose2_array(np.zeros((308, 2)))
    with pytest.raises(HandGestureError):
        validate_pose2_array(np.zeros((200, 3)))


# ---------------------------------------------------------------------------
# Finger-extension geometry (scale-invariant within-hand ratios)
# ---------------------------------------------------------------------------

def test_finger_extended_and_folded() -> None:
    hand = _hand(_OPEN)
    assert _finger_extended(hand, 5, 6, 7, 8, 1.25) is True
    hand = _hand(_FIST)
    assert _finger_extended(hand, 5, 6, 7, 8, 1.25) is False


def test_thumb_extended_and_folded() -> None:
    hand = _hand(_OPEN)
    assert _thumb_extended(hand, 2, 3, 4, 1.10) is True
    hand = _hand(_FIST)
    assert _thumb_extended(hand, 2, 3, 4, 1.10) is False


def test_extension_scale_invariant() -> None:
    """The extension decision must not depend on the absolute hand size."""
    for scale in (0.5, 1.0, 2.5):
        assert _finger_extended(_hand(_OPEN, scale), 5, 6, 7, 8, 1.25) is True
        assert _finger_extended(_hand(_FIST, scale), 5, 6, 7, 8, 1.25) is False


# ---------------------------------------------------------------------------
# Gesture classes
# ---------------------------------------------------------------------------

def test_open_palm_class() -> None:
    cls, pay = _gesture_class(_hand(_OPEN))
    assert cls == "open-palm"
    assert pay["fingers_extended_count"] == 4


def test_fist_class() -> None:
    cls, _ = _gesture_class(_hand(_FIST))
    assert cls == "fist"


def test_pointing_class() -> None:
    cls, _ = _gesture_class(_hand(_POINT))
    assert cls == "pointing"


def test_peace_sign_is_relaxed_curl() -> None:
    cls, _ = _gesture_class(_hand(_PEACE))
    assert cls == "relaxed-curl"


# ---------------------------------------------------------------------------
# Side matching + hand-raised (pose2-gated)
# ---------------------------------------------------------------------------

def test_match_sides_distinct_two_hands() -> None:
    pose = _pose()
    sides = _match_sides(pose, [(130.0, 90.0), (270.0, 90.0)])
    assert set(s for s in sides if s) == {"left", "right"}


def test_match_sides_single_wrist_leftover_opposite() -> None:
    pose = _pose(wrist_l=(140.0, 90.0), wrist_r=None)
    sides = _match_sides(pose, [(135.0, 90.0), (300.0, 90.0)])
    assert sides[0] == "left"
    assert sides[1] == "right"  # leftover gets the opposite side


def test_match_sides_no_wrists_none() -> None:
    pose = _pose(wrist_l=None, wrist_r=None)
    assert _match_sides(pose, [(100.0, 90.0), (300.0, 90.0)]) == [None, None]


def test_hand_raised_above_shoulder() -> None:
    pose = _pose(wrist_r=(230.0, 80.0))  # right wrist 120px above shoulders
    r = _hand_raised(pose, (235.0, 85.0), "right")
    assert r["measured"] is True
    assert r["hand_raised"] is True


def test_hand_raised_not_raised() -> None:
    pose = _pose(wrist_r=(230.0, 240.0))  # wrist below the shoulder line
    r = _hand_raised(pose, (235.0, 235.0), "right")
    assert r["measured"] is True
    assert r["hand_raised"] is False


def test_hand_raised_abstains_without_wrist() -> None:
    pose = _pose(wrist_r=None)
    r = _hand_raised(pose, (235.0, 85.0), "right")
    assert r["measured"] is False


def test_hand_raised_abstains_without_shoulders() -> None:
    pose = _pose(shoulder_l=None, shoulder_r=None)
    r = _hand_raised(pose, (235.0, 85.0), "right")
    assert r["measured"] is False


# ---------------------------------------------------------------------------
# Item-level abstention (blank frame, real model — safe non-sensitive probe)
# ---------------------------------------------------------------------------

def test_compute_abstains_on_blank_frame() -> None:
    from research_harness.hand_gesture import _HandLandmarkerRuntime
    model = "/mnt/nas-ai-models/research/stratum/models/hand-gesture/hand_landmarker.task"
    try:
        _HandLandmarkerRuntime.reset()
        rgb = np.full((120, 120, 3), 220, dtype=np.uint8)
        r = compute_hand_gesture(rgb, _pose(), model_asset_path=model)
        assert r["abstained"] is True
        assert "no hand detected" in r["abstention_reason"]
    finally:
        _HandLandmarkerRuntime.reset()


def test_compute_invalid_rgb_raises() -> None:
    with pytest.raises(HandGestureError):
        compute_hand_gesture(np.zeros((10, 10), dtype=np.uint8), _pose(),
                             model_asset_path="/nonexistent.task")


# ---------------------------------------------------------------------------
# Rendering (only coarse categorical facts)
# ---------------------------------------------------------------------------

def test_render_abstention() -> None:
    lines = render_hand_gesture({"abstained": True, "abstention_reason": "no hands"})
    assert lines and "no hands" in lines[0]


def test_render_gesture_claim() -> None:
    cfg = {
        "abstained": False,
        "gesture_claim": "left hand fist",
        "raised_flags": [{"side": "left", "hand_raised": True, "measured": True}],
    }
    lines = render_hand_gesture(cfg)
    assert any("left hand fist" in ln for ln in lines)
    assert any("raised" in ln for ln in lines)


def test_render_silences_degenerate_count_band() -> None:
    """One-vs-two-hands is payload-only (probe max_share 0.929 >= 0.75)."""
    cfg = {"abstained": False, "gesture_claim": "right hand fist",
           "hands_detected": 1, "raised_flags": []}
    lines = render_hand_gesture(cfg)
    assert not any("hands visible" in ln for ln in lines)


def test_render_empty_config() -> None:
    assert render_hand_gesture(None) == []