"""TDD coverage for the hairstyle evidence specialist (arm #82).

Deterministic hair-length + hair-arrangement bands from seg2 (DOME-29 Hair)
+ pose2 (GOLIATH-308 shoulder/neck), scale-invariant. Only the coarse bands
are verbalized; raw fractions / pixel spans stay in the payload. Pure and
tested without any model; no GPU needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.hairstyle import (
    MIN_CLASS_PX,
    HairstyleError,
    compute_hairstyle,
    render_hairstyle,
    validate_pose2_array,
    validate_seg2_array,
)

from stratum2.config import DOME_29, GOLIATH_308

_HAIR = DOME_29.index("Hair")
_G = {name: i for i, name in enumerate(GOLIATH_308)}


def _seg(h: int = 1000, w: int = 800) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _pose(*, ls=(300.0, 500.0, 0.9), rs=(500.0, 500.0, 0.9),
          neck=(400.0, 480.0, 0.9)) -> np.ndarray:
    """Default upright torso: shoulders at y=500 (width 200), neck above."""
    pose = np.zeros((308, 3), dtype=float)
    pose[:, 2] = 1.0
    pose[_G["left_shoulder"]] = ls
    pose[_G["right_shoulder"]] = rs
    pose[_G["neck"]] = neck
    return pose


def _hair_region(seg: np.ndarray, *, top: int, bot: int, cx: int = 400,
                 half_w: int = 60) -> np.ndarray:
    """Paint a rectangular Hair block [top..bot] rows x [cx-half_w..cx+half_w] cols."""
    seg[top:bot + 1, cx - half_w:cx + half_w + 1] = _HAIR
    return seg


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

def test_validate_seg2() -> None:
    with pytest.raises(HairstyleError):
        validate_seg2_array(np.zeros((5, 5, 1), dtype=np.uint8))
    with pytest.raises(HairstyleError):
        validate_seg2_array(np.zeros((5, 5), dtype=np.float32))
    with pytest.raises(HairstyleError):
        validate_seg2_array("nope")


def test_validate_pose2() -> None:
    with pytest.raises(HairstyleError):
        validate_pose2_array(np.zeros((308, 2)))
    with pytest.raises(HairstyleError):
        validate_pose2_array(np.zeros((200, 3), dtype=float))


# ---------------------------------------------------------------------------
# Length band
# ---------------------------------------------------------------------------

def test_long_hair_below_shoulders() -> None:
    """Hair extends well below the shoulder line -> long."""
    seg = _seg()
    _hair_region(seg, top=300, bot=800, cx=400, half_w=60)  # bot 800 vs shoulder 500
    r = compute_hairstyle(seg, _pose())
    assert r["abstained"] is False
    assert r["hair_length_band"] == "long"
    assert r["hair_below_shoulder_ratio"] is not None
    assert r["hair_below_shoulder_ratio"] >= 0.60


def test_short_hair_above_shoulders() -> None:
    """Hair entirely above the shoulder line -> short (bsr = 0)."""
    seg = _seg()
    _hair_region(seg, top=300, bot=460, cx=400, half_w=60)  # bot above shoulder 500
    r = compute_hairstyle(seg, _pose())
    assert r["hair_length_band"] == "short"
    assert r["hair_below_shoulder_ratio"] == pytest.approx(0.0, abs=1e-6)


def test_shoulder_length() -> None:
    """Hair hanging modestly below the shoulder line -> shoulder-length."""
    seg = _seg()
    _hair_region(seg, top=300, bot=560, cx=400, half_w=60)  # 60px below / width 200 -> 0.30
    r = compute_hairstyle(seg, _pose())
    assert r["hair_length_band"] == "shoulder-length"


def test_length_is_scale_invariant() -> None:
    """Scaling the frame + hair + pose must keep the same length band."""
    seg = _seg(2000, 1600)
    _hair_region(seg, top=600, bot=1600, cx=800, half_w=120)
    big_pose = _pose(
        ls=(600.0, 1000.0, 0.9), rs=(1000.0, 1000.0, 0.9),
        neck=(800.0, 960.0, 0.9),
    )
    r_big = compute_hairstyle(seg, big_pose)
    seg_s = _seg(1000, 800)
    _hair_region(seg_s, top=300, bot=800, cx=400, half_w=60)
    r_small = compute_hairstyle(seg_s, _pose())
    assert r_big["hair_length_band"] == r_small["hair_length_band"] == "long"
    assert r_big["hair_below_shoulder_ratio"] == pytest.approx(
        r_small["hair_below_shoulder_ratio"], abs=1e-3
    )


# ---------------------------------------------------------------------------
# Arrangement band
# ---------------------------------------------------------------------------

def test_down_arrangement_hair_below_shoulders() -> None:
    """Material hair below the shoulder line -> down."""
    seg = _seg()
    _hair_region(seg, top=300, bot=850, cx=400, half_w=70)
    r = compute_hairstyle(seg, _pose())
    assert r["hair_arrangement_band"] == "down"
    assert r["hair_below_shoulder_fraction"] >= 0.10


def test_kept_up_arrangement_hair_above_shoulders() -> None:
    """Hair kept above the shoulder line -> kept-up (short/bun/tie)."""
    seg = _seg()
    _hair_region(seg, top=350, bot=470, cx=400, half_w=50)
    r = compute_hairstyle(seg, _pose())
    assert r["hair_arrangement_band"] == "kept-up"
    assert r["hair_below_shoulder_fraction"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------

def test_no_hair_abstains() -> None:
    seg = _seg()  # empty frame
    r = compute_hairstyle(seg, _pose())
    assert r["abstained"] is True
    assert "absent" in (r["abstention_reason"] or "")
    assert r["hair_length_band"] is None
    assert r["hair_arrangement_band"] is None


def test_insufficient_hair_pixels_abstains() -> None:
    seg = _seg()
    seg[400:410, 395:405] = _HAIR  # 100 px < MIN_CLASS_PX floor
    r = compute_hairstyle(seg, _pose())
    assert r["abstained"] is True
    assert "floor" in (r["abstention_reason"] or "")


def test_unreliable_shoulders_abstains() -> None:
    seg = _seg()
    _hair_region(seg, top=300, bot=800, cx=400, half_w=60)
    pose = np.zeros((308, 3), dtype=float)  # all conf 0 -> shoulders unreliable
    r = compute_hairstyle(seg, pose)
    assert r["abstained"] is True
    assert "keypoints" in (r["abstention_reason"] or "")


def test_hair_present_but_shoulders_bad_honest() -> None:
    """Hair present 24/24 but shoulder/neck unreliable -> honest abstain, hair_present True."""
    seg = _seg()
    _hair_region(seg, top=300, bot=800, cx=400, half_w=60)
    pose = np.zeros((308, 3), dtype=float)
    r = compute_hairstyle(seg, pose)
    assert r["hair_present"] is True
    assert r["abstained"] is True


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_long_down() -> None:
    seg = _seg()
    _hair_region(seg, top=300, bot=850, cx=400, half_w=60)
    r = compute_hairstyle(seg, _pose())
    lines = render_hairstyle(r)
    assert any("long" in ln for ln in lines)
    assert any("down" in ln for ln in lines)


def test_render_short_kept_up() -> None:
    seg = _seg()
    _hair_region(seg, top=350, bot=460, cx=400, half_w=50)
    r = compute_hairstyle(seg, _pose())
    lines = render_hairstyle(r)
    assert any("short" in ln for ln in lines)
    assert any("above the shoulders" in ln for ln in lines)


def test_render_not_measured_empty() -> None:
    # Empty dict = dimension not measured -> no fabricated hairstyle claim.
    assert render_hairstyle({}) == []


def test_render_abstain() -> None:
    r = {"abstained": True, "abstention_reason": "shoulder/neck unreliable"}
    lines = render_hairstyle(r)
    assert any("abstain" in ln for ln in lines)


def test_render_no_pixel_values_in_prose() -> None:
    seg = _seg()
    _hair_region(seg, top=300, bot=800, cx=400, half_w=60)
    r = compute_hairstyle(seg, _pose())
    joined = " ".join(render_hairstyle(r))
    # no long integers that look like px counts (span ratios are < 10)
    assert "800" not in joined and "200" not in joined


# ---------------------------------------------------------------------------
# Payload honesty: raw values present, prose has only bands
# ---------------------------------------------------------------------------

def test_payload_raw_values_present() -> None:
    seg = _seg()
    _hair_region(seg, top=300, bot=850, cx=400, half_w=60)
    r = compute_hairstyle(seg, _pose())
    assert r["hair_below_shoulder_ratio"] is not None
    assert r["hair_below_shoulder_fraction"] is not None
    assert r["hair_span_ratio"] is not None
    assert r["hair_centroid_row_fraction"] is not None


def test_threshold_sanity() -> None:
    assert MIN_CLASS_PX >= 100
