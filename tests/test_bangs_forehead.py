"""TDD coverage for the bangs-forehead evidence specialist (arm #110).

Deterministic bangs / partial-fringe / swept-back band from seg2 (DOME-29
Hair over the eye-line-anchored forehead band) + pose2 (GOLIATH-308 eye
line), scale-invariant. Only the coarse band is verbalized; the raw normalized
coverage ratio stays in the payload. Pure and tested without any model; no GPU
needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.bangs_forehead import (
    BangsForeheadError,
    compute_bangs_forehead,
    render_bangs_forehead,
    validate_pose2_array,
    validate_seg2_array,
)

from stratum2.config import DOME_29, GOLIATH_308

_FACE_NECK = DOME_29.index("Face_Neck")
_HAIR = DOME_29.index("Hair")
_G = {name: i for i, name in enumerate(GOLIATH_308)}


def _seg(h: int = 1000, w: int = 800) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _pose(*, eye_y: float = 500.0) -> np.ndarray:
    """Default pose: eyes at y=eye_y, confident."""
    pose = np.zeros((308, 3), dtype=float)
    pose[:, 2] = 1.0
    pose[_G["right_eye"]] = (440.0, eye_y, 0.95)
    pose[_G["left_eye"]] = (540.0, eye_y, 0.95)
    return pose


def _face(seg: np.ndarray, *, top: int = 300, bot: int = 900, cx: int = 400,
          half_w: int = 100) -> np.ndarray:
    """Paint the seg2 Face_Neck region as a rectangle [top..bot] x [cx±half_w]."""
    seg[top:bot + 1, cx - half_w:cx + half_w + 1] = _FACE_NECK
    return seg


def _hair(seg: np.ndarray, *, top: int, bot: int, cx: int = 400,
          half_w: int = 100) -> np.ndarray:
    seg[top:bot + 1, cx - half_w:cx + half_w + 1] = _HAIR
    return seg


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

def test_validate_seg2() -> None:
    with pytest.raises(BangsForeheadError):
        validate_seg2_array(np.zeros((5, 5, 1), dtype=np.uint8))
    with pytest.raises(BangsForeheadError):
        validate_seg2_array("nope")


def test_validate_pose2() -> None:
    with pytest.raises(BangsForeheadError):
        validate_pose2_array(np.zeros((308, 2)))
    with pytest.raises(BangsForeheadError):
        validate_pose2_array(np.zeros((200, 3), dtype=float))


# ---------------------------------------------------------------------------
# Band classification
# ---------------------------------------------------------------------------

def test_bangs_high_forehead_coverage() -> None:
    """Hair draping over the forehead band right above the eyes -> bangs."""
    # face: rows 300..899; eyes at 500; forehead band = 500-0.2*400=420..500
    # paint hair over exactly the band rows 420..499 (full width) -> coverage ~1
    seg = _face(_seg())
    _hair(seg, top=400, bot=499, cx=400, half_w=100)
    r = compute_bangs_forehead(seg, _pose())
    assert r["abstained"] is False
    assert r["bangs_band"] == "bangs"
    assert r["forehead_hair_coverage"] >= 0.35


def test_swept_back_low_forehead_coverage() -> None:
    """Hair well above the forehead (scalp) -> swept-back."""
    seg = _face(_seg())
    # hair only in the scalp rows (well above the forehead band: band top ~420)
    _hair(seg, top=300, bot=360, cx=400, half_w=100)
    r = compute_bangs_forehead(seg, _pose())
    assert r["abstained"] is False
    assert r["bangs_band"] == "swept-back"
    assert r["forehead_hair_coverage"] <= 0.12


def test_partial_fringe_mid_coverage() -> None:
    """Some hair in the band -> partial-fringe."""
    seg = _face(_seg())
    # paint hair over ~1/4 of the band rows -> partial coverage (~0.25)
    _hair(seg, top=440, bot=460, cx=400, half_w=100)
    r = compute_bangs_forehead(seg, _pose())
    assert r["abstained"] is False
    assert r["bangs_band"] == "partial-fringe"
    assert 0.12 < r["forehead_hair_coverage"] < 0.35


def test_scale_invariance() -> None:
    """Scaling frame + face + hair must keep the same band."""
    def build(h: int, w: int, sf: float):
        seg = _seg(h, w)
        top, bot, cx, half_w = [int(v * sf) for v in (300, 900, 400, 100)]
        _face(seg, top=top, bot=bot, cx=cx, half_w=half_w)
        _hair(seg, top=int(400 * sf), bot=int(499 * sf), cx=cx, half_w=half_w)
        pose = _pose(eye_y=500.0 * sf)
        return compute_bangs_forehead(seg, pose)

    r_small = build(1000, 800, 1.0)
    r_big = build(2000, 1600, 2.0)
    assert r_small["bangs_band"] == r_big["bangs_band"]
    # The band is a fraction of face height (scale-invariant) up to integer
    # row discretization; allow a sub-2% anti-fragility tolerance.
    assert r_small["forehead_hair_coverage"] == pytest.approx(
        r_big["forehead_hair_coverage"], abs=0.02)


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------

def test_no_face_region_abstains() -> None:
    seg = _seg()  # no Face_Neck pixels
    _hair(seg, top=400, bot=499)
    r = compute_bangs_forehead(seg, _pose())
    assert r["abstained"] is True
    assert "Face_Neck" in (r["abstention_reason"] or "")


def test_no_hair_abstains() -> None:
    seg = _face(_seg())
    r = compute_bangs_forehead(seg, _pose())
    assert r["abstained"] is True
    assert "Hair" in (r["abstention_reason"] or "")


def test_unreliable_eyes_abstains() -> None:
    seg = _face(_seg())
    _hair(seg, top=400, bot=499)
    pose = np.zeros((308, 3), dtype=float)  # all conf 0
    r = compute_bangs_forehead(seg, pose)
    assert r["abstained"] is True
    assert "eye-line" in (r["abstention_reason"] or "")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_bangs() -> None:
    seg = _face(_seg())
    _hair(seg, top=400, bot=499)
    lines = render_bangs_forehead(compute_bangs_forehead(seg, _pose()))
    assert any("bangs" in ln for ln in lines)


def test_render_swept_back() -> None:
    seg = _face(_seg())
    _hair(seg, top=300, bot=360)
    lines = render_bangs_forehead(compute_bangs_forehead(seg, _pose()))
    assert any("swept back" in ln for ln in lines)


def test_render_not_measured_empty() -> None:
    assert render_bangs_forehead({}) == []


def test_render_abstain() -> None:
    lines = render_bangs_forehead({"abstained": True, "abstention_reason": "no face"})
    assert any("abstain" in ln for ln in lines)


def test_payload_honest_no_px_in_prose() -> None:
    seg = _face(_seg())
    _hair(seg, top=400, bot=499)
    joined = " ".join(render_bangs_forehead(compute_bangs_forehead(seg, _pose())))
    assert "800" not in joined and "500" not in joined  # no raw px in prose
