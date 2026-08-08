"""TDD coverage for the face-visibility evidence specialist (arm #84).

Deterministic face-prominence band (clearly-visible / partially-framed /
hair-dominant) from seg2 DOME-29 Face_Neck + Hair, scale-invariant. Only the
coarse band is verbalized; the raw face-share ratio stays payload-only. Pure
and tested without any model; no GPU needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.face_visibility import (
    CLEARLY_VISIBLE_MIN,
    FRAMED_MIN,
    FaceVisibilityError,
    compute_face_visibility,
    render_face_visibility,
    validate_seg2_array,
)

from stratum2.config import DOME_29

_FACE = DOME_29.index("Face_Neck")
_HAIR = DOME_29.index("Hair")


def _seg(h: int = 400, w: int = 400) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _paint(seg: np.ndarray, cls: int, r0: int, r1: int, c0: int, c1: int) -> None:
    seg[r0:r1, c0:c1] = cls


def _face_only() -> np.ndarray:
    """A face region alone (no hair) -> share=1 -> clearly-visible."""
    seg = _seg()
    _paint(seg, _FACE, 100, 240, 150, 250)
    return seg


def _face_with_some_hair() -> np.ndarray:
    """Face + hair filling the local head window around it -> moderately framed.

    Face is a 140x100 block; hair is painted over the whole rest of the local
    head window (20px margin around the face bbox), so the face share lands in
    the [0.45, 0.65) band.
    """
    seg = _seg()
    face_r0, face_r1, face_c0, face_c1 = 100, 240, 150, 250
    _paint(seg, _FACE, face_r0, face_r1, face_c0, face_c1)
    win_r0, win_r1, win_c0, win_c1 = face_r0 - 20, face_r1 + 20, face_c0 - 20, face_c1 + 20
    seg[win_r0:win_r1, win_c0:win_c1] = _HAIR
    seg[face_r0:face_r1, face_c0:face_c1] = _FACE  # face back on top (hard labels)
    return seg


def _face_mostly_hair() -> np.ndarray:
    """A small face region inside a huge hair mass -> hair-dominant."""
    seg = _seg()
    _paint(seg, _FACE, 160, 200, 185, 215)  # 40x30 = 1200 px (above floor)
    _paint(seg, _HAIR, 120, 280, 160, 240)   # hair covering the whole head window
    seg[160:200, 185:215] = _FACE
    return seg


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

def test_validate_seg2() -> None:
    with pytest.raises(FaceVisibilityError):
        validate_seg2_array(np.zeros((5, 5, 1), dtype=np.uint8))
    with pytest.raises(FaceVisibilityError):
        validate_seg2_array(np.zeros((5, 5), dtype=np.float32))
    with pytest.raises(FaceVisibilityError):
        validate_seg2_array("nope")


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------

def test_face_only_clearly_visible() -> None:
    r = compute_face_visibility(_face_only())
    assert r["abstained"] is False
    assert r["face_visibility_band"] == "clearly-visible"
    assert r["face_share_of_head"] is not None
    assert r["face_share_of_head"] >= CLEARLY_VISIBLE_MIN


def test_some_hair_partially_framed() -> None:
    r = compute_face_visibility(_face_with_some_hair())
    assert r["face_visibility_band"] == "partially-framed"
    assert FRAMED_MIN <= r["face_share_of_head"] < CLEARLY_VISIBLE_MIN


def test_mostly_hair_hair_dominant() -> None:
    r = compute_face_visibility(_face_mostly_hair())
    assert r["face_visibility_band"] == "hair-dominant"
    assert r["face_share_of_head"] < FRAMED_MIN


def test_scale_invariant() -> None:
    """Scaling the frame + regions must keep the same band and share."""
    small = _face_with_some_hair()
    r_small = compute_face_visibility(small)
    big = _seg(800, 800)
    face_r0, face_r1, face_c0, face_c1 = 200, 480, 300, 500
    _paint(big, _FACE, face_r0, face_r1, face_c0, face_c1)
    win_r0, win_r1, win_c0, win_c1 = face_r0 - 40, face_r1 + 40, face_c0 - 40, face_c1 + 40
    big[win_r0:win_r1, win_c0:win_c1] = _HAIR
    big[face_r0:face_r1, face_c0:face_c1] = _FACE
    r_big = compute_face_visibility(big)
    assert r_small["face_visibility_band"] == r_big["face_visibility_band"]
    assert r_big["face_share_of_head"] == pytest.approx(
        r_small["face_share_of_head"], abs=1e-3
    )


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------

def test_no_face_abstains() -> None:
    r = compute_face_visibility(_seg())
    assert r["abstained"] is True
    assert r["face_visibility_band"] is None
    assert "face" in (r["abstention_reason"] or "").lower()


def test_tiny_face_abstains() -> None:
    seg = _seg()
    _paint(seg, _FACE, 200, 210, 200, 210)  # 100 px < floor
    r = compute_face_visibility(seg)
    assert r["abstained"] is True


def test_face_no_hair_in_window_still_measures() -> None:
    """Face with zero hair around it -> share 1.0 (clearly-visible, honest)."""
    r = compute_face_visibility(_face_only())
    assert r["face_share_of_head"] == pytest.approx(1.0, abs=1e-4)
    assert r["face_visibility_band"] == "clearly-visible"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_clearly_visible() -> None:
    lines = render_face_visibility(compute_face_visibility(_face_only()))
    assert any("clearly visible" in ln for ln in lines)


def test_render_partially_framed() -> None:
    lines = render_face_visibility(compute_face_visibility(_face_with_some_hair()))
    assert any("framed" in ln for ln in lines)


def test_render_hair_dominant() -> None:
    lines = render_face_visibility(compute_face_visibility(_face_mostly_hair()))
    assert any("hair dominates" in ln for ln in lines)


def test_render_not_measured_empty() -> None:
    assert render_face_visibility({}) == []


def test_render_abstain() -> None:
    r = {"abstained": True, "abstention_reason": "no face"}
    lines = render_face_visibility(r)
    assert any("abstain" in ln for ln in lines)


def test_render_no_ratio_in_prose() -> None:
    r = compute_face_visibility(_face_with_some_hair())
    joined = " ".join(render_face_visibility(r))
    assert "0.5" not in joined  # ratio stays payload-only


# ---------------------------------------------------------------------------
# Threshold sanity
# ---------------------------------------------------------------------------

def test_thresholds_sane() -> None:
    assert 0.0 < FRAMED_MIN < CLEARLY_VISIBLE_MIN <= 1.0
