"""TDD coverage for the facial-expression evidence specialist (arm #81).

Deterministic smile/expression band from pose2 GOLIATH-308 mouth-corner +
eye-center keypoints, scale-invariant (normalized by inter-eye distance).
Only the coarse band is verbalized; raw ratios stay payload-only. Pure and
tested without any model; no GPU needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.facial_expression import (
    FacialExpressionError,
    compute_facial_expression,
    render_facial_expression,
    validate_pose2_array,
)

from stratum2.config import GOLIATH_308

_G = {name: i for i, name in enumerate(GOLIATH_308)}


def _pose(*, spread_px: float = 60.0, open_px: float = 8.0,
          corner_elev: float = 0.0, eye_ref: float = 200.0, scale: float = 1.0) -> np.ndarray:
    """Construct a frontal face with a controllable mouth geometry.

    eye centers symmetric around (400,200) separated by eye_ref -> the scale
    denom; mouth below. All coordinates scaled by `scale`.
    """
    pose = np.zeros((308, 3), dtype=float)
    pose[:, 2] = 1.0
    half_ref = eye_ref / 2.0
    pose[_G["l_center_of_iris"]] = ((400 - half_ref) * scale, 200 * scale, 0.9)
    pose[_G["r_center_of_iris"]] = ((400 + half_ref) * scale, 200 * scale, 0.9)
    mx = 400.0 * scale
    mouth_y = 500.0 * scale
    corner_y = mouth_y - corner_elev * scale
    sp = spread_px * scale
    op = open_px * scale
    pose[_G["l_outer_corner_of_mouth"]] = (mx - sp / 2, corner_y, 0.9)
    pose[_G["r_outer_corner_of_mouth"]] = (mx + sp / 2, corner_y, 0.9)
    inner_half = sp * 0.7 / 2
    pose[_G["l_inner_corner_of_mouth"]] = (mx - inner_half, corner_y, 0.9)
    pose[_G["r_inner_corner_of_mouth"]] = (mx + inner_half, corner_y, 0.9)
    pose[_G["midpoint_3_of_upper_outer_lip"]] = (mx, mouth_y - op / 2, 0.9)
    pose[_G["midpoint_3_of_lower_outer_lip"]] = (mx, mouth_y + op / 2, 0.9)
    return pose


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

def test_validate_pose2() -> None:
    with pytest.raises(FacialExpressionError):
        validate_pose2_array(np.zeros((308, 2)))
    with pytest.raises(FacialExpressionError):
        validate_pose2_array(np.zeros((200, 3), dtype=float))


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------

def test_open_smile() -> None:
    # mouth opening 70px / eye_ref 200 = 0.35 >= 0.28 -> open-smile
    pose = _pose(spread_px=120, open_px=70)
    r = compute_facial_expression(pose)
    assert r["abstained"] is False
    assert r["expression_band"] == "open-smile"
    assert r["openness_ratio"] >= 0.28


def test_slight_smile() -> None:
    # closed mouth (open 8/200 = 0.04) but corners raised +25px/200 = 0.125
    pose = _pose(spread_px=110, open_px=8, corner_elev=25)
    r = compute_facial_expression(pose)
    assert r["expression_band"] == "slight-smile"
    assert r["openness_ratio"] < 0.28
    assert r["corner_elevation_ratio"] >= 0.05


def test_neutral() -> None:
    # closed mouth, level corners, narrow spread
    pose = _pose(spread_px=70, open_px=6, corner_elev=0)
    r = compute_facial_expression(pose)
    assert r["expression_band"] == "neutral"
    assert r["corner_elevation_ratio"] < 0.05


def test_scale_invariant() -> None:
    """Doubling all geometry keeps the same band + ratios (pure ratio)."""
    small = _pose(spread_px=110, open_px=8, corner_elev=25)
    r_small = compute_facial_expression(small)
    big = _pose(spread_px=110, open_px=8, corner_elev=25, scale=2.0)
    r_big = compute_facial_expression(big)
    assert r_small["expression_band"] == r_big["expression_band"]
    assert r_big["spread_ratio"] == pytest.approx(r_small["spread_ratio"], abs=1e-3)


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------

def test_no_mouth_abstains() -> None:
    pose = np.zeros((308, 3), dtype=float)
    r = compute_facial_expression(pose)
    assert r["abstained"] is True
    assert r["expression_band"] is None
    assert r["abstention_reason"]


def test_eye_fallback_still_measures() -> None:
    """No eye keypoints -> local-scale fallback, still classified."""
    pose = _pose(spread_px=110, open_px=8, corner_elev=25)
    for n in GOLIATH_308:
        if "center_of_iris" in n or "center_of_pupil" in n:
            pose[_G[n]] = (0.0, 0.0, 0.0)  # zero out eyes
    r = compute_facial_expression(pose)
    assert r["abstained"] is False
    assert r["reference_fallback"] is True
    assert r["expression_band"] in ("slight-smile", "neutral")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_open_smile() -> None:
    r = compute_facial_expression(_pose(spread_px=120, open_px=70))
    lines = render_facial_expression(r)
    assert any("open smile" in ln for ln in lines)


def test_render_neutral() -> None:
    r = compute_facial_expression(_pose(spread_px=70, open_px=6, corner_elev=0))
    lines = render_facial_expression(r)
    assert any("neutral" in ln for ln in lines)


def test_render_not_measured_empty() -> None:
    assert render_facial_expression({}) == []


def test_render_abstain() -> None:
    r = {"abstained": True, "abstention_reason": "mouth occluded"}
    lines = render_facial_expression(r)
    assert any("abstain" in ln for ln in lines)


def test_render_no_ratio_in_prose() -> None:
    r = compute_facial_expression(_pose(spread_px=110, open_px=8, corner_elev=25))
    joined = " ".join(render_facial_expression(r))
    assert "0." not in joined  # normalized ratios stay payload-only
