"""Tests for the arm-#126 background-color deterministic specialist.

NEW evidence part (background-color): scale-invariant dominant hue-family band
(warm / cool / neutral) over the seg2 DOME-29 Background region from the
decoded source RGB. Registered 2026-08-11 via the gated propose-dimensions
channel, selected by the SELECTOR EXPLOIT slot. Capability probe on the
frozen cohort (background-color-calibration-probe.json): 24/24 measured,
warm 11 / cool 6 / neutral 7 (max_share 0.4583 < 0.75 NON-degenerate),
coverage floor 8/24 MET, 0 abstentions.

Only the coarse hue-family band is verbalized; raw family shares, mean luma,
and mean RGB stay payload-only. Lightness is deliberately payload-only
(setting #34 already grounds a light/mid/dark tone band).
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.background_color import (
    BackgroundColorError,
    compute_background_color,
    render_background_color,
    validate_seg2_array,
)

SKIN = 7  # DOME-29 Left_Upper_Arm (a limb/skin class > 0 -> foreground)


def _seg_with_subject(*, shape=(240, 240)) -> np.ndarray:
    """A seg2 canvas with a foreground subject block + background surround."""
    seg = np.zeros(shape, dtype=np.uint8)
    seg[60:180, 60:180] = SKIN  # subject fills the middle -> background ring
    return seg


def _solid_rgb(color) -> np.ndarray:
    return np.full((240, 240, 3), color, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

def test_validate_seg2() -> None:
    with pytest.raises(BackgroundColorError):
        compute_background_color(np.zeros((240, 240, 3), dtype=np.uint8),
                                 np.zeros((240, 240, 3), dtype=np.uint8))
    with pytest.raises(BackgroundColorError):
        compute_background_color(np.zeros((240, 240), dtype=np.uint8),
                                 np.zeros((240, 240), dtype=np.uint8))


def test_validate_seg2_array_direct() -> None:
    with pytest.raises(BackgroundColorError):
        validate_seg2_array(np.zeros((240, 240, 3), dtype=np.uint8))
    with pytest.raises(BackgroundColorError):
        validate_seg2_array(np.zeros((240, 240), dtype=np.float64))


def test_validate_rgb_shape() -> None:
    seg = _seg_with_subject()
    with pytest.raises(BackgroundColorError):
        compute_background_color(seg, np.zeros((240, 240), dtype=np.uint8))
    with pytest.raises(BackgroundColorError):
        compute_background_color(seg, np.zeros((120, 120, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Abstention behavior (pure paths)
# ---------------------------------------------------------------------------

def test_abstains_no_foreground() -> None:
    seg = np.zeros((240, 240), dtype=np.uint8)
    out = compute_background_color(seg, _solid_rgb([120, 120, 120]))
    assert out["abstained"] is True
    assert out["subject_present"] is False
    assert out["background_color_band"] is None
    assert "no foreground" in out["abstention_reason"]


def test_abstains_below_gates() -> None:
    seg = _seg_with_subject()
    out = compute_background_color(seg, np.full((240, 240, 3), 200, dtype=np.uint8),
                                   min_bg_px=100_000)
    assert out["abstained"] is True
    assert out["background_color_band"] is None
    assert "too small" in out["abstention_reason"]


# ---------------------------------------------------------------------------
# Hue-family determinism (pure)
# ---------------------------------------------------------------------------

def test_warm_orange() -> None:
    seg = _seg_with_subject()
    out = compute_background_color(seg, _solid_rgb([255, 150, 40]))
    assert out["abstained"] is False
    assert out["background_color_band"] == "warm"
    assert out["bg_family_shares"]["warm"] > 0.9


def test_cool_blue() -> None:
    seg = _seg_with_subject()
    out = compute_background_color(seg, _solid_rgb([40, 120, 200]))
    assert out["abstained"] is False
    assert out["background_color_band"] == "cool"
    assert out["bg_family_shares"]["cool"] > 0.9


def test_neutral_grey() -> None:
    seg = _seg_with_subject()
    out = compute_background_color(seg, _solid_rgb([120, 120, 120]))
    assert out["abstained"] is False
    assert out["background_color_band"] == "neutral"
    assert out["bg_family_shares"]["neutral"] > 0.9


def test_scale_invariance_band_unchanged_across_resolution() -> None:
    """The hue-family band is scale-invariant (same colors, any frame size)."""
    for shape in ((120, 160), (240, 320), (480, 640)):
        seg = np.zeros(shape, dtype=np.uint8)
        seg[shape[0] // 4:3 * shape[0] // 4, shape[1] // 4:3 * shape[1] // 4] = SKIN
        out = compute_background_color(seg, np.full((*shape, 3), [255, 150, 40], dtype=np.uint8))
        assert out["background_color_band"] == "warm"


def test_mixed_backdrop_dominant_family_wins() -> None:
    """A mostly-cool-with-some-warm backdrop resolves to cool (dominant)."""
    seg = _seg_with_subject()
    rgb = np.full((240, 240, 3), [40, 120, 200], dtype=np.uint8)  # cool
    rgb[10:30, 20:40] = [255, 150, 40]  # small warm patch
    out = compute_background_color(seg, rgb)
    assert out["background_color_band"] == "cool"
    assert out["bg_family_shares"]["cool"] > out["bg_family_shares"]["warm"]


def test_lightness_stays_payload_only() -> None:
    """Lightness is measured but must NOT leak into the verbalized claims."""
    seg = _seg_with_subject()
    dark = compute_background_color(seg, _solid_rgb([20, 20, 20]))
    light = compute_background_color(seg, _solid_rgb([230, 230, 230]))
    assert dark["background_lightness_payload"] == "dark"
    assert light["background_lightness_payload"] == "light"
    # Both are neutral hue family; the payload difference must not change the
    # verbalized claim set (render output must be identical).
    assert render_background_color(dark) == render_background_color(light)


# ---------------------------------------------------------------------------
# Render determinism (pure)
# ---------------------------------------------------------------------------

def test_render_warm() -> None:
    text = render_background_color({"abstained": False, "background_color_band": "warm"})
    assert "warm" in text
    assert "red/yellow/orange" in text


def test_render_cool() -> None:
    text = render_background_color({"abstained": False, "background_color_band": "cool"})
    assert "cool" in text
    assert "blue/green/purple" in text


def test_render_neutral() -> None:
    text = render_background_color({"abstained": False, "background_color_band": "neutral"})
    assert "neutral" in text
    assert "achromatic" in text


def test_render_abstained_surfaces_reason() -> None:
    text = render_background_color({
        "abstained": True,
        "abstention_reason": "background region too small",
    })
    assert "abstain" in text
    assert "too small" in text


def test_render_none_config() -> None:
    assert "not measured" in render_background_color(None)


def test_render_missing_band_does_not_crash() -> None:
    text = render_background_color({"subject_present": True, "abstained": False})
    assert isinstance(text, str)
