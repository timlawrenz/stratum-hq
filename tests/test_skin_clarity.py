"""Tests for the arm-#125 skin-clarity deterministic specialist.

NEW evidence part (skin-clarity): illumination-invariant high-frequency
blemish/unevenness energy density over the seg2 exposed-skin region
(mean |Laplacian| / skin-luminance-std, canonical 512). Registered 2026-08-11
via the gated propose-dimensions channel (brainstorm-new-data), selected by
the SELECTOR EXPLOIT slot. Capability probe on the frozen cohort
(skin-clarity-calibration-probe.json): 24/24 measured, clear 8 / even 8 /
blemished 8 (max_share 0.3333 < 0.75 NON-degenerate), coverage floor 8/24
MET, 0 abstentions.

Only the coarse clear/even/blemished band is verbalized; raw HP density,
percentiles, tail-ratio, spot-fraction, and luma-std stay payload-only.
Non-redundance: skin-color #31 = tone, image-quality #95 = whole-image IQA,
texture #35 = garment material, image-focus #75 = optical acutance; this axis
is region-specific skin clarity (blemish/unevenness density).
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.skin_clarity import (
    CLEAR_MAX,
    EVEN_MAX,
    SkinClarityError,
    compute_skin_clarity,
    render_skin_clarity,
    validate_seg2_array,
)

SKIN = 3  # DOME-29 Face_Neck (a skin class > 0 -> foreground)
_TORSO = 22


def _seg_with_skin(*, shape=(240, 240), skin_class=SKIN) -> np.ndarray:
    seg = np.zeros(shape, dtype=np.uint8)
    seg[60:180, 60:180] = skin_class  # subject fills the middle
    return seg


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

def test_validate_seg2() -> None:
    with pytest.raises(SkinClarityError):
        compute_skin_clarity(np.zeros((240, 240, 3), dtype=np.uint8),
                             np.zeros((240, 240), dtype=np.uint8))
    with pytest.raises(SkinClarityError):
        compute_skin_clarity(np.zeros((240, 240), dtype=np.uint8),
                             np.zeros((240, 240), dtype=np.uint8))


def test_validate_seg2_array_direct() -> None:
    with pytest.raises(SkinClarityError):
        validate_seg2_array(np.zeros((240, 240, 3), dtype=np.uint8))
    with pytest.raises(SkinClarityError):
        validate_seg2_array(np.zeros((240, 240), dtype=np.float64))


def test_validate_rgb_shape() -> None:
    seg = _seg_with_skin()
    with pytest.raises(SkinClarityError):
        compute_skin_clarity(seg, np.zeros((240, 240), dtype=np.uint8))
    with pytest.raises(SkinClarityError):
        compute_skin_clarity(seg, np.zeros((120, 120, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Abstention behavior (pure paths)
# ---------------------------------------------------------------------------

def test_abstains_no_foreground() -> None:
    seg = np.zeros((240, 240), dtype=np.uint8)
    out = compute_skin_clarity(seg, np.full((240, 240, 3), 120, dtype=np.uint8))
    assert out["abstained"] is True
    assert out["subject_present"] is False
    assert out["skin_clarity_band"] is None
    assert "no foreground" in out["abstention_reason"]


def test_abstains_below_gates() -> None:
    seg = _seg_with_skin()
    out = compute_skin_clarity(seg, np.full((240, 240, 3), 120, dtype=np.uint8),
                               min_class_px=100_000)
    assert out["abstained"] is True
    assert out["skin_clarity_band"] is None
    assert "below presence gates" in out["abstention_reason"]


def test_abstains_flat_contrast() -> None:
    """Blown-out / uniform skin must abstain (degenerate luminance ~0)."""
    seg = _seg_with_skin()
    rgb = np.full((240, 240, 3), 250, dtype=np.uint8)  # near-blown-out flat
    out = compute_skin_clarity(seg, rgb)
    assert out["abstained"] is True
    assert "degenerate contrast" in out["abstention_reason"]


def test_abstains_tiny_interior() -> None:
    seg = np.zeros((240, 240), dtype=np.uint8)
    seg[119:121, 119:121] = SKIN  # too small to survive erosion + floor
    out = compute_skin_clarity(seg, np.full((240, 240, 3), 120, dtype=np.uint8))
    assert out["abstained"] is True


# ---------------------------------------------------------------------------
# Band determinism (pure) — smooth vs spotty skin
# ---------------------------------------------------------------------------

def test_smooth_skin_clear_band() -> None:
    """Uniform smooth skin with a tiny structural texture => clear (low D1)."""
    seg = _seg_with_skin()
    rgb = _smooth_rgb(seg)
    out = compute_skin_clarity(seg, rgb)
    assert out["abstained"] is False
    assert out["skin_clarity_band"] == "clear"
    assert out["skin_hp_density_luma"] is not None


def test_spotty_skin_blemished_band() -> None:
    """Skin with strong high-frequency spot noise => blemished (high D1)."""
    seg = _seg_with_skin()
    rgb = _spotty_rgb(seg)
    out = compute_skin_clarity(seg, rgb)
    assert out["abstained"] is False
    assert out["skin_clarity_band"] == "blemished"
    smooth = compute_skin_clarity(seg, _smooth_rgb(seg))
    assert out["skin_hp_density_luma"] > smooth["skin_hp_density_luma"]


def _smooth_rgb(seg: np.ndarray) -> np.ndarray:
    """A genuinely smooth skin block: a gentle linear luminance gradient only
    (large-scale shading, no small high-frequency spots)."""
    rgb = np.full((*seg.shape, 3), 160, dtype=np.uint8)
    skin = seg > 0
    rows, cols = np.nonzero(skin)
    if rows.size == 0:
        return rgb
    # Linear brightness ramp across the block (0..255→low-F structure-less).
    ramp = np.linspace(140, 180, seg.shape[1]).astype(np.uint8)
    for c in range(seg.shape[1]):
        rgb[skin[:, c], c] = ramp[c]
    return rgb


def _spotty_rgb(seg: np.ndarray) -> np.ndarray:
    rgb = np.full((*seg.shape, 3), 160, dtype=np.uint8)
    skin = seg > 0
    yy, xx = np.nonzero(skin)
    rng = np.random.default_rng(0)
    picks = rng.integers(0, yy.size, size=max(1, yy.size // 8))
    for idx in picks:
        rgb[yy[idx], xx[idx]] = 255  # bright spots = high-F blemish texture
    return rgb


# ---------------------------------------------------------------------------
# Render determinism (pure)
# ---------------------------------------------------------------------------

def test_render_clear() -> None:
    text = render_skin_clarity({"abstained": False, "skin_clarity_band": "clear"})
    assert "clear" in text


def test_render_even() -> None:
    text = render_skin_clarity({"abstained": False, "skin_clarity_band": "even"})
    assert "even" in text


def test_render_blemished() -> None:
    text = render_skin_clarity({"abstained": False, "skin_clarity_band": "blemished"})
    assert "blemish" in text  # band name + natural-language rendering


def test_render_abstained_surfaces_reason() -> None:
    text = render_skin_clarity({
        "abstained": True,
        "abstention_reason": "degenerate contrast",
    })
    assert "abstain" in text
    assert "degenerate contrast" in text


def test_render_none_config() -> None:
    assert "not measured" in render_skin_clarity(None)


def test_render_missing_band_does_not_crash() -> None:
    text = render_skin_clarity({"subject_present": True, "abstained": False})
    assert isinstance(text, str)


# ---------------------------------------------------------------------------
# Band-cut sanity (pure, no threshold-fitting escape hatch)
# ---------------------------------------------------------------------------

def test_cuts_are_disclosed_calibration() -> None:
    """CLEAR_MAX < EVEN_MAX and both are strictly positive — the verbalized
    cuts are calibrated on the frozen cohort, not a degenerate all-in-one cut."""
    assert 0 < CLEAR_MAX < EVEN_MAX


def test_payload_never_leaks_into_render() -> None:
    """Raw density / luma-std must not appear as caption text."""
    seg = _seg_with_skin()
    out = compute_skin_clarity(seg, _smooth_rgb(seg))
    text = render_skin_clarity(out)
    assert "density" not in text
    assert "luma" not in text
    assert "std" not in text
