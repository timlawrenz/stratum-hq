"""Tests for the arm-#94 hair-texture deterministic specialist."""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.hair_texture import (
    HairTextureError,
    compute_hair_texture,
    render_hair_texture,
)

HAIR = 4  # DOME-29 Hair
SKIN = 7  # Left_Upper_Arm (a limb/skin class > 0 -> foreground)


def _seg_with_hair(*, shape=(240, 240), hair_rows=None, hair_cols=None) -> np.ndarray:
    """A seg2 canvas with a foreground subject + a rectangular Hair region."""
    seg = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    seg[40:200, 40:200] = SKIN  # subject
    hrows = hair_rows if hair_rows is not None else (60, 140)
    hcols = hair_cols if hair_cols is not None else (80, 160)
    seg[hrows[0]:hrows[1], hcols[0]:hcols[1]] = HAIR
    return seg


def _synthetic_straight_hair(shape=(240, 240)):
    """Vertical bright/dark stripes -> single dominant horizontal-gradient axis."""
    seg = _seg_with_hair(shape=shape)
    img = np.zeros((*shape, 3), dtype=np.uint8)
    rows, cols = np.nonzero(seg == HAIR)
    for r, c in zip(rows, cols):
        # vertical stripes: brightness varies with column -> gradients along x
        img[r, c] = (80 + 120 * ((c // 3) % 2),) * 3
    img[seg == SKIN] = 128
    return seg, img


def _synthetic_curly_hair(shape=(240, 240)):
    """Checkerboard micro-pattern -> dispersed gradient orientations."""
    seg = _seg_with_hair(shape=shape)
    img = np.zeros((*shape, 3), dtype=np.uint8)
    rows, cols = np.nonzero(seg == HAIR)
    for r, c in zip(rows, cols):
        img[r, c] = (80 + 120 * ((r // 2 + c // 2) % 2),) * 3
    img[seg == SKIN] = 128
    return seg, img


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

def test_validate_seg2() -> None:
    with pytest.raises(HairTextureError):
        compute_hair_texture(np.zeros((240, 240, 3), dtype=np.uint8),
                             np.zeros((240, 240, 3), dtype=np.uint8))
    with pytest.raises(HairTextureError):
        compute_hair_texture(np.zeros((240, 240), dtype=np.uint8),
                             np.zeros((240, 240), dtype=np.uint8))


def test_misaligned_rgb_aborts() -> None:
    seg = _seg_with_hair()
    with pytest.raises(HairTextureError):
        compute_hair_texture(seg, np.zeros((100, 100, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Bands + abstention
# ---------------------------------------------------------------------------

def test_no_foreground_abstains() -> None:
    out = compute_hair_texture(np.zeros((240, 240), dtype=np.uint8),
                               np.zeros((240, 240, 3), dtype=np.uint8))
    assert out["abstained"] is True
    assert out["subject_present"] is False


def test_no_hair_abstains() -> None:
    seg = np.zeros((240, 240), dtype=np.uint8)
    seg[40:200, 40:200] = SKIN
    out = compute_hair_texture(seg, np.full((240, 240, 3), 128, dtype=np.uint8))
    assert out["abstained"] is True
    assert "Hair region absent" in out["abstention_reason"]


def test_smooth_hair_abstains() -> None:
    """Uniform hair interior -> texture-unresolvable honest abstention."""
    seg = _seg_with_hair()
    img = np.full((240, 240, 3), 128, dtype=np.uint8)
    out = compute_hair_texture(seg, img)
    assert out["abstained"] is True
    assert "unresolvable" in out["abstention_reason"]


def test_straight_vs_curly_discriminates() -> None:
    """Straight synthetic hair must have HIGHER R2 than curly synthetic hair."""
    seg_s, img_s = _synthetic_straight_hair()
    seg_c, img_c = _synthetic_curly_hair()
    out_s = compute_hair_texture(seg_s, img_s)
    out_c = compute_hair_texture(seg_c, img_c)
    assert out_s["abstained"] is False
    assert out_c["abstained"] is False
    assert out_s["r2_concentration"] > out_c["r2_concentration"]
    assert out_s["hair_texture_band"] in ("straight", "wavy")
    assert out_c["hair_texture_band"] in ("wavy", "curly")
    # Ordering must hold strictly: straight > wavy > curly on R2.
    assert out_s["r2_concentration"] > 0.5


def test_rotation_invariance() -> None:
    """Rotating the synthetic straight hair keeps the SAME band (R2 rotation-invariant)."""
    seg, img = _synthetic_straight_hair()
    out1 = compute_hair_texture(seg, img)
    rot_img = np.rot90(img, k=1)
    rot_seg = np.rot90(seg, k=1)
    out2 = compute_hair_texture(rot_seg, rot_img)
    assert out1["hair_texture_band"] == out2["hair_texture_band"]
    assert abs(out1["r2_concentration"] - out2["r2_concentration"]) < 0.05


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_no_config_empty() -> None:
    assert render_hair_texture({}) == []


def test_render_abstain() -> None:
    lines = render_hair_texture({"abstained": True, "abstention_reason": "too smooth"})
    assert any("abstain" in ln for ln in lines)


def test_render_band_text() -> None:
    lines = render_hair_texture({"hair_present": True, "hair_texture_band": "curly"})
    assert any("curly" in ln for ln in lines)


def test_render_no_raw_stats_in_prose() -> None:
    """Raw R2/entropy numbers must never leak into rendered claims."""
    lines = render_hair_texture({"hair_present": True, "hair_texture_band": "wavy",
                                 "r2_concentration": 0.5, "orientation_entropy": 1.2})
    joined = " ".join(lines)
    assert "0.5" not in joined


def test_synthetic_curly_is_curly_or_wavy() -> None:
    seg_c, img_c = _synthetic_curly_hair()
    out = compute_hair_texture(seg_c, img_c)
    assert out["hair_texture_band"] in ("curly", "wavy")
    assert out["r2_concentration"] < 0.75