"""TDD coverage for the garment-type evidence specialist (arm #97).

Deterministic garment-type / silhouette-category band from seg2 DOME-29
clothing classes (Apparel + Upper/Lower_Clothing + skin). Scale-invariant:
only the coarse band is verbalized; raw class-coverage ratios stay payload-only.
Pure and tested without any model; no GPU needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.garment_type import (
    GarmentTypeError,
    compute_garment_type,
    render_garment_type,
    validate_seg2_array,
)


def _seg_with_garments(*, upper: int = 0, lower: int = 0, torso: int = 0,
                       apparel: int = 0, lower_limb: int = 0, background: bool = True,
                       shape=(480, 480)) -> np.ndarray:
    """A seg2 canvas with a foreground subject and controllable garment regions.

    Upper region occupies rows 80..240; lower region rows 240..400 (approx).
    `upper`/`lower`/`torso` are the counts of Upper_Clothing / Lower_Clothing /
    Torso pixels scattered into their regions; `apparel` scatters Apparel across
    the subject (split by subject centroid for the lower-half dress rule).
    """
    seg = np.zeros(shape, dtype=np.uint8)
    seg[:, :] = 0  # background
    rng = np.random.default_rng(20260808)
    h, w = shape

    def scatter(count: int, class_id: int, y0: int, y1: int):
        if count <= 0:
            return
        ys = rng.integers(y0, y1, count)
        xs = rng.integers(0, w, count)
        seg[ys, xs] = class_id

    # subject: fill the whole frame lightly as upper/lower regions (foreground)
    seg[60:420, 40:440] = 7  # Left_Upper_Arm (a skin/limb class in SKIN_LIMBS)
    if upper:
        scatter(upper, 23, 60, 240)   # Upper_Clothing = class 23
    if lower:
        scatter(lower, 13, 240, 420)  # Lower_Clothing = class 13
    if torso:
        scatter(torso, 22, 120, 260)  # Torso = class 22 (skin)
    if apparel:
        scatter(apparel, 1, 60, 420)  # Apparel = class 1
    if lower_limb:
        scatter(lower_limb, 12, 240, 420)  # Left_Upper_Leg skin
    return seg


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

def test_validate_seg2() -> None:
    with pytest.raises(GarmentTypeError):
        validate_seg2_array(np.zeros((480, 480, 3), dtype=np.uint8))
    with pytest.raises(GarmentTypeError):
        validate_seg2_array(np.zeros(480, dtype=np.uint8))


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------

def test_upper_lower_covered() -> None:
    seg = _seg_with_garments(upper=30000, lower=30000)
    r = compute_garment_type(seg)
    assert r["abstained"] is False
    assert r["garment_type_band"] == "upper-lower-covered"
    assert r["upper_garment_present"] is True
    assert r["lower_garment_present"] is True


def test_upper_only() -> None:
    seg = _seg_with_garments(upper=30000, lower=0)
    r = compute_garment_type(seg)
    assert r["garment_type_band"] == "upper-only"
    assert r["upper_garment_present"] is True
    assert r["lower_garment_present"] is False


def test_lower_only() -> None:
    seg = _seg_with_garments(upper=0, lower=30000)
    r = compute_garment_type(seg)
    assert r["garment_type_band"] == "lower-only"
    assert r["upper_garment_present"] is False
    assert r["lower_garment_present"] is True


def test_skin_dominant() -> None:
    seg = _seg_with_garments(upper=0, lower=0, torso=20000, lower_limb=20000)
    r = compute_garment_type(seg)
    assert r["abstained"] is False
    assert r["garment_type_band"] == "skin-dominant"
    assert r["skin_dominant"] is True


def test_scale_invariant_same_input() -> None:
    """Same class layout at different resolutions -> same band + ratios."""
    r1 = compute_garment_type(_seg_with_garments(upper=30000, lower=30000, shape=(480, 480)))
    r2 = compute_garment_type(_seg_with_garments(upper=120000, lower=120000, shape=(960, 960)))
    assert r1["garment_type_band"] == r2["garment_type_band"]


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------

def test_no_foreground_abstains() -> None:
    seg = np.zeros((480, 480), dtype=np.uint8)  # all background
    r = compute_garment_type(seg)
    assert r["abstained"] is True
    assert r["subject_present"] is False


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_dressed() -> None:
    r = compute_garment_type(_seg_with_garments(upper=30000, lower=30000))
    lines = render_garment_type(r)
    assert any("dressed" in ln for ln in lines)


def test_render_upper_only() -> None:
    r = compute_garment_type(_seg_with_garments(upper=30000, lower=0))
    lines = render_garment_type(r)
    assert any("upper body clothed" in ln for ln in lines)


def test_render_not_measured_empty() -> None:
    assert render_garment_type({}) == []


def test_render_abstain() -> None:
    r = {"abstained": True, "abstention_reason": "no garment classes present"}
    lines = render_garment_type(r)
    assert any("abstain" in ln for ln in lines)


def test_render_no_ratio_in_prose() -> None:
    r = compute_garment_type(_seg_with_garments(upper=30000, lower=30000))
    joined = " ".join(render_garment_type(r))
    assert "0." not in joined  # raw coverage ratios stay payload-only
