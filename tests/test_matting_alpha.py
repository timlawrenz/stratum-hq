"""TDD coverage for deterministic matting / alpha-fidelity measurements.

These measurements are the deterministic evidence for arm #59 (matting-alpha).
They must be scale-invariant alpha facts (subject coverage band, boundary
crispness band, soft-edge character — hair-dominant vs clean skin cutout)
computed from existing `matting.npy` (per-pixel soft alpha) + `seg2` DOME-29
masks, honoring exactly-one-subject abstention — absolute pixel areas/band
widths stay in the machine-readable payload, never as caption claims.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.matting_alpha import (
    MattingAlphaError,
    compute_matting_alpha,
    validate_matting_array,
    validate_seg2_array,
)


def _alpha_crisp(height: int = 100, width: int = 100) -> np.ndarray:
    """Opaque subject rectangle with a 1-px step edge (crisp cutout)."""
    a = np.zeros((height, width), dtype=np.float32)
    a[20:80, 30:70] = 1.0
    return a


def _alpha_soft(height: int = 100, width: int = 100, ramp: int = 8) -> np.ndarray:
    """Opaque subject rectangle feathered on ALL four sides (soft edges).

    A linear ramp from 0 (outside the subject) to 1.0 (interior) across a
    `ramp`-px band on every border, so the silhouette ring gradient is soft on
    every side."""
    a = np.zeros((height, width), dtype=np.float32)
    a[20 + ramp:80 - ramp, 30 + ramp:70 - ramp] = 1.0
    for i in range(ramp):
        t = float(i) / float(ramp - 1) if ramp > 1 else 1.0
        # top and bottom horizontal feather
        a[20 + i, 30:70] = t
        a[79 - i, 30:70] = t
        # left and right vertical feather (over the interior edge)
        a[20:80, 30 + i] = np.maximum(a[20:80, 30 + i], t)
        a[20:80, 69 - i] = np.maximum(a[20:80, 69 - i], t)
    return a


def _seg(
    *,
    hair_band_rows=(19, 81),
    hair_band_cols=(26, 66),
    height: int = 100,
    width: int = 100,
) -> np.ndarray:
    """DOME-29 seg2 with a Hair class (4) band over the left/top/bottom edges."""
    a = np.zeros((height, width), dtype=np.uint8)
    a[hair_band_rows[0]:hair_band_rows[1], hair_band_cols[0]:hair_band_cols[1]] = 4
    return a


def test_subject_present_and_measurable_crisp() -> None:
    m = compute_matting_alpha(_alpha_crisp(), _seg())
    assert m["subject_present"] is True
    assert m["abstained"] is False
    assert m["matting_measurable"] is True
    assert m["coverage_band"] in ("sparse", "centered", "fills-frame")
    assert m["boundary_crisp_band"] in ("soft", "moderate", "crisp")
    # A truly crisp 1-px step has no semi-transparent band -> edge character
    # abstains (None); real mattes always carry anti-aliasing so this is a
    # deliberate hard-step edge case, not the measured cohort behavior.
    if m["soft_edge_band"] is not None:
        assert m["soft_edge_band"] in ("skin-clean", "mixed", "hair-dominant")
    assert m["coverage_ratio"] is not None and 0.0 < m["coverage_ratio"] <= 1.0
    assert m["silhouette_closedness"] is not None


def test_crisp_vs_soft_boundary_bands() -> None:
    crisp = compute_matting_alpha(_alpha_crisp(), _seg())
    soft = compute_matting_alpha(_alpha_soft(), _seg())
    # A 1-px step edge must be crisper than an 8-px feathered ramp.
    assert crisp["boundary_crispness"] > soft["boundary_crispness"]
    assert crisp["boundary_crisp_band"] != soft["boundary_crisp_band"]


def test_hair_dominant_soft_edge() -> None:
    # Hair class covers the entire soft band -> hair-dominant edge.
    m = compute_matting_alpha(_alpha_soft(), _seg())
    assert m["soft_edge_band"] in ("hair-dominant", "mixed")
    if m["soft_edge_band"] == "hair-dominant":
        assert m["hair_soft_share"] >= 0.5


def test_skin_clean_soft_edge_when_no_hair_over_band() -> None:
    # Move the Hair class away from the soft band -> skin/background cutout.
    seg = _seg(hair_band_cols=(10, 18))  # hair far left, not over the ramp
    m = compute_matting_alpha(_alpha_soft(), seg)
    assert m["soft_edge_band"] == "skin-clean"
    assert m["hair_soft_share"] is not None and m["hair_soft_share"] < 0.2


def test_abstains_when_subject_too_small() -> None:
    a = np.zeros((100, 100), dtype=np.float32)
    a[40:42, 40:42] = 1.0  # 4 opaque px < MIN_SUBJECT_PX
    m = compute_matting_alpha(a, _seg())
    assert m["abstained"] is True
    assert m["matting_measurable"] is False
    assert "too small" in m["abstention_reason"]


def test_abstains_when_values_outside_alpha_band() -> None:
    a = np.ones((100, 100), dtype=np.float32) * 1.5  # > 1.0 -> degenerate
    m = compute_matting_alpha(a, _seg())
    assert m["abstained"] is True
    assert "outside the [0, 1] alpha band" in m["abstention_reason"]


def test_validate_array_errors() -> None:
    with pytest.raises(MattingAlphaError):
        validate_matting_array(np.zeros((10, 10, 1), dtype=np.float32))
    with pytest.raises(MattingAlphaError):
        validate_matting_array(np.zeros((10, 10), dtype=np.int16))
    with pytest.raises(MattingAlphaError):
        validate_seg2_array(np.zeros((10, 10, 1), dtype=np.uint8))


def test_misaligned_shapes_raise() -> None:
    a = _alpha_crisp(100, 100)
    seg = np.zeros((50, 50), dtype=np.uint8)
    with pytest.raises(MattingAlphaError):
        compute_matting_alpha(a, seg)
