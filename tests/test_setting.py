"""TDD coverage for deterministic setting/environment measurements (arm #34).

These measurements are the deterministic evidence for arm #34 (setting). They
must be scale-invariant, caption-relevant setting facts (background coverage,
quantized dominant color, tone/vibrancy/pattern bands) computed from the
existing `seg2` + source pixels, honoring exactly-one-subject abstention —
absolute camera-frame values stay in the machine-readable payload, never as
caption claims.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.setting import (
    SettingError,
    compute_setting,
    validate_seg2_array,
)

BACKGROUND = 0  # must match DOME-29: class 0 is Background


def _seg(mask: dict[int, set[tuple[int, int]]], height: int = 100, width: int = 100) -> np.ndarray:
    a = np.zeros((height, width), dtype=np.uint8)
    for cls, cells in mask.items():
        for (y, x) in cells:
            a[y, x] = cls
    return a


def _rgb(color: tuple[int, int, int], height: int = 100, width: int = 100) -> np.ndarray:
    a = np.empty((height, width, 3), dtype=np.uint8)
    a[:] = color
    return a


def _full_frame_subject() -> np.ndarray:
    """subject occupies a centered third; everything else is Background (0)."""
    a = np.zeros((100, 100), dtype=np.uint8)
    a[35:65, 25:75] = 1
    return a


def test_validate_seg2_array_rejects_wrong_ndim() -> None:
    with pytest.raises(SettingError):
        validate_seg2_array(np.zeros((64, 64, 3), dtype=np.uint8))


def test_validate_seg2_array_rejects_float() -> None:
    with pytest.raises(SettingError):
        validate_seg2_array(np.zeros((64, 64), dtype=np.float32))


def test_compute_setting_rejects_misaligned_rgb() -> None:
    seg = _full_frame_subject()
    with pytest.raises(SettingError):
        compute_setting(seg, _rgb((200, 200, 200), height=99, width=99))


def test_compute_setting_abstains_when_no_subject() -> None:
    seg = np.zeros((100, 100), dtype=np.uint8)  # all Background, no subject
    m = compute_setting(seg, _rgb((120, 120, 120)))
    assert m["subject_present"] is False
    assert m["abstained"] is True
    assert m["setting_measurable"] is False
    assert m["abstention_reason"] == "no foreground subject detected"


def test_compute_setting_abstains_when_no_background() -> None:
    seg = np.ones((100, 100), dtype=np.uint8)  # subject fills frame, no Background
    m = compute_setting(seg, _rgb((120, 120, 120)))
    assert m["subject_present"] is True
    assert m["abstained"] is True
    assert m["setting_measurable"] is False
    assert "background" in m["abstention_reason"]


def test_compute_setting_measures_beige_background() -> None:
    seg = _full_frame_subject()
    m = compute_setting(seg, _rgb((210, 190, 160)))  # beige anchor
    assert m["subject_present"] is True
    assert m["abstained"] is False
    assert m["setting_measurable"] is True
    assert 0.4 < m["background_coverage"] < 0.9
    assert m["dominant_background_color"] == "beige"
    assert m["dominant_background_hex"] == "#d2bea0"
    assert m["background_tone_band"] in {"light", "mid", "dark"}
    assert m["background_vibrancy_band"] in {"muted", "moderate", "vivid"}
    assert m["background_pattern_band"] in {"solid", "some variation", "busy"}


def test_compute_setting_detects_busy_background() -> None:
    seg = _full_frame_subject()
    rgb = _rgb((210, 190, 160))
    # paint a strong checkerboard over the Background region
    bg = seg == BACKGROUND
    y, x = np.nonzero(bg)
    for yy, xx in zip(y[::7], x[::7]):
        rgb[yy, xx] = (10, 10, 10)
    m = compute_setting(seg, rgb)
    assert m["setting_measurable"] is True
    # a heavily patterned background must never be called "solid"
    assert m["background_pattern_band"] != "solid"


def test_compute_setting_has_scale_invariant_names_only() -> None:
    """No absolute pixel / raw-RGB caption-facing fields on a positive result."""
    seg = _full_frame_subject()
    m = compute_setting(seg, _rgb((95, 135, 70)))  # green anchor
    assert m["setting_measurable"] is True
    public_keys = {
        "subject_present", "abstained", "abstention_reason", "setting_measurable",
        "background_coverage", "dominant_background_color", "dominant_background_hex",
        "background_tone_band", "background_vibrancy_band", "background_pattern_band",
        "background_deviant_fraction", "background_mean_luma",
        "measured_bg_px", "measured_fg_px",
    }
    assert set(m) == public_keys