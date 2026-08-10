"""TDD coverage for the image-focus / depth-of-field evidence specialist (arm #75).

Deterministic focus-quality bands from source RGB + seg2 region split, all
scale-invariant (subject-vs-frame interior acutance band + background-vs-
subject DOF band). Raw acutance numbers stay in the machine-readable payload.
Pure and tested without any model; no GPU needed.

Fixture convention (arm #35 pitfall: period-2 patterns alias to zero under
central differences): a period-4 checkerboard is the sharp texture, blurred
regions are heavy Gaussian smears of the same checkerboard.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from research_harness.image_focus import (
    CANONICAL_SIDE,
    ImageFocusError,
    compute_image_focus,
    render_image_focus,
    validate_image_focus_inputs,
)


def _checker(h: int, w: int, amp: float = 60.0, period: int = 12) -> np.ndarray:
    """Checkerboard in a gray base (period 12 — period 2/4 aliases to zero
    under LANCZOS downsample to the canonical 512 side, arm #35 pitfall)."""
    base = np.ones((h, w), dtype=np.float32) * 128.0
    yy = np.arange(h)[:, None] // period
    xx = np.arange(w)[None, :] // period
    check = ((yy + xx) % 2) == 0
    img = base.copy()
    img[check] += amp
    img[np.logical_not(check)] -= amp
    return np.clip(img, 0, 255).astype(np.uint8)


def _to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[:, :, None], 3, axis=2)


def _frame(h=1000, w=750):
    return h, w


def _subject_in(h, w, y0, y1, x0, x1):
    seg = np.zeros((h, w), dtype=np.uint8)
    seg[y0:y1, x0:x1] = 1
    return seg


def _sharp_subject_blurred_bg(
    h=1000, w=750, subj_amp=60.0, bg_amp=60.0, bg_sigma=4.0
):
    """Sharp period-12 subject inside a smoothed background."""
    img = _checker(h, w, amp=subj_amp)
    bg = _checker(h, w, amp=bg_amp)
    bg = gaussian_filter(bg, sigma=bg_sigma)
    seg = np.zeros((h, w), dtype=np.uint8)
    seg[250:750, 150:600] = 1
    img[seg == 0] = bg[seg == 0]
    return _to_rgb(img), seg


def test_validate_inputs() -> None:
    h, w = _frame()
    rgb = _to_rgb(_checker(h, w))
    with pytest.raises(ImageFocusError):
        validate_image_focus_inputs(rgb[..., 0], np.zeros((h, w), dtype=np.uint8))
    with pytest.raises(ImageFocusError):
        validate_image_focus_inputs(rgb.astype(np.float32), np.zeros((h, w), dtype=np.uint8))
    with pytest.raises(ImageFocusError):
        validate_image_focus_inputs(rgb, np.zeros((h, w, 1), dtype=np.uint8))
    with pytest.raises(ImageFocusError):
        validate_image_focus_inputs(rgb, np.zeros((h + 1, w), dtype=np.uint8))


def test_empty_subject_abstains() -> None:
    h, w = _frame()
    rgb = _to_rgb(_checker(h, w))
    r = compute_image_focus(rgb, np.zeros((h, w), dtype=np.uint8))
    assert r["abstained"] is True
    assert "subject" in r["abstention_reason"]


def test_fullbleed_keeps_subject_band_dof_abstains() -> None:
    h, w = _frame()
    # No background region at all: the subject band still measures, the DOF
    # axis honestly abstains (nothing to compare the subject against).
    img = _checker(h, w, amp=60.0)
    seg = np.ones((h, w), dtype=np.uint8)
    r = compute_image_focus(_to_rgb(img), seg)
    assert r["abstained"] is False
    assert r["dof_abstained"] is True
    assert r["dof_band"] is None
    assert "background" in r["dof_abstention_reason"]
    assert r["subject_focus_band"] in ("subject-crisp", "subject-comparable", "subject-softer")


def test_blurred_background_is_shallow_dof() -> None:
    rgb, seg = _sharp_subject_blurred_bg(bg_sigma=4.0)
    r = compute_image_focus(rgb, seg)
    assert r["abstained"] is False
    assert r["dof_band"] == "background-blurred"
    assert r["subject_focus_band"] in ("subject-crisp", "subject-comparable")
    assert r["dof_ratio"] < 0.45


def test_sharp_background_is_deep_focus() -> None:
    h, w = _frame()
    img = _checker(h, w, amp=60.0)
    seg = _subject_in(h, w, 250, 750, 150, 600)
    r = compute_image_focus(_to_rgb(img), seg)
    assert r["abstained"] is False
    assert r["dof_band"] == "background-sharp"
    assert r["dof_ratio"] >= 0.80


def test_soft_subject_is_detected() -> None:
    h, w = _frame()
    subj = gaussian_filter(_checker(h, w, amp=60.0), sigma=6.0)
    bg = _checker(h, w, amp=60.0)  # background stays sharp
    seg = _subject_in(h, w, 250, 750, 150, 600)
    img = subj.copy()
    img[seg == 0] = bg[seg == 0]
    r = compute_image_focus(_to_rgb(img), seg)
    assert r["abstained"] is False
    # subject is the softest part of the frame
    assert r["subject_focus_band"] == "subject-softer"
    assert r["subject_vs_frame_ratio"] is not None
    assert r["subject_vs_frame_ratio"] < 0.9


def test_flat_subject_does_not_crash_dof() -> None:
    """A fully-flat subject region (no interior texture) must abstain on the
    DOF axis with a surfaced reason, never divide by zero."""
    h, w = _frame()
    subj = np.full((h, w), 128.0, dtype=np.float32)
    bg = _checker(h, w, amp=60.0)
    seg = _subject_in(h, w, 250, 750, 150, 600)
    img = subj.copy()
    img[seg == 0] = bg[seg == 0]
    r = compute_image_focus(_to_rgb(np.clip(img, 0, 255).astype(np.uint8)), seg)
    assert r["abstained"] is False
    assert r["dof_abstained"] is True
    assert "texture" in r["dof_abstention_reason"]


def test_render_abstention() -> None:
    lines = render_image_focus({"abstained": True, "abstention_reason": "regions too small"})
    assert lines and "abstain" in lines[0]


def test_render_not_measured_emits_nothing() -> None:
    # Non-image-focus runs pass None/{}: no claim, never a fabricated focus.
    assert render_image_focus({}) == []
    assert render_image_focus(None) == []


def test_render_bands() -> None:
    r = compute_image_focus(*_sharp_subject_blurred_bg(bg_sigma=4.0))
    lines = render_image_focus(r)
    text = " ".join(lines)
    assert "image-focus" in text
    assert "depth-of-field" in text
    assert "blurred" in text or "softer" in text


def test_payload_contains_raw_but_prose_not() -> None:
    h, w = _frame()
    rgb, seg = _sharp_subject_blurred_bg(bg_sigma=4.0)
    r = compute_image_focus(rgb, seg)
    assert "subject_acutance_median" in r
    assert "dof_ratio" in r
    assert "canonical_dims" in r
    prose = " ".join(render_image_focus(r))
    # raw acutance numbers never appear in prose
    assert "acutance_median" not in prose
    assert "dof_ratio" not in prose


def test_canonical_resampling_is_invariant() -> None:
    """The same scene at two native resolutions lands in the same bands."""
    h, w = 2000, 1500
    rgb_big = np.repeat(
        np.repeat(np.asarray(_sharp_subject_blurred_bg()[0]), 2, axis=0), 2, axis=1
    )[:h, :w]
    seg_big = np.repeat(
        np.repeat(np.asarray(_sharp_subject_blurred_bg()[1]), 2, axis=0), 2, axis=1
    )[:h, :w]
    rgb_small = rgb_big[::2, ::2]
    seg_small = seg_big[::2, ::2]
    rb = compute_image_focus(rgb_big, seg_big)
    rs = compute_image_focus(rgb_small, seg_small)
    assert rb["dof_band"] == rs["dof_band"]
    assert rb["subject_focus_band"] == rs["subject_focus_band"]
    assert rb["canonical_dims"] == rs["canonical_dims"]
    assert rb["canonical_dims"][0] <= CANONICAL_SIDE