"""TDD coverage for the camera-viewing-angle / framing evidence specialist (arm #74).

Deterministic camera-relative framing bands from seg2 subject mask + full-frame
geometry. Only scale-invariant bands are verbalized; raw bbox extents / frame
shares stay in the machine-readable payload. Pure and tested without any model;
no GPU needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.camera_viewing_angle import (
    CAMERA_HIGH,
    CAMERA_LOW,
    CameraViewingAngleError,
    compute_camera_viewing_angle,
    render_camera_viewing_angle,
    validate_seg2_array,
)


def _frame(h=1000, w=750):
    return h, w


def _subject_in(h, w, y0, y1, x0, x1):
    seg = np.zeros((h, w), dtype=np.uint8)
    seg[y0:y1, x0:x1] = 1
    return seg


def test_validate_seg2() -> None:
    with pytest.raises(CameraViewingAngleError):
        validate_seg2_array(np.zeros((5, 5, 1), dtype=np.uint8))
    with pytest.raises(CameraViewingAngleError):
        validate_seg2_array(np.zeros((5, 5), dtype=np.float32))


def test_empty_mask_abstains() -> None:
    h, w = _frame()
    r = compute_camera_viewing_angle(np.zeros((h, w), dtype=np.uint8), h, w)
    assert r["abstained"] is True
    assert "no subject" in r["abstention_reason"]


def test_degenerate_bbox_abstains() -> None:
    h, w = _frame()
    seg = np.zeros((h, w), dtype=np.uint8)
    seg[500, 500] = 1  # single-pixel subject
    r = compute_camera_viewing_angle(seg, h, w)
    assert r["abstained"] is True


def test_fullbleed_is_legitimate_framing() -> None:
    h, w = _frame()
    # Full-bleed subject (touches both frame edges) is legitimate framing:
    # tight headroom + large shot share, NOT an abstention.
    seg = _subject_in(h, w, 0, 999, 0, 749)
    r = compute_camera_viewing_angle(seg, h, w)
    assert r["abstained"] is False
    assert r["headroom_band"] == "tight"
    assert r["shot_scale_band"] == "close-up" or r["shot_scale_band"] == "full-body"


def test_closeup_band() -> None:
    h, w = _frame()
    # subject fills 80% of the frame height, centered -> full-body
    seg = _subject_in(h, w, 100, 900, 200, 600)
    r = compute_camera_viewing_angle(seg, h, w)
    assert r["abstained"] is False
    assert r["shot_scale_band"] == "full-body"


def test_midshot_band() -> None:
    h, w = _frame()
    # subject spans ~40% of frame height near the center -> mid-shot
    seg = _subject_in(h, w, 300, 700, 250, 550)
    r = compute_camera_viewing_angle(seg, h, w)
    assert r["shot_scale_band"] == "mid-shot"


def test_closeup_small_share() -> None:
    h, w = _frame()
    # subject spans ~20% of frame height (small head-and-shoulders) -> close-up
    seg = _subject_in(h, w, 400, 600, 300, 480)
    r = compute_camera_viewing_angle(seg, h, w)
    assert r["shot_scale_band"] == "close-up"


def test_eye_level_centered() -> None:
    h, w = _frame()
    # subject vertically centered -> camera at eye level
    seg = _subject_in(h, w, 350, 650, 250, 500)
    r = compute_camera_viewing_angle(seg, h, w)
    assert r["camera_height_band"] == "eye-level"


def test_low_camera_high_subject() -> None:
    h, w = _frame()
    # subject occupies the UPPER half of the frame -> camera below (looking up)
    seg = _subject_in(h, w, 100, 450, 250, 500)
    r = compute_camera_viewing_angle(seg, h, w)
    assert "camera below" in r["camera_height_band"]


def test_high_camera_low_subject() -> None:
    h, w = _frame()
    # subject occupies the LOWER half of the frame -> camera above (looking down)
    seg = _subject_in(h, w, 550, 900, 250, 500)
    r = compute_camera_viewing_angle(seg, h, w)
    assert "camera above" in r["camera_height_band"]


def test_headroom_bands() -> None:
    h, w = _frame()
    seg = _subject_in(h, w, 50, 850, 250, 500)   # tight headroom (y0 near top)
    r = compute_camera_viewing_angle(seg, h, w)
    assert r["headroom_band"] == "tight"
    seg2 = _subject_in(h, w, 400, 800, 250, 500)  # wide headroom (y0 around 0.4)
    r2 = compute_camera_viewing_angle(seg2, h, w)
    assert r2["headroom_band"] == "wide"


def test_render_abstention() -> None:
    lines = render_camera_viewing_angle({"abstained": True, "abstention_reason": "cropped"})
    assert lines and "abstain" in lines[0]


def test_render_bands() -> None:
    # headroom-only prose (shot-scale + camera-height are payload-only)
    r = compute_camera_viewing_angle(
        _subject_in(1000, 750, 300, 700, 250, 500), 1000, 750
    )
    lines = render_camera_viewing_angle(r)
    text = " ".join(lines)
    assert "headroom" in text
    assert "shot" not in text.lower()  # shotgun-scale not verbalized
    assert "eye-level" not in text.lower()


def test_payload_contains_raw_but_prose_not() -> None:
    h, w = _frame()
    r = compute_camera_viewing_angle(_subject_in(h, w, 350, 650, 250, 500), h, w)
    assert "subject_frame_height_share" in r  # raw share in payload
    prose = " ".join(render_camera_viewing_angle(r))
    # raw numeric shares/bbox never appear in prose
    assert "0." not in prose
    assert "350" not in prose
