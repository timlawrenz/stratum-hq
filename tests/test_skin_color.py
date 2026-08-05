"""TDD coverage for deterministic DOME-29 skin-tone measurements (arm #31).

These measurements are the deterministic evidence for arm #31 (skin-color).
They must be continuous exposure-coverage fractions + deterministic quantized
dominant-tone names (never absolute pixel claims), computed from existing
`seg2` masks + source pixels, honoring exactly-one-subject abstention.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.skin_color import (
    FACE_NECK,
    TORSO,
    SkinColorError,
    compute_skin_tone,
    validate_seg2_array,
)


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


def test_validate_seg2_array_accepts_2d_uint8() -> None:
    a = np.zeros((64, 64), dtype=np.uint8)
    validate_seg2_array(a)  # must not raise


def test_validate_seg2_array_rejects_wrong_ndim() -> None:
    with pytest.raises(SkinColorError):
        validate_seg2_array(np.zeros((64, 64, 3), dtype=np.uint8))


def test_validate_seg2_array_rejects_non_2d_1d() -> None:
    with pytest.raises(SkinColorError):
        validate_seg2_array(np.zeros((64,), dtype=np.uint8))


def test_compute_skin_tone_no_subject_abstains() -> None:
    seg = np.zeros((80, 80), dtype=np.uint8)
    m = compute_skin_tone(seg, _rgb((200, 30, 40), height=80, width=80))
    assert m["subject_present"] is False
    assert m["abstained"] is True
    assert m["exposed_skin_present"] is False
    assert m["skin_tone_name"] is None


def test_compute_skin_tone_measures_exposed_torso_tone() -> None:
    seg = np.zeros((120, 120), dtype=np.uint8)
    seg[20:100, 20:100] = TORSO  # exposed torso fills most of the subject
    img = _rgb((60, 60, 60), height=120, width=120)
    img[20:100, 20:100] = (228, 190, 164)  # "light"
    m = compute_skin_tone(seg, img)
    assert m["subject_present"] is True
    assert m["exposed_skin_present"] is True
    assert m["skin_coverage"] > 0.5
    assert m["skin_tone_name"] == "light"
    assert m["skin_tone_hex"].startswith("#")


def test_compute_skin_tone_abstains_for_covered_subject() -> None:
    seg = np.zeros((100, 100), dtype=np.uint8)
    # Subject is entirely Apparel(1)/Upper_Clothing(23) -> no exposed skin.
    seg[20:80, 20:80] = 23
    seg[20:80, 20:80] = 1
    m = compute_skin_tone(seg, _rgb((120, 90, 70)))
    assert m["subject_present"] is True
    assert m["exposed_skin_present"] is False
    assert m["skin_tone_name"] is None


def test_compute_skin_tone_abstains_for_tiny_region_over_real_subject() -> None:
    seg = np.zeros((100, 100), dtype=np.uint8)
    seg[20:80, 20:80] = 23  # big clothed subject
    seg[40:41, 40:41] = TORSO  # 1px skin splice -> below floor
    m = compute_skin_tone(seg, _rgb((120, 90, 70)))
    assert m["subject_present"] is True
    assert m["exposed_skin_present"] is False
    assert m["skin_tone_name"] is None


def test_compute_skin_tone_face_body_regions() -> None:
    seg = np.zeros((120, 120), dtype=np.uint8)
    seg[40:100, 20:100] = TORSO  # body skin
    seg[10:40, 40:80] = FACE_NECK  # face/neck skin above
    img = _rgb((200, 160, 120), height=120, width=120)
    m = compute_skin_tone(seg, img)
    assert m["exposed_skin_present"] is True
    assert m["face_tone_name"] is not None
    assert m["body_tone_name"] is not None
    assert m["face_body_agree"] in (True, False)
    assert "face_neck" in m["measured_regions"]


def test_compute_skin_tone_misaligned_image_rejected() -> None:
    seg = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(SkinColorError):
        compute_skin_tone(seg, np.zeros((100, 101, 3), dtype=np.uint8))


def test_skin_class_constants_pinned_to_dome29() -> None:
    # Cross-check against the authoritative config so drift is caught.
    from stratum2.config import DOME_29

    assert DOME_29[3] == "Face_Neck"
    assert DOME_29[22] == "Torso"
    assert DOME_29[4] == "Hair"
    assert DOME_29[7] == "Left_Lower_Arm"
