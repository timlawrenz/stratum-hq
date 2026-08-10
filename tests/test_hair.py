"""TDD coverage for deterministic DOME-29 hair measurements (arm #30).

These measurements are the deterministic evidence for arm #30 (hair).
They must be continuous coverage fractions + deterministic dominant colors
+ scale-invariant position/length facts (never absolute pixel claims),
computed from existing `seg2` masks + source pixels, honoring exactly-one-
subject abstention.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.hair import (
    FACE_NECK,
    HAIR,
    HairError,
    compute_hair,
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
    with pytest.raises(HairError):
        validate_seg2_array(np.zeros((64, 64, 3), dtype=np.uint8))


def test_validate_seg2_array_rejects_non_2d_1d() -> None:
    with pytest.raises(HairError):
        validate_seg2_array(np.zeros((64,), dtype=np.uint8))


def test_compute_hair_no_subject_abstains() -> None:
    seg = np.zeros((80, 80), dtype=np.uint8)
    m = compute_hair(seg, _rgb((200, 30, 40), height=80, width=80))
    assert m["subject_present"] is False
    assert m["abstained"] is True
    assert m["hair_present"] is False
    assert m["hair_dominant_color_name"] is None


def test_compute_hair_measures_present_hair_and_color() -> None:
    seg = np.zeros((120, 120), dtype=np.uint8)
    # Foreground subject (Torso) fills the frame; Hair sits in the top band.
    seg[30:110, 10:110] = 22  # Torso
    seg[0:30, 30:90] = HAIR
    img = _rgb((30, 30, 30), height=120, width=120)
    img[0:30, 30:90] = (215, 185, 130)  # blonde
    m = compute_hair(seg, img)
    assert m["subject_present"] is True
    assert m["hair_present"] is True
    assert m["hair_coverage"] > 0.05
    assert m["hair_dominant_color_name"] in {"blonde", "light blonde"}
    assert m["hair_dominant_hex"].startswith("#")
    # Hair is in the upper part of the frame.
    assert m["hair_position"] == "top"


def test_compute_hair_abstains_for_tiny_region_over_real_subject() -> None:
    seg = np.zeros((100, 100), dtype=np.uint8)
    seg[20:80, 20:80] = 22  # big torso subject
    seg[40:41, 40:41] = HAIR  # 1px hair splice -> below floor
    m = compute_hair(seg, _rgb((100, 100, 100)))
    assert m["subject_present"] is True
    assert m["hair_present"] is False
    assert m["hair_dominant_color_name"] is None
    assert m["hair_face_extent_ratio"] is None


def test_compute_hair_length_proxy_with_face() -> None:
    seg = np.zeros((120, 120), dtype=np.uint8)
    seg[40:100, 20:100] = 22  # Torso body
    # Hair spans much taller than the face (long-hair proxy).
    seg[0:60, 30:90] = HAIR
    # Face_Neck painted last so it remains a distinct face region.
    seg[30:50, 40:80] = FACE_NECK
    m = compute_hair(seg, _rgb((90, 60, 40), height=120, width=120))
    assert m["subject_present"] is True
    assert m["hair_present"] is True
    assert m["hair_position"] == "top"
    assert m["hair_face_extent_ratio"] is not None
    assert m["hair_face_extent_ratio"] > 1.0


def test_compute_hair_misaligned_image_rejected() -> None:
    seg = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(HairError):
        compute_hair(seg, np.zeros((100, 101, 3), dtype=np.uint8))


def test_compute_hair_middle_position_band() -> None:
    seg = np.zeros((100, 100), dtype=np.uint8)
    seg[30:80, 30:80] = 22
    seg[35:55, 40:60] = HAIR  # centroid around row 45 -> middle band
    m = compute_hair(seg, _rgb((40, 40, 40)))
    assert m["subject_present"] is True
    assert m["hair_present"] is True
    assert m["hair_position"] == "middle"
