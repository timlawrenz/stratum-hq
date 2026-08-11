"""TDD coverage for deterministic DOME-29 clothing/apparel measurements (arm #29).

These measurements are the deterministic evidence for arm #29 (clothing).
They must be continuous coverage fractions + deterministic dominant colors
(never closed taxonomies or fabricated attributes), computed from existing
`seg2` masks + source pixels, honoring exactly-one-subject abstention.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.clothing import (
    APPAREL,
    LOWER_CLOTHING,
    UPPER_CLOTHING,
    ClothingError,
    compute_clothing,
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
    with pytest.raises(ClothingError):
        validate_seg2_array(np.zeros((64, 64, 3), dtype=np.uint8))


def test_compute_clothing_no_subject_abstains() -> None:
    seg = np.zeros((80, 80), dtype=np.uint8)
    m = compute_clothing(seg, _rgb((200, 30, 40), height=80, width=80))
    assert m["subject_present"] is False
    assert m["abstained"] is True
    assert m["garments"] == []


def test_compute_clothing_measures_present_garment_and_color() -> None:
    seg = np.zeros((100, 100), dtype=np.uint8)
    # A large red upper-clothing region over a background of black.
    seg[10:90, 10:90] = UPPER_CLOTHING
    img = _rgb((30, 30, 30))
    img[10:90, 10:90] = (200, 30, 40)
    m = compute_clothing(seg, img)
    assert m["subject_present"] is True
    names = {g["class"] for g in m["garments"]}
    assert "upper_clothing" in names
    upper = m["classes"][UPPER_CLOTHING]
    assert upper["present"] is True
    assert upper["coverage"] > 0.5
    assert upper["dominant_color_name"] == "red"
    assert upper["dominant_hex"].startswith("#")


def test_compute_clothing_abstains_for_tiny_class_over_real_subject() -> None:
    seg = np.zeros((100, 100), dtype=np.uint8)
    seg[20:80, 20:80] = APPAREL  # big apparel = real subject foreground
    # tiny lower-clothing splice
    seg[40:41, 40:41] = LOWER_CLOTHING
    m = compute_clothing(seg, _rgb((100, 100, 100)))
    assert m["subject_present"] is True
    apparel = m["classes"][APPAREL]
    assert apparel["present"] is True
    low = m["classes"][LOWER_CLOTHING]
    assert low["present"] is False  # below floor -> abstain, not a claim
    assert low["dominant_color_name"] is None


def test_compute_clothing_rejects_misaligned_pixels() -> None:
    seg = np.zeros((80, 80), dtype=np.uint8)
    with pytest.raises(ClothingError):
        compute_clothing(seg, _rgb((10, 10, 10), height=64, width=64))


def test_clothing_serialization_verbalizes_garments_only_no_px(tmp_path: object = None) -> None:
    from research_harness.stage_b import _serialize_clothing

    seg = np.zeros((100, 100), dtype=np.uint8)
    seg[10:90, 10:90] = UPPER_CLOTHING
    img = _rgb((30, 30, 30))
    img[10:90, 10:90] = (200, 30, 40)
    m = compute_clothing(seg, img)
    text = _serialize_clothing(m)
    assert "upper clothing present" in text
    assert "dominant color red" in text
    # raw px / hex must NOT be verbalized into caption prompts
    assert "frame_coverage" not in text
    assert "#" not in text
    assert "px" not in text.lower()

    # abstention path
    m2 = compute_clothing(np.zeros((100, 100), dtype=np.uint8), _rgb((30, 30, 30)))
    t2 = _serialize_clothing(m2)
    assert "abstain from clothing claims" in t2
