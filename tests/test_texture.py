"""TDD coverage for deterministic texture/material measurements (arm #35).

These measurements are the deterministic evidence for arm #35 (texture). They
must be scale-invariant, caption-relevant texture facts (dominant fabric
class surface/pattern bands; dominant skin class surface band) computed from
the existing `seg2` + source pixels, honoring exactly-one-subject abstention —
absolute camera-frame values (gradient magnitudes, pixel counts) stay in the
machine-readable payload, never as caption claims.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.texture import (
    FABRIC_CLASSES,
    SKIN_CLASSES,
    TextureError,
    compute_texture,
    validate_seg2_array,
)

UPPER_CLOTHING = 23  # must match DOME-29
LOWER_CLOTHING = 13
APPAREL = 1
TORSO = 22
FACE_NECK = 3
LEFT_UPPER_LEG = 12
BACKGROUND = 0


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
    a[35:65, 25:75] = UPPER_CLOTHING
    return a


def test_validate_seg2_array_rejects_wrong_ndim() -> None:
    with pytest.raises(TextureError):
        validate_seg2_array(np.zeros((64, 64, 3), dtype=np.uint8))


def test_validate_seg2_array_rejects_float() -> None:
    with pytest.raises(TextureError):
        validate_seg2_array(np.zeros((64, 64), dtype=np.float32))


def test_compute_texture_rejects_misaligned_rgb() -> None:
    seg = _full_frame_subject()
    with pytest.raises(TextureError):
        compute_texture(seg, _rgb((200, 200, 200), height=99, width=99))


def test_compute_texture_abstains_when_no_subject() -> None:
    seg = np.zeros((100, 100), dtype=np.uint8)  # all Background, no subject
    m = compute_texture(seg, _rgb((120, 120, 120)))
    assert m["subject_present"] is False
    assert m["abstained"] is True
    assert m["texture_measurable"] is False
    assert m["abstention_reason"] == "no foreground subject detected"


def test_compute_texture_abstains_when_no_measurable_region() -> None:
    # Foreground exists but is tiny (below raw-pixel floor).
    seg = np.zeros((100, 100), dtype=np.uint8)
    seg[50:52, 50:52] = UPPER_CLOTHING
    m = compute_texture(seg, _rgb((120, 120, 120)))
    assert m["subject_present"] is True
    assert m["abstained"] is True
    assert m["texture_measurable"] is False
    assert "fabric or skin" in m["abstention_reason"]


def test_compute_texture_measures_smooth_fabric() -> None:
    seg = _full_frame_subject()
    m = compute_texture(seg, _rgb((90, 90, 90)))  # uniform fabric
    assert m["subject_present"] is True
    assert m["abstained"] is False
    assert m["texture_measurable"] is True
    assert m["fabric_class"] == "upper_clothing"
    assert m["fabric_texture_band"] == "smooth"
    assert m["fabric_pattern_band"] == "solid"
    assert m["fabric_edge_fraction"] < 0.08
    assert m["fabric_deviant_fraction"] < 0.15


def test_compute_texture_measures_textured_fabric() -> None:
    seg = _full_frame_subject()
    rgb = _rgb((90, 90, 90))
    # Paint 2px-wide dark stripes every 4 columns over the fabric region:
    # real edges (period 4 resolves under central differences) => textured,
    # and strong two-tone deviance => busy pattern.
    fab = seg == UPPER_CLOTHING
    y, x = np.nonzero(fab)
    stripe_cols = {c for c in range(x.min(), x.max() + 1) if (c - x.min()) % 4 < 2}
    sel = fab.copy()
    sel[y, x] = False
    for yy, xx in zip(y, x):
        if xx in stripe_cols:
            rgb[yy, xx] = (10, 10, 10)
    m = compute_texture(seg, rgb)
    assert m["texture_measurable"] is True
    assert m["fabric_texture_band"] != "smooth"
    assert m["fabric_pattern_band"] != "solid"


def test_compute_texture_skin_only_portrait() -> None:
    """Topless portrait: no garment classes => fabric abstains, skin measures."""
    a = np.zeros((100, 100), dtype=np.uint8)
    a[35:65, 25:75] = TORSO  # skin fills the subject region
    m = compute_texture(a, _rgb((200, 160, 140)))
    assert m["texture_measurable"] is True
    assert m["fabric_class"] is None
    assert m["fabric_texture_band"] is None
    assert m["fabric_pattern_band"] is None
    assert m["skin_class"] == "torso"
    assert m["skin_texture_band"] in {"smooth", "some texture", "textured"}
    assert m["measured_skin_px"] > 0


def test_compute_texture_prefers_dominant_region_class() -> None:
    """With both fabric and skin present, each reports its dominant class."""
    a = np.zeros((100, 100), dtype=np.uint8)
    a[20:50, 10:90] = UPPER_CLOTHING   # fabric, upper half
    a[50:90, 10:90] = TORSO            # skin, lower half (larger region)
    rgb = _rgb((120, 120, 120))
    # make the fabric busy so its bands are distinct from skin
    fab = a == UPPER_CLOTHING
    y, x = np.nonzero(fab)
    for yy, xx in zip(y[::2], x[::2]):
        rgb[yy, xx] = (20, 20, 20)
    m = compute_texture(a, rgb)
    assert m["texture_measurable"] is True
    assert m["fabric_class"] == "upper_clothing"
    assert m["skin_class"] == "torso"
    assert m["fabric_pattern_band"] != "solid"
    assert m["skin_texture_band"] in {"smooth", "some texture", "textured"}


def test_compute_texture_has_scale_invariant_names_only() -> None:
    """No absolute pixel / raw-gradient caption-facing fields on a positive result."""
    seg = _full_frame_subject()
    m = compute_texture(seg, _rgb((95, 135, 70)))
    assert m["texture_measurable"] is True
    public_keys = {
        "subject_present", "abstained", "abstention_reason", "texture_measurable",
        "fabric_class", "fabric_coverage", "fabric_edge_fraction",
        "fabric_deviant_fraction", "fabric_texture_band", "fabric_pattern_band",
        "skin_class", "skin_coverage", "skin_edge_fraction", "skin_mean_gradient",
        "skin_texture_band", "measured_fabric_px", "measured_skin_px", "measured_fg_px",
    }
    assert set(m) == public_keys


def test_class_pins_match_dome29() -> None:
    from stratum2.config import DOME_29

    assert DOME_29[0] == "Background"
    assert DOME_29[APPAREL] == "Apparel"
    assert DOME_29[LOWER_CLOTHING] == "Lower_Clothing"
    assert DOME_29[UPPER_CLOTHING] == "Upper_Clothing"
    assert DOME_29[TORSO] == "Torso"
    assert DOME_29[FACE_NECK] == "Face_Neck"
    assert set(FABRIC_CLASSES.values()) <= set(range(len(DOME_29)))
    assert set(SKIN_CLASSES.values()) <= set(range(len(DOME_29)))
