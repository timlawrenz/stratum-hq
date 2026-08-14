"""TDD coverage for deterministic lighting measurements (arm #33).

These measurements are the deterministic evidence for arm #33 (lighting). They
must be scale-invariant, caption-relevant lighting facts (luma bands, dynamic
range band, shadow fraction, direction name) computed from existing `normal2` +
`seg2` + source pixels, honoring exactly-one-subject abstention — absolute
camera-frame values stay in the machine-readable payload, never as caption
claims.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.lighting import (
    LightingError,
    compute_lighting,
    validate_normal2_array,
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


def _front_facing_normal() -> np.ndarray:
    """Unit normal pointing at the camera across the whole frame."""
    n = np.zeros((100, 100, 3), dtype=np.float16)
    n[..., 2] = 1.0
    return n


def _subject(seg2: np.ndarray, *, nz: float = 0.9, laterality: float = 0.0) -> np.ndarray:
    """Build a normal map with a shading gradient over the foreground to make the
    light-direction fit meaningful, plus background normals away from camera."""
    n = np.zeros((seg2.shape[0], seg2.shape[1], 3), dtype=np.float16)
    fg = seg2 > 0
    n[fg, 0] = np.float16(laterality)
    n[fg, 2] = np.float16(nz)
    # normalize rows to unit length where needed
    norms = np.sqrt(n[..., 0] ** 2 + n[..., 1] ** 2 + n[..., 2] ** 2)
    norms3 = np.where(norms > 0, norms, 1.0)[..., None]
    nzr = np.zeros_like(n)
    valid = norms > 0
    nzr[valid] = n[valid] / norms3[valid]
    return nzr.astype(np.float16)


def test_validate_normal2_array_accepts_3d_three_channel() -> None:
    a = np.zeros((64, 64, 3), dtype=np.float16)
    validate_normal2_array(a)  # must not raise


def test_validate_normal2_array_rejects_2d() -> None:
    with pytest.raises(LightingError):
        validate_normal2_array(np.zeros((64, 64), dtype=np.float16))


def test_validate_normal2_array_rejects_wrong_channels() -> None:
    with pytest.raises(LightingError):
        validate_normal2_array(np.zeros((64, 64, 2), dtype=np.float16))


def test_validate_seg2_array_rejects_wrong_ndim() -> None:
    with pytest.raises(LightingError):
        validate_seg2_array(np.zeros((64, 64, 3), dtype=np.uint8))


def test_compute_lighting_no_subject_abstains() -> None:
    seg = np.zeros((80, 80), dtype=np.uint8)
    n = np.zeros((80, 80, 3), dtype=np.float16)
    n[..., 2] = 1.0
    m = compute_lighting(n, seg, _rgb((50, 50, 50), height=80, width=80))
    assert m["subject_present"] is False
    assert m["abstained"] is True
    assert m["lighting_measurable"] is False
    assert m["luma_band"] is None


def test_compute_lighting_measures_bright_front_lit() -> None:
    seg = np.zeros((120, 120), dtype=np.uint8)
    seg[20:100, 20:100] = 22  # Torso (skin) subject
    img = _rgb((220, 220, 220), height=120, width=120)  # bright
    normal = _subject(seg, nz=0.95)
    m = compute_lighting(seg2=seg, normal2=normal, image_rgb=img)
    assert m["subject_present"] is True
    assert m["abstained"] is False
    assert m["lighting_measurable"] is True
    assert m["luma_band"] == "brightly lit"
    assert m["dynamic_range"] is not None
    assert m["shadow_fraction"] == 0.0
    assert m["light_direction"] is not None


def test_compute_lighting_abstains_for_tiny_subject() -> None:
    seg = np.zeros((100, 100), dtype=np.uint8)
    seg[49:51, 49:51] = 22  # 4px subject -> below floor
    m = compute_lighting(_front_facing_normal(), seg, _rgb((120, 120, 120)))
    assert m["subject_present"] is True
    assert m["abstained"] is True
    assert m["lighting_measurable"] is False


def test_compute_lighting_direction_side_key() -> None:
    seg = np.zeros((120, 120), dtype=np.uint8)
    seg[20:100, 20:100] = 22
    img = _rgb((150, 150, 150), height=120, width=120)
    # Laterality 0.6 horizontal component -> a strong left/right cue.
    normal = _subject(seg, nz=0.8, laterality=0.6)
    m = compute_lighting(seg2=seg, normal2=normal, image_rgb=img)
    assert m["lighting_measurable"] is True
    assert "left" in m["light_direction"] or "right" in m["light_direction"]
    assert m["light_vector"] is not None


def test_compute_lighting_abstains_direction_on_degenerate_normals() -> None:
    seg = np.zeros((100, 100), dtype=np.uint8)
    seg[20:80, 20:80] = 22
    img = _rgb((120, 120, 120))
    normal = np.zeros((100, 100, 3), dtype=np.float16)  # no valid unit normals
    m = compute_lighting(seg2=seg, normal2=normal, image_rgb=img)
    assert m["subject_present"] is True
    # Luma/dynamic-range still measurable; direction abstains explicitly.
    assert m["lighting_measurable"] is True
    assert m["light_direction"] == "undetermined"
    assert m["abstention_reason"] is not None


def test_compute_lighting_misaligned_image_rejected() -> None:
    seg = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(LightingError):
        compute_lighting(_front_facing_normal(), seg, np.zeros((100, 101, 3), dtype=np.uint8))


def test_compute_lighting_misaligned_normal_rejected() -> None:
    seg = np.zeros((100, 100), dtype=np.uint8)
    img = _rgb((100, 100, 100))
    with pytest.raises(LightingError):
        compute_lighting(np.zeros((100, 99, 3), dtype=np.float16), seg, img)


def test_compute_lighting_scale_invariance_verbalized_bands() -> None:
    """Doubling the absolute luma level inside the same band must not change the
    verbalized caption fact (only the machine-readable payload value)."""
    seg = np.zeros((120, 120), dtype=np.uint8)
    seg[20:100, 20:100] = 22
    normal = _subject(seg, nz=0.9)
    img_a = _rgb((200, 200, 200), height=120, width=120)
    img_b = _rgb((215, 215, 215), height=120, width=120)
    ma = compute_lighting(seg2=seg, normal2=normal, image_rgb=img_a)
    mb = compute_lighting(seg2=seg, normal2=normal, image_rgb=img_b)
    # Both fall in the bright band; band+band decisions are invariant to small
    # absolute shifts. (Pixel values are NOT caption claims.)
    assert ma["luma_band"] == mb["luma_band"] == "brightly lit"
    assert ma["light_direction"] == mb["light_direction"]
    assert ma["light_direction"] == "from the front"


def test_lighting_class_constants_pinned_to_dome29() -> None:
    from stratum2.config import DOME_29

    assert DOME_29[22] == "Torso"
    assert DOME_29[4] == "Hair"
    assert DOME_29[3] == "Face_Neck"
