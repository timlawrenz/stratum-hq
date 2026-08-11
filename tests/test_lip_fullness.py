"""TDD coverage for the lip-fullness evidence specialist (arm #122).

Deterministic lip-fullness band (thin / medium / full) from the
already-qualified MediaPipe FaceLandmarker 478-point mesh (same model as
face-geometry #60 / gaze-head #68 / eyebrow-position #111 / nose-geometry
#121). Only the coarse band is verbalized; normalized ratios stay in the
machine-readable payload. The pure plan-banding (`_apply_bands`) and render
are tested without the model; the compute path abstains cleanly on a blank
non-face frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.lip_fullness import (
    FULL_MIN,
    MIN_MOUTH_PX,
    LipFullnessError,
    THIN_MAX,
    _apply_bands,
    compute_lip_fullness,
    render_lip_fullness,
    validate_rgb_array,
    validate_seg2_array,
)


def test_band_three_way_fullness() -> None:
    assert _apply_bands({"lip_fullness_ratio": THIN_MAX - 0.001})["lip_fullness_band"] == "thin"
    assert _apply_bands({"lip_fullness_ratio": FULL_MIN + 0.001})["lip_fullness_band"] == "full"
    mid = (THIN_MAX + FULL_MIN) / 2
    assert _apply_bands({"lip_fullness_ratio": mid})["lip_fullness_band"] == "medium"
    # Exact boundary values classify deterministically (< thin, > full).
    assert _apply_bands({"lip_fullness_ratio": THIN_MAX})["lip_fullness_band"] == "medium"
    assert _apply_bands({"lip_fullness_ratio": FULL_MIN})["lip_fullness_band"] == "medium"


def test_band_missing_ratio_is_payload_only() -> None:
    f = _apply_bands({"abstained": False})
    assert f["lip_fullness_band"] is None
    assert f["banding_unavailable"] is True
    assert render_lip_fullness(f) == ["lip-fullness: measured but not banded (payload)"]


def test_band_payload_fields_preserved() -> None:
    f = _apply_bands({
        "lip_fullness_ratio": 0.30,
        "upper_lip_ratio": 0.15,
        "lower_lip_ratio": 0.15,
        "mouth_width_px": 90.0,
    })
    assert f["lip_fullness_band"] == "thin"
    assert f["upper_lip_ratio"] == 0.15
    assert f["mouth_width_px"] == 90.0


def test_render_abstention() -> None:
    r = render_lip_fullness({"abstained": True, "abstention_reason": "no face detected"})
    assert len(r) == 1 and "abstain" in r[0] and "no face detected" in r[0]


def test_render_band_phrases() -> None:
    assert "full" in render_lip_fullness({"lip_fullness_band": "full"})[0]
    assert "thin" in render_lip_fullness({"lip_fullness_band": "thin"})[0]
    assert "medium" in render_lip_fullness({"lip_fullness_band": "medium"})[0]
    assert render_lip_fullness({}) == []


def test_validate_arrays() -> None:
    with pytest.raises(LipFullnessError):
        validate_rgb_array(np.zeros((5, 5), dtype=np.uint8))
    with pytest.raises(LipFullnessError):
        validate_rgb_array(np.zeros((5, 5, 3), dtype=np.float32))
    with pytest.raises(LipFullnessError):
        validate_seg2_array(np.zeros((5, 5, 1), dtype=np.uint8))


def test_misaligned_shapes_raise() -> None:
    seg2 = np.zeros((10, 20), dtype=np.uint8)
    rgb = np.zeros((11, 20, 3), dtype=np.uint8)
    with pytest.raises(LipFullnessError):
        compute_lip_fullness(seg2, rgb, model_asset_path="/nonexistent.task")


def test_compute_abstains_on_blank_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank non-face frame must abstain cleanly (no crash, no nonsense)."""
    monkeypatch.setattr("research_harness.lip_fullness._detect_mesh_on",
                        lambda arr, model_asset_path: None)
    seg2 = np.zeros((64, 64), dtype=np.uint8)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8) + 200
    res = compute_lip_fullness(seg2, rgb, model_asset_path="/nonexistent.task")
    assert res["abstained"] is True
    assert res["abstention_reason"]


def test_min_mouth_px_floor_positive() -> None:
    assert MIN_MOUTH_PX > 0
    assert 0.0 < THIN_MAX < FULL_MIN


def test_set_band_floors_recalibrates() -> None:
    """The probe recalibration hook must take effect (nose-geometry pattern)."""
    from research_harness import lip_fullness
    old_thin, old_full = THIN_MAX, FULL_MIN
    try:
        lip_fullness.set_band_floors(0.28, 0.45)
        assert _apply_bands({"lip_fullness_ratio": 0.30})["lip_fullness_band"] == "medium"
        assert _apply_bands({"lip_fullness_ratio": 0.47})["lip_fullness_band"] == "full"
    finally:
        lip_fullness.set_band_floors(old_thin, old_full)