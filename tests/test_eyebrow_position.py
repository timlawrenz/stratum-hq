"""TDD coverage for the eyebrow-position evidence specialist (arm #111).

Deterministic brow-arch-elevation band from the already-qualified MediaPipe
FaceLandmarker 478-point mesh (same model as face-geometry #60 / gaze-head
#68). Only the coarse calibrated band (neutral / raised / furrowed-low) is
verbalized; per-side ratios stay in the machine-readable payload. The pure
plan-banding (`_apply_bands`) and render are tested without the model; the
compute path abstains cleanly on a blank non-face frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.eyebrow_position import (
    FURROWED_HIGH,
    MIN_EYE_PX,
    RAISED_LOW,
    EyebrowPositionError,
    _apply_bands,
    compute_eyebrow_position,
    render_eyebrow_position,
    validate_rgb_array,
    validate_seg2_array,
)


def test_band_three_way() -> None:
    assert _apply_bands({"arch_elevation": RAISED_LOW + 0.01})["eyebrow_position_band"] == "raised"
    assert _apply_bands({"arch_elevation": FURROWED_HIGH - 0.01})["eyebrow_position_band"] == "furrowed"
    assert _apply_bands({"arch_elevation": (FURROWED_HIGH + RAISED_LOW) / 2})["eyebrow_position_band"] == "neutral"
    # Exact boundary values classify deterministically (>= raised, < furrowed).
    assert _apply_bands({"arch_elevation": RAISED_LOW})["eyebrow_position_band"] == "raised"
    assert _apply_bands({"arch_elevation": FURROWED_HIGH})["eyebrow_position_band"] == "neutral"


def test_band_missing_arch_is_payload_only() -> None:
    f = _apply_bands({"abstained": False})
    assert f["eyebrow_position_band"] is None
    assert f["banding_unavailable"] is True
    # banding_unavailable never claims a brow state in prose.
    assert render_eyebrow_position(f) == [
        "eyebrow-position: measured but not banded (payload)"
    ]


def test_band_payload_fields_preserved() -> None:
    f = _apply_bands({
        "arch_elevation": 1.0,
        "inner_brow_elevation": -0.02,
        "per_side": [{"side": "l"}, {"side": "r"}],
    })
    assert f["eyebrow_position_band"] == "raised"
    assert f["inner_brow_elevation"] == -0.02


def test_render_abstention() -> None:
    r = render_eyebrow_position({"abstained": True, "abstention_reason": "no face detected"})
    assert len(r) == 1 and "abstain" in r[0] and "no face detected" in r[0]


def test_render_band_phrases() -> None:
    assert "raised" in render_eyebrow_position({"eyebrow_position_band": "raised"})[0]
    assert "furrowed" in render_eyebrow_position({"eyebrow_position_band": "furrowed"})[0]
    assert "neutral" in render_eyebrow_position({"eyebrow_position_band": "neutral"})[0]
    assert render_eyebrow_position({}) == []


def test_validate_arrays() -> None:
    with pytest.raises(EyebrowPositionError):
        validate_rgb_array(np.zeros((5, 5), dtype=np.uint8))
    with pytest.raises(EyebrowPositionError):
        validate_rgb_array(np.zeros((5, 5, 3), dtype=np.float32))
    with pytest.raises(EyebrowPositionError):
        validate_seg2_array(np.zeros((5, 5, 1), dtype=np.uint8))


def test_misaligned_shapes_raise() -> None:
    seg2 = np.zeros((10, 20), dtype=np.uint8)
    rgb = np.zeros((11, 20, 3), dtype=np.uint8)
    with pytest.raises(EyebrowPositionError):
        compute_eyebrow_position(seg2, rgb, model_asset_path="/nonexistent.task")


def test_compute_abstains_on_blank_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank non-face frame must abstain cleanly (no crash, no nonsense)."""
    monkeypatch.setattr("research_harness.eyebrow_position._detect_mesh_on",
                        lambda arr, model_asset_path: None)
    seg2 = np.zeros((64, 64), dtype=np.uint8)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8) + 200
    res = compute_eyebrow_position(seg2, rgb, model_asset_path="/nonexistent.task")
    assert res["abstained"] is True
    assert res["abstention_reason"]


def test_min_eye_px_floor_positive() -> None:
    assert MIN_EYE_PX > 0
