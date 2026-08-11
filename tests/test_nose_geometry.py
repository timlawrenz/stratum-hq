"""TDD coverage for the nose-geometry evidence specialist (arm #121).

Deterministic nose-width / nose-length bands from the already-qualified
MediaPipe FaceLandmarker 478-point mesh (same model as face-geometry #60 /
gaze-head #68 / eyebrow-position #111). Only the coarse calibrated bands
(width narrow/average/wide; length short/average/long) are verbalized;
normalized ratios stay in the machine-readable payload. The pure plan-banding
(`_apply_bands`) and render are tested without the model; the compute path
abstains cleanly on a blank non-face frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.nose_geometry import (
    LENGTH_LONG_MIN,
    LENGTH_SHORT_MAX,
    MIN_IPD_PX,
    NoseGeometryError,
    WIDTH_NARROW_MAX,
    WIDTH_WIDE_MIN,
    _apply_bands,
    compute_nose_geometry,
    render_nose_geometry,
    validate_rgb_array,
    validate_seg2_array,
)


def test_band_three_way_width() -> None:
    assert _apply_bands({"nose_width_ratio": WIDTH_NARROW_MAX - 0.001})["nose_width_band"] == "narrow"
    assert _apply_bands({"nose_width_ratio": WIDTH_WIDE_MIN + 0.001})["nose_width_band"] == "wide"
    mid = (WIDTH_NARROW_MAX + WIDTH_WIDE_MIN) / 2
    assert _apply_bands({"nose_width_ratio": mid})["nose_width_band"] == "average"
    # Exact boundary values classify deterministically (< narrow, > wide).
    assert _apply_bands({"nose_width_ratio": WIDTH_NARROW_MAX})["nose_width_band"] == "average"
    assert _apply_bands({"nose_width_ratio": WIDTH_WIDE_MIN})["nose_width_band"] == "average"


def test_band_three_way_length() -> None:
    assert _apply_bands({"nose_length_ratio": LENGTH_SHORT_MAX - 0.001})["nose_length_band"] == "short"
    assert _apply_bands({"nose_length_ratio": LENGTH_LONG_MIN + 0.001})["nose_length_band"] == "long"
    mid = (LENGTH_SHORT_MAX + LENGTH_LONG_MIN) / 2
    assert _apply_bands({"nose_length_ratio": mid})["nose_length_band"] == "average"


def test_band_missing_ratio_is_payload_only() -> None:
    f = _apply_bands({"abstained": False})
    assert f["nose_width_band"] is None
    assert f["banding_unavailable"] is True
    assert render_nose_geometry(f) == ["nose-geometry: measured but not banded (payload)"]


def test_band_payload_fields_preserved() -> None:
    f = _apply_bands({
        "nose_width_ratio": 0.9,
        "nose_length_ratio": 1.5,
        "ipd_px": 120.0,
    })
    assert f["nose_width_band"] == "narrow"
    assert f["nose_length_band"] == "average"
    assert f["ipd_px"] == 120.0


def test_render_abstention() -> None:
    r = render_nose_geometry({"abstained": True, "abstention_reason": "no face detected"})
    assert len(r) == 1 and "abstain" in r[0] and "no face detected" in r[0]


def test_render_band_phrases() -> None:
    assert "narrow" in render_nose_geometry({"nose_width_band": "narrow"})[0]
    assert "wide" in render_nose_geometry({"nose_width_band": "wide"})[0]
    assert "long" in render_nose_geometry({"nose_length_band": "long"})[0]
    assert render_nose_geometry({}) == []


def test_validate_arrays() -> None:
    with pytest.raises(NoseGeometryError):
        validate_rgb_array(np.zeros((5, 5), dtype=np.uint8))
    with pytest.raises(NoseGeometryError):
        validate_rgb_array(np.zeros((5, 5, 3), dtype=np.float32))
    with pytest.raises(NoseGeometryError):
        validate_seg2_array(np.zeros((5, 5, 1), dtype=np.uint8))


def test_misaligned_shapes_raise() -> None:
    seg2 = np.zeros((10, 20), dtype=np.uint8)
    rgb = np.zeros((11, 20, 3), dtype=np.uint8)
    with pytest.raises(NoseGeometryError):
        compute_nose_geometry(seg2, rgb, model_asset_path="/nonexistent.task")


def test_compute_abstains_on_blank_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank non-face frame must abstain cleanly (no crash, no nonsense)."""
    monkeypatch.setattr("research_harness.nose_geometry._detect_mesh_on",
                        lambda arr, model_asset_path: None)
    seg2 = np.zeros((64, 64), dtype=np.uint8)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8) + 200
    res = compute_nose_geometry(seg2, rgb, model_asset_path="/nonexistent.task")
    assert res["abstained"] is True
    assert res["abstention_reason"]


def test_min_ipd_floor_positive() -> None:
    assert MIN_IPD_PX > 0
