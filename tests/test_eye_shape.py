"""TDD coverage for the eye-shape evidence specialist (arm #123).

Deterministic eye-shape band (almond / medium / round) from the
already-qualified MediaPipe FaceLandmarker 478-point mesh (same model as
face-geometry #60 / gaze-head #68 / lip-fullness #122). Only the coarse band
is verbalized; normalized fissure-aspect ratios and per-eye payloads stay in
the machine-readable payload. The pure plan-banding (`_apply_bands`) and
render are tested without the model; the compute path abstains cleanly on a
blank non-face frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.eye_shape import (
    ALMOND_MAX,
    MIN_EYE_PX,
    ROUND_MIN,
    EyeShapeError,
    _apply_bands,
    compute_eye_shape,
    render_eye_shape,
    validate_rgb_array,
    validate_seg2_array,
)


def test_band_three_way_shape() -> None:
    assert _apply_bands({"eye_shape_ratio": ALMOND_MAX - 0.001})["eye_shape_band"] == "almond"
    assert _apply_bands({"eye_shape_ratio": ROUND_MIN + 0.001})["eye_shape_band"] == "round"
    mid = (ALMOND_MAX + ROUND_MIN) / 2
    assert _apply_bands({"eye_shape_ratio": mid})["eye_shape_band"] == "medium"
    # Exact boundary values classify deterministically (< almond, > round).
    assert _apply_bands({"eye_shape_ratio": ALMOND_MAX})["eye_shape_band"] == "medium"
    assert _apply_bands({"eye_shape_ratio": ROUND_MIN})["eye_shape_band"] == "medium"


def test_band_missing_ratio_is_payload_only() -> None:
    f = _apply_bands({"abstained": False})
    assert f["eye_shape_band"] is None
    assert f["banding_unavailable"] is True
    assert render_eye_shape(f) == ["eye-shape: measured but not banded (payload)"]


def test_band_payload_fields_preserved() -> None:
    f = _apply_bands({
        "eye_shape_ratio": 0.26,
        "eyes_measured": 2,
        "left_eye": {"aspect": 0.26, "canthal_tilt_deg": 2.5},
        "right_eye": {"aspect": 0.27, "canthal_tilt_deg": 3.1},
    })
    assert f["eye_shape_band"] == "almond"
    assert f["eyes_measured"] == 2
    assert f["left_eye"]["canthal_tilt_deg"] == 2.5


def test_measure_duplicates_matching_eye_geometry() -> None:
    """A synthetic eye pair with an elongated fissure must read almond."""
    pts = np.zeros((478, 2), dtype=np.float64)
    # Left eye: outer (33) at (100, 100), inner (133) at (140, 101) -> width 40
    # Fissure height 10 (upper lid 159 at y=95, lower lid 145 at y=105) -> aspect 0.25
    pts[33] = (100.0, 100.0)
    pts[133] = (140.0, 101.0)
    pts[159] = (120.0, 95.0)
    pts[145] = (120.0, 105.0)
    # Right eye: outer (362) at (200, 100), inner (263) at (240, 101)
    pts[362] = (200.0, 100.0)
    pts[263] = (240.0, 101.0)
    pts[386] = (220.0, 95.0)
    pts[374] = (220.0, 105.0)
    from research_harness.eye_shape import _measure
    m = _measure(pts)
    assert m["measureable"] is True
    assert m["eyes_measured"] == 2
    assert abs(m["eye_shape_ratio"] - 0.25) < 0.02
    assert "asymmetry_disclosed" not in m


def test_measure_single_eye_discloses_asymmetry() -> None:
    """One degenerate eye (zero width) must fall back to the single eye."""
    pts = np.zeros((478, 2), dtype=np.float64)
    pts[33] = (100.0, 100.0)
    pts[133] = (140.0, 101.0)
    pts[159] = (120.0, 95.0)
    pts[145] = (120.0, 105.0)
    # Right eye all zeros -> width 0 -> None
    from research_harness.eye_shape import _measure
    m = _measure(pts)
    assert m["measureable"] is True
    assert m["eyes_measured"] == 1
    assert m["asymmetry_disclosed"] is True


def test_measure_no_measurable_eye_abstains() -> None:
    """A face mesh with degenerate eyes must abstain with a surfaced reason."""
    pts = np.zeros((478, 2), dtype=np.float64)
    # Eyes too small (width 4 < MIN_EYE_PX=8)
    pts[33] = (100.0, 100.0)
    pts[133] = (104.0, 100.0)
    pts[159] = (102.0, 98.0)
    pts[145] = (102.0, 102.0)
    pts[362] = (110.0, 100.0)
    pts[263] = (114.0, 100.0)
    pts[386] = (112.0, 98.0)
    pts[374] = (112.0, 102.0)
    from research_harness.eye_shape import _measure
    m = _measure(pts)
    assert m["measureable"] is False
    assert "reason" in m


def test_measure_closed_eye_caught_by_plausibility_gate() -> None:
    """A collapsed fissure (aspect ~0, closed-eye signature) must abstain."""
    pts = np.zeros((478, 2), dtype=np.float64)
    pts[33] = (100.0, 100.0)
    pts[133] = (140.0, 101.0)
    pts[159] = (120.0, 100.4)  # height 0.8 / width ~40 -> aspect 0.02 < 0.05
    pts[145] = (120.0, 101.2)
    pts[362] = (200.0, 100.0)
    pts[263] = (240.0, 101.0)
    pts[386] = (220.0, 100.4)
    pts[374] = (220.0, 101.2)
    from research_harness.eye_shape import _measure
    m = _measure(pts)
    assert m["measureable"] is False


def test_render_abstention() -> None:
    r = render_eye_shape({"abstained": True, "abstention_reason": "no face detected"})
    assert len(r) == 1 and "abstain" in r[0] and "no face detected" in r[0]


def test_render_band_phrases() -> None:
    assert "almond" in render_eye_shape({"eye_shape_band": "almond"})[0]
    assert "round" in render_eye_shape({"eye_shape_band": "round"})[0]
    assert "medium" in render_eye_shape({"eye_shape_band": "medium"})[0]
    assert render_eye_shape({}) == []


def test_validate_arrays() -> None:
    with pytest.raises(EyeShapeError):
        validate_rgb_array(np.zeros((5, 5), dtype=np.uint8))
    with pytest.raises(EyeShapeError):
        validate_rgb_array(np.zeros((5, 5, 3), dtype=np.float32))
    with pytest.raises(EyeShapeError):
        validate_seg2_array(np.zeros((5, 5, 1), dtype=np.uint8))


def test_misaligned_shapes_raise() -> None:
    seg2 = np.zeros((10, 20), dtype=np.uint8)
    rgb = np.zeros((11, 20, 3), dtype=np.uint8)
    with pytest.raises(EyeShapeError):
        compute_eye_shape(seg2, rgb, model_asset_path="/nonexistent.task")


def test_compute_abstains_on_blank_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank non-face frame must abstain cleanly (no crash, no nonsense)."""
    monkeypatch.setattr("research_harness.eye_shape._detect_mesh_on",
                        lambda arr, model_asset_path: None)
    seg2 = np.zeros((64, 64), dtype=np.uint8)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8) + 200
    res = compute_eye_shape(seg2, rgb, model_asset_path="/nonexistent.task")
    assert res["abstained"] is True
    assert res["abstention_reason"]


def test_min_eye_px_floor_and_floor_ordering() -> None:
    assert MIN_EYE_PX > 0
    assert 0.0 < ALMOND_MAX < ROUND_MIN
    assert ALMOND_MAX > 0.05  # above the plausibility floor so bands can fire


def test_set_band_floors_recalibrates() -> None:
    """The probe recalibration hook must take effect (nose-geometry pattern)."""
    from research_harness import eye_shape
    old_almond, old_round = ALMOND_MAX, ROUND_MIN
    try:
        eye_shape.set_band_floors(0.28, 0.45)
        assert _apply_bands({"eye_shape_ratio": 0.30})["eye_shape_band"] == "medium"
        assert _apply_bands({"eye_shape_ratio": 0.47})["eye_shape_band"] == "round"
    finally:
        eye_shape.set_band_floors(old_almond, old_round)