"""TDD coverage for the face-geometry evidence specialist (arm #60).

Deterministic wrappers over a local open-weight MediaPipe FaceLandmarker
(new model class). Only scale-invariant ratios are verbalized; landmark
coordinates / pixel bbox stay in the machine-readable payload. The plan-banding
logic (`_apply_bands`, `_band`) and render are pure and tested without the
model; the compute path abstains cleanly on a blank non-face frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.face_geometry import (
    EYE_CLOSE,
    EYE_WIDE,
    JAW_NARROW,
    JAW_WIDE,
    MIDFACE_PLAUSIBLE,
    MIDFACE_SHORT,
    MIDFACE_TALL,
    MOUTH_NARROW,
    MOUTH_WIDE,
    FaceGeometryError,
    _apply_bands,
    _band,
    compute_face_geometry,
    render_face_geometry,
    validate_rgb_array,
    validate_seg2_array,
)


def test_band_three_way() -> None:
    assert _band(0.30, EYE_CLOSE, EYE_WIDE, "close-set", "wide-set", "average") == "close-set"
    assert _band(0.46, EYE_CLOSE, EYE_WIDE, "close-set", "wide-set", "average") == "average"
    assert _band(0.60, EYE_CLOSE, EYE_WIDE, "close-set", "wide-set", "average") == "wide-set"
    assert _band(None, EYE_CLOSE, EYE_WIDE, "close-set", "wide-set", "average") is None


def test_apply_bands_calibrated() -> None:
    f = _apply_bands({
        "eye_spacing_face_width": 0.430,   # close-set
        "mouth_face_width": 0.500,         # wide
        "jaw_face_width": 0.760,           # narrow
        "midface_share": 0.50,             # average
    })
    assert f["eye_spacing_band"] == "close-set"
    assert f["mouth_band"] == "wide"
    assert f["jaw_band"] == "narrow"
    assert f["midface_band"] == "average"
    assert f.get("midface_plausibility_abstained") is None


def test_apply_bands_midface_plausibility_gate() -> None:
    lo, hi = MIDFACE_PLAUSIBLE
    f = _apply_bands({"eye_spacing_face_width": 0.46, "midface_share": hi + 1.5})
    assert f["midface_band"] is None
    assert f["midface_plausibility_abstained"] is True
    # In-plausible values get a band.
    g = _apply_bands({"eye_spacing_face_width": 0.46, "midface_share": (lo + hi) / 2})
    assert g["midface_band"] == "average"


def test_apply_bands_threshold_values() -> None:
    # Exact boundary values classify deterministically (>= wide, < narrow/low).
    f = _apply_bands({"eye_spacing_face_width": EYE_WIDE, "mouth_face_width": MOUTH_WIDE,
                      "jaw_face_width": JAW_WIDE, "midface_share": MIDFACE_TALL})
    assert f["eye_spacing_band"] == "wide-set"
    assert f["mouth_band"] == "wide"
    assert f["jaw_band"] == "wide"
    assert f["midface_band"] == "tall"
    g = _apply_bands({"eye_spacing_face_width": EYE_CLOSE - 0.01,
                      "mouth_face_width": MOUTH_NARROW - 0.01,
                      "jaw_face_width": JAW_NARROW - 0.01,
                      "midface_share": MIDFACE_SHORT - 0.01})
    assert g["eye_spacing_band"] == "close-set"
    assert g["mouth_band"] == "narrow"
    assert g["jaw_band"] == "narrow"
    assert g["midface_band"] == "short"


def test_validate_arrays() -> None:
    with pytest.raises(FaceGeometryError):
        validate_rgb_array(np.zeros((5, 5), dtype=np.uint8))
    with pytest.raises(FaceGeometryError):
        validate_rgb_array(np.zeros((5, 5, 3), dtype=np.float32))
    with pytest.raises(FaceGeometryError):
        validate_seg2_array(np.zeros((5, 5, 1), dtype=np.uint8))
    with pytest.raises(FaceGeometryError):
        validate_seg2_array(np.zeros((5, 5), dtype=np.float32))


def test_misaligned_shapes_raise() -> None:
    with pytest.raises(FaceGeometryError):
        compute_face_geometry(
            np.zeros((50, 50), dtype=np.uint8),
            np.zeros((100, 100, 3), dtype=np.uint8),
            model_asset_path="unused",
        )


def test_render_abstention() -> None:
    lines = render_face_geometry({"abstained": True, "abstention_reason": "no face"})
    assert lines and "abstain" in lines[0]
    lines2 = render_face_geometry({})
    assert lines2 == [] or lines2[0]


def test_render_bands() -> None:
    lines = render_face_geometry(_apply_bands({
        "eye_spacing_face_width": 0.43,
        "mouth_face_width": 0.50,
        "jaw_face_width": 0.76,
        "midface_share": 0.50,
    }))
    text = " ".join(lines)
    assert "close" in text and "wide" in text and "narrow" in text


def test_compute_abstains_on_blank_frame(tmp_path) -> None:
    """Blank non-face frame must abstain cleanly (no nonsense landmarks)."""
    from research_harness.face_geometry import _FaceLandmarkerRuntime
    # Model asset is the frozen one; a blank frame is a safe non-sensitive probe.
    model = "/mnt/nas-ai-models/research/stratum/models/face-geometry/face_landmarker.task"
    try:
        _FaceLandmarkerRuntime.reset()
        rgb = np.full((80, 80, 3), 200, dtype=np.uint8)
        seg = np.zeros((80, 80), dtype=np.uint8)  # no Face_Neck region
        r = compute_face_geometry(seg, rgb, model_asset_path=model)
        assert r["abstained"] is True
        assert "Face_Neck" in r["abstention_reason"]
    finally:
        _FaceLandmarkerRuntime.reset()
