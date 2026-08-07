"""TDD coverage for the gaze / head-orientation evidence specialist (arm #68).

Deterministic head-orientation bands from the validated MediaPipe
FaceLandmarker 478-point mesh (reused from arm #60) via the canonical six-point
PnP head-pose fit. Only scale-invariant direction bands are verbalized; raw
yaw/pitch/roll degrees and the pixel bbox stay in the machine-readable payload.
The banding/render/validation logic is pure and tested without the model; the
compute path abstains cleanly on a blank non-face frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.gaze_head import (
    GAZE_HEAD_MODEL_ASSET,
    GazeHeadError,
    PITCH_CENTER,
    PITCH_LEVEL_HALF,
    _apply_orientation_bands,
    _band_pitch,
    _band_roll,
    _band_yaw,
    _rotation_to_euler,
    compute_gaze_head,
    render_gaze_head,
    validate_rgb_array,
    validate_seg2_array,
)


def test_yaw_bands() -> None:
    assert _band_yaw(0.0) == "facing camera"
    assert _band_yaw(5.0) == "facing camera"
    assert _band_yaw(12.0) == "partially turned"   # boundary, not facing
    assert _band_yaw(20.0) == "partially turned"
    assert _band_yaw(35.0) == "profile or turned away"  # boundary, turned
    assert _band_yaw(60.0) == "profile or turned away"
    assert _band_yaw(-60.0) == "profile or turned away"  # sign-agnostic
    assert _band_yaw(None) is None


def test_pitch_bands() -> None:
    lo = PITCH_CENTER - PITCH_LEVEL_HALF
    hi = PITCH_CENTER + PITCH_LEVEL_HALF
    assert _band_pitch((lo + hi) / 2) == "level"
    assert _band_pitch(PITCH_CENTER) == "level"
    assert _band_pitch(hi + 0.05) == "tilted down"
    assert _band_pitch(hi + 10.0) == "tilted down"
    assert _band_pitch(lo - 0.05) == "tilted up"
    assert _band_pitch(lo - 10.0) == "tilted up"
    assert _band_pitch(None) is None


def test_roll_bands() -> None:
    assert _band_roll(0.0) == "level"
    assert _band_roll(5.0) == "level"
    assert _band_roll(12.0) == "tilted"
    assert _band_roll(30.0) == "tilted"
    assert _band_roll(-30.0) == "tilted"
    assert _band_roll(None) is None


def test_apply_orientation_bands_payload_preserved() -> None:
    f = _apply_orientation_bands({"yaw": 45.0, "pitch": 20.0, "roll": 30.0})
    assert f["yaw_band"] == "profile or turned away"
    assert f["pitch_band"] == "tilted down"
    assert f["roll_band"] == "tilted"
    # raw degrees survive for the payload
    assert f["yaw"] == 45.0 and f["pitch"] == 20.0 and f["roll"] == 30.0


def test_rotation_to_euler_identity() -> None:
    import numpy as np

    yaw, pitch, roll = _rotation_to_euler(np.eye(3))
    assert abs(yaw) < 1e-6
    assert abs(pitch) < 1e-6
    assert abs(roll) < 1e-6


def test_rotation_to_euler_yaw_rotation() -> None:
    """A pure rotation about the camera Y axis yields a yaw-only angle."""
    ang = np.radians(30.0)
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    yaw, pitch, roll = _rotation_to_euler(rot)
    assert abs(yaw) - 30.0 < 1.0
    assert abs(pitch) < 1.0
    assert abs(roll) < 1.0


def test_validate_arrays() -> None:
    with pytest.raises(GazeHeadError):
        validate_rgb_array(np.zeros((5, 5), dtype=np.uint8))
    with pytest.raises(GazeHeadError):
        validate_rgb_array(np.zeros((5, 5, 3), dtype=np.float32))
    with pytest.raises(GazeHeadError):
        validate_seg2_array(np.zeros((5, 5, 1), dtype=np.uint8))
    with pytest.raises(GazeHeadError):
        validate_seg2_array(np.zeros((5, 5), dtype=np.float32))


def test_misaligned_shapes_raise() -> None:
    with pytest.raises(GazeHeadError):
        compute_gaze_head(
            np.zeros((50, 50), dtype=np.uint8),
            np.zeros((100, 100, 3), dtype=np.uint8),
            model_asset_path="unused",
        )


def test_render_abstention() -> None:
    lines = render_gaze_head({"abstained": True, "abstention_reason": "no face"})
    assert lines and "abstain" in lines[0]
    lines2 = render_gaze_head({})
    assert len(lines2) >= 1


def test_render_bands() -> None:
    lines = render_gaze_head(_apply_orientation_bands(
        {"yaw": 45.0, "pitch": 0.0, "roll": 0.0}
    ))
    text = " ".join(lines)
    assert "profile" in text or "turned" in text
    lines2 = render_gaze_head(_apply_orientation_bands(
        {"yaw": 5.0, "pitch": 0.0, "roll": 0.0}
    ))
    assert "facing the camera" in " ".join(lines2)


def test_compute_abstains_on_blank_frame() -> None:
    """Blank non-face frame must abstain cleanly (no nonsense landmarks)."""
    from research_harness.face_geometry import _FaceLandmarkerRuntime

    model = GAZE_HEAD_MODEL_ASSET
    try:
        _FaceLandmarkerRuntime.reset()
        rgb = np.full((80, 80, 3), 200, dtype=np.uint8)
        seg = np.zeros((80, 80), dtype=np.uint8)  # no Face_Neck region
        r = compute_gaze_head(seg, rgb, model_asset_path=model)
        assert r["abstained"] is True
        assert "Face_Neck" in r["abstention_reason"]
    finally:
        _FaceLandmarkerRuntime.reset()
