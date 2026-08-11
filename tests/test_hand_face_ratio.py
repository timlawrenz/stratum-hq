"""TDD coverage for the hand-face-ratio evidence specialist (arm #124).

Relational NEW evidence part fusing the already-qualified MediaPipe 21-point
HandLandmarker (arm #109) + 478-point FaceLandmarker (arm #60) into a
scale-invariant hand-size-vs-face band (small / typical / large). Only the
coarse band is verbalized; ratios / per-hand payloads stay machine-readable.
The pure plan-banding (`_apply_bands`) and render are tested without the
models; the compute path abstains cleanly on a blank non-face frame and on a
face-only frame (no hand).
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.hand_face_ratio import (
    HAND_FACE_PLAUSIBLE,
    LARGE_MIN,
    SMALL_MAX,
    HandFaceRatioError,
    _apply_bands,
    _measure,
    compute_hand_face_ratio,
    render_hand_face_ratio,
    validate_rgb_array,
    validate_seg2_array,
)

FRAME = 64


class _Pt:
    """Minimal MediaPipe landmark stub (x, y)."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def _hand_mesh(points: np.ndarray) -> list[_Pt]:
    """Map (21, 2) pixel-coordinate points to normalized landmarks on FRAME."""
    return [_Pt(float(px) / FRAME, float(py) / FRAME) for px, py in points]


def test_band_three_way_size() -> None:
    assert _apply_bands({"hand_face_ratio": SMALL_MAX - 0.001})["hand_size_band"] == "small"
    assert _apply_bands({"hand_face_ratio": LARGE_MIN + 0.001})["hand_size_band"] == "large"
    mid = (SMALL_MAX + LARGE_MIN) / 2
    assert _apply_bands({"hand_face_ratio": mid})["hand_size_band"] == "typical"
    # Exact boundary values classify deterministically.
    assert _apply_bands({"hand_face_ratio": SMALL_MAX})["hand_size_band"] == "typical"
    assert _apply_bands({"hand_face_ratio": LARGE_MIN})["hand_size_band"] == "typical"


def test_band_missing_ratio_is_payload_only() -> None:
    f = _apply_bands({"abstained": False})
    assert f["hand_size_band"] is None
    assert f["banding_unavailable"] is True
    assert render_hand_face_ratio(f) == ["hand-face-ratio: measured but not banded (payload)"]


def test_band_payload_fields_preserved() -> None:
    f = _apply_bands({
        "hand_face_ratio": 0.55,
        "hands_measured": 1,
        "face_detection_via": "full_frame",
        "hand_detection_via": "full_frame",
        "per_hand": [{"index": 0, "measureable": True, "hand_face_ratio": 0.55}],
    })
    assert f["hand_size_band"] == "small"
    assert f["hands_measured"] == 1
    assert f["per_hand"][0]["hand_face_ratio"] == 0.55


def test_measure_typical_hand() -> None:
    """A synthetic hand with palm length / face width ~0.8 -> typical."""
    hand = np.zeros((21, 2), dtype=np.float64)
    hand[0] = (0.0, 0.0)       # wrist
    hand[9] = (0.8, 0.0)       # middle MCP -> palm length 0.8
    hand[5] = (0.2, 0.5)       # index MCP
    hand[17] = (0.2, -0.5)     # pinky MCP
    m = _measure(hand, face_width_px=1.0)
    assert m["measureable"] is True
    assert abs(m["hand_face_ratio"] - 0.8) < 1e-6
    assert abs(m["palm_length_px"] - 0.8) < 1e-6
    assert abs(m["palm_width_face_width_payload"] - 1.0) < 1e-6


def test_measure_small_hand() -> None:
    hand = np.zeros((21, 2), dtype=np.float64)
    hand[0] = (0.0, 0.0)
    hand[9] = (0.5, 0.0)              # palm length 0.5 vs face width 1.0 -> small
    m = _measure(hand, face_width_px=1.0)
    assert m["measureable"] is True
    assert _apply_bands({"hand_face_ratio": m["hand_face_ratio"]})["hand_size_band"] == "small"


def test_measure_large_hand() -> None:
    hand = np.zeros((21, 2), dtype=np.float64)
    hand[0] = (0.0, 0.0)
    hand[9] = (1.2, 0.0)              # palm length 1.2 vs face width 1.0 -> large
    m = _measure(hand, face_width_px=1.0)
    assert m["measureable"] is True
    assert _apply_bands({"hand_face_ratio": m["hand_face_ratio"]})["hand_size_band"] == "large"


def test_measure_outside_plausible_depth_band_abstains() -> None:
    """A hand held right at the lens (ratio >> 1.3) must abstain (confounded)."""
    hand = np.zeros((21, 2), dtype=np.float64)
    hand[0] = (0.0, 0.0)
    hand[9] = (5.0, 0.0)              # palm length 5x face width -> implausible
    m = _measure(hand, face_width_px=1.0)
    assert m["measureable"] is False
    assert "outside the human-plausible band" in m["reason"]


def test_measure_degenerate_hand_abstains() -> None:
    hand = np.zeros((21, 2), dtype=np.float64)   # zero extent
    m = _measure(hand, face_width_px=1.0)
    assert m is None


def test_plausible_band_sane() -> None:
    assert HAND_FACE_PLAUSIBLE[0] < SMALL_MAX < LARGE_MIN < HAND_FACE_PLAUSIBLE[1]


def test_render_abstention() -> None:
    r = render_hand_face_ratio({"abstained": True, "abstention_reason": "no face detected"})
    assert len(r) == 1 and "abstain" in r[0] and "no face detected" in r[0]


def test_render_band_phrases() -> None:
    assert "small" in render_hand_face_ratio({"hand_size_band": "small"})[0]
    assert "large" in render_hand_face_ratio({"hand_size_band": "large"})[0]
    assert "typical" in render_hand_face_ratio({"hand_size_band": "typical"})[0]
    assert render_hand_face_ratio({}) == []


def test_validate_arrays() -> None:
    with pytest.raises(HandFaceRatioError):
        validate_rgb_array(np.zeros((5, 5), dtype=np.uint8))
    with pytest.raises(HandFaceRatioError):
        validate_rgb_array(np.zeros((5, 5, 3), dtype=np.float32))
    with pytest.raises(HandFaceRatioError):
        validate_seg2_array(np.zeros((5, 5, 1), dtype=np.uint8))


def test_misaligned_shapes_raise() -> None:
    seg2 = np.zeros((10, 20), dtype=np.uint8)
    rgb = np.zeros((11, 20, 3), dtype=np.uint8)
    with pytest.raises(HandFaceRatioError):
        compute_hand_face_ratio(
            rgb, seg2,
            face_model_asset_path="/nonexistent.task",
            hand_model_asset_path="/nonexistent.task",
        )


def test_compute_abstains_on_blank_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank non-face frame must abstain cleanly (no crash, no nonsense)."""
    monkeypatch.setattr(
        "research_harness.hand_face_ratio._face_on_full_frame",
        lambda rgb, model_asset_path: None,
    )
    monkeypatch.setattr(
        "research_harness.hand_face_ratio._face_on_seg_crop",
        lambda seg2, rgb, model_asset_path: None,
    )
    seg2 = np.zeros((64, 64), dtype=np.uint8)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8) + 200
    res = compute_hand_face_ratio(
        rgb, seg2,
        face_model_asset_path="/nonexistent.task",
        hand_model_asset_path="/nonexistent.task",
    )
    assert res["abstained"] is True
    assert res["abstention_reason"]
    assert "no face detected" in res["abstention_reason"]


def test_compute_abstains_when_no_hand(monkeypatch: pytest.MonkeyPatch) -> None:
    """A face present but no hand anywhere -> honest abstention."""
    monkeypatch.setattr(
        "research_harness.hand_face_ratio._face_on_full_frame",
        lambda rgb, model_asset_path: {
            "pts": np.zeros((478, 2), dtype=np.float64),
            "face_width_px": 100.0,
            "via": "full_frame",
        },
    )
    monkeypatch.setattr(
        "research_harness.hand_face_ratio._face_on_seg_crop",
        lambda seg2, rgb, model_asset_path: None,
    )
    monkeypatch.setattr(
        "research_harness.hand_face_ratio._detect_hands",
        lambda arr, model_asset_path: [],
    )
    seg2 = np.zeros((64, 64), dtype=np.uint8)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8) + 200
    res = compute_hand_face_ratio(
        rgb, seg2,
        face_model_asset_path="/nonexistent.task",
        hand_model_asset_path="/nonexistent.task",
    )
    assert res["abstained"] is True
    assert "no hand detected" in res["abstention_reason"]


def test_compute_banded_item(monkeypatch: pytest.MonkeyPatch) -> None:
    """A face + one measurable hand in the plausible band -> a banded item."""
    # Hand mesh in full-frame px (FRAME=64): wrist (0,0), middle MCP (32,16)
    # -> palm 32px; index/pinky spread for a sane bbox. Face width 64px ->
    # ratio 32/64 = 0.5 -> small.
    hand = np.zeros((21, 2), dtype=np.float64)
    hand[0] = (0.0, 0.0)
    hand[9] = (32.0, 16.0)
    hand[5] = (8.0, 28.0)
    hand[17] = (8.0, 4.0)

    monkeypatch.setattr(
        "research_harness.hand_face_ratio._face_on_full_frame",
        lambda rgb, model_asset_path: {
            "pts": np.zeros((478, 2), dtype=np.float64),
            "face_width_px": 64.0,          # face width 64px -> ratio 32/64 = 0.5 -> small
            "via": "full_frame",
        },
    )
    monkeypatch.setattr(
        "research_harness.hand_face_ratio._face_on_seg_crop",
        lambda seg2, rgb, model_asset_path: None,
    )
    monkeypatch.setattr(
        "research_harness.hand_face_ratio._detect_hands",
        lambda arr, model_asset_path: [_hand_mesh(hand)],
    )
    seg2 = np.zeros((64, 64), dtype=np.uint8)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8) + 200
    res = compute_hand_face_ratio(
        rgb, seg2,
        face_model_asset_path="/nonexistent.task",
        hand_model_asset_path="/nonexistent.task",
    )
    assert res["abstained"] is False
    assert res["hands_measured"] == 1
    assert res["hand_size_band"] == "small"
    # palm = |(32,16) - (0,0)| = 35.78px, face width 64px -> 0.559 (rounded to 4dp)
    assert abs(res["hand_face_ratio"] - round(np.hypot(32.0, 16.0) / 64.0, 4)) < 1e-6


def test_set_band_floors_recalibrates() -> None:
    """The probe recalibration hook must take effect (nose-geometry pattern)."""
    from research_harness import hand_face_ratio
    old_small, old_large = SMALL_MAX, LARGE_MIN
    try:
        hand_face_ratio.set_band_floors(0.55, 1.05)
        # 0.60 sits between the new 0.55/1.05 cuts.
        assert _apply_bands({"hand_face_ratio": 0.60})["hand_size_band"] == "typical"
        assert _apply_bands({"hand_face_ratio": 0.50})["hand_size_band"] == "small"
        assert _apply_bands({"hand_face_ratio": 1.10})["hand_size_band"] == "large"
    finally:
        hand_face_ratio.set_band_floors(old_small, old_large)
