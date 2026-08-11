"""TDD coverage for the eyebrow-thickness evidence specialist (arm #127).

Deterministic eyebrow-thickness band (thin / medium / thick) from the
already-qualified MediaPipe FaceLandmarker 478-point mesh (same model as
face-geometry #60 / eyebrow-position #111 / lip-fullness #122 / eye-shape
#123). Only the coarse band is verbalized; normalized brow-vertical-extent
ratios and per-side payloads stay in the machine-readable payload. The pure
plan-banding (`_apply_bands`) and render are tested without the model; the
compute path abstains cleanly on a blank non-face frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.eyebrow_thickness import (
    MIN_EYE_PX,
    MIN_PAIRS_PER_SIDE,
    THICK_MIN,
    THIN_MAX,
    EyebrowThicknessError,
    _apply_bands,
    compute_eyebrow_thickness,
    render_eyebrow_thickness,
    validate_rgb_array,
    validate_seg2_array,
)


def test_band_three_way_thickness() -> None:
    assert _apply_bands({"brow_thickness": THIN_MAX - 0.001})["eyebrow_thickness_band"] == "thin"
    assert _apply_bands({"brow_thickness": THICK_MIN + 0.001})["eyebrow_thickness_band"] == "thick"
    mid = (THIN_MAX + THICK_MIN) / 2
    assert _apply_bands({"brow_thickness": mid})["eyebrow_thickness_band"] == "medium"
    # Exact boundary values classify deterministically (< thin, > thick).
    assert _apply_bands({"brow_thickness": THIN_MAX})["eyebrow_thickness_band"] == "medium"
    assert _apply_bands({"brow_thickness": THICK_MIN})["eyebrow_thickness_band"] == "medium"


def test_band_missing_ratio_is_payload_only() -> None:
    f = _apply_bands({"abstained": False})
    assert f["eyebrow_thickness_band"] is None
    assert f["banding_unavailable"] is True
    assert render_eyebrow_thickness(f) == [
        "eyebrow-thickness: measured but not banded (payload)"
    ]


def test_band_payload_fields_preserved() -> None:
    f = _apply_bands({
        "brow_thickness": 0.10,
        "sides_measured": 2,
        "per_side": [
            {"side": "l", "brow_thickness": 0.10, "pairs_measured": 5},
            {"side": "r", "brow_thickness": 0.11, "pairs_measured": 5},
        ],
    })
    assert f["eyebrow_thickness_band"] == "thin"
    assert f["sides_measured"] == 2
    assert f["per_side"][0]["pairs_measured"] == 5


def _synthetic_mesh(
    eye_w: float = 40.0,
    brow_thickness_ratio: float = 0.20,
) -> np.ndarray:
    """Build a synthetic 478-point mesh with plausible eyes and brows.

    Left eye outer (33) at x0=100, inner (133) at x0+eye_w. The eye line is
    horizontal (y=200). The brow upper arc sits above the line and the lower
    edge sits between the upper arc and the line, so that
    (upper - lower) / eye_w == brow_thickness_ratio.
    """
    pts = np.zeros((478, 2), dtype=np.float64)
    for out_i, in_i, upper, lower, x0 in (
        (33, 133, (70, 63, 105, 66, 107), (46, 53, 52, 65, 55), 100.0),
        (362, 263, (336, 296, 334, 293, 300), (276, 283, 282, 295, 285), 220.0),
    ):
        xs = np.linspace(x0 + 2.0, x0 + eye_w - 2.0, 5)
        pts[out_i] = (x0, 200.0)
        pts[in_i] = (x0 + eye_w, 200.0)
        # Upper arc height: 0.30 of eye width above the eye line.
        upper_y = 200.0 - 0.30 * eye_w
        # Lower edge: upper_y + thickness (so extent = thickness / eye_w).
        lower_y = upper_y + brow_thickness_ratio * eye_w
        for i, (u, l, x) in enumerate(zip(upper, lower, xs)):
            pts[u] = (x, upper_y)
            pts[l] = (x, lower_y)
    return pts


def test_measure_duplicates_matching_brow_geometry() -> None:
    """A synthetic brow band with extent 0.20 of eye width must read 0.20."""
    pts = _synthetic_mesh(brow_thickness_ratio=0.20)
    from research_harness.eyebrow_thickness import _side_measure
    l = _side_measure(pts, "l")
    r = _side_measure(pts, "r")
    assert l["measureable"] is True
    assert r["measureable"] is True
    assert abs(l["brow_thickness"] - 0.20) < 0.02
    assert abs(r["brow_thickness"] - 0.20) < 0.02
    assert l["pairs_measured"] == 5


def test_measure_thick_brow_reads_thick_band() -> None:
    pts = _synthetic_mesh(brow_thickness_ratio=0.35)
    from research_harness.eyebrow_thickness import _side_measure as _sm
    sides = [_sm(pts, "l"), _sm(pts, "r")]
    item = float(np.mean([s["brow_thickness"] for s in sides]))
    band = _apply_bands({"brow_thickness": item})["eyebrow_thickness_band"]
    assert band == "thick"


def test_measure_single_side_discloses_asymmetry() -> None:
    """One degenerate side (tiny eye width) must fall back to the single side."""
    pts = _synthetic_mesh(brow_thickness_ratio=0.20)
    # Collapse the right eye width below MIN_EYE_PX by moving 362 onto 263.
    pts[362] = pts[263] = (240.0, 200.0)
    from research_harness.eyebrow_thickness import _side_measure
    r = _side_measure(pts, "r")
    assert r["measureable"] is False
    l = _side_measure(pts, "l")
    assert l["measureable"] is True


def test_measure_no_measurable_side_abstains() -> None:
    """Both eyes degenerate (width 4 < MIN_EYE_PX) must yield no side."""
    pts = _synthetic_mesh(eye_w=4.0, brow_thickness_ratio=0.20)
    from research_harness.eyebrow_thickness import _side_measure
    l = _side_measure(pts, "l")
    r = _side_measure(pts, "r")
    assert l["measureable"] is False
    assert r["measureable"] is False


def test_measure_implausible_extent_abstains() -> None:
    """Brow lower edge BELOW the eye line (extent negative) is implausible."""
    pts = np.zeros((478, 2), dtype=np.float64)
    for out_i, in_i, upper, lower in (
        (33, 133, (70, 63, 105, 66, 107), (46, 53, 52, 65, 55)),
        (362, 263, (336, 296, 334, 293, 300), (276, 283, 282, 295, 285)),
    ):
        pts[out_i] = (100.0, 200.0)
        pts[in_i] = (140.0, 200.0)
        for u in upper:
            pts[u] = (110.0, 190.0)  # upper arc above the line
        for l in lower:
            pts[l] = (110.0, 230.0)  # lower edge far BELOW the line -> extent ~1.0 > 0.50
    from research_harness.eyebrow_thickness import _side_measure
    l = _side_measure(pts, "l")
    assert l["measureable"] is False
    assert "brow landmarks" in l["reason"]


def test_render_abstention() -> None:
    r = render_eyebrow_thickness({
        "abstained": True,
        "abstention_reason": "no face detected",
    })
    assert len(r) == 1 and "abstain" in r[0] and "no face detected" in r[0]


def test_render_band_phrases() -> None:
    assert "thin" in render_eyebrow_thickness({"eyebrow_thickness_band": "thin"})[0]
    assert "thick" in render_eyebrow_thickness({"eyebrow_thickness_band": "thick"})[0]
    assert "medium" in render_eyebrow_thickness({"eyebrow_thickness_band": "medium"})[0]
    assert render_eyebrow_thickness({}) == []


def test_validate_arrays() -> None:
    with pytest.raises(EyebrowThicknessError):
        validate_rgb_array(np.zeros((5, 5), dtype=np.uint8))
    with pytest.raises(EyebrowThicknessError):
        validate_rgb_array(np.zeros((5, 5, 3), dtype=np.float32))
    with pytest.raises(EyebrowThicknessError):
        validate_seg2_array(np.zeros((5, 5, 1), dtype=np.uint8))


def test_misaligned_shapes_raise() -> None:
    seg2 = np.zeros((10, 20), dtype=np.uint8)
    rgb = np.zeros((11, 20, 3), dtype=np.uint8)
    with pytest.raises(EyebrowThicknessError):
        compute_eyebrow_thickness(seg2, rgb, model_asset_path="/nonexistent.task")


def test_compute_abstains_on_blank_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank non-face frame must abstain cleanly (no crash, no nonsense)."""
    monkeypatch.setattr(
        "research_harness.eyebrow_thickness._detect_mesh_on",
        lambda arr, model_asset_path: None,
    )
    seg2 = np.zeros((64, 64), dtype=np.uint8)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8) + 200
    res = compute_eyebrow_thickness(seg2, rgb, model_asset_path="/nonexistent.task")
    assert res["abstained"] is True
    assert res["abstention_reason"]


def test_floor_constants_sane() -> None:
    assert MIN_EYE_PX > 0
    assert MIN_PAIRS_PER_SIDE >= 2
    assert 0.0 < THIN_MAX < THICK_MIN
    assert THIN_MAX > 0.02  # above the plausibility floor so thin can fire


def test_set_band_floors_recalibrates() -> None:
    """The probe recalibration hook must take effect (eyebrow-position pattern)."""
    from research_harness import eyebrow_thickness
    old_thin, old_thick = THIN_MAX, THICK_MIN
    try:
        eyebrow_thickness.set_band_floors(0.12, 0.28)
        assert _apply_bands({"brow_thickness": 0.10})["eyebrow_thickness_band"] == "thin"
        assert _apply_bands({"brow_thickness": 0.30})["eyebrow_thickness_band"] == "thick"
    finally:
        eyebrow_thickness.set_band_floors(old_thin, old_thick)