"""TDD coverage for the cheek-prominence evidence specialist (arm #128).

Deterministic cheek-prominence band (subtle / moderate / prominent) from the
already-qualified MediaPipe FaceLandmarker 478-point mesh (same model as
face-geometry #60 / eyebrow-position #111 / lip-fullness #122 / eye-shape
#123 / eyebrow-thickness #127). The primary zygomatic pair is the
zygomatic-arch landmarks 116/345 (NOT face-geometry #60's face-width cheek
pair 234/454 — reusing that pair would make cheek/jaw the exact reciprocal of
the validated jaw/face-width axis, a degenerate redundancy); the jaw pair
172/397 is shared with #60. Only the coarse band is verbalized; raw spans /
ratios stay in the machine-readable payload. The pure plan-banding
(`_apply_bands`) and render are tested without the model; the compute path
abstains cleanly on a blank non-face frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.cheek_prominence import (
    MIDLINE_TOL,
    MIN_SPAN_PX,
    PROMINENT_MIN,
    RATIO_PLAUSIBLE,
    SUBTLE_MAX,
    CheekProminenceError,
    _apply_bands,
    compute_cheek_prominence,
    render_cheek_prominence,
    validate_rgb_array,
    validate_seg2_array,
)


def test_band_three_way_prominence() -> None:
    assert _apply_bands({"cheek_bone_ratio": SUBTLE_MAX - 0.001})["cheek_prominence_band"] == "subtle"
    assert _apply_bands({"cheek_bone_ratio": PROMINENT_MIN + 0.001})["cheek_prominence_band"] == "prominent"
    mid = (SUBTLE_MAX + PROMINENT_MIN) / 2
    assert _apply_bands({"cheek_bone_ratio": mid})["cheek_prominence_band"] == "moderate"
    # Exact boundary values classify deterministically (< subtle, > prominent).
    assert _apply_bands({"cheek_bone_ratio": SUBTLE_MAX})["cheek_prominence_band"] == "moderate"
    assert _apply_bands({"cheek_bone_ratio": PROMINENT_MIN})["cheek_prominence_band"] == "moderate"


def test_band_missing_ratio_is_payload_only() -> None:
    f = _apply_bands({"abstained": False})
    assert f["cheek_prominence_band"] is None
    assert f["banding_unavailable"] is True
    assert render_cheek_prominence(f) == [
        "cheek-prominence: measured but not banded (payload)"
    ]


def test_band_payload_fields_preserved() -> None:
    f = _apply_bands({
        "cheek_bone_ratio": 1.10,
        "zygomatic_span_px": 44.0,
        "jaw_span_px": 40.0,
    })
    assert f["cheek_prominence_band"] == "subtle"
    assert f["zygomatic_span_px"] == 44.0


def _synthetic_mesh(
    zygomatic_span: float = 60.0,
    jaw_span: float = 40.0,
    jaw_dx_shift: float = 0.0,
) -> np.ndarray:
    """Build a synthetic 478-point mesh with plausible cheekbones and jaw.

    Zygomatic pair (116/345) centered at x=140, y=150; jaw pair (172/397)
    centered at x=140 + jaw_dx_shift, y=220. Ratio == zygomatic/jaw.
    """
    pts = np.zeros((478, 2), dtype=np.float64)
    zx0, zx1 = 140.0 - zygomatic_span / 2.0, 140.0 + zygomatic_span / 2.0
    pts[116] = (zx0, 150.0)
    pts[345] = (zx1, 150.0)
    jx0 = 140.0 + jaw_dx_shift - jaw_span / 2.0
    jx1 = 140.0 + jaw_dx_shift + jaw_span / 2.0
    pts[172] = (jx0, 220.0)
    pts[397] = (jx1, 220.0)
    return pts


def test_measure_duplicates_matching_cheekbone_geometry() -> None:
    """A synthetic zygomatic/jaw span ratio of 1.5 must read 1.5."""
    from research_harness.cheek_prominence import _measure
    m = _measure(_synthetic_mesh(zygomatic_span=60.0, jaw_span=40.0))
    assert m["measureable"] is True
    assert abs(m["cheek_bone_ratio"] - 1.5) < 1e-9
    assert abs(m["zygomatic_span_px"] - 60.0) < 1e-9
    assert abs(m["jaw_span_px"] - 40.0) < 1e-9


def test_measure_prominent_ratio_reads_prominent_band() -> None:
    from research_harness.cheek_prominence import _measure
    m = _measure(_synthetic_mesh(zygomatic_span=60.0, jaw_span=40.0))
    assert m["measureable"] is True
    assert _apply_bands(m)["cheek_prominence_band"] == "prominent"


def test_measure_ratio_outside_plausible_band_abstains() -> None:
    """A collapsed ratio (profile / occlusion) must abstain with a reason."""
    from research_harness.cheek_prominence import _measure
    m = _measure(_synthetic_mesh(zygomatic_span=30.0, jaw_span=40.0))
    assert m["measureable"] is False
    assert "human-plausible" in m["reason"]
    m2 = _measure(_synthetic_mesh(zygomatic_span=100.0, jaw_span=40.0))
    assert m2["measureable"] is False
    assert "human-plausible" in m2["reason"]


def test_measure_degenerate_spans_abstain() -> None:
    from research_harness.cheek_prominence import _measure
    m = _measure(_synthetic_mesh(zygomatic_span=4.0, jaw_span=40.0))
    assert m["measureable"] is False
    assert "degenerate" in m["reason"]
    m2 = _measure(_synthetic_mesh(zygomatic_span=60.0, jaw_span=3.0))
    assert m2["measureable"] is False
    assert "degenerate" in m2["reason"]


def test_measure_midline_misalignment_abstains() -> None:
    """Shifting the jaw pair off the mid-sagittal line is landmark failure."""
    from research_harness.cheek_prominence import _measure
    m = _measure(_synthetic_mesh(
        zygomatic_span=60.0, jaw_span=40.0, jaw_dx_shift=30.0,
    ))
    assert m["measureable"] is False
    assert "mid-sagittal misalignment" in m["reason"]


def test_render_abstention() -> None:
    r = render_cheek_prominence({
        "abstained": True,
        "abstention_reason": "no face detected",
    })
    assert len(r) == 1 and "abstain" in r[0] and "no face detected" in r[0]


def test_render_band_phrases() -> None:
    assert "subtle" in render_cheek_prominence({"cheek_prominence_band": "subtle"})[0]
    assert "prominent" in render_cheek_prominence({"cheek_prominence_band": "prominent"})[0]
    assert "moderate" in render_cheek_prominence({"cheek_prominence_band": "moderate"})[0]
    assert render_cheek_prominence({}) == []


def test_validate_arrays() -> None:
    with pytest.raises(CheekProminenceError):
        validate_rgb_array(np.zeros((5, 5), dtype=np.uint8))
    with pytest.raises(CheekProminenceError):
        validate_rgb_array(np.zeros((5, 5, 3), dtype=np.float32))
    with pytest.raises(CheekProminenceError):
        validate_seg2_array(np.zeros((5, 5, 1), dtype=np.uint8))


def test_misaligned_shapes_raise() -> None:
    seg2 = np.zeros((10, 20), dtype=np.uint8)
    rgb = np.zeros((11, 20, 3), dtype=np.uint8)
    with pytest.raises(CheekProminenceError):
        compute_cheek_prominence(seg2, rgb, model_asset_path="/nonexistent.task")


def test_compute_abstains_on_blank_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank non-face frame must abstain cleanly (no crash, no nonsense)."""
    monkeypatch.setattr(
        "research_harness.cheek_prominence._detect_mesh_on",
        lambda arr, model_asset_path: None,
    )
    seg2 = np.zeros((64, 64), dtype=np.uint8)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8) + 200
    res = compute_cheek_prominence(seg2, rgb, model_asset_path="/nonexistent.task")
    assert res["abstained"] is True
    assert res["abstention_reason"]


def test_floor_constants_sane() -> None:
    assert MIN_SPAN_PX > 0
    assert 0.0 < SUBTLE_MAX < PROMINENT_MIN
    assert SUBTLE_MAX > RATIO_PLAUSIBLE[0]  # subtle band reachable in-plausibility
    assert PROMINENT_MIN < RATIO_PLAUSIBLE[1]  # prominent band reachable in-plausibility
    assert 0.0 < MIDLINE_TOL < 1.0


def test_set_band_floors_recalibrates() -> None:
    """The probe recalibration hook must take effect (eyebrow-position pattern)."""
    from research_harness import cheek_prominence
    old_subtle, old_prominent = SUBTLE_MAX, PROMINENT_MIN
    try:
        cheek_prominence.set_band_floors(1.20, 1.60)
        assert _apply_bands({"cheek_bone_ratio": 1.10})["cheek_prominence_band"] == "subtle"
        assert _apply_bands({"cheek_bone_ratio": 1.70})["cheek_prominence_band"] == "prominent"
        assert _apply_bands({"cheek_bone_ratio": 1.40})["cheek_prominence_band"] == "moderate"
    finally:
        cheek_prominence.set_band_floors(old_subtle, old_prominent)