"""TDD coverage for the apparent-age evidence specialist (arm #73).

Pure logic tests (`_age_band`, `render_apparent_age`, validators, crop bbox)
run without the model; the compute path surfaces a clean abstention when the
MiVOLO inference path fails (simulated by monkeypatching `_infer_age` to raise /
return an implausible age). Only the coarse scale-invariant band is verbalized;
raw ages / gender probe stay in the machine-readable payload.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.apparent_age import (
    AGE_EARLY_TWENTIES_MAX,
    AGE_LATE_TEENS_MAX,
    AGE_MID_TWENTIES_MAX,
    ApparentAgeError,
    _age_band,
    _crop,
    _crop_mask_bbox,
    compute_apparent_age,
    render_apparent_age,
    validate_rgb_array,
    validate_seg2_array,
)


def test_age_band_calibrated_four_way() -> None:
    assert _age_band(AGE_LATE_TEENS_MAX - 0.1) == "late-teens-to-early-twenties"
    assert _age_band((AGE_LATE_TEENS_MAX + AGE_EARLY_TWENTIES_MAX) / 2) == "early-twenties"
    assert _age_band((AGE_EARLY_TWENTIES_MAX + AGE_MID_TWENTIES_MAX) / 2) == "mid-twenties"
    assert _age_band(AGE_MID_TWENTIES_MAX + 0.1) == "late-twenties-to-thirties"


def test_age_band_no_band_is_75_percent_share() -> None:
    # Calibration invariant: the 2026-08-07 probe measured 2/6/12/4 (max share
    # 50.0%), i.e. the bands on the real cohort never exceed the 75% degeneracy
    # line. Mirror the probe ages through `_age_band` and assert max share.
    from collections import Counter

    probe_ages = [26.46, 20.87, 25.03, 24.4, 26.29, 26.72, 26.06, 24.85, 27.87,
                  24.86, 25.19, 27.53, 29.37, 32.89, 29.63, 26.02, 26.19, 25.5,
                  25.57, 19.82, 24.25, 29.28, 26.66, 26.64]
    c = Counter(_age_band(a) for a in probe_ages)
    max_share = max(c.values()) / len(probe_ages)
    assert max_share <= 0.75


def test_validate_seg2_array() -> None:
    validate_seg2_array(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ApparentAgeError):
        validate_seg2_array(np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(ApparentAgeError):
        validate_seg2_array("nope")


def test_validate_rgb_array() -> None:
    validate_rgb_array(np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(ApparentAgeError):
        validate_rgb_array(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ApparentAgeError):
        validate_rgb_array(np.zeros((8, 8, 3), dtype=np.float32))


def test_crop_mask_bbox_and_crop() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 30:50] = 1
    bbox = _crop_mask_bbox(mask, margin=5, img_h=100, img_w=100)
    assert bbox == (15, 44, 25, 54)
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = _crop(rgb, bbox)
    assert crop.shape == (29, 29, 3)


def test_render_apparent_age_band() -> None:
    r = render_apparent_age({"age_band": "mid-twenties", "age_years": 26.2})
    assert r and "mid-twenties" in r[0]
    assert "26.2" not in r[0]  # raw float age never verbalized


def test_render_apparent_age_abstained() -> None:
    r = render_apparent_age({"abstained": True, "abstention_reason": "face too small"})
    assert r and "abstain" in r[0]


def test_render_apparent_age_not_measured_no_claim() -> None:
    # #74/#75 no-claim pattern: a non-apparent-age run must never fabricate an
    # age claim (this keeps the context4k dossier_evidence_ids test green).
    assert render_apparent_age({}) == []
    assert render_apparent_age(None) == []


def test_compute_abstains_on_mivolo_failure(monkeypatch) -> None:
    def _boom(rgb, face_crop, body_crop, model_dir):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr("research_harness.apparent_age._infer_age", _boom)
    seg2 = np.zeros((64, 64), dtype=np.uint8)
    seg2[10:30, 20:44] = 3  # Face_Neck region present
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    m = compute_apparent_age(seg2, rgb)
    assert m["abstained"] is True
    assert "failed on all face candidates" in m["abstention_reason"]


def test_compute_abstains_on_implausible_age(monkeypatch) -> None:
    def _implausible(rgb, face_crop, body_crop, model_dir):
        return {"age_years": -5.0, "gender_probe": None}

    monkeypatch.setattr("research_harness.apparent_age._infer_age", _implausible)
    seg2 = np.zeros((64, 64), dtype=np.uint8)
    seg2[10:30, 20:44] = 3
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    m = compute_apparent_age(seg2, rgb)
    assert m["abstained"] is True


def test_compute_mismatched_shapes() -> None:
    with pytest.raises(ApparentAgeError, match="pixel-aligned"):
        compute_apparent_age(np.zeros((10, 10), dtype=np.uint8),
                             np.zeros((20, 20, 3), dtype=np.uint8))
