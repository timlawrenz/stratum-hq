"""Tests for the arm-#112 jewelry learned specialist (CLIP zero-shot).

NEW evidence part (jewelry) + NEW MODEL CLASS (CLIP ViT-L/14 zero-shot over
the seg2 Face_Neck crop). Validated 2026-08-10 by a capability probe on the
frozen cohort (jewelry-calibration-probe.json): 23/24 measured, earrings
sub-band NON-degenerate (max_share 0.739 < 0.75), necklace sub-band DEGENERATE
(max_share 0.783 >= 0.75) -> silenced payload-only (arm-#74 precedent),
VERBALIZED merged jewelry-presence band (earrings OR necklace) 8/23
(max_share 0.652), coverage floor 8/24 MET. Only the merged band is
verbalized; raw probabilities and sub-band decisions stay payload-only.

The validation/render/abstention logic is pure and tested without the model;
the compute path runs the classifier on a Face_Neck crop (CPU, owned
hardware) to verify the full pipeline emits a proper structure.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.jewelry import (
    ABSTAIN_CONFIDENCE,
    EARRINGS_LABELS,
    JewelryError,
    NECKLACE_LABELS,
    compute_jewelry,
    render_jewelry,
    validate_seg2_array,
)

MODEL_DIR = "/mnt/nas-ai-models/research/stratum/models/scene-category"

FACE_NECK = 3  # DOME-29 Face_Neck
SKIN = 7  # Left_Upper_Arm (a limb/skin class > 0 -> foreground)


def _seg_with_face_neck(*, shape=(240, 240)) -> np.ndarray:
    """A seg2 canvas with a foreground subject + a rectangular Face_Neck region."""
    seg = np.zeros(shape, dtype=np.uint8)
    seg[40:200, 40:200] = SKIN  # subject
    seg[60:140, 80:160] = FACE_NECK
    return seg


def _noise_rgb(shape=(240, 240)) -> np.ndarray:
    rng = np.random.default_rng(7)
    return np.ascontiguousarray(rng.integers(80, 180, (*shape, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

def test_validate_seg2() -> None:
    with pytest.raises(JewelryError):
        compute_jewelry(np.zeros((240, 240, 3), dtype=np.uint8),
                        np.zeros((240, 240, 3), dtype=np.uint8))
    with pytest.raises(JewelryError):
        compute_jewelry(np.zeros((240, 240), dtype=np.uint8),
                        np.zeros((240, 240), dtype=np.uint8))


def test_validate_seg2_array_direct() -> None:
    with pytest.raises(JewelryError):
        validate_seg2_array(np.zeros((240, 240, 3), dtype=np.uint8))
    with pytest.raises(JewelryError):
        validate_seg2_array(np.zeros((240, 240), dtype=np.float64))


def test_validate_rgb_shape() -> None:
    seg = _seg_with_face_neck()
    with pytest.raises(JewelryError):
        compute_jewelry(seg, np.zeros((240, 240), dtype=np.uint8))  # not (H,W,3)
    with pytest.raises(JewelryError):
        compute_jewelry(seg, np.zeros((120, 120, 3), dtype=np.uint8))  # misaligned


# ---------------------------------------------------------------------------
# Abstention behavior (pure paths)
# ---------------------------------------------------------------------------

def test_abstains_no_foreground() -> None:
    seg = np.zeros((240, 240), dtype=np.uint8)
    rgb = _noise_rgb()
    out = compute_jewelry(seg, rgb, model_asset_dir=MODEL_DIR)
    assert out["abstained"] is True
    assert out["subject_present"] is False
    assert out["jewelry_band"] is None
    assert "no foreground" in out["abstention_reason"]


def test_abstains_face_neck_below_floor() -> None:
    seg = _seg_with_face_neck()
    rgb = _noise_rgb()
    out = compute_jewelry(seg, rgb, min_px=10_000, model_asset_dir=MODEL_DIR)
    assert out["abstained"] is True
    assert out["jewelry_band"] is None
    assert "below the raw-pixel floor" in out["abstention_reason"]


def test_abstains_empty_crop() -> None:
    rgb = _noise_rgb()
    # A zero-width Face_Neck box has face_neck_px == 0 < min_px -> abstains.
    seg3 = np.zeros((240, 240), dtype=np.uint8)
    seg3[40:200, 40:200] = SKIN
    seg3[100:140, 100:100] = FACE_NECK  # zero-width -> no pixels
    out = compute_jewelry(seg3, rgb, min_px=1, model_asset_dir=MODEL_DIR)
    assert out["abstained"] is True
    assert out["jewelry_band"] is None
    assert out["face_neck_px"] == 0

# ---------------------------------------------------------------------------
# Render determinism (pure)
# ---------------------------------------------------------------------------

def test_render_absent_never_fabricates() -> None:
    text = render_jewelry({
        "subject_present": True,
        "abstained": False,
        "abstention_reason": None,
        "jewelry_band": "no-jewelry",
    })
    assert "NO jewelry detected" in text
    assert "do NOT describe earrings" in text


def test_render_present_grounds_only_named_evidence() -> None:
    text = render_jewelry({
        "subject_present": True,
        "abstained": False,
        "abstention_reason": None,
        "jewelry_band": "jewelry-present",
    })
    assert "IS wearing jewelry" in text
    assert "never specific materials or brands" in text


def test_render_abstained_surfaces_reason() -> None:
    text = render_jewelry({
        "abstained": True,
        "abstention_reason": "Face_Neck region absent or below the raw-pixel floor",
    })
    assert "abstained" in text
    assert "below the raw-pixel floor" in text


def test_render_none_config() -> None:
    assert "not measured" in render_jewelry(None)


def test_render_missing_band_does_not_crash() -> None:
    text = render_jewelry({"subject_present": True, "abstained": False})
    assert isinstance(text, str)


# ---------------------------------------------------------------------------
# Full CLI compute path (real CLIP on CPU, owned hardware)
# ---------------------------------------------------------------------------

def test_compute_on_real_clip_emits_structure() -> None:
    seg = _seg_with_face_neck()
    rgb = _noise_rgb()
    out = compute_jewelry(seg, rgb, model_asset_dir=MODEL_DIR)
    assert out["subject_present"] is True
    assert out["abstained"] is False
    assert out["face_neck_px"] >= 6400
    assert out["jewelry_band"] in ("jewelry-present", "no-jewelry")
    assert out["earrings_band"] in ("earrings", "none")
    assert out["necklace_band"] in ("necklace", "none")
    assert 0.0 <= out["confidence"] <= 1.0
    assert len(out["earrings_probabilities"]) == len(EARRINGS_LABELS) == 2
    assert len(out["necklace_probabilities"]) == len(NECKLACE_LABELS) == 2
    assert out["abstain_confidence"] == ABSTAIN_CONFIDENCE