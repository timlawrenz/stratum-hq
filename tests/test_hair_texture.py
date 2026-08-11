"""Tests for the arm-#94 hair-texture learned specialist (CLIP re-scope).

RE-SCOPED 2026-08-10 under the open-world sourcing directive: the deterministic
gradient-orientation axis measured DEGENERATE (11/24 all one band, max_share
1.00), so the arm now runs the open-weight CLIP ViT-L/14 zero-shot classifier
over the seg2 Hair crop with a closed straight/wavy/curly/coily vocabulary.
The validation/render/abstention logic is pure and tested without the model;
the compute path runs the classifier on a hair crop (CPU, owned hardware) to
verify the full pipeline emits a proper structure.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.hair_texture import (
    ABSTAIN_CONFIDENCE,
    HairTextureError,
    TEXTURE_LABELS,
    compute_hair_texture,
    render_hair_texture,
    validate_seg2_array,
)

MODEL_DIR = "/mnt/nas-ai-models/research/stratum/models/scene-category"

HAIR = 4  # DOME-29 Hair
SKIN = 7  # Left_Upper_Arm (a limb/skin class > 0 -> foreground)


def _seg_with_hair(*, shape=(240, 240)) -> np.ndarray:
    """A seg2 canvas with a foreground subject + a rectangular Hair region."""
    seg = np.zeros(shape, dtype=np.uint8)
    seg[40:200, 40:200] = SKIN  # subject
    seg[60:140, 80:160] = HAIR
    return seg


def _noise_rgb(shape=(240, 240)) -> np.ndarray:
    rng = np.random.default_rng(7)
    return np.ascontiguousarray(rng.integers(80, 180, (*shape, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

def test_validate_seg2() -> None:
    with pytest.raises(HairTextureError):
        compute_hair_texture(np.zeros((240, 240, 3), dtype=np.uint8),
                             np.zeros((240, 240, 3), dtype=np.uint8))
    with pytest.raises(HairTextureError):
        compute_hair_texture(np.zeros((240, 240), dtype=np.uint8),
                             np.zeros((240, 240), dtype=np.uint8))


def test_validate_seg2_array_direct() -> None:
    with pytest.raises(HairTextureError):
        validate_seg2_array(np.zeros((240, 240, 3), dtype=np.uint8))
    with pytest.raises(HairTextureError):
        validate_seg2_array(np.zeros((240, 240), dtype=np.float32))


def test_misaligned_rgb_aborts() -> None:
    seg = _seg_with_hair()
    with pytest.raises(HairTextureError):
        compute_hair_texture(seg, np.zeros((100, 100, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Bands + abstention
# ---------------------------------------------------------------------------

def test_no_foreground_abstains() -> None:
    out = compute_hair_texture(np.zeros((240, 240), dtype=np.uint8),
                               np.zeros((240, 240, 3), dtype=np.uint8))
    assert out["abstained"] is True
    assert out["subject_present"] is False


def test_no_hair_abstains() -> None:
    seg = np.zeros((240, 240), dtype=np.uint8)
    seg[40:200, 40:200] = SKIN
    out = compute_hair_texture(seg, np.full((240, 240, 3), 128, dtype=np.uint8))
    assert out["abstained"] is True
    assert "Hair region absent" in out["abstention_reason"]


def test_tiny_hair_abstains() -> None:
    seg = np.zeros((240, 240), dtype=np.uint8)
    seg[40:200, 40:200] = SKIN
    seg[100:110, 100:110] = HAIR  # 100 px < 200 floor
    out = compute_hair_texture(seg, np.full((240, 240, 3), 128, dtype=np.uint8))
    assert out["abstained"] is True


def test_compute_runs_on_noise_hair_crop() -> None:
    """End-to-end pipeline smoke: the classifier over a hair crop emits a
    well-formed structure (CPU, owned hardware, read-only)."""
    seg = _seg_with_hair()
    rgb = _noise_rgb()
    result = compute_hair_texture(seg, rgb, model_asset_dir=MODEL_DIR)
    assert "abstained" in result
    assert "confidence" in result
    assert "hair_texture_band" in result
    assert result["hair_present"] is True
    probs = result.get("probabilities")
    if not result["abstained"]:
        assert result["hair_texture_band"] in ("straight", "wavy", "curly", "coily")
    if probs:
        assert abs(sum(probs) - 1.0) < 1e-2
        assert len(result.get("logits", [])) == len(probs)
        assert len(probs) == len(TEXTURE_LABELS)


def test_abstention_floor_constant() -> None:
    assert ABSTAIN_CONFIDENCE == 0.35


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_no_config_empty() -> None:
    assert render_hair_texture({}) == []


def test_render_abstain() -> None:
    lines = render_hair_texture({"abstained": True, "abstention_reason": "low confidence"})
    assert any("abstain" in ln for ln in lines)


def test_render_band_text() -> None:
    for band, word in (("straight", "straight"), ("wavy", "wavy"),
                       ("curly", "curly"), ("coily", "coily")):
        lines = render_hair_texture({"hair_present": True, "hair_texture_band": band})
        assert any(word in ln for ln in lines)


def test_render_no_raw_stats_in_prose() -> None:
    """Raw probabilities/logits must never leak into rendered claims."""
    lines = render_hair_texture({"hair_present": True, "hair_texture_band": "wavy",
                                 "confidence": 0.5, "probabilities": [0.4, 0.6]})
    joined = " ".join(lines)
    assert "0.5" not in joined
    assert "0.4" not in joined