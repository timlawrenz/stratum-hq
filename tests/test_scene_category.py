"""TDD coverage for the scene-category evidence specialist (arm #69).

NEW-MODEL-CLASS specialist: open-weight CLIP ViT-L/14 zero-shot semantic
scene classifier over the full-frame source with a frozen closed scene-category
set. Only the scale-invariant category label (or a surfaced abstention) is
verbalized; similarity logits / probabilities stay in the machine-readable
payload. The banding/render/validation logic is pure and tested without the
model; the compute path runs the classifier on a small synthetic frame (CPU,
owned hardware) to verify the full pipeline emits a proper structure.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.scene_category import (
    ABSTAIN_CONFIDENCE,
    SceneCategoryError,
    _zero_shot_probabilities,
    compute_scene_category,
    render_scene_category,
    validate_rgb_array,
)

MODEL_DIR = "/mnt/nas-ai-models/research/stratum/models/scene-category"


def test_validate_arrays() -> None:
    with pytest.raises(SceneCategoryError):
        validate_rgb_array(np.zeros((5, 5), dtype=np.uint8))
    with pytest.raises(SceneCategoryError):
        validate_rgb_array(np.zeros((5, 5, 3), dtype=np.float32))
    with pytest.raises(SceneCategoryError):
        validate_rgb_array(np.zeros((5, 5, 4), dtype=np.uint8))


def test_render_empty_is_no_claim() -> None:
    # Not measured: must NOT fabricate a scene label.
    assert render_scene_category({}) == []
    assert render_scene_category({"category": None}) == []


def test_render_abstention() -> None:
    lines = render_scene_category({"abstained": True, "abstention_reason": "low confidence"})
    assert lines and "abstain" in lines[0]


def test_render_label() -> None:
    lines = render_scene_category({"category": "outdoor beach", "confidence": 0.9})
    assert lines == ["scene-category: the setting is a outdoor beach"]


def test_render_below_floor_abstains() -> None:
    lines = render_scene_category({
        "category": "bedroom", "confidence": 0.05,
        "abstain_confidence": ABSTAIN_CONFIDENCE,
    })
    assert lines and "abstain" in lines[0]


def test_zero_shot_probabilities_softmax_structure() -> None:
    """The probability vector is a non-negative normalized closed-set softmax."""
    rng = np.random.default_rng(1)
    rgb = np.ascontiguousarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
    probs, logits = _zero_shot_probabilities(rgb, model_asset_dir=MODEL_DIR)
    assert len(probs) >= 10
    assert abs(sum(probs) - 1.0) < 1e-3
    assert all(p >= 0.0 for p in probs)
    assert len(logits) == len(probs)


def test_compute_runs_on_synthetic_frame() -> None:
    """End-to-end pipeline smoke: the classifier over a synthetic frame emits a
    well-formed structure (CPU, owned hardware, read-only)."""
    rng = np.random.default_rng(3)
    rgb = np.ascontiguousarray(rng.integers(80, 180, (256, 256, 3), dtype=np.uint8))
    # a bright "outdoor sky + field-ish" block to nudge the classifier
    rgb[0:120, :, :] = (135, 200, 235)
    rgb[150:256, :, :] = (80, 160, 60)

    result = compute_scene_category(rgb, model_asset_dir=MODEL_DIR)
    assert "abstained" in result
    assert "confidence" in result
    assert "category" in result
    # Structure invariants: probabilities normalized, logits aligned.
    probs = result.get("probabilities")
    if not result["abstained"]:
        assert result["category"] is not None
    if probs:
        assert abs(sum(probs) - 1.0) < 1e-2
        assert len(result.get("logits", [])) == len(probs)
