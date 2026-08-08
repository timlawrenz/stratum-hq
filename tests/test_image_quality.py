"""TDD coverage for the image-quality evidence specialist (arm #95).

NEW-MODEL-CLASS specialist: open-weight CLIP ViT-L/14 implementing the
zero-shot CLIP-IQA scoring method (Wang et al., AAAI 2023) — a no-reference
perceptual-quality measurement. Only the coarse scale-invariant quality band
(sharp / moderate / degraded) or a surfaced abstention is verbalized; the raw
CLIP-IQA score, aspect probabilities, and logits stay in the machine-readable
payload. The banding/render/validation logic is pure and tested without the
model; the compute path runs the scorer on a small synthetic frame (CPU,
owned hardware) to verify the full pipeline emits a proper structure.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.image_quality import (
    MODERATE_FLOOR,
    SHARP_FLOOR,
    ImageQualityError,
    _quality_band,
    compute_image_quality,
    render_image_quality,
    validate_rgb_array,
)


def test_validate_arrays() -> None:
    with pytest.raises(ImageQualityError):
        validate_rgb_array(np.zeros((5, 5), dtype=np.uint8))
    with pytest.raises(ImageQualityError):
        validate_rgb_array(np.zeros((5, 5, 3), dtype=np.float32))
    with pytest.raises(ImageQualityError):
        validate_rgb_array(np.zeros((5, 5, 4), dtype=np.uint8))


def test_render_empty_is_no_claim() -> None:
    # Not measured: must NOT fabricate a quality band.
    assert render_image_quality({}) == []
    assert render_image_quality({"quality_band": None}) == []


def test_render_abstention() -> None:
    lines = render_image_quality({"abstained": True, "abstention_reason": "model failure"})
    assert lines and "abstain" in lines[0]


def test_render_bands() -> None:
    assert render_image_quality({"quality_band": "sharp"}) == [
        "image-quality: the photo appears sharp and crisp"
    ]
    assert render_image_quality({"quality_band": "moderate"}) == [
        "image-quality: the photo appears of moderate quality"
    ]
    assert render_image_quality({"quality_band": "degraded"}) == [
        "image-quality: the photo appears degraded / low quality"
    ]


def test_quality_band_floors() -> None:
    assert _quality_band(SHARP_FLOOR) == "sharp"
    assert _quality_band(SHARP_FLOOR - 1e-6) == "moderate"
    assert _quality_band(MODERATE_FLOOR) == "moderate"
    assert _quality_band(MODERATE_FLOOR - 1e-6) == "degraded"
    assert _quality_band(1.0) == "sharp"
    assert _quality_band(0.0) == "degraded"


def test_compute_abstains_on_implausible_score(monkeypatch) -> None:
    # An out-of-range score must abstain, never fabricate a band.
    monkeypatch.setattr(
        "research_harness.image_quality._clip_iqa_score",
        lambda rgb, model_asset_dir: {
            "score": 1.7,  # outside [0, 1]
            "aspect_scores": [1.0, 1.0, 0.9, 0.9],
            "aspect_logits": [[10.0, -10.0]] * 4,
            "prompt_pairs": [],
        },
    )
    q = compute_image_quality(np.zeros((8, 8, 3), dtype=np.uint8), model_asset_dir="unused")
    assert q["abstained"] is True
    assert q["quality_band"] is None
    assert "outside" in q["abstention_reason"]


def test_compute_structure_and_band() -> None:
    # Deterministic score path: sharp score -> sharp band, payload fields present.
    import research_harness.image_quality as iq
    from unittest.mock import patch

    with patch.object(iq, "_clip_iqa_score", return_value={
        "score": 0.81,
        "aspect_scores": [0.8, 0.85, 0.79, 0.80],
        "aspect_logits": [[3.0, -3.0]] * 4,
        "prompt_pairs": list(iq.QUALITY_PROMPT_PAIRS),
    }):
        q = iq.compute_image_quality(np.zeros((8, 8, 3), dtype=np.uint8), model_asset_dir="unused")
    assert q["abstained"] is False
    assert q["quality_band"] == "sharp"
    assert q["score"] == 0.81
    assert q["sharp_floor"] == SHARP_FLOOR
    assert q["moderate_floor"] == MODERATE_FLOOR


def test_compute_real_model_pipeline() -> None:
    # Full pipeline on a small synthetic frame through the real local CLIP
    # asset (CPU, owned hardware). Verifies the model loads and emits a proper
    # structure — the same asset the frozen plan binds.
    import research_harness.image_quality as iq

    rng = np.random.default_rng(20260808)
    rgb = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    q = iq.compute_image_quality(rgb, model_asset_dir=iq.IMAGE_QUALITY_MODEL_ASSET)
    assert q["abstained"] in (True, False)
    if not q["abstained"]:
        assert q["quality_band"] in ("sharp", "moderate", "degraded")
        assert 0.0 <= q["score"] <= 1.0
    assert "score" in q
    assert "aspect_scores" in q


def test_unresolvable_model_asset_raises() -> None:
    # A missing model asset must surface as a compute error (fail closed), not
    # a silent fabricated band.
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    with pytest.raises(ImageQualityError):
        compute_image_quality(rgb, model_asset_dir="/nonexistent/model-asset")