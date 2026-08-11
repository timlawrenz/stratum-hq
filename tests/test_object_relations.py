"""TDD coverage for the object-relations evidence specialist (arm #61).

NEW-MODEL-CLASS specialist: open-weight Grounding DINO (text-grounded
open-vocabulary detector) over the full frame + seg2 subject mask. Only
scale-invariant facts are verbalized (count band, placement band, canonical
class list); normalized boxes/scores stay in the machine-readable payload. The
banding/render/canonical logic is pure and tested without the model; the
compute path runs the detector on a small synthetic frame (CPU, owned
hardware) to verify the full pipeline emits a proper structure.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.object_relations import (
    DENSE,
    MODERATE,
    SPARSE,
    ObjectRelationsError,
    _count_band,
    _is_subject_self,
    _placement_band,
    canonical_class,
    compute_object_relations,
    render_object_relations,
    validate_rgb_array,
    validate_seg2_array,
)

MODEL_DIR = "/mnt/nas-ai-models/research/stratum/models/object-relations"


def test_count_band_thresholds() -> None:
    assert _count_band(0) == "none"
    assert _count_band(1) == "sparse"
    assert _count_band(SPARSE) == "sparse"
    assert _count_band(SPARSE + 1) == "moderate"
    assert _count_band(MODERATE) == "moderate"
    assert _count_band(MODERATE + 1) == "moderate"
    assert _count_band(DENSE) == "dense"
    assert _count_band(DENSE + 10) == "dense"


def test_placement_band() -> None:
    assert _placement_band(3, 1, 0, 4) == "foreground"
    assert _placement_band(1, 3, 0, 4) == "background"
    assert _placement_band(2, 2, 0, 4) == "mix"
    assert _placement_band(2, 1, 2, 5) == "mix"
    assert _placement_band(0, 0, 0, 0) == "none"
    assert _placement_band(1, 1, 1, 3) == "mix"


def test_canonical_class_mapping() -> None:
    assert canonical_class("body of water") == "body of water"
    assert canonical_class("window window frame door") == "window"
    assert canonical_class("sneakers") == "sneakers"
    assert canonical_class("Boat Deck") == "boat deck"
    # Unknown phrase falls through to itself.
    assert canonical_class("flurbafnord") == "flurbafnord"


def test_subject_self_guard() -> None:
    assert _is_subject_self("body") is True
    assert _is_subject_self("person") is True
    assert _is_subject_self("woman") is True
    # 'body of water' is a legitimate scene object, NOT subject-self.
    assert _is_subject_self("body of water") is False
    assert _is_subject_self("tree") is False


def test_validate_arrays() -> None:
    with pytest.raises(ObjectRelationsError):
        validate_rgb_array(np.zeros((5, 5), dtype=np.uint8))
    with pytest.raises(ObjectRelationsError):
        validate_rgb_array(np.zeros((5, 5, 3), dtype=np.float32))
    with pytest.raises(ObjectRelationsError):
        validate_seg2_array(np.zeros((5, 5, 1), dtype=np.uint8))
    with pytest.raises(ObjectRelationsError):
        validate_seg2_array(np.zeros((5, 5), dtype=np.float32))


def test_misaligned_shapes_raise() -> None:
    with pytest.raises(ObjectRelationsError):
        compute_object_relations(
            np.zeros((50, 50), dtype=np.uint8),
            np.zeros((100, 100, 3), dtype=np.uint8),
            model_asset_dir="unused",
        )


def test_render_empty_is_no_claim() -> None:
    # Not measured: must NOT fabricate a "no objects" claim.
    assert render_object_relations({}) == []
    assert render_object_relations({"count_band": None}) == []


def test_render_abstention() -> None:
    lines = render_object_relations({"abstained": True, "abstention_reason": "model failure"})
    assert lines and "abstain" in lines[0]


def test_render_bands() -> None:
    lines = render_object_relations({
        "count": 3, "count_band": "moderate", "placement_band": "background",
        "classes": ["tree", "body of water"],
    })
    text = " ".join(lines)
    assert "several scene objects" in text
    assert "background" in text
    assert "tree" in text

    none_lines = render_object_relations({
        "count": 0, "count_band": "none", "placement_band": "none", "classes": [],
    })
    assert any("no scene objects" in line for line in none_lines)


def test_compute_runs_on_synthetic_frame() -> None:
    """End-to-end pipeline smoke: detector over a tiny synthetic frame emits a
    well-formed structure (CPU, owned hardware, read-only)."""
    rng = np.random.default_rng(3)
    h, w = 256, 256
    rgb = np.full((h, w, 3), 120, dtype=np.uint8)
    # a simple rectangular "plant-like" blob
    rgb[180:230, 60:110] = (40, 160, 60)
    rgb[200:240, 70:100] = (60, 40, 20)
    rgb = np.ascontiguousarray(rgb)
    seg = np.zeros((h, w), dtype=np.uint8)
    seg[0:h, 0:w] = 0  # background; synth subject blob
    seg[150:250, 150:250] = 1  # 'Torso' subject-ish region

    result = compute_object_relations(seg, rgb, model_asset_dir=MODEL_DIR)
    assert "count_band" in result
    assert "placement_band" in result
    assert result["abstained"] is False
    # Structure invariants: normalized boxes in payload, bands in prose.
    for det in result.get("detections", []):
        box = det["box_normalized"]
        assert len(box) == 4
        assert all(0.0 <= v <= 1.0 for v in box)
        assert det["placement"] in ("in-front", "behind", "mixed")
