"""TDD coverage for the dossier -> full-prose caption2 renderer (#79 prerequisite)."""

from __future__ import annotations

import pytest

from research_harness.dossier import assemble_dossier
from research_harness.dossier_caption import (
    CAPTION2_SYSTEM,
    DossierCaptionError,
    caption2_prompt,
    caption2_variants,
    render_caption2,
)


def _measurements() -> dict:
    """Synthetic fixture mirroring tests/test_dossier.py shapes (no live models)."""
    return {
        "proportions": {
            "subject_present": True,
            "between_shoulders": 300.0,
            "between_hips": 190.0,
            "shoulder_hip_ratio": 1.58,
            "shoulder_hip_ratio_abstention_reason": None,
            "torso_length": 220.0,
            "left_leg_length": 260.0,
            "right_leg_length": 255.0,
            "leg_torso_ratio": 1.17,
        },
        "clothing": {
            "subject_present": True,
            "garments": [
                {"class": "upper_clothing", "coverage": 0.35, "dominant_color_name": "blue", "dominant_hex": "#2346ae"},
                {"class": "lower_clothing", "coverage": 0.30, "dominant_color_name": "black", "dominant_hex": "#0f0f0f"},
            ],
        },
        "hair": {
            "subject_present": True,
            "hair_present": True,
            "hair_coverage": 0.18,
            "hair_dominant_color_name": "dark brown",
            "hair_position": "middle",
            "hair_face_extent_ratio": 1.63,
        },
        "skin": {
            "subject_present": True,
            "exposed_skin_present": True,
            "skin_tone_name": "brown",
            "skin_tone_hex": "#926e58",
            "face_tone_name": "dark brown",
            "face_tone_hex": "#745343",
            "skin_coverage": 0.60,
            "face_body_agree": False,
        },
        "lighting": {
            "lighting_measurable": True,
            "luma_band": "moderately lit",
            "dynamic_range_band": "high contrast",
            "shadow_band": "some shadow",
            "surround_band": "backlit rim-lit",
            "light_direction": "from the front-left",
            "mean_luma": 0.40,
            "surround_ratio": 1.49,
            "light_residual": 0.24,
        },
        # object-relations dimension used by the before/after (specialist) tests
        "object_relations": {
            "count": 1,
            "count_band": "sparse",
            "placement_band": "foreground",
            "classes": ["skateboard"],
            "class_counts": {"skateboard": 1},
            "n_front": 1,
            "n_behind": 0,
            "n_mixed": 0,
            "box_threshold": 0.25,
            "text_threshold": 0.25,
            "detections": [
                {"label": "skateboard", "box": [0.1, 0.2, 0.3, 0.4], "score": 0.6}
            ],
        },
        "determinations": {
            "schema_version": 1,
            "subject": {"n_detections": 1, "detector_anomaly": "none", "note": "x"},
            "body_parts_visible": [
                {"part": "face", "pixel_frac": 0.2, "kp_conf": 0.8},
                {"part": "torso", "pixel_frac": 0.1, "kp_conf": 0.7},
            ],
            "orientation": {"upright_deg": 16.9},
            "relations": ["left arm extended downward", "face turned toward camera"],
        },
    }


def _dossier() -> dict:
    return assemble_dossier(image_id="item-1", **_measurements())


def _fake_backend(prompt: str) -> str:
    """Deterministic fake aggregator: emits a fixed prose caption."""
    return (
        "A woman with long dark hair wears a blue top and black bottoms, "
        "her skin tone brown, lit by soft key light from the front-left, "
        "with a skateboard in the foreground. This is a grounded caption."
    )


def test_render_caption2_deterministic_single_paragraph() -> None:
    out = render_caption2(_dossier())
    assert out["via"] == "deterministic"
    text = out["text"]
    # single flow of prose, no section headers, no bullet markers, no prefixes
    assert "\n" not in text
    assert "## " not in text
    assert "[body-type" not in text
    assert "body type:" not in text
    # ends sentences
    assert text.rstrip().endswith(".")
    # token accounting present and deterministic
    assert out["token_count"] > 0
    again = render_caption2(_dossier())
    assert again["text"] == text
    assert again["token_count"] == out["token_count"]


def test_render_caption2_keeps_scale_invariant_content() -> None:
    text = render_caption2(_dossier())["text"]
    # ratios / bands / color NAMES are verbalized (scale-invariant)
    assert "1.58" in text
    assert "1.17" in text
    assert "dark brown" in text
    # absolute pixel keys and hex triplets are NOT verbalized (honesty)
    assert "between_shoulders" not in text
    assert "#2346ae" not in text
    assert "px" not in text.lower()


def test_render_caption2_raises_on_empty() -> None:
    empty_dossier = {"sections": {}, "evidence_ids": []}
    with pytest.raises(DossierCaptionError, match="empty"):
        render_caption2(empty_dossier)


def test_render_caption2_aggregator_path_returns_backend_text() -> None:
    out = render_caption2(_dossier(), backend=_fake_backend)
    assert out["via"] == "aggregator"
    assert "This is a grounded caption." in out["text"]
    assert out["token_count"] > 0


def test_caption2_prompt_contains_system_and_ground_truth() -> None:
    d = _dossier()
    prompt = caption2_prompt(d)
    assert CAPTION2_SYSTEM.split(".")[0] in prompt
    # ground-truth claims must be in the prompt
    assert "GROUND-TRUTH DETERMINATIONS" in prompt
    assert "skateboard" in prompt
    assert "shoulder:hip" in prompt or "1.58" in prompt


def test_caption2_prompt_never_verbalizes_payload_hex() -> None:
    d = _dossier()
    prompt = caption2_prompt(d)
    # the payload block is explicitly labelled non-verbalized and contains machine
    # values, but the ground-truth block itself must not leak hex/pixel keys.
    gt_section = prompt.split("GROUND-TRUTH DETERMINATIONS:")[1].split("MACHINE-READABLE")[0]
    assert "#" not in gt_section
    assert "between_shoulders" not in gt_section


def test_caption2_variants_before_after_specialist_axis() -> None:
    d = _dossier()
    variants = caption2_variants(d, specialist_evidence_id="object-relations:v1")
    assert variants["exclusion_seen"] is True
    before = variants["before"]["text"]
    after = variants["after"]["text"]
    # The specialist's skateboard claim is dropped in `before` and present in `after`.
    assert "skateboard" not in before
    assert "skateboard" in after
    # Same deterministic path otherwise -> the axis is ONLY the specialist.
    assert variants["before"]["via"] == variants["after"]["via"] == "deterministic"
    # They differ (non-degenerate) and both are honest single paragraphs.
    assert before != after
    assert "## " not in before and "## " not in after


def test_caption2_variants_exclusion_seen_false_when_specialist_absent() -> None:
    # Remove the object-relations section -> exclusion is a no-op, flagged honestly.
    d = _dossier()
    d["sections"] = {k: v for k, v in d["sections"].items() if k != "object-relations"}
    d["evidence_ids"] = [e for e in d["evidence_ids"] if e != "object-relations:v1"]
    variants = caption2_variants(d, specialist_evidence_id="object-relations:v1")
    assert variants["exclusion_seen"] is False
    assert variants["before"]["text"] == variants["after"]["text"]


def test_caption2_monotonic_before_after_with_aggregator() -> None:
    # With a backend, before/after keep the same aggregator; the withdrawn claim
    # is only in the prompt, so the fake emits identical prose (caller reports).
    d = _dossier()
    variants = caption2_variants(d, specialist_evidence_id="object-relations:v1", backend=_fake_backend)
    assert variants["before"]["via"] == variants["after"]["via"] == "aggregator"
