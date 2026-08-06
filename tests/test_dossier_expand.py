"""TDD coverage for the honest evidence-bound dossier expansion (arm #36 pre-gate)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_harness.dossier import DossierError, assemble_dossier, build_evidence_payload
from research_harness.dossier_expand import (
    ExpansionError,
    expand_dossier,
    floor_gap_analysis,
    honesty_check,
)


def _program() -> dict:
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "research" / "program.json").read_text(encoding="utf-8"))


def _measurements() -> dict:
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
            "hair_dominant_hex": "#4d3b34",
            "hair_frame_coverage": 0.02,
            "hair_position": "middle",
            "hair_face_extent_ratio": 1.6,
        },
        "skin": {
            "subject_present": True,
            "exposed_skin_present": True,
            "skin_tone_name": "brown",
            "skin_tone_hex": "#926e58",
            "face_tone_name": "brown",
            "face_tone_hex": "#926e58",
            "skin_coverage": 0.60,
            "face_body_agree": True,
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


def _expanded() -> dict:
    m = _measurements()
    dossier = assemble_dossier(image_id="item-1", **m)
    payload = build_evidence_payload(image_id="item-1", source_sha256="a" * 64, **m)
    return expand_dossier(dossier, payload)


def test_expansion_grows_tokens_and_is_deterministic() -> None:
    e = _expanded()
    assert e["token_count"] > e["base_token_count"], "expansion must add bounded honest detail"
    assert e["expansion_multiplier"] and e["expansion_multiplier"] > 1.0
    e2 = _expanded()
    assert e["token_count"] == e2["token_count"]  # deterministic


def test_expansion_restates_but_does_not_verbalize_pixels_or_hex() -> None:
    e = _expanded()
    violations = honesty_check(e["expanded_text"])
    assert violations == [], f"honesty violations: {violations}"
    assert "between_shoulders" not in e["expanded_text"]
    assert "#2346ae" not in e["expanded_text"]  # hex stays machine-readable


def test_expansion_preserves_evidence_linkage() -> None:
    e = _expanded()
    exp_dossier = e["expanded_dossier"]
    for section, claims in (exp_dossier.get("sections") or {}).items():
        for claim in claims:
            assert claim["evidence_ids"], f"{section} claim lost evidence linkage: {claim!r}"
    assert "body-type-proportions:v1" in exp_dossier["evidence_ids"]
    assert "relational-determinations:v1" in exp_dossier["evidence_ids"]
    # provenance detail is present
    assert any("Evidence source:" in c["text"] for claims in exp_dossier["sections"].values() for c in claims)


def test_expansion_of_abstaining_item_keeps_abstention() -> None:
    m = _measurements()
    m["proportions"] = {"subject_present": False}
    m["clothing"] = {"subject_present": False}
    dossier = assemble_dossier(image_id="item-1", **m)
    payload = build_evidence_payload(image_id="item-1", **m)
    e = expand_dossier(dossier, payload)
    text = e["expanded_text"].lower()
    assert "abstain" in text  # abstentions must survive expansion, never become facts


def test_honesty_check_detects_pixel_and_hex_violations() -> None:
    assert any("px" in v or "pixel" in v.lower() for v in honesty_check("leg spans 240 px at shoulder level"))
    assert any("hex" in v.lower() for v in honesty_check("garment color #2346ae"))
    assert honesty_check("shoulder:hip ratio 1.58, coverage 0.35") == []


def test_floor_gap_analysis_reports_honest_unreachability() -> None:
    prog = _program()
    expanded_floor = prog["representation"]["expanded_dossier_target_tokens"]
    g = floor_gap_analysis(
        expanded_prose_tokens=3_000,
        payload_tokens=2_000,
        claim_count=21,
        expanded_floor=expanded_floor,
    )
    assert g["total_dossier_record_tokens"] == 5_000
    assert g["expanded_floor_reached"] is False
    assert g["max_honest_floor_reached"] is False
    assert g["expanded_floor_gap"] == expanded_floor - 5_000
    # A truly enormous capture WOULD clear the analytic ceiling (sanity of the bound)
    g2 = floor_gap_analysis(expanded_prose_tokens=120_000, payload_tokens=0, claim_count=1, expanded_floor=expanded_floor)
    assert g2["expanded_floor_reached"] is True


def test_build_compression_bundle_still_refuses_under_budget_after_expansion() -> None:
    """The honesty gate must NOT be weakened by the expander: even an expanded (but still
    small) dossier must be refused by build_compression_bundle."""
    from research_harness.dossier import build_compression_bundle, compress_dossier_to_context

    m = _measurements()
    dossier = assemble_dossier(image_id="item-1", **m)
    payload = build_evidence_payload(image_id="item-1", **m)
    expanded = expand_dossier(dossier, payload)
    context = compress_dossier_to_context(expanded["expanded_dossier"])
    with pytest.raises(DossierError, match="expanded_dossier.token_count is below the program minimum"):
        build_compression_bundle(
            image_id="item-1",
            dossier=expanded["expanded_dossier"],
            context=context,
            program=_program(),
        )
