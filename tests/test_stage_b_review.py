"""TDD coverage for the independent Stage-B adversarial reviewer pass."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_harness.stage_b_review import (
    ReviewSettings,
    StageBReviewError,
    _parse_review_json,
    _aggregate,
    build_review_plan,
)


def _fake_review_records() -> list[dict]:
    return [
        {
            "image_id": "img-1",
            "condition_id": "legacy-raw-no-evidence",
            "model": "reviewer-qwen3vl-32b",
            "supported": ["hair described"],
            "unsupported": ["skateboard inside bowl"],
            "omissions": ["tattoo left arm"],
            "contradictions": [],
            "abstentions": [],
        },
        {
            "image_id": "img-1",
            "condition_id": "context-raw-geometry",
            "model": "reviewer-qwen3vl-32b",
            "supported": ["torso angle ~17deg", "left leg downward"],
            "unsupported": [],
            "omissions": [],
            "contradictions": [],
            "abstentions": [],
        },
    ]


def test_parse_review_json_normalizes_fields() -> None:
    raw = {
        "supported_claims": ["a", "b"],
        "unsupported_claims": [],
        "omissions": ["c"],
        "contradictions": [],
        "abstentions": ["x"],
    }
    parsed = _parse_review_json(raw)
    assert parsed["supported"] == ["a", "b"]
    assert parsed["omissions"] == ["c"]
    assert set(parsed) == {"supported", "unsupported", "omissions", "contradictions", "abstentions"}
    for key, value in parsed.items():
        assert isinstance(value, list)
        assert all(isinstance(item, str) for item in value)


def test_parse_review_json_rejects_malformed() -> None:
    with pytest.raises(StageBReviewError):
        _parse_review_json({"supported_claims": "not-a-list"})
    with pytest.raises(StageBReviewError):
        _parse_review_json({})


def test_aggregate_groups_by_condition() -> None:
    aggregate = _aggregate(_fake_review_records())
    assert set(aggregate) == {"legacy-raw-no-evidence", "context-raw-geometry"}
    row = aggregate["legacy-raw-no-evidence"]
    assert row["supported"] == ["hair described"]
    assert row["unsupported"] == ["skateboard inside bowl"]
    assert row["omissions"] == ["tattoo left arm"]
    assert row["counts"] == {"items": 1, "supported": 1, "unsupported": 1, "omissions": 1, "contradictions": 0, "abstentions": 0}


def test_build_review_plan_is_checked(tmp_path: Path) -> None:
    # Minimal synthetic run root: records + stage-b-plan + outputs + evidence marker.
    run_root = tmp_path / "run"
    (run_root / "outputs" / "legacy-raw-no-evidence").mkdir(parents=True)
    (run_root / "outputs" / "context-raw-geometry").mkdir(parents=True)
    records = [
        {"image_id": "img-1", "condition_id": "legacy-raw-no-evidence",
         "source_relative_path": "img-1.jpg", "source_sha256": "a" * 64,
         "caption": "one", "generation_fingerprint": "fp1"},
        {"image_id": "img-1", "condition_id": "context-raw-geometry",
         "source_relative_path": "img-1.jpg", "source_sha256": "a" * 64,
         "caption": "two", "generation_fingerprint": "fp1"},
    ]
    (run_root / "records.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )
    (run_root / "stage-b-plan.json").write_text(json.dumps({"comparison_plan_fingerprint": "fp1"}))
    (run_root / "outputs" / "legacy-raw-no-evidence" / "img-1.txt").write_text("one\n")
    (run_root / "outputs" / "context-raw-geometry" / "img-1.txt").write_text("two\n")

    settings = ReviewSettings(model_name="gemma4:e4b",
                              digest="c6eb396dbd59",
                              endpoint="http://127.0.0.1:11434/api/generate",
                              temperature=0.0, seed=20260804, num_predict=1400,
                              review_items="all")
    plan = build_review_plan(settings, run_root, candidate_fingerprint="cand1")
    assert plan["review_plan_id"] == "stage-b-adversarial-review-v1"
    assert plan["reviewer_model_id"] == "local-ollama-gemma4-e4b-c6eb396dbd59"
    assert plan["independent_of_generator"] is True
    assert plan["item_count"] == 1
    assert plan["condition_count"] == 2
    assert plan["target_n"] == 2  # 1 item x 2 conditions
