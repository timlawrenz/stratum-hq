"""Synthetic-fixture tests for the observer-only Stage-B output verifier."""

from __future__ import annotations

import hashlib
import json

import pytest

from research_harness import ContractError
from research_harness.stage_b_verify import verify_stage_b_output_root


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value, field="fingerprint") -> str:
    return _sha(_canonical_json({k: v for k, v in value.items() if k != field}))


def _conditions():
    conds = [
        {
            "id": "legacy-bucketed-no-evidence",
            "input_view": {"id": "legacy-bucketed-crop-view-v1"},
            "prompt": {"id": "legacy-caption-prompt-v1"},
            "evidence": {"id": "no-specialist-evidence-v1", "kind": "none"},
        },
        {
            "id": "legacy-raw-no-evidence",
            "input_view": {"id": "legacy-raw-view-v1"},
            "prompt": {"id": "legacy-caption-prompt-v1"},
            "evidence": {"id": "no-specialist-evidence-v1", "kind": "none"},
        },
        {
            "id": "context-raw-no-evidence",
            "input_view": {"id": "legacy-raw-view-v1"},
            "prompt": {"id": "context-caption-prompt-v1"},
            "evidence": {"id": "no-specialist-evidence-v1", "kind": "none"},
        },
        {
            "id": "context-raw-geometry",
            "input_view": {"id": "legacy-raw-view-v1"},
            "prompt": {"id": "context-caption-prompt-v1"},
            "evidence": {"id": "in-memory-geometry-v1", "kind": "specialist_bundle"},
        },
    ]
    for condition in conds:
        condition["input_view"]["fingerprint"] = _fingerprint(condition["input_view"])
        condition["prompt"]["fingerprint"] = _fingerprint(condition["prompt"])
        condition["evidence"]["fingerprint"] = _fingerprint(condition["evidence"])
    return conds


def build_root(tmp_path, *, corrupt=None) -> tuple:
    """Build a synthetic Stage-B output root and return (root, plan)."""
    cap_a = "A synthetic caption for image a."
    cap_b = "A synthetic caption for image b."
    image_ids = ["img-a", "img-b"]
    conditions = _conditions()

    plan = {
        "schema_version": 1,
        "kind": "stage-b-comparison-plan",
        "comparison_plan_id": "synthetic-stage-b-plan-v1",
        "candidate_manifest_fingerprint": "0" * 64,
        "conditions": conditions,
        "metric_self_audit": {
            "known_case_item_id": "img-a",
            "null_output_id": "empty-caption-null-v1",
        },
    }
    plan["comparison_plan_fingerprint"] = _fingerprint(plan, "comparison_plan_fingerprint")

    records = []
    review = []
    for image_id, caption in zip(image_ids, [cap_a, cap_b]):
        for condition in conditions:
            cid = condition["id"]
            rendered_prompt = f"Render for {cid}"
            record = {
                "schema_version": 1,
                "record_id": f"{cid}:{image_id}",
                "image_id": image_id,
                "source_relative_path": f"{image_id}.jpg",
                "source_sha256": _sha(image_id),
                "condition_id": cid,
                "input_view": dict(condition["input_view"]),
                "prompt": {
                    **condition["prompt"],
                    "rendered_sha256": _sha(rendered_prompt),
                    "rendered_text": rendered_prompt,
                },
                "evidence": dict(condition["evidence"]),
                "evidence_payload": {"x": 1} if condition["evidence"]["kind"] == "specialist_bundle" else None,
                "selected_evidence_input_artifact_sha256": {"pose2.npy": _sha("p2"), "seg2.npy": _sha("s2")},
                "output_relative_path": f"outputs/{cid}/{image_id}.txt",
                "caption_sha256": _sha(caption),
                "caption": caption,
                "caption_word_count": len(caption.split()),
            }
            records.append(record)
            review.append(
                {
                    "record_id": record["record_id"],
                    "image_id": image_id,
                    "condition_id": cid,
                    "output_relative_path": record["output_relative_path"],
                    "review_status": "unreviewed",
                    "supported_claims": [],
                    "unsupported_claims": [],
                    "omissions": [],
                    "contradictions": [],
                    "abstentions": [],
                    "verdict": "PENDING",
                }
            )

    run = {
        "schema_version": 1,
        "status": "PENDING_INDEPENDENT_REVIEW",
        "semantic_verdict": "PENDING",
        "candidate_manifest_fingerprint": plan["candidate_manifest_fingerprint"],
        "comparison_plan_fingerprint": plan["comparison_plan_fingerprint"],
        "record_count": len(records),
        "metric_self_audit": {"status": "PENDING_HUMAN_SELF_AUDIT"},
    }
    scheduler = {
        "job_id": "synthetic-stage-b",
        "scheduler_result": {"status": "completed", "started_at_utc": "T", "finished_at_utc": "T"},
    }

    root = tmp_path / "stage-b-run"
    (root / "outputs").mkdir(parents=True)
    for record in records:
        out = root / record["output_relative_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(record["caption"] + "\n", encoding="utf-8")

    def write(name, value):
        (root / name).write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )

    write("stage-b-plan.json", plan)
    write("run-provenance.json", run)
    write("scheduler-provenance.json", scheduler)
    with (root / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_canonical_json(record) + "\n")
    with (root / "review-queue.jsonl").open("w", encoding="utf-8") as handle:
        for row in review:
            handle.write(_canonical_json(row) + "\n")

    if corrupt == "plan-fingerprint":
        plan_mutated = dict(plan)
        plan_mutated["comparison_plan_fingerprint"] = "1" * 64
        write("stage-b-plan.json", plan_mutated)
    if corrupt == "evidence-fingerprint":
        records_mutated = list(records)
        records_mutated[0]["evidence"]["fingerprint"] = "2" * 64
        with (root / "records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records_mutated:
                handle.write(_canonical_json(record) + "\n")
    if corrupt == "review-scored":
        review_mutated = list(review)
        review_mutated[0]["verdict"] = "PASS"
        with (root / "review-queue.jsonl").open("w", encoding="utf-8") as handle:
            for row in review_mutated:
                handle.write(_canonical_json(row) + "\n")
    if corrupt == "missing-output":
        target = root / "outputs" / conditions[0]["id"] / "img-a.txt"
        target.unlink()

    return root, plan


def test_verify_accepts_wellformed_root(tmp_path):
    root, _plan = build_root(tmp_path)
    report = verify_stage_b_output_root(root)
    assert report["verified"] is True
    assert report["record_count"] == 8
    assert report["review_pending_count"] == 8
    assert report["checks_failed"] == 0


def test_verify_rejects_plan_fingerprint_drift(tmp_path):
    root, _plan = build_root(tmp_path, corrupt="plan-fingerprint")
    with pytest.raises(ContractError, match="comparison_plan_fingerprint"):
        verify_stage_b_output_root(root)


def test_verify_rejects_evidence_fingerprint_drift(tmp_path):
    root, _plan = build_root(tmp_path, corrupt="evidence-fingerprint")
    with pytest.raises(ContractError, match="evidence.fingerprint"):
        verify_stage_b_output_root(root)


def test_verify_rejects_fabricated_review_verdict(tmp_path):
    root, _plan = build_root(tmp_path, corrupt="review-scored")
    with pytest.raises(ContractError, match="review"):
        verify_stage_b_output_root(root)


def test_verify_rejects_missing_output_file(tmp_path):
    root, _plan = build_root(tmp_path, corrupt="missing-output")
    with pytest.raises(ContractError, match="output file missing"):
        verify_stage_b_output_root(root)


def test_verify_rejects_missing_root(tmp_path):
    with pytest.raises(ContractError, match="must be an existing directory"):
        verify_stage_b_output_root(tmp_path / "does-not-exist")
