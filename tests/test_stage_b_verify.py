"""Synthetic-fixture tests for the observer-only Stage-B output verifier."""

from __future__ import annotations

import hashlib
import json

import pytest

from research_harness import ContractError
from research_harness.stage_b_verify import (
    check_stage_b_evidence_axis,
    check_stage_b_self_audit_readiness,
    verify_stage_b_output_root,
)


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


def build_root(
    tmp_path, *, corrupt=None, materialize_null=False, empty_caption_as_null=False
) -> tuple:
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
            if condition["evidence"]["kind"] == "specialist_bundle":
                evidence_payload = {
                    "subject": {"id": image_id, "frame_frac": 0.5},
                    "relations": [{"part": "face", "kp_conf": 0.9}],
                    "image_id": image_id,
                }
            else:
                evidence_payload = None
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
                "evidence_payload": evidence_payload,
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

    # Optionally materialize the declared null-output self-audit fixture. The null fixture may
    # either be its own empty-caption record (record_id == null_output_id) or an empty-caption
    # record that the null self-audit scores as an abstention.
    if materialize_null:
        null_caption = "" if empty_caption_as_null else "This is a deliberately non-empty null sentinel caption."
        null_cid = conditions[0]["id"]
        null_record = {
            "schema_version": 1,
            "record_id": plan["metric_self_audit"]["null_output_id"],
            "image_id": "img-a",
            "source_relative_path": "img-a.jpg",
            "source_sha256": _sha("img-a"),
            "condition_id": null_cid,
            "input_view": dict(conditions[0]["input_view"]),
            "prompt": {
                **conditions[0]["prompt"],
                "rendered_sha256": _sha("null"),
                "rendered_text": "null",
            },
            "evidence": dict(conditions[0]["evidence"]),
            "evidence_payload": None,
            "selected_evidence_input_artifact_sha256": {"pose2.npy": _sha("p2"), "seg2.npy": _sha("s2")},
            "output_relative_path": f"outputs/{null_cid}/{plan['metric_self_audit']['null_output_id']}.txt",
            "caption_sha256": _sha(null_caption),
            "caption": null_caption,
            "caption_word_count": len(null_caption.split()),
        }
        records.append(null_record)
        review.append(
            {
                "record_id": null_record["record_id"],
                "image_id": "img-a",
                "condition_id": null_cid,
                "output_relative_path": null_record["output_relative_path"],
                "review_status": "unreviewed",
                "supported_claims": [],
                "unsupported_claims": [],
                "omissions": [],
                "contradictions": [],
                "abstentions": [],
                "verdict": "PENDING",
            }
        )
        out = root / null_record["output_relative_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(null_caption + "\n", encoding="utf-8")
        run["record_count"] = len(records)

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
    if corrupt == "evidence-boilerplate":
        records_mutated = list(records)
        for record in records_mutated:
            if record["condition_id"] == "context-raw-geometry":
                record["evidence_payload"] = {
                    "subject": {"id": "img-a", "frame_frac": 0.5},
                    "relations": [{"part": "face", "kp_conf": 0.9}],
                    "image_id": "img-a",
                }
        with (root / "records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records_mutated:
                handle.write(_canonical_json(record) + "\n")
    if corrupt == "evidence-payload-on-noevidence":
        records_mutated = list(records)
        for record in records_mutated:
            if record["condition_id"] == "context-raw-no-evidence":
                record["evidence_payload"] = {
                    "subject": {"id": record["image_id"], "frame_frac": 0.5},
                    "relations": [{"part": "face", "kp_conf": 0.9}],
                    "image_id": record["image_id"],
                }
        with (root / "records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records_mutated:
                handle.write(_canonical_json(record) + "\n")
    if corrupt == "evidence-missing-inputs":
        records_mutated = list(records)
        for record in records_mutated:
            if record["condition_id"] == "context-raw-geometry":
                record["selected_evidence_input_artifact_sha256"] = {"pose2.npy": _sha("p2")}
        with (root / "records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records_mutated:
                handle.write(_canonical_json(record) + "\n")
    if corrupt == "evidence-empty-payload":
        records_mutated = list(records)
        for record in records_mutated:
            if record["condition_id"] == "context-raw-geometry":
                record["evidence_payload"] = {}
        with (root / "records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records_mutated:
                handle.write(_canonical_json(record) + "\n")

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


def test_verify_accepts_root_with_materialized_null_fixture(tmp_path):
    root, _plan = build_root(tmp_path, materialize_null=True)
    report = verify_stage_b_output_root(root)
    assert report["verified"] is True
    assert report["record_count"] == 9


def test_readiness_flags_missing_null_fixture(tmp_path):
    root, _plan = build_root(tmp_path)
    report = check_stage_b_self_audit_readiness(root)
    assert report["readiness_verdict"] == "NOT_READY"
    assert report["known_case_present"] is True
    assert report["known_case_item_id"] == "img-a"
    assert report["null_present"] is False
    assert report["null_record_present"] is False
    assert report["empty_caption_record_count"] == 0
    assert any("null_output_id" in finding for finding in report["missing_fixtures"])
    assert "not executable as specified" in report["summary"]


def test_readiness_ready_when_null_fixture_materialized(tmp_path):
    root, _plan = build_root(tmp_path, materialize_null=True)
    report = check_stage_b_self_audit_readiness(root)
    assert report["readiness_verdict"] == "READY"
    assert report["known_case_present"] is True
    assert report["null_present"] is True
    assert report["null_record_present"] is True
    assert report["missing_fixtures"] == []
    assert report["summary"] == "self-audit fixtures materialized"


def test_readiness_ready_for_empty_caption_abstention_record(tmp_path):
    root, _plan = build_root(tmp_path, materialize_null=True, empty_caption_as_null=True)
    report = check_stage_b_self_audit_readiness(root)
    assert report["readiness_verdict"] == "READY"
    assert report["null_record_present"] is True
    assert report["empty_caption_record_count"] == 1


def test_readiness_raises_on_undeclared_fixture(tmp_path):
    root, plan = build_root(tmp_path)
    plan_mutated = dict(plan)
    plan_mutated["metric_self_audit"] = {"known_case_item_id": "img-a"}
    (root / "stage-b-plan.json").write_text(
        json.dumps(plan_mutated, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="null_output_id"):
        check_stage_b_self_audit_readiness(root)


def test_evidence_axis_ok_on_wellformed_root(tmp_path):
    root, _plan = build_root(tmp_path)
    report = check_stage_b_evidence_axis(root)
    assert report["evidence_axis_ok"] is True
    assert report["checks_failed"] == 0
    assert report["evidence_condition_ids"] == ["context-raw-geometry"]
    assert report["no_evidence_condition_ids"] == [
        "legacy-bucketed-no-evidence",
        "legacy-raw-no-evidence",
        "context-raw-no-evidence",
    ]
    assert report["evidence_record_count"] == 2
    assert report["no_evidence_record_count"] == 6


def test_evidence_axis_rejects_boilerplate_payloads(tmp_path):
    root, _plan = build_root(tmp_path, corrupt="evidence-boilerplate")
    report = check_stage_b_evidence_axis(root)
    assert report["evidence_axis_ok"] is False
    assert any("distinct per-image" in finding for finding in report["findings"])


def test_evidence_axis_rejects_payload_on_noevidence(tmp_path):
    root, _plan = build_root(tmp_path, corrupt="evidence-payload-on-noevidence")
    report = check_stage_b_evidence_axis(root)
    assert report["evidence_axis_ok"] is False
    assert any("null evidence_payload" in finding for finding in report["findings"])


def test_evidence_axis_rejects_missing_core_inputs(tmp_path):
    root, _plan = build_root(tmp_path, corrupt="evidence-missing-inputs")
    report = check_stage_b_evidence_axis(root)
    assert report["evidence_axis_ok"] is False
    assert any("pose2.npy and seg2.npy" in finding for finding in report["findings"])


def test_evidence_axis_rejects_empty_payload(tmp_path):
    root, _plan = build_root(tmp_path, corrupt="evidence-empty-payload")
    report = check_stage_b_evidence_axis(root)
    assert report["evidence_axis_ok"] is False
    assert any("non-empty evidence_payload" in finding for finding in report["findings"])


def test_evidence_axis_rejects_missing_root(tmp_path):
    with pytest.raises(ContractError, match="must be an existing directory"):
        check_stage_b_evidence_axis(tmp_path / "does-not-exist")
