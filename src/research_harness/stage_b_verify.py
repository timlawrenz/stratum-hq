"""Read-only structural verification of a completed Stage-B output root.

This is an observer-side check only. It never invokes a model, GPU, or
scheduler; never mutates any corpus or derived tree; and never re-runs
Stage B. It re-derives the same bindings a reviewer needs to confirm that
a completed run's records, plan, and provenance are structurally sound
*before* any claim-support self-audit or adversarial review is accepted.

The verifier treats a `records.jsonl` row's evidence/prompt/input-view
fingerprints as binding to the frozen execution plan found in the same
root. It does not fabricate a metric judgment: a run with all review rows
`PENDING` is reported as unreviewed, never as PASS/FAIL.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import ContractError


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_fingerprint(value: Mapping[str, Any], field: str) -> str:
    payload = {key: item for key, item in value.items() if key != field}
    return _sha256(_canonical_json(payload).encode("utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"unable to read {path}: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return parsed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError(f"unable to read {path}: {exc}") from exc
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid JSON line {index + 1} in {path}: {exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise ContractError(f"line {index + 1} in {path} must contain a JSON object")
        rows.append(parsed)
    return rows


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise ContractError(f"missing Stage-B output artifact: {path}")
    return path


def verify_stage_b_output_root(root: Path) -> dict[str, Any]:
    """Structurally verify a completed Stage-B output root.

    Returns a structured report. Raises ContractError on any binding or
    completeness failure. The report records the run's declared status and
    review state; it never asserts an empirical PASS/FAIL.
    """
    if not root.exists():
        raise ContractError(f"Stage-B output root must be an existing directory: {root}")
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"Stage-B output root is unavailable: {root}: {exc}") from exc
    if not root.is_dir():
        raise ContractError(f"Stage-B output root must be an existing directory: {root}")

    plan = _read_json(_require_file(root / "stage-b-plan.json"))
    run = _read_json(_require_file(root / "run-provenance.json"))
    scheduler = _read_json(_require_file(root / "scheduler-provenance.json"))
    records = _read_jsonl(_require_file(root / "records.jsonl"))
    review = _read_jsonl(_require_file(root / "review-queue.jsonl"))

    checks: dict[str, int] = {"ok": 0, "bad": 0}
    findings: list[str] = []

    def check(condition: bool, label: str) -> None:
        checks["ok" if condition else "bad"] += 1
        if not condition:
            findings.append(label)

    # 1. Plan fingerprint binds its content.
    check(
        _canonical_fingerprint(plan, "comparison_plan_fingerprint")
        == plan.get("comparison_plan_fingerprint"),
        "comparison_plan_fingerprint must bind plan content",
    )

    # 2. Records mirror the plan fingerprint and candidate manifest fingerprint.
    check(
        run.get("comparison_plan_fingerprint") == plan.get("comparison_plan_fingerprint"),
        "run-provenance comparison_plan_fingerprint must match plan",
    )
    check(
        run.get("candidate_manifest_fingerprint") == plan.get("candidate_manifest_fingerprint"),
        "run-provenance candidate_manifest_fingerprint must match plan",
    )

    # 3. Condition set in records matches the plan condition set exactly.
    plan_conditions = {condition["id"] for condition in plan.get("conditions", []) if isinstance(condition, dict)}
    record_conditions = {record.get("condition_id") for record in records}
    check(record_conditions == plan_conditions, "records condition set must match plan condition set")

    # 4. Every record is complete and has at least both core evidence inputs.
    for record in records:
        check(
            isinstance(record.get("record_id"), str)
            and isinstance(record.get("image_id"), str)
            and isinstance(record.get("condition_id"), str)
            and isinstance(record.get("source_sha256"), str)
            and isinstance(record.get("caption"), str)
            and record.get("caption"),
            f"{record.get('record_id')}: record must be complete and non-empty",
        )
        check(
            isinstance(record.get("evidence_input_artifact_sha256"), dict)
            or isinstance(record.get("selected_evidence_input_artifact_sha256"), dict),
            f"{record.get('record_id')}: evidence input artifacts must be recorded",
        )

    # 5. Evidence fingerprints are canonical (excluding asserted fingerprint member).
    for record in records:
        evidence = record.get("evidence")
        check(
            isinstance(evidence, dict)
            and isinstance(evidence.get("fingerprint"), str)
            and _canonical_fingerprint(evidence, "fingerprint") == evidence["fingerprint"],
            f"{record.get('record_id')}: evidence.fingerprint must bind canonical evidence content",
        )

    # 6. Prompt and input-view fingerprints bind to the frozen plan.
    plan_by_condition = {
        condition["id"]: condition
        for condition in plan.get("conditions", [])
        if isinstance(condition, dict)
    }
    for record in records:
        condition = plan_by_condition.get(record.get("condition_id"))
        if condition is None:
            check(False, f"{record.get('record_id')}: condition missing from plan")
            continue
        prompt = record.get("prompt") or {}
        view = record.get("input_view") or {}
        check(
            prompt.get("fingerprint") == (condition.get("prompt") or {}).get("fingerprint"),
            f"{record.get('record_id')}: prompt.fingerprint must bind plan prompt",
        )
        check(
            view.get("fingerprint") == (condition.get("input_view") or {}).get("fingerprint"),
            f"{record.get('record_id')}: input_view.fingerprint must bind plan input view",
        )
        check(
            isinstance(prompt.get("rendered_sha256"), str)
            and isinstance(prompt.get("rendered_text"), str)
            and str(_sha256(prompt["rendered_text"].encode("utf-8"))) == prompt["rendered_sha256"],
            f"{record.get('record_id')}: rendered_sha256 must bind rendered_text",
        )

    # 7. Output files exist with non-empty content; caption digest covers the recorded text.
    for record in records:
        rel_path = record.get("output_relative_path")
        output_path = root / rel_path if isinstance(rel_path, str) else None
        if not isinstance(rel_path, str) or not output_path or not output_path.is_file():
            check(False, f"{record.get('record_id')}: output file missing")
            continue
        text = output_path.read_text(encoding="utf-8")
        check(text.strip() != "", f"{record.get('record_id')}: output file must be non-empty")
        check(
            _sha256(record.get("caption", "").encode("utf-8")) == record.get("caption_sha256"),
            f"{record.get('record_id')}: caption_sha256 must bind recorded caption",
        )

    # 8. Every review-queue row matches a record and is unreviewed (no fabricated verdict).
    record_ids = {record.get("record_id") for record in records}
    check(len(review) == len(records), "review-queue row count must equal record count")
    for row in review:
        check(row.get("record_id") in record_ids, f"{row.get('record_id')}: review row must match a record")
        review_status = row.get("review_status")
        verdict = row.get("verdict")
        if review_status in {"unreviewed", "pending"}:
            check(
                verdict == "PENDING",
                f"{row.get('record_id')}: an unreviewed run must keep verdict PENDING (no fabricated PASS/FAIL)",
            )
        else:
            check(
                False,
                f"{row.get('record_id')}: review_status must remain unreviewed until self-audit and adversarial review complete",
            )

    # 9. Scheduler provenance is recorded.
    scheduler_result = scheduler.get("scheduler_result") or {}
    check(
        isinstance(scheduler_result, dict) and scheduler_result.get("status") in {"completed", "failed", "completed_review_pending", "completed_review_pending_v1"},
        "scheduler-provenance must record a lifecycle status",
    )

    report = {
        "root": str(root),
        "record_count": len(records),
        "review_pending_count": sum(
            1 for row in review if row.get("review_status") in {"unreviewed", "pending"}
        ),
        "checks_passed": checks["ok"],
        "checks_failed": checks["bad"],
        "findings": findings,
        "run_status": run.get("status"),
        "scheduler_status": scheduler_result.get("status") if isinstance(scheduler_result, dict) else None,
        "semantic_verdict": run.get("semantic_verdict"),
        "verified": checks["bad"] == 0,
    }
    if report["findings"]:
        raise ContractError(
            "Stage-B output root structural verification failed: " + "; ".join(report["findings"])
        )
    return report


def check_stage_b_self_audit_readiness(root: Path) -> dict[str, Any]:
    """Observer-only readiness report for a completed Stage-B run's pre-registered metric self-audit.

    The frozen plan declares a `metric_self_audit` with a `known_case_item_id` and a
    `null_output_id`. This check reports, read-only, whether those fixtures are actually
    materialized by the run's records so the pre-registered known-case and null/abstention
    self-audit steps can execute as specified. It never fabricates a verdict: a missing fixture
    means the corresponding self-audit step is not executable as pre-registered, which is a
    metric-readiness finding, not a model/quality PASS or FAIL.

    The check is deliberately independent of `verify_stage_b_output_root` structural binding:
    a run can be structurally sound (all fingerprints bind, all files present) while still
    lacking the declared known-case or null-output fixture needed by the self-audit.
    """
    if not root.exists():
        raise ContractError(f"Stage-B output root must be an existing directory: {root}")
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"Stage-B output root is unavailable: {root}: {exc}") from exc
    if not root.is_dir():
        raise ContractError(f"Stage-B output root must be an existing directory: {root}")

    plan = _read_json(_require_file(root / "stage-b-plan.json"))
    records = _read_jsonl(_require_file(root / "records.jsonl"))

    audit = plan.get("metric_self_audit") or {}
    known_case = audit.get("known_case_item_id")
    null_output = audit.get("null_output_id")

    if not isinstance(known_case, str) or not known_case:
        raise ContractError("comparison parity plan metric_self_audit.known_case_item_id must be declared")
    if not isinstance(null_output, str) or not null_output:
        raise ContractError("comparison parity plan metric_self_audit.null_output_id must be declared")

    record_ids = {record.get("record_id") for record in records}
    image_ids = {record.get("image_id") for record in records}
    empty_records = [
        record.get("record_id")
        for record in records
        if not isinstance(record.get("caption"), str) or not record["caption"].strip()
    ]

    # The known-case self-audit scores one generated output against its original selected
    # source, so it is materialized when the declared item id is a record's image_id.
    known_case_present = known_case in image_ids
    known_case_record_keys = sorted(
        [r["record_id"] for r in records if r.get("image_id") == known_case and isinstance(r.get("record_id"), str)]
    )

    # The null self-audit scores the declared empty-caption null output and confirms it is
    # recorded as an abstention. It is materialized when the declared id is itself a record_id
    # or when at least one record carries an empty caption. Absence means the step cannot run
    # as pre-registered on this run's records.
    null_record_present = null_output in record_ids
    null_empty_present = len(empty_records) > 0
    null_present = null_record_present or null_empty_present

    missing = []
    if not known_case_present:
        missing.append(f"known_case_item_id {known_case!r} is not any record image_id")
    if not null_present:
        missing.append(
            f"null_output_id {null_output!r} is neither a record_id nor materialized as an empty-caption record"
        )

    return {
        "root": str(root),
        "record_count": len(records),
        "known_case_item_id": known_case,
        "known_case_present": known_case_present,
        "known_case_record_keys": known_case_record_keys,
        "null_output_id": null_output,
        "null_record_present": null_record_present,
        "empty_caption_record_count": len(empty_records),
        "null_present": null_present,
        "missing_fixtures": missing,
        "summary": (
            "self-audit fixtures materialized"
            if not missing
            else "self-audit fixtures missing; pre-registered step not executable as specified: " + "; ".join(missing)
        ),
        "readiness_verdict": "READY" if not missing else "NOT_READY",
    }
