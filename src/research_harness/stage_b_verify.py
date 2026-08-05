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


def check_stage_b_evidence_axis(root: Path) -> dict[str, Any]:
    """Observer-only report on whether a completed Stage-B run's evidence axis is real.

    The evidence-only contrast compares a no-evidence condition (kind ``none``, null payload)
    against an evidence-bearing condition (kind ``specialist_bundle``). A run only *exercises*
    that axis if its evidence-bearing records carry non-trivial, per-image payload content
    (so the contrast is not two identical empties) and every no-evidence record carries a null
    payload (so the axis is not silently confounded by stray evidence on the baseline side).

    This check is deliberately structural and observer-only: it verifies payload presence,
    per-condition isolation, and per-image distinctness from the records already in the root.
    It does not inspect a source image, run a model, or judge whether the payload is *accurate*;
    semantic claim-support remains the reserved human/adversarial-review step. A report stamped
    here is not an authorization and not a PASS/FAIL.
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

    conditions = plan.get("conditions") or []
    if not isinstance(conditions, list) or not all(isinstance(c, dict) for c in conditions):
        raise ContractError("stage-b-plan.json conditions must be a list of objects")
    if not conditions:
        raise ContractError("stage-b-plan.json must declare at least one condition")

    evidence_condition_ids: list[str] = []
    no_evidence_condition_ids: list[str] = []
    for condition in conditions:
        cid = condition.get("id")
        evidence = condition.get("evidence") or {}
        kind = evidence.get("kind")
        if not isinstance(cid, str) or not cid:
            raise ContractError("each comparison condition must declare a string id")
        if kind == "specialist_bundle":
            evidence_condition_ids.append(cid)
        elif kind == "none":
            no_evidence_condition_ids.append(cid)
        else:
            raise ContractError(
                f"condition {cid!r} must declare evidence.kind 'specialist_bundle' or 'none'"
            )

    if not evidence_condition_ids:
        raise ContractError("stage-b-plan.json must declare at least one evidence-bearing condition")
    if not no_evidence_condition_ids:
        raise ContractError("stage-b-plan.json must declare at least one no-evidence condition")

    by_condition: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        cid = record.get("condition_id")
        if not isinstance(cid, str):
            continue
        by_condition.setdefault(cid, []).append(record)

    # Every evidence-bearing record must carry a non-trivial payload; every no-evidence
    # record must carry a null payload. This guarantees the axis isolates evidence placement.
    bad = 0
    findings: list[str] = []
    checks: dict[str, int] = {"ok": 0, "bad": 0}

    def check(condition: bool, label: str) -> None:
        nonlocal bad
        checks["ok" if condition else "bad"] += 1
        if not condition:
            bad += 1
            findings.append(label)

    for condition_id in sorted(evidence_condition_ids):
        group = by_condition.get(condition_id, [])
        check(len(group) > 0, f"evidence condition {condition_id!r} has no records")
        distinct_payloads: set[str] = set()
        for record in group:
            cid = record.get("condition_id")
            payload = record.get("evidence_payload")
            # Payloads must be real JSON objects (never the 'none' sentinel, null, []).
            check(
                isinstance(payload, dict) and len(payload) > 0,
                f"{record.get('record_id')}: evidence-bearing record must carry a non-empty evidence_payload object",
            )
            if isinstance(payload, dict) and len(payload) > 0:
                distinct_payloads.add(_canonical_json(payload))
            inputs = record.get("selected_evidence_input_artifact_sha256")
            check(
                isinstance(inputs, dict)
                and "pose2.npy" in inputs
                and "seg2.npy" in inputs,
                f"{record.get('record_id')}: evidence-bearing record must record pose2.npy and seg2.npy inputs",
            )
        # Distinctness guards against a boilerplate/bloated copy of one shared payload.
        check(
            len(distinct_payloads) >= 2,
            f"evidence condition {condition_id!r} must carry more than one distinct per-image payload",
        )

    for condition_id in sorted(no_evidence_condition_ids):
        group = by_condition.get(condition_id, [])
        check(len(group) > 0, f"no-evidence condition {condition_id!r} has no records")
        for record in group:
            payload = record.get("evidence_payload")
            check(
                payload is None,
                f"{record.get('record_id')}: no-evidence record must carry a null evidence_payload",
            )
            inputs = record.get("selected_evidence_input_artifact_sha256")
            check(
                isinstance(inputs, dict)
                and "pose2.npy" in inputs
                and "seg2.npy" in inputs,
                f"{record.get('record_id')}: no-evidence record should still list the core inputs it binds",
            )

    return {
        "root": str(root),
        "evidence_condition_ids": evidence_condition_ids,
        "no_evidence_condition_ids": no_evidence_condition_ids,
        "evidence_record_count": sum(len(by_condition.get(c, [])) for c in evidence_condition_ids),
        "no_evidence_record_count": sum(len(by_condition.get(c, [])) for c in no_evidence_condition_ids),
        "checks_passed": checks["ok"],
        "checks_failed": checks["bad"],
        "findings": findings,
        "summary": (
            "evidence axis isolated and materialized" if bad == 0
            else "evidence axis NOT isolated: " + "; ".join(findings)
        ),
        "evidence_axis_ok": bad == 0,
    }


def _token_set(text: str) -> set[str]:
    """Lowercased alphanumeric token set used only for divergence stats."""
    return set(__import__("re").findall(r"[a-zA-Z0-9]+", text.lower()))


def _token_jaccard(left: str, right: str) -> float:
    tokens_left = _token_set(left)
    tokens_right = _token_set(right)
    union = tokens_left | tokens_right
    if not union:
        return 0.0
    return len(tokens_left & tokens_right) / len(union)


def check_stage_b_contrast_divergence(root: Path) -> dict[str, Any]:
    """Observer-only report on whether a completed Stage-B run's declared one-axis contrasts
    produced distinguishable output captions.

    The frozen plan declares one-axis contrasts (a baseline and a variant condition differing
    on exactly one declared axis, e.g. ``input_view``, ``prompt``, or ``evidence``). A run can
    only *support* a declared contrast if its output captions actually differ across that pair:
    a wholesale byte-collapse (every baseline/variant pair identical) would mean the aggregator
    ignored the axis, making the contrast vacuous regardless of how the plan and inputs were
    bound. This check also verifies each condition emitted more than one distinct caption across
    images (no per-condition boilerplate), the output-level twin of the per-image evidence-payload
    distinctness check.

    The check is deliberately structural and observer-only: it measures presence and statistical
    divergence of the *recorded* caption text only. It does not judge caption quality, claim
    support, semantic accuracy, or which output is better; claimed supported/unsupported scoring,
    known-case/null self-audit, and adversarial review remain the reserved human steps. A report
    produced here is not an authorization and not a PASS/FAIL.
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

    plan_condition_ids = {
        condition.get("id")
        for condition in (plan.get("conditions") or [])
        if isinstance(condition, dict) and isinstance(condition.get("id"), str)
    }
    contrasts = plan.get("contrasts")
    if not isinstance(contrasts, list) or not contrasts:
        raise ContractError("stage-b-plan.json must declare at least one one-axis contrast in 'contrasts'")

    by_image_and_condition: dict[tuple[str, str], str] = {}
    for record in records:
        image_id = record.get("image_id")
        condition_id = record.get("condition_id")
        caption = record.get("caption")
        if (
            isinstance(image_id, str)
            and isinstance(condition_id, str)
            and isinstance(caption, str)
        ):
            by_image_and_condition[(image_id, condition_id)] = caption

    checks: dict[str, int] = {"ok": 0, "bad": 0}
    findings: list[str] = []
    contrast_reports: list[dict[str, Any]] = []

    def check(condition: bool, label: str) -> None:
        checks["ok" if condition else "bad"] += 1
        if not condition:
            findings.append(label)

    for contrast in contrasts:
        if not isinstance(contrast, dict):
            check(False, "each contrast must be an object")
            continue
        contrast_id = contrast.get("id")
        baseline = contrast.get("baseline_condition")
        variant = contrast.get("variant_condition")
        changed_axes = contrast.get("changed_axes") or []
        if not isinstance(contrast_id, str) or not contrast_id:
            check(False, "each contrast must declare a string id")
            continue
        if not isinstance(baseline, str) or not isinstance(variant, str):
            check(False, f"contrast {contrast_id!r} must declare baseline_condition and variant_condition")
            continue
        check(
            baseline in plan_condition_ids and variant in plan_condition_ids,
            f"contrast {contrast_id!r} must reference declared plan conditions",
        )
        if not isinstance(changed_axes, list):
            check(False, f"contrast {contrast_id!r} changed_axes must be a list")
            continue
        # Single-axis guard: the contrast must declare exactly one changed axis.
        check(
            len(changed_axes) == 1,
            f"contrast {contrast_id!r} must declare exactly one changed axis",
        )

        pair_captions: list[tuple[str, str]] = []
        pair_image_ids: set[str] = set()
        for image_id, condition_id in by_image_and_condition:
            if condition_id == baseline or condition_id == variant:
                pair_image_ids.add(image_id)
        for image_id in sorted(pair_image_ids):
            baseline_caption = by_image_and_condition.get((image_id, baseline))
            variant_caption = by_image_and_condition.get((image_id, variant))
            if isinstance(baseline_caption, str) and isinstance(variant_caption, str):
                pair_captions.append((baseline_caption, variant_caption))

        check(
            len(pair_captions) > 0,
            f"contrast {contrast_id!r} has no images with both baseline and variant records",
        )
        if not pair_captions:
            continue

        identical_count = sum(1 for left, right in pair_captions if left == right)
        distinct_count = len(pair_captions) - identical_count
        # A contrast that produces byte-identical output for EVERY image is vacuous.
        check(
            distinct_count >= 1,
            f"contrast {contrast_id!r} is vacuous: all {len(pair_captions)} baseline/variant caption pairs are byte-identical",
        )

        jaccards = sorted(_token_jaccard(left, right) for left, right in pair_captions)
        contrast_reports.append(
            {
                "id": contrast_id,
                "baseline_condition": baseline,
                "variant_condition": variant,
                "changed_axes": changed_axes,
                "image_count": len(pair_captions),
                "identical_pair_count": identical_count,
                "token_jaccard_min": round(jaccards[0], 4) if jaccards else None,
                "token_jaccard_median": round(jaccards[len(jaccards) // 2], 4) if jaccards else None,
                "token_jaccard_max": round(jaccards[-1], 4) if jaccards else None,
            }
        )

    # Per-condition distinctness: a condition that emitted the same caption for every image
    # carries no per-image signal and cannot support the declared per-image comparison.
    condition_boilerplate: list[str] = []
    for condition_id in sorted(plan_condition_ids):
        condition_captions = [
            caption
            for (image_id, cid), caption in by_image_and_condition.items()
            if cid == condition_id
        ]
        check(
            len(condition_captions) > 0,
            f"condition {condition_id!r} has no records",
        )
        distinct_captions = len({caption for caption in condition_captions})
        check(
            distinct_captions >= 2,
            f"condition {condition_id!r} collapsed to a single boilerplate caption across {len(condition_captions)} records",
        )
        if distinct_captions < 2:
            condition_boilerplate.append(condition_id)

    bad = checks["bad"]
    return {
        "root": str(root),
        "contrast_count": len(contrast_reports),
        "contrasts": contrast_reports,
        "condition_boilerplate_ids": condition_boilerplate,
        "checks_passed": checks["ok"],
        "checks_failed": checks["bad"],
        "findings": findings,
        "summary": (
            "all declared one-axis contrasts produced distinguishable captions"
            if bad == 0
            else "contrast divergence NOT confirmed: " + "; ".join(findings)
        ),
        "contrast_divergence_ok": bad == 0,
    }


# Instruction-bearing fragments that must never appear inside a data-only evidence
# slot of a rendered prompt. They are the role/task/semantic-expansion directives
# that legitimate data-only evidence serialization must not carry (see the
# executor-level controlled-comparison audit). Kept deliberately short and specific;
# a match means the evidence slot is not data-only, so the evidence axis is not
# cleanly isolated at the rendered-input boundary.
_EVIDENCE_SLOT_INSTRUCTION_MARKERS = (
    "Your job is to VERBALIZE the geometry and ADD what the determinations omit",
    "Name the posture or activity if obvious",
    "Translate the measured relations",
    "Describe mood, lighting quality, color palette",
    "Describe the setting and environment",
    "Subject & Pose",
    "Semantics:",
    "Visuals:",
    "Background:",
    "Below is a block of DETERMINATIONS",
    "These are ground truth",
    "You must NEVER contradict",
)
_EVIDENCE_SLOT_MARKER = "DECLARED SPECIALIST EVIDENCE:"
# The context-grounded template's own role/task tail follows the evidence slot. It
# legitimately repeats phrases like "Write strictly objective prose" / "Start the
# description immediately", so the check clips the slot at this boundary and scans only
# the evidence block itself, never the surrounding template prose.
_EVIDENCE_SLOT_TAIL = "Use declared specialist evidence only as bounded support;"


def _evidence_slot(rendered_text: str) -> str | None:
    """Return the evidence slot from a rendered prompt (clipped at the template tail).

    Returns ``None`` when the declared-specialist-evidence marker is not present (the
    prompt does not follow the context-grounded template shape).
    """
    marker_index = rendered_text.find(_EVIDENCE_SLOT_MARKER)
    if marker_index == -1:
        return None
    slot = rendered_text[marker_index + len(_EVIDENCE_SLOT_MARKER):].strip()
    tail_index = slot.find(_EVIDENCE_SLOT_TAIL)
    if tail_index != -1:
        slot = slot[:tail_index].strip()
    return slot


def check_stage_b_evidence_prompt_clean(root: Path) -> dict[str, Any]:
    """Observer-only report on whether a completed Stage-B run's evidence slot is data-only.

    The other evidence checks verify the run's *recorded* `evidence_payload` field and the
    output-level divergence of its declared contrasts. They do not inspect the exact rendered
    prompt that was sent to the aggregator. This check reads `prompt.rendered_text` from the
    records and inspects the declared specialist-evidence slot: a data-only evidence block must
    contain only the evidence content itself and must not smuggle role text, task instructions,
    semantic-expansion guidance, or detector/evaluator metadata into the prompt.

    Specifically, for every evidence-bearing condition (kind ``specialist_bundle``) the check:

    - locates the ``DECLARED SPECIALIST EVIDENCE:`` slot in each record's rendered prompt;
    - verifies the slot is present and non-empty;
    - verifies the slot contains per-image distinct content (not one shared block);
    - flags any instruction-bearing marker inside the slot.

    A flagged marker means the evidence-only contrast changes embedded instructions together
    with the evidence itself, so the evidence axis is not cleanly isolated at the model-input
    boundary. This is a metric-readiness finding, never a semantic/quality PASS or FAIL, and
    never an authorization. No model, GPU, scheduler, corpus, or derived-tree action occurs.
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

    conditions = plan.get("conditions") or []
    evidence_ids: list[str] = []
    none_ids: list[str] = []
    for condition in conditions:
        if not isinstance(condition, dict):
            raise ContractError("each comparison condition must be an object")
        cid = condition.get("id")
        evidence = condition.get("evidence") or {}
        kind = evidence.get("kind") if isinstance(evidence, dict) else None
        if not isinstance(cid, str) or not cid:
            raise ContractError("each comparison condition must declare a string id")
        if kind == "specialist_bundle":
            evidence_ids.append(cid)
        elif kind == "none":
            none_ids.append(cid)
        else:
            raise ContractError(
                f"condition {cid!r} must declare evidence.kind 'specialist_bundle' or 'none'"
            )
    if not evidence_ids:
        raise ContractError("stage-b-plan.json must declare at least one evidence-bearing condition")
    if not none_ids:
        raise ContractError("stage-b-plan.json must declare at least one no-evidence condition")

    by_condition: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        cid = record.get("condition_id")
        if isinstance(cid, str):
            by_condition.setdefault(cid, []).append(record)

    checks: dict[str, int] = {"ok": 0, "bad": 0}
    findings: list[str] = []
    condition_reports: list[dict[str, Any]] = []

    def check(condition: bool, label: str) -> None:
        checks["ok" if condition else "bad"] += 1
        if not condition:
            findings.append(label)

    for condition_id in sorted(evidence_ids):
        group = by_condition.get(condition_id, [])
        check(len(group) > 0, f"evidence condition {condition_id!r} has no records")
        slots: list[str] = []
        leaky_records: list[dict[str, Any]] = []
        for record in group:
            rendered = record.get("prompt") or {}
            text = rendered.get("rendered_text")
            check(
                bool(isinstance(text, str) and text),
                f"{record.get('record_id')}: evidence-bearing record must carry rendered prompt text",
            )
            if not isinstance(text, str):
                continue
            slot = _evidence_slot(text)
            check(
                slot is not None,
                f"{record.get('record_id')}: evidence-bearing rendered prompt must declare an evidence slot",
            )
            if slot is None:
                continue
            check(slot != "", f"{record.get('record_id')}: evidence slot must be non-empty")
            if not slot:
                continue
            slots.append(slot)
            found = [marker for marker in _EVIDENCE_SLOT_INSTRUCTION_MARKERS if marker in slot]
            if found:
                leaky_records.append(
                    {"record_id": record.get("record_id"), "image_id": record.get("image_id"), "markers": found}
                )
        check(len(slots) > 0, f"evidence condition {condition_id!r} has no readable evidence slots")
        distinct_slots = len({slot for slot in slots})
        check(
            distinct_slots >= 2,
            f"evidence condition {condition_id!r} evidence slot collapsed to one shared block across records",
        )
        for row in leaky_records:
            check(
                False,
                f"{row['record_id']}: evidence slot contains instruction-bearing text: {', '.join(row['markers'])}",
            )
        condition_reports.append(
            {
                "condition": condition_id,
                "record_count": len(group),
                "distinct_slot_count": distinct_slots,
                "instruction_leak_count": len(leaky_records),
                "leaky_records": leaky_records,
            }
        )

    for condition_id in sorted(none_ids):
        group = by_condition.get(condition_id, [])
        for record in group:
            rendered = record.get("prompt") or {}
            text = rendered.get("rendered_text")
            slot = _evidence_slot(text) if isinstance(text, str) else None
            if slot is None:
                # A no-evidence condition may legitimately be rendered without the
                # context-grounded marker (e.g. the legacy prompt template). Nothing to
                # inspect; the evidence-prompt check only binds evidence conditions.
                continue
            found = [marker for marker in _EVIDENCE_SLOT_INSTRUCTION_MARKERS if marker in slot]
            check(
                not found,
                f"{record.get('record_id')}: no-evidence evidence slot contains instruction-bearing text: {', '.join(found)}",
            )

    bad = checks["bad"]
    return {
        "root": str(root),
        "evidence_condition_count": len(evidence_ids),
        "conditions": condition_reports,
        "checks_passed": checks["ok"],
        "checks_failed": checks["bad"],
        "findings": findings,
        "summary": (
            "all evidence slots are data-only in their rendered prompts"
            if bad == 0
            else "evidence prompts NOT data-only: " + "; ".join(findings)
        ),
        "evidence_prompt_clean": bad == 0,
    }


# Record-level fields that may carry a per-image digest of the exact view bytes
# fed to the aggregator. Absence of all of them means the run's records do not
# document the input-view materialization per image.
_VIEW_DIGEST_RECORD_KEYS = ("input_view_sha256", "view_sha256", "view_content_sha256")
# Keys inside the input_view object itself that may carry the same per-image
# digest (never the identity/fingerprint members).
_VIEW_DIGEST_VIEW_KEYS = ("content_sha256", "sha256", "view_sha256", "bytes_sha256")


def _record_view_digest(record: Mapping[str, Any]) -> str | None:
    """Return the record's per-image view-content digest, if any is recorded."""
    for key in _VIEW_DIGEST_RECORD_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    view = record.get("input_view")
    if isinstance(view, Mapping):
        for key in _VIEW_DIGEST_VIEW_KEYS:
            value = view.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def check_stage_b_input_view_axis(root: Path) -> dict[str, Any]:
    """Observer-only report on whether a completed Stage-B run's declared
    input-view-only contrast is isolated and materialized at the input level.

    The evidence-axis check inspects the recorded evidence payload, and the
    contrast-divergence check inspects output captions. Neither inspects the
    input-view side: whether the run's own records demonstrate that the
    bucketed and raw conditions actually fed different view bytes to the
    aggregator. This check covers that side in three layers:

    - *Declaration*: the plan must declare exactly two distinct view components
      — the input-view-only baseline condition's view used by exactly one
      condition and the variant condition's view shared by every other
      condition — with distinct fingerprints, plus an ``input-view-only``
      contrast whose single changed axis is ``input_view``.
    - *Binding*: every record's ``input_view`` {id, fingerprint} must match its
      condition's declaration exactly.
    - *Materialization*: the run must record a per-image view-content digest
      (e.g. ``input_view_sha256`` on the record or ``content_sha256`` inside
      the ``input_view`` object). When digests exist, per image the baseline
      view digest must differ from the variant view digest (the axis is
      actually exercised) and records sharing one view id must share one digest
      (stimulus isolation for the prompt/evidence contrasts).

    A run that declares and binds the axis but records no per-image view
    digest is reported ``input_view_axis_materialized: false``: the run's own
    records cannot demonstrate that the bucketed and raw conditions fed
    different views, so the input-view-only contrast is declared-but-not-input-
    documented. This is a metric-readiness finding, never a semantic/quality
    PASS or FAIL, and never an authorization. No model, GPU, scheduler, corpus,
    or derived-tree action occurs.
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

    conditions = plan.get("conditions") or []
    if not isinstance(conditions, list) or not all(isinstance(c, dict) for c in conditions):
        raise ContractError("stage-b-plan.json conditions must be a list of objects")
    if not conditions:
        raise ContractError("stage-b-plan.json must declare at least one condition")

    checks: dict[str, int] = {"ok": 0, "bad": 0}
    findings: list[str] = []

    def check(condition: bool, label: str) -> None:
        checks["ok" if condition else "bad"] += 1
        if not condition:
            findings.append(label)

    # --- Declaration layer -------------------------------------------------
    condition_views: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        cid = condition.get("id")
        view = condition.get("input_view")
        if not isinstance(cid, str) or not cid:
            check(False, "each comparison condition must declare a string id")
            continue
        if not isinstance(view, Mapping) or not isinstance(view.get("id"), str) or not isinstance(
            view.get("fingerprint"), str
        ):
            check(False, f"condition {cid!r} must declare input_view id and fingerprint")
            continue
        condition_views[cid] = {"id": view["id"], "fingerprint": view["fingerprint"]}

    view_by_id: dict[str, list[str]] = {}
    for cid, view in condition_views.items():
        view_by_id.setdefault(view["id"], []).append(cid)
    view_fingerprints = {
        view["id"]: view["fingerprint"] for view in condition_views.values()
    }

    contrasts = plan.get("contrasts")
    if not isinstance(contrasts, list) or not contrasts:
        raise ContractError("stage-b-plan.json must declare at least one one-axis contrast in 'contrasts'")
    input_view_contrast = None
    for contrast in contrasts:
        if not isinstance(contrast, Mapping):
            continue
        if contrast.get("changed_axes") == ["input_view"]:
            input_view_contrast = contrast
            break
    check(input_view_contrast is not None, "plan must declare an input-view-only contrast with changed_axes [input_view]")

    baseline_view_id: str | None = None
    variant_view_id: str | None = None
    if input_view_contrast is not None:
        baseline_cid = input_view_contrast.get("baseline_condition")
        variant_cid = input_view_contrast.get("variant_condition")
        check(
            isinstance(baseline_cid, str) and isinstance(variant_cid, str),
            "input-view-only contrast must declare baseline_condition and variant_condition",
        )
        if isinstance(baseline_cid, str) and isinstance(variant_cid, str):
            check(
                baseline_cid in condition_views and variant_cid in condition_views,
                "input-view-only contrast must reference declared plan conditions",
            )
            if baseline_cid in condition_views and variant_cid in condition_views:
                baseline_view_id = condition_views[baseline_cid]["id"]
                variant_view_id = condition_views[variant_cid]["id"]

    if baseline_view_id is not None and variant_view_id is not None:
        check(
            baseline_view_id != variant_view_id,
            "input-view-only contrast must pair two distinct view components",
        )
        check(
            len({view["id"] for view in condition_views.values()}) == 2,
            "plan must declare exactly two distinct input-view components (one baseline view, one variant view)",
        )
        check(
            len(view_by_id.get(baseline_view_id, [])) == 1,
            f"baseline view {baseline_view_id!r} must be used by exactly one condition (the input-view-only baseline)",
        )
        check(
            len(view_by_id.get(variant_view_id, [])) == len(conditions) - 1,
            f"variant view {variant_view_id!r} must be shared by every other condition",
        )
        check(
            view_fingerprints.get(baseline_view_id) != view_fingerprints.get(variant_view_id),
            "baseline and variant view components must carry distinct fingerprints",
        )

    # --- Binding layer ------------------------------------------------------
    plan_view_by_condition = {
        cid: view for cid, view in condition_views.items()
    }
    for record in records:
        cid = record.get("condition_id")
        view = record.get("input_view")
        declared = plan_view_by_condition.get(cid) if isinstance(cid, str) else None
        if declared is None:
            check(False, f"{record.get('record_id')}: condition missing from plan")
            continue
        check(
            isinstance(view, Mapping)
            and view.get("id") == declared["id"]
            and view.get("fingerprint") == declared["fingerprint"],
            f"{record.get('record_id')}: input_view must bind the declared plan view component",
        )

    # --- Materialization layer ----------------------------------------------
    digest_by_record: dict[str, str] = {}
    missing_digest_count = 0
    for record in records:
        digest = _record_view_digest(record)
        if digest is None:
            missing_digest_count += 1
            check(
                False,
                f"{record.get('record_id')}: no per-image view-content digest recorded (expected e.g. input_view_sha256)",
            )
        else:
            digest_by_record[record.get("record_id", "")] = digest

    view_digests: dict[tuple[str, str], set[str]] = {}
    for record in records:
        view = record.get("input_view")
        digest = _record_view_digest(record)
        if isinstance(view, Mapping) and isinstance(view.get("id"), str) and digest is not None:
            image_id = record.get("image_id")
            if isinstance(image_id, str):
                view_digests.setdefault((image_id, view["id"]), set()).add(digest)

    for (image_id, view_id), digests in sorted(view_digests.items()):
        check(
            len(digests) == 1,
            f"{image_id}: view {view_id!r} must record one consistent per-image digest (got {len(digests)})",
        )

    if baseline_view_id is not None and variant_view_id is not None and digest_by_record:
        for image_id in {
            record.get("image_id")
            for record in records
            if isinstance(record.get("image_id"), str)
        }:
            baseline_digests = view_digests.get((image_id, baseline_view_id), set())
            variant_digests = view_digests.get((image_id, variant_view_id), set())
            if baseline_digests and variant_digests:
                check(
                    baseline_digests != variant_digests,
                    f"{image_id}: baseline and variant view digests must differ (the views fed must actually differ)",
                )

    materialized = missing_digest_count == 0

    bad = checks["bad"]
    return {
        "root": str(root),
        "record_count": len(records),
        "view_ids": sorted(view_by_id),
        "baseline_view_id": baseline_view_id,
        "variant_view_id": variant_view_id,
        "condition_view_ids": {cid: view["id"] for cid, view in condition_views.items()},
        "input_view_axis_declared": (
            baseline_view_id is not None
            and variant_view_id is not None
            and baseline_view_id != variant_view_id
            and len({view["id"] for view in condition_views.values()}) == 2
        ),
        "input_view_axis_materialized": materialized,
        "per_image_view_digest_count": len(digest_by_record),
        "checks_passed": checks["ok"],
        "checks_failed": checks["bad"],
        "findings": findings,
        "summary": (
            "input-view axis declared, bound, and input-materialized per image"
            if bad == 0
            else "input-view axis NOT fully documented: " + "; ".join(findings)
        ),
        "input_view_axis_ok": bad == 0,
    }
