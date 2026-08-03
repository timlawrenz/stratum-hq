"""Fail-closed contracts for autonomous, evidence-driven research.

The module validates durable, machine-readable control-plane state. It does not
select research directions, issue shell commands, launch GPU work, or mutate a
dataset. Those actions remain separately reviewed and auditable.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any


class ContractError(ValueError):
    """Raised when autonomous work would violate an explicit program contract."""


_RESEARCH_METADATA = re.compile(
    r"<!--\s*research-harness:\s*(\{.*?\})\s*-->", re.DOTALL
)
_DURATION = re.compile(r"^(?P<amount>(?:\d+(?:\.\d*)?|\.\d+))(?P<unit>[hm]?)$")
_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PLACEHOLDERS = frozenset({"REPLACE", "TODO", "TBD", "N/A", "NONE", "UNKNOWN"})
_SUPPORTED_KINDS = frozenset({"program", "arm", "hold", "postmortem"})
_REQUIRED_ARM_FIELDS = (
    "hypothesis",
    "falsified_if",
    "pre_registered_gate",
    "metric_version",
    "data_snapshot",
    "valid_non_improving_experiments",
)
_REQUIRED_GPU_LIFECYCLE = (
    "request",
    "poll_and_claim",
    "launch",
    "verify",
    "activate",
    "heartbeat",
    "release",
)
_REQUIRED_COMPACT_ARTIFACT_ROLES = (
    "structured",
    "human_readable",
    "provenance",
)
_REQUIRED_AUTONOMY_DENIALS = (
    "autonomous_merge_allowed",
    "autonomous_direct_main_push_allowed",
    "autonomous_gpu_execution_allowed",
    "autonomous_model_installation_allowed",
    "autonomous_canonical_source_write_allowed",
)
_REQUIRED_TREE_FLAGS = (
    "require_program_root",
    "require_parent_issue",
    "require_selection_rationale",
)
_STRATUM_PROFILE = "stratum-single-woman-v1"


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{field} must be an object")
    return value


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty string")
    return value.strip()


def _require_meaningful_string(value: Any, field: str) -> str:
    result = _require_nonempty_string(value, field)
    normalized = result.upper()
    if normalized in _PLACEHOLDERS or result.startswith("{{") or result.startswith("<"):
        raise ContractError(f"{field} must not be a placeholder")
    return result


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{field} must be a boolean")
    return value


def _require_false(value: Any, field: str) -> None:
    if _require_bool(value, field) is not False:
        raise ContractError(f"{field} must be false")


def _require_positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{field} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ContractError(f"{field} must be a positive finite number")
    return result


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{field} must be a positive integer")
    return value


def _require_absolute_path(value: Any, field: str) -> str:
    raw = _require_nonempty_string(value, field)
    path = PurePosixPath(raw)
    if not path.is_absolute() or ".." in path.parts:
        raise ContractError(f"{field} must be an absolute path without '..'")
    normalized = str(path)
    if normalized == "/":
        raise ContractError(f"{field} must not be the filesystem root")
    return normalized.rstrip("/")


def _is_at_or_below(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


def _parse_duration_hours(value: Any, field: str) -> float:
    raw = _require_nonempty_string(value, field)
    match = _DURATION.fullmatch(raw)
    if match is None:
        raise ContractError(f"{field} must be a finite duration such as '2h' or '30m'")
    amount = float(match.group("amount"))
    if not math.isfinite(amount) or amount <= 0:
        raise ContractError(f"{field} must be a positive finite duration")
    if match.group("unit") == "m":
        amount /= 60.0
    return amount


def _labels(issue: Mapping[str, Any]) -> set[str]:
    labels = issue.get("labels", [])
    if not isinstance(labels, list):
        raise ContractError("issue labels must be a list")
    result: set[str] = set()
    for label in labels:
        if isinstance(label, str):
            result.add(label)
        elif isinstance(label, Mapping) and isinstance(label.get("name"), str):
            result.add(label["name"])
        else:
            raise ContractError("each issue label must be a string or an object with name")
    return result


def _issue_number(issue: Mapping[str, Any]) -> int:
    return _require_positive_int(issue.get("number"), "issue number")


def _issue_state(issue: Mapping[str, Any]) -> str:
    state = issue.get("state", "OPEN")
    if not isinstance(state, str):
        raise ContractError("issue state must be a string")
    normalized = state.upper()
    if normalized not in {"OPEN", "CLOSED"}:
        raise ContractError("issue state must be OPEN or CLOSED")
    return normalized


def _issue_metadata(issue: Mapping[str, Any]) -> Mapping[str, Any]:
    body = issue.get("body", "")
    if not isinstance(body, str):
        raise ContractError("issue body must be a string")
    match = _RESEARCH_METADATA.search(body)
    if match is None:
        number = issue.get("number", "unknown")
        raise ContractError(f"research issue #{number} lacks research-harness metadata")
    try:
        metadata = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        number = issue.get("number", "unknown")
        raise ContractError(f"research issue #{number} has invalid metadata JSON: {exc.msg}") from exc
    return _require_mapping(metadata, "research-harness metadata")


def _metadata_kind(metadata: Mapping[str, Any]) -> str:
    kind = _require_nonempty_string(metadata.get("kind"), "research-harness metadata.kind")
    if kind not in _SUPPORTED_KINDS:
        raise ContractError(f"unsupported research-harness metadata kind: {kind}")
    return kind


def _metadata_issue_reference(value: Any, field: str) -> int:
    return _require_positive_int(value, field)


def _validate_arm_metadata(metadata: Mapping[str, Any]) -> None:
    for field in _REQUIRED_ARM_FIELDS[:-1]:
        _require_meaningful_string(metadata.get(field), f"research arm {field}")
    failures = metadata["valid_non_improving_experiments"]
    if isinstance(failures, bool) or not isinstance(failures, int) or failures < 0:
        raise ContractError("valid_non_improving_experiments must be a non-negative integer")
    if "parent_issue" in metadata:
        _metadata_issue_reference(metadata["parent_issue"], "research arm parent_issue")
    if "postmortem_issue" in metadata:
        _metadata_issue_reference(metadata["postmortem_issue"], "research arm postmortem_issue")
    if "selection_rationale" in metadata:
        _require_meaningful_string(metadata["selection_rationale"], "research arm selection_rationale")
    if "surveyed_issue_numbers" in metadata:
        surveyed = metadata["surveyed_issue_numbers"]
        if not isinstance(surveyed, list) or not surveyed:
            raise ContractError("research arm surveyed_issue_numbers must be a non-empty list")
        values = [_metadata_issue_reference(item, "research arm surveyed_issue_numbers item") for item in surveyed]
        if len(values) != len(set(values)):
            raise ContractError("research arm surveyed_issue_numbers must not contain duplicates")


def _validate_hold_metadata(metadata: Mapping[str, Any]) -> None:
    for field in ("trigger", "risk", "decision"):
        _require_meaningful_string(metadata.get(field), f"research hold {field}")


def _validate_postmortem_metadata(metadata: Mapping[str, Any]) -> None:
    _metadata_issue_reference(metadata.get("parent_issue"), "research postmortem parent_issue")
    _require_meaningful_string(metadata.get("decision"), "research postmortem decision")
    _require_meaningful_string(metadata.get("evidence_summary"), "research postmortem evidence_summary")


def _require_string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{field} must be a non-empty list of strings")
    result = [_require_nonempty_string(item, f"{field} item") for item in value]
    if len(result) != len(set(result)):
        raise ContractError(f"{field} must not contain duplicates")
    return result


def _validate_stratum_policy_profile(program: Mapping[str, Any]) -> None:
    source = _require_mapping(program["canonical_source"], "canonical_source")
    if source["path"] != "/mnt/nas-ai-models/training-data/crawlr/approved":
        raise ContractError("stratum policy profile requires crawlr/approved as canonical source")
    if source["subject_invariant"] != "exactly_one_curated_woman":
        raise ContractError("stratum policy profile requires exactly_one_curated_woman")

    content = _require_mapping(program["content_policy"], "content_policy")
    if content["model_execution"] != "local_first":
        raise ContractError("stratum policy profile requires local_first image-model execution")

    artifact = _require_mapping(program["artifact_policy"], "artifact_policy")
    roots = artifact["approved_output_roots"]
    if "/mnt/nas-ai-models/research" not in roots:
        raise ContractError("stratum policy profile requires the NAS research artifact root")

    representation = _require_mapping(program["representation"], "representation")
    expected_tokens = {
        "expanded_dossier_target_tokens": 100_000,
        "expanded_dossier_min_tokens": 100_000,
        "compact_context_target_tokens": 4_000,
        "compact_context_min_tokens": 4_000,
        "legacy_text_encoder_max_tokens": 512,
    }
    for key, expected in expected_tokens.items():
        if representation[key] != expected:
            raise ContractError(f"stratum policy profile requires {key}={expected}")

    scheduler = _require_mapping(program["gpu_scheduler"], "gpu_scheduler")
    if scheduler.get("scheduler_project") != "stratum-contextual-specialist-research":
        raise ContractError("stratum policy profile requires the stratum scheduler project")
    if scheduler.get("max_job_duration_hours") != 24:
        raise ContractError("stratum policy profile requires max_job_duration_hours=24")
    resources = _require_mapping(scheduler["resources"], "gpu_scheduler.resources")
    expected_resources = {
        "4090": {"host_route": "local", "total_vram_gb": 24, "usable_vram_gb": 24},
        "strix": {
            "host_route": "ssh:max395",
            "total_vram_gb": 110,
            "usable_vram_gb": 100,
            "evergreen_reserved_vram_gb": 10,
        },
    }
    for name, expected in expected_resources.items():
        resource = _require_mapping(resources.get(name), f"gpu_scheduler.resources.{name}")
        for key, value in expected.items():
            if resource.get(key) != value:
                raise ContractError(f"stratum policy profile requires {name}.{key}={value!r}")

    tree = _require_mapping(program["research_tree"], "research_tree")
    if any(tree[flag] is not True for flag in _REQUIRED_TREE_FLAGS):
        raise ContractError("stratum policy profile requires strict research-tree validation")


def validate_program(program: Mapping[str, Any]) -> None:
    """Validate the durable program configuration before autonomous work begins."""
    program = _require_mapping(program, "program")
    if program.get("schema_version") != 1:
        raise ContractError("program schema_version must be 1")
    _require_nonempty_string(program.get("program_id"), "program_id")

    source = _require_mapping(program.get("canonical_source"), "canonical_source")
    _require_absolute_path(source.get("path"), "canonical_source.path")
    _require_nonempty_string(source.get("subject_invariant"), "canonical_source.subject_invariant")
    _require_nonempty_string(source.get("detector_disagreement"), "canonical_source.detector_disagreement")
    if "derived_tree" in source:
        _require_absolute_path(source["derived_tree"], "canonical_source.derived_tree")

    content_policy = _require_mapping(program.get("content_policy"), "content_policy")
    if content_policy.get("model_execution") != "local_first":
        raise ContractError("content_policy.model_execution must be local_first")
    _require_false(
        content_policy.get("autonomous_external_image_model_allowed"),
        "content_policy.autonomous_external_image_model_allowed",
    )
    _require_nonempty_string(content_policy.get("reason"), "content_policy.reason")
    _require_nonempty_string(
        content_policy.get("external_model_requirement"),
        "content_policy.external_model_requirement",
    )

    artifact_policy = _require_mapping(program.get("artifact_policy"), "artifact_policy")
    output_roots = _require_string_list(
        artifact_policy.get("approved_output_roots"), "artifact_policy.approved_output_roots"
    )
    normalized_output_roots = [
        _require_absolute_path(root, "artifact_policy.approved_output_roots item")
        for root in output_roots
    ]
    source_path = _require_absolute_path(source["path"], "canonical_source.path")
    if any(_is_at_or_below(root, source_path) for root in normalized_output_roots):
        raise ContractError("artifact_policy.approved_output_roots must not be inside canonical_source.path")
    _require_false(
        artifact_policy.get("canonical_source_write_allowed"),
        "artifact_policy.canonical_source_write_allowed",
    )

    representation = _require_mapping(program.get("representation"), "representation")
    expanded = _require_positive_number(
        representation.get("expanded_dossier_target_tokens"),
        "representation.expanded_dossier_target_tokens",
    )
    expanded_min = _require_positive_number(
        representation.get("expanded_dossier_min_tokens"),
        "representation.expanded_dossier_min_tokens",
    )
    compact = _require_positive_number(
        representation.get("compact_context_target_tokens"),
        "representation.compact_context_target_tokens",
    )
    compact_min = _require_positive_number(
        representation.get("compact_context_min_tokens"),
        "representation.compact_context_min_tokens",
    )
    legacy = _require_positive_number(
        representation.get("legacy_text_encoder_max_tokens"),
        "representation.legacy_text_encoder_max_tokens",
    )
    if expanded_min > expanded:
        raise ContractError("expanded_dossier_min_tokens must not exceed expanded_dossier_target_tokens")
    if compact_min > compact:
        raise ContractError("compact_context_min_tokens must not exceed compact_context_target_tokens")
    if expanded <= compact or expanded_min < compact_min:
        raise ContractError("expanded dossier budget must exceed compact-context budget")
    if compact <= legacy or compact_min <= legacy:
        raise ContractError("compact-context budget must exceed legacy_text_encoder_max_tokens")
    artifacts = _require_mapping(
        representation.get("compact_artifacts"), "representation.compact_artifacts"
    )
    for role in _REQUIRED_COMPACT_ARTIFACT_ROLES:
        _require_nonempty_string(artifacts.get(role), f"representation.compact_artifacts.{role}")

    specialists = _require_mapping(program.get("specialists"), "specialists")
    if specialists.get("policy") != "open_world":
        raise ContractError("specialists.policy must be open_world")
    declarations = specialists.get("required_declaration_fields")
    if not isinstance(declarations, list) or not all(isinstance(item, str) for item in declarations):
        raise ContractError("specialists.required_declaration_fields must be a list of strings")
    required_declarations = {
        "scope",
        "inputs",
        "output_semantics",
        "provenance",
        "abstention_policy",
        "qualification_gate",
    }
    if not required_declarations.issubset(set(declarations)):
        raise ContractError("specialist declaration fields are incomplete")

    tree = _require_mapping(program.get("research_tree"), "research_tree")
    for flag in _REQUIRED_TREE_FLAGS:
        _require_bool(tree.get(flag), f"research_tree.{flag}")

    autonomy = _require_mapping(program.get("autonomy"), "autonomy")
    if autonomy.get("mode") != "draft_pr_only":
        raise ContractError("autonomy.mode must be draft_pr_only")
    for field in _REQUIRED_AUTONOMY_DENIALS:
        _require_false(autonomy.get(field), f"autonomy.{field}")
    for field in ("authorized_without_new_human_approval", "requires_hold"):
        _require_string_list(autonomy.get(field), f"autonomy.{field}")

    scheduler = _require_mapping(program.get("gpu_scheduler"), "gpu_scheduler")
    _require_nonempty_string(scheduler.get("command"), "gpu_scheduler.command")
    if scheduler.get("execution_mode") != "observer_only":
        raise ContractError("gpu_scheduler.execution_mode must be observer_only")
    _require_positive_number(scheduler.get("max_job_duration_hours"), "gpu_scheduler.max_job_duration_hours")
    _require_nonempty_string(scheduler.get("scheduler_project"), "gpu_scheduler.scheduler_project")
    launchers = _require_string_list(scheduler.get("allowed_launchers"), "gpu_scheduler.allowed_launchers")
    if not launchers:
        raise ContractError("gpu_scheduler.allowed_launchers must not be empty")
    resources = _require_mapping(scheduler.get("resources"), "gpu_scheduler.resources")
    if not resources:
        raise ContractError("gpu_scheduler.resources must declare at least one accelerator")
    for accelerator, resource_value in resources.items():
        _require_nonempty_string(accelerator, "gpu_scheduler resource name")
        resource = _require_mapping(resource_value, f"gpu_scheduler.resources.{accelerator}")
        _require_nonempty_string(resource.get("host_route"), f"resources.{accelerator}.host_route")
        total = _require_positive_number(resource.get("total_vram_gb"), f"resources.{accelerator}.total_vram_gb")
        usable = _require_positive_number(resource.get("usable_vram_gb"), f"resources.{accelerator}.usable_vram_gb")
        if usable > total:
            raise ContractError(f"resources.{accelerator}.usable_vram_gb must not exceed total_vram_gb")
        if "evergreen_reserved_vram_gb" in resource:
            reserved = _require_positive_number(
                resource["evergreen_reserved_vram_gb"],
                f"resources.{accelerator}.evergreen_reserved_vram_gb",
            )
            if usable > total - reserved:
                raise ContractError(
                    f"resources.{accelerator}.usable_vram_gb must account for evergreen reservation"
                )

    profile = program.get("policy_profile")
    if profile is not None:
        if profile != _STRATUM_PROFILE:
            raise ContractError(f"unsupported policy_profile: {profile}")
        _validate_stratum_policy_profile(program)


def validate_research_tree(snapshot: Mapping[str, Any], program: Mapping[str, Any]) -> None:
    """Validate GitHub issue-tree state without imposing FIFO ordering."""
    validate_program(program)
    snapshot = _require_mapping(snapshot, "research tree snapshot")
    issues = snapshot.get("issues")
    if not isinstance(issues, list):
        raise ContractError("research tree snapshot.issues must be a list")

    records: list[dict[str, Any]] = []
    by_number: dict[int, dict[str, Any]] = {}
    for raw_issue in issues:
        issue = _require_mapping(raw_issue, "issue")
        labels = _labels(issue)
        if "research" not in labels:
            continue
        number = _issue_number(issue)
        if number in by_number:
            raise ContractError(f"research tree contains duplicate issue #{number}")
        state = _issue_state(issue)
        metadata = _issue_metadata(issue)
        kind = _metadata_kind(metadata)
        if kind == "arm":
            _validate_arm_metadata(metadata)
        elif kind == "hold":
            _validate_hold_metadata(metadata)
            if "research:hold" not in labels:
                raise ContractError("research hold metadata requires research:hold label")
        elif kind == "postmortem":
            _validate_postmortem_metadata(metadata)
            if "research:postmortem" not in labels:
                raise ContractError("research postmortem metadata requires research:postmortem label")
        record = {
            "number": number,
            "state": state,
            "labels": labels,
            "metadata": metadata,
            "kind": kind,
        }
        records.append(record)
        by_number[number] = record

    if not records:
        raise ContractError("research tree contains no research issues")

    tree_policy = _require_mapping(program["research_tree"], "research_tree")
    if tree_policy["require_program_root"]:
        roots = [record for record in records if record["kind"] == "program" and record["state"] == "OPEN"]
        if len(roots) != 1:
            raise ContractError("strict research tree must have exactly one open program root")

    if tree_policy["require_parent_issue"]:
        for record in records:
            if record["kind"] != "arm":
                continue
            parent = _metadata_issue_reference(record["metadata"].get("parent_issue"), "research arm parent_issue")
            parent_record = by_number.get(parent)
            if parent_record is None or parent_record["kind"] not in {"program", "arm"}:
                raise ContractError("research arm parent_issue must reference a program or arm issue")

    for record in records:
        if record["kind"] != "arm":
            continue
        failures = record["metadata"]["valid_non_improving_experiments"]
        if failures < 3:
            continue
        postmortem_number = _metadata_issue_reference(
            record["metadata"].get("postmortem_issue"), "research arm postmortem_issue"
        )
        postmortem = by_number.get(postmortem_number)
        if postmortem is None or postmortem["kind"] != "postmortem":
            raise ContractError("an arm with three valid non-improving experiments requires a linked postmortem")
        if postmortem["metadata"]["parent_issue"] != record["number"]:
            raise ContractError("linked postmortem must reference the stalled arm")
        if record["state"] != "CLOSED" or "research:active" in record["labels"]:
            raise ContractError("an arm with three valid non-improving experiments must be closed and inactive")

    active_arms = [
        record
        for record in records
        if record["kind"] == "arm"
        and record["state"] == "OPEN"
        and "research:active" in record["labels"]
    ]
    open_holds = [
        record
        for record in records
        if record["kind"] == "hold" and record["state"] == "OPEN"
    ]
    if open_holds:
        if len(active_arms) > 1:
            raise ContractError("a held program may not have multiple active research arms")
        return
    if len(active_arms) != 1:
        raise ContractError("research tree must have exactly one active research arm")

    active = active_arms[0]
    if tree_policy["require_selection_rationale"]:
        metadata = active["metadata"]
        _require_meaningful_string(metadata.get("selection_rationale"), "research arm selection_rationale")
        surveyed = metadata.get("surveyed_issue_numbers")
        if not isinstance(surveyed, list):
            raise ContractError("research arm surveyed_issue_numbers must be a list")
        surveyed_numbers = {
            _metadata_issue_reference(item, "research arm surveyed_issue_numbers item")
            for item in surveyed
        }
        open_research_numbers = {record["number"] for record in records if record["state"] == "OPEN"}
        if not open_research_numbers.issubset(surveyed_numbers):
            raise ContractError("active arm must record a survey of the whole open research tree")


def validate_compression_bundle(bundle: Mapping[str, Any], program: Mapping[str, Any]) -> None:
    """Require compact context to retain an evidence path for every claim."""
    validate_program(program)
    bundle = _require_mapping(bundle, "compression bundle")
    if bundle.get("schema_version") != 1:
        raise ContractError("compression bundle schema_version must be 1")
    _require_nonempty_string(bundle.get("image_id"), "compression bundle image_id")

    dossier = _require_mapping(bundle.get("expanded_dossier"), "expanded_dossier")
    expanded_tokens = _require_positive_number(dossier.get("token_count"), "expanded_dossier.token_count")
    representation = _require_mapping(program["representation"], "representation")
    if expanded_tokens < float(representation["expanded_dossier_min_tokens"]):
        raise ContractError("expanded_dossier.token_count is below the program minimum")
    evidence_ids = dossier.get("evidence_ids")
    if not isinstance(evidence_ids, list) or not evidence_ids or not all(
        isinstance(item, str) and item for item in evidence_ids
    ):
        raise ContractError("expanded_dossier.evidence_ids must be a non-empty list of strings")
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ContractError("expanded_dossier.evidence_ids must not contain duplicates")
    evidence_set = set(evidence_ids)

    context = _require_mapping(bundle.get("compact_context"), "compact_context")
    compact_tokens = _require_positive_number(context.get("token_count"), "compact_context.token_count")
    target = float(representation["compact_context_target_tokens"])
    minimum = float(representation["compact_context_min_tokens"])
    if compact_tokens < minimum:
        raise ContractError("compact_context.token_count is below the program minimum")
    if compact_tokens > target:
        raise ContractError("compact_context.token_count exceeds the program target")
    if compact_tokens > expanded_tokens:
        raise ContractError("compact context cannot exceed the expanded dossier")
    if compact_tokens <= float(representation["legacy_text_encoder_max_tokens"]):
        raise ContractError(
            "compact context is not a first-class long-context artifact; it fits the legacy encoder"
        )

    claims = context.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ContractError("compact_context.claims must be a non-empty list")
    for claim in claims:
        claim = _require_mapping(claim, "compact context claim")
        _require_meaningful_string(claim.get("text"), "compact context claim.text")
        claim_evidence = claim.get("evidence_ids")
        if not isinstance(claim_evidence, list) or not claim_evidence:
            raise ContractError("each compact-context claim requires supporting evidence")
        if not all(isinstance(item, str) and item in evidence_set for item in claim_evidence):
            raise ContractError("compact-context claim references unknown supporting evidence")

    artifacts = _require_mapping(bundle.get("artifacts"), "compression artifacts")
    configured_artifacts = _require_mapping(representation["compact_artifacts"], "representation.compact_artifacts")
    for role in _REQUIRED_COMPACT_ARTIFACT_ROLES:
        actual = _require_nonempty_string(artifacts.get(role), f"artifacts.{role}")
        if actual != configured_artifacts[role]:
            raise ContractError(f"artifacts.{role} must match the configured compact artifact name")


def validate_gpu_manifest(manifest: Mapping[str, Any], program: Mapping[str, Any]) -> None:
    """Validate a scheduler-bound job manifest before a supervisor may inspect it."""
    validate_program(program)
    manifest = _require_mapping(manifest, "GPU manifest")
    if manifest.get("schema_version") != 1:
        raise ContractError("GPU manifest schema_version must be 1")
    job_id = _require_nonempty_string(manifest.get("job_id"), "job_id")
    if not _JOB_ID.fullmatch(job_id):
        raise ContractError("job_id may contain only letters, digits, '.', '_', and '-'")
    target = _require_nonempty_string(manifest.get("target_gpu"), "target_gpu")
    scheduler = _require_mapping(program["gpu_scheduler"], "gpu_scheduler")
    resources = _require_mapping(scheduler["resources"], "gpu_scheduler.resources")
    if target not in resources:
        raise ContractError(f"target_gpu {target!r} is not declared by the program")
    resource = _require_mapping(resources[target], f"gpu_scheduler.resources.{target}")
    requested_vram = _require_positive_number(manifest.get("requested_vram_gb"), "requested_vram_gb")
    if requested_vram > float(resource["usable_vram_gb"]):
        raise ContractError(f"requested_vram_gb exceeds usable_vram_gb for {target}")
    duration_hours = _parse_duration_hours(manifest.get("maximum_duration"), "maximum_duration")
    if duration_hours > float(scheduler["max_job_duration_hours"]):
        raise ContractError("maximum_duration exceeds gpu_scheduler.max_job_duration_hours")
    approved_issue = _require_positive_int(manifest.get("approved_issue"), "approved_issue")
    if manifest.get("manifest_state") != "approved":
        raise ContractError("GPU manifest must be explicitly approved")
    authorization = _require_mapping(manifest.get("authorization"), "GPU manifest authorization")
    if authorization.get("mode") != "human_reviewed":
        raise ContractError("GPU manifest authorization.mode must be human_reviewed")
    _require_meaningful_string(authorization.get("approved_by"), "GPU manifest authorization.approved_by")
    if _metadata_issue_reference(authorization.get("approval_issue"), "GPU manifest authorization.approval_issue") != approved_issue:
        raise ContractError("GPU manifest authorization.approval_issue must match approved_issue")

    expected_route = resource["host_route"]
    if manifest.get("host_route") != expected_route:
        raise ContractError(f"{target} GPU manifest must use host_route {expected_route!r}")
    launcher = _require_nonempty_string(manifest.get("launcher_id"), "launcher_id")
    if launcher not in scheduler["allowed_launchers"]:
        raise ContractError("GPU manifest references an unregistered launcher")
    scheduler_project = _require_nonempty_string(manifest.get("scheduler_project"), "scheduler_project")
    if scheduler_project != scheduler["scheduler_project"]:
        raise ContractError("scheduler_project must match gpu_scheduler.scheduler_project")
    for forbidden in ("command", "shell_command", "inline_script"):
        if forbidden in manifest:
            raise ContractError(f"GPU manifest may not contain arbitrary {forbidden}")
    output_root = _require_absolute_path(manifest.get("output_root"), "output_root")
    artifact_policy = _require_mapping(program["artifact_policy"], "artifact_policy")
    approved_roots = [
        _require_absolute_path(root, "artifact_policy.approved_output_roots item")
        for root in artifact_policy["approved_output_roots"]
    ]
    if not any(_is_at_or_below(output_root, root) for root in approved_roots):
        raise ContractError("output_root must be under an approved output root")

    lifecycle = manifest.get("scheduler_lifecycle")
    if not isinstance(lifecycle, list) or tuple(lifecycle) != _REQUIRED_GPU_LIFECYCLE:
        raise ContractError(
            "scheduler_lifecycle must be request → poll_and_claim → launch → verify → "
            "activate → heartbeat → release"
        )