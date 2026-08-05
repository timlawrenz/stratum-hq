"""Freeze a coverage-balanced, source-hashed candidate cohort without inference.

The freeze deliberately separates two phases:

1. Re-audit existing derived artifacts and select from filename/derived metadata
   only. No source image bytes are opened during this phase.
2. Read each selected source exactly once to bind its SHA-256 and dimensions.

It is a pre-compute provenance artifact, not Stage-B execution or permission to
invoke a model, use a GPU, mutate either corpus tree, or make an empirical claim.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from PIL import Image

from .contracts import ContractError, validate_program
from .core_coverage import audit_core_coverage, compact_coverage_report

COVERAGE_BALANCED_SELECTION_SALT = "stratum-first500-coverage-design-v1"
ASPECT_ORDER = ("portrait", "squareish", "landscape")
DEFAULT_QUOTAS = {"portrait": 12, "squareish": 6, "landscape": 6}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a SHA-256 hex digest") from exc
    return value.lower()


def _resolved_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} must be an existing directory: {path}") from exc
    if not resolved.is_dir():
        raise ValueError(f"{label} must be an existing directory: {path}")
    return resolved


def _require_quotas(quotas: Mapping[str, int]) -> dict[str, int]:
    if set(quotas) != set(ASPECT_ORDER):
        raise ValueError(f"quotas must contain exactly: {', '.join(ASPECT_ORDER)}")
    normalized: dict[str, int] = {}
    for aspect in ASPECT_ORDER:
        count = quotas[aspect]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"quota for {aspect} must be a positive integer")
        normalized[aspect] = count
    return normalized


def _safe_flat_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("candidate source_relative_path must be a non-empty string")
    if "\\" in value or any(ord(character) == 0 for character in value):
        raise ValueError("candidate source_relative_path must be normalized POSIX")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name != value or value in {".", ".."}:
        raise ValueError("candidate source_relative_path must be a flat normalized relative path")
    return value


def rank_source_relative_path(source_relative_path: str) -> str:
    """Return the deterministic selection rank for one normalized source path."""
    normalized = _safe_flat_relative_path(source_relative_path)
    payload = bytearray(COVERAGE_BALANCED_SELECTION_SALT.encode("utf-8"))
    payload.extend(bytearray(1))
    payload.extend(normalized.encode("utf-8"))
    return _sha256_bytes(bytes(payload))


def _select_items(
    detailed_report: Mapping[str, Any], quotas: Mapping[str, int]
) -> tuple[list[tuple[str, dict[str, Any]]], int, int]:
    raw_items = detailed_report.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("detailed core-coverage report must contain items")

    primary_by_aspect: dict[str, list[dict[str, Any]]] = {aspect: [] for aspect in ASPECT_ORDER}
    detector_anomaly_holdout_count = 0
    for raw_item in raw_items:
        if not isinstance(raw_item, dict) or raw_item.get("core_complete") is not True:
            continue
        quality = raw_item.get("quality_status")
        coverage = raw_item.get("coverage")
        if not isinstance(quality, dict) or not isinstance(coverage, dict):
            continue
        if quality.get("detector_disagreement") is True:
            detector_anomaly_holdout_count += 1
            continue
        if quality.get("pose2_detection_count") != 1:
            continue
        aspect = coverage.get("aspect_bucket")
        if aspect not in primary_by_aspect:
            continue
        _safe_flat_relative_path(raw_item.get("source_relative_path"))
        primary_by_aspect[aspect].append(raw_item)

    selected: list[tuple[str, dict[str, Any]]] = []
    for aspect in ASPECT_ORDER:
        ranked = sorted(
            primary_by_aspect[aspect],
            key=lambda item: rank_source_relative_path(item["source_relative_path"]),
        )
        required = quotas[aspect]
        if len(ranked) < required:
            raise ValueError(
                f"quota shortfall for {aspect}: need {required}, have {len(ranked)} primary candidates"
            )
        selected.extend((aspect, item) for item in ranked[:required])

    return selected, sum(len(items) for items in primary_by_aspect.values()), detector_anomaly_holdout_count


def _source_identity(source_root: Path, source_relative_path: str) -> dict[str, Any]:
    normalized = _safe_flat_relative_path(source_relative_path)
    source_path = source_root / normalized
    try:
        resolved = source_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"selected source is unavailable: {normalized}") from exc
    if not resolved.is_file() or not resolved.is_relative_to(source_root):
        raise ValueError(f"selected source escapes canonical root: {normalized}")

    # One full disk read supplies both the content hash and image-header dimensions.
    payload = source_path.read_bytes()
    try:
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
            source_format = image.format
    except (OSError, ValueError, SyntaxError) as exc:
        raise ValueError(f"selected source cannot be decoded for dimensions: {normalized}") from exc
    if not isinstance(source_format, str) or not source_format:
        raise ValueError(f"selected source has no recognized format: {normalized}")
    if width <= 0 or height <= 0:
        raise ValueError(f"selected source has invalid dimensions: {normalized}")
    return {
        "source_relative_path": normalized,
        "source_sha256": _sha256_bytes(payload),
        "source_dimensions": {"width": int(width), "height": int(height)},
        "source_format": source_format,
        "source_byte_read_count": 1,
    }


def _artifact_availability(item: Mapping[str, Any]) -> tuple[dict[str, bool], dict[str, str]]:
    raw_artifacts = item.get("artifacts")
    if not isinstance(raw_artifacts, dict):
        raise ValueError("detailed core-coverage item lacks artifact facts")
    availability: dict[str, bool] = {}
    statuses: dict[str, str] = {}
    for name, raw_fact in sorted(raw_artifacts.items()):
        if not isinstance(name, str) or not isinstance(raw_fact, dict):
            raise ValueError("detailed core-coverage item has malformed artifact facts")
        status = raw_fact.get("status")
        if not isinstance(status, str):
            raise ValueError("detailed core-coverage artifact status must be a string")
        availability[name] = status == "readable"
        statuses[name] = status
    if not availability:
        raise ValueError("detailed core-coverage item has no artifact facts")
    return availability, statuses


def freeze_coverage_balanced_manifest(
    source_root: Path,
    derived_root: Path,
    *,
    expected_membership_sha256: str,
    expected_detail_sha256: str,
    quotas: Mapping[str, int] = DEFAULT_QUOTAS,
    limit: int = 500,
    program_id: str = "stratum-contextual-specialist-research",
    parent_issue: int = 4,
    manifest_id: str = "first500-coverage-balanced-candidate-v1",
) -> dict[str, Any]:
    """Build an in-memory, source-hashed candidate manifest.

    The expected audit identities are checked before source-byte reads. This
    fail-closes if the first-N membership or the per-item derived metadata has
    drifted from the design basis.
    """
    source_root = _resolved_directory(source_root, "source_root")
    derived_root = _resolved_directory(derived_root, "derived_root")
    expected_membership_sha256 = _require_sha256(
        expected_membership_sha256, "expected_membership_sha256"
    )
    expected_detail_sha256 = _require_sha256(expected_detail_sha256, "expected_detail_sha256")
    normalized_quotas = _require_quotas(quotas)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if not isinstance(parent_issue, int) or isinstance(parent_issue, bool) or parent_issue <= 0:
        raise ValueError("parent_issue must be a positive integer")
    if not isinstance(program_id, str) or not program_id.strip() or program_id != program_id.strip():
        raise ValueError("program_id must be a canonical non-empty string")
    if not isinstance(manifest_id, str) or not manifest_id.strip() or manifest_id != manifest_id.strip():
        raise ValueError("manifest_id must be a canonical non-empty string")

    detailed_report = audit_core_coverage(source_root, derived_root, limit=limit)
    compact_report = compact_coverage_report(detailed_report)
    cohort = compact_report["cohort"]
    detail_provenance = compact_report["detail_provenance"]
    observed_membership = cohort.get("membership_sha256")
    observed_detail = detail_provenance.get("item_details_sha256")
    if observed_membership != expected_membership_sha256:
        raise ValueError(
            "coverage audit membership identity drifted before source reads: "
            f"expected {expected_membership_sha256}, observed {observed_membership}"
        )
    if observed_detail != expected_detail_sha256:
        raise ValueError(
            "coverage audit detail identity drifted before source reads: "
            f"expected {expected_detail_sha256}, observed {observed_detail}"
        )

    selected, primary_pool_count, detector_anomaly_holdout_count = _select_items(
        detailed_report, normalized_quotas
    )
    items: list[dict[str, Any]] = []
    for aspect, coverage_item in selected:
        source_relative_path = _safe_flat_relative_path(coverage_item.get("source_relative_path"))
        identity = _source_identity(source_root, source_relative_path)
        availability, statuses = _artifact_availability(coverage_item)
        quality = coverage_item["quality_status"]
        items.append(
            {
                "image_id": coverage_item["image_id"],
                **identity,
                "selection": {
                    "aspect_bucket": aspect,
                    "rank_sha256": rank_source_relative_path(source_relative_path),
                },
                "artifact_availability": availability,
                "artifact_readability_status": statuses,
                "quality_status": {
                    "pose2_detection_count": quality["pose2_detection_count"],
                    "detector_disagreement": False,
                    "caption_semantics": "excluded",
                },
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "first500-coverage-balanced-candidate-manifest",
        "status": "PENDING_PRE_COMPUTE_NON_EXECUTING",
        "manifest_id": manifest_id,
        "program_id": program_id,
        "parent_issue": parent_issue,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "canonical_source_root": str(source_root),
        "derived_artifact_root": str(derived_root),
        "freeze_implementation": {
            "module": "research_harness.coverage_freeze",
            "source_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        },
        "audit_binding": {
            "limit": limit,
            "membership_sha256": observed_membership,
            "item_details_sha256": observed_detail,
            "eligible_source_count": cohort["eligible_source_count"],
            "selected_first_n_count": cohort["selected_count"],
            "source_content_read_count_in_audit": compact_report["source_content_read_count"],
            "audit_implementation": compact_report["implementation"],
        },
        "selection": {
            "source_cohort": "first eligible canonical filenames in bytewise POSIX order",
            "primary_pool_rule": "complete existing core artifacts and exactly one pose2 detection",
            "detector_anomaly_handling": "holdout_quality_abstention_not_caption_content",
            "primary_pool_count": primary_pool_count,
            "detector_anomaly_holdout_count": detector_anomaly_holdout_count,
            "quotas": normalized_quotas,
            "ranking": "SHA-256 of UTF-8 selection salt plus normalized source_relative_path; lowest digest per aspect quota",
            "ranking_salt_description": "stratum-first500-coverage-design-v1 followed by one NUL separator",
        },
        "source_byte_read_count": len(items),
        "items": items,
        "limitations": [
            "This is a non-executing pre-compute candidate manifest, not Stage-B authorization.",
            "Only selected source images were read, exactly once each, to bind hashes and dimensions after deterministic selection.",
            "Artifact availability/readability is not semantic validation, model provenance, or a caption-quality result.",
            "Detector disagreement remains a quality/anomaly holdout and is not caption or representation content.",
            "Existing determinations/caption2/t52 availability does not make an evidence-only comparison executable or make t52 a context4k substitute.",
        ],
        "explicit_non_authorizations": [
            "model invocation or download",
            "GPU or scheduler action",
            "canonical or derived corpus mutation",
            "artifact generation or backfill",
            "Stage-B execution",
            "merge or direct main push",
            "empirical PASS or FAIL claim",
        ],
    }
    manifest["manifest_fingerprint"] = _sha256_bytes(_canonical_json(manifest).encode("utf-8"))
    return manifest


def write_coverage_balanced_manifest(
    manifest: Mapping[str, Any],
    output_path: Path,
    *,
    allowed_output_roots: tuple[Path, ...],
    protected_roots: tuple[Path, ...],
) -> None:
    """Publish a new manifest atomically without corpus writes or replacement."""
    if not allowed_output_roots:
        raise ValueError("at least one allowed output root is required")
    allowed = tuple(_resolved_directory(root, "allowed output root") for root in allowed_output_roots)
    protected = tuple(_resolved_directory(root, "protected corpus root") for root in protected_roots)
    output = output_path.resolve(strict=False)
    if any(output.is_relative_to(root) for root in protected):
        raise ValueError("output must not be inside a protected corpus root")
    if not any(output.is_relative_to(root) for root in allowed):
        raise ValueError("output must be inside an allowed output root")
    if output.exists():
        raise ValueError("coverage-balanced manifest already exists; choose a new versioned path")
    if output.is_dir():
        raise ValueError("coverage-balanced manifest output must be a file")

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (_canonical_json(dict(manifest)) + "\n").encode("utf-8")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        try:
            # A hard-link publish is create-only: unlike Path.replace it cannot
            # overwrite a concurrently created versioned manifest on NFS.
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError as exc:
            raise ValueError("coverage-balanced manifest already exists; choose a new versioned path") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_program(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read program JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("program JSON must be an object")
    try:
        validate_program(value)
    except ContractError as exc:
        raise ValueError(f"invalid program: {exc}") from exc
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a source-hashed coverage-balanced candidate cohort without inference"
    )
    parser.add_argument("program", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-membership-sha256", required=True)
    parser.add_argument("--expected-detail-sha256", required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--manifest-id", default="first500-coverage-balanced-candidate-v1")
    parser.add_argument("--parent-issue", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        program = _read_program(args.program)
        canonical_source = program["canonical_source"]
        source_root = Path(canonical_source["path"])
        derived_root = Path(canonical_source["derived_tree"])
        allowed_roots = tuple(Path(root) for root in program["artifact_policy"]["approved_output_roots"])
        manifest = freeze_coverage_balanced_manifest(
            source_root,
            derived_root,
            expected_membership_sha256=args.expected_membership_sha256,
            expected_detail_sha256=args.expected_detail_sha256,
            limit=args.limit,
            program_id=program["program_id"],
            parent_issue=args.parent_issue,
            manifest_id=args.manifest_id,
        )
        write_coverage_balanced_manifest(
            manifest,
            args.output,
            allowed_output_roots=allowed_roots,
            protected_roots=(source_root, derived_root),
        )
    except ValueError as exc:
        print(f"coverage-balanced-freeze: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "manifest_id": manifest["manifest_id"],
                "manifest_fingerprint": manifest["manifest_fingerprint"],
                "selected_count": len(manifest["items"]),
                "source_byte_read_count": manifest["source_byte_read_count"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
