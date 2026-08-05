"""Read-only coverage audits for existing Stratum core artifacts.

The audit deliberately enumerates source *names* without opening source-image
bytes. It probes only named artifacts in a caller-supplied derived tree and
writes reports only outside both protected roots. It is a feasibility instrument,
not model execution, artifact generation, or a semantic data claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

CORE_ARTIFACTS = (
    "pose2.npy",
    "seg2.npy",
    "normal2.npy",
    "pointmap.npy",
    "matting.npy",
)
LATER_CHAIN_ARTIFACTS = (
    "determinations.json",
    "caption2.txt",
    "t52_hidden.npy",
    "t52_mask.npy",
)
LEGACY_CHAIN_ARTIFACTS = (
    "caption.txt",
    "t5_hidden.npy",
    "t5_mask.npy",
)
IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


def _resolved_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} must be an existing directory: {path}") from exc
    if not resolved.is_dir():
        raise ValueError(f"{label} must be an existing directory: {path}")
    return resolved


def _ordered_eligible_source_paths(source_root: Path) -> list[Path]:
    """List flat source candidates by bytewise relative-path order without opens."""
    candidates: list[Path] = []
    for candidate in source_root.iterdir():
        if candidate.suffix.lower() not in IMAGE_EXTENSIONS or not candidate.is_file():
            continue
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if not resolved.is_relative_to(source_root):
            continue
        candidates.append(candidate)
    return sorted(
        candidates,
        key=lambda path: os.fsencode(path.relative_to(source_root).as_posix()),
    )


def _probe_npy(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing"}
    array: np.ndarray[Any, Any] | None = None
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        probe = np.asarray(array)
        if probe.size:
            # Touch bounded endpoints so a valid header alone cannot masquerade
            # as a readable array. This remains an availability probe, not an
            # integrity or semantic validation of the full tensor.
            _ = probe.reshape(-1)[0]
            _ = probe.reshape(-1)[-1]
        return {
            "status": "readable",
            "shape": [int(dimension) for dimension in probe.shape],
            "dtype": str(probe.dtype),
        }
    except (OSError, ValueError, TypeError):
        return {"status": "unreadable"}
    finally:
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()


def _probe_json(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {"status": "missing"}
    try:
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {"status": "unreadable"}
    return {"status": "readable"}


def _probe_text(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {"status": "missing"}
    try:
        value = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {"status": "unreadable"}
    return {"status": "readable" if value.strip() else "empty"}


def _probe_artifact(artifact_name: str, derived_dir: Path) -> dict[str, Any]:
    path = derived_dir / artifact_name
    if artifact_name.endswith(".npy"):
        return _probe_npy(path)
    if artifact_name.endswith(".json"):
        return _probe_json(path)
    if artifact_name.endswith(".txt"):
        return _probe_text(path)
    raise ValueError(f"unsupported audit artifact: {artifact_name}")


def _readable(artifact: dict[str, Any]) -> bool:
    return artifact.get("status") == "readable"


def _aspect_bucket(matting: dict[str, Any]) -> str:
    shape = matting.get("shape")
    if not _readable(matting) or not isinstance(shape, list) or len(shape) < 2:
        return "unavailable"
    height, width = shape[0], shape[1]
    if not isinstance(height, int) or not isinstance(width, int) or height <= 0:
        return "unavailable"
    ratio = width / height
    if ratio < 0.9:
        return "portrait"
    if ratio <= 1.1:
        return "squareish"
    return "landscape"


def _pose_detection_count(pose: dict[str, Any]) -> int | None:
    shape = pose.get("shape")
    if not _readable(pose) or not isinstance(shape, list) or len(shape) != 3:
        return None
    count = shape[0]
    return count if isinstance(count, int) and count >= 0 else None


def _chain_complete(artifacts: dict[str, dict[str, Any]], names: tuple[str, ...]) -> bool:
    return all(_readable(artifacts[name]) for name in names)


def _cohort_membership_sha256(relative_paths: list[str]) -> str:
    digest = hashlib.sha256()
    for relative_path in relative_paths:
        digest.update(os.fsencode(relative_path))
        digest.update(b"\n")
    return digest.hexdigest()


def audit_core_coverage(source_root: Path, derived_root: Path, *, limit: int) -> dict[str, Any]:
    """Audit a deterministic first-N source-name cohort against existing artifacts.

    Source files are listed and `stat`-checked only; source image bytes, pixels,
    hashes, and dimensions are never read. All artifact reads are confined to
    the matching derived-item directories.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")

    source_root = _resolved_directory(source_root, "source_root")
    derived_root = _resolved_directory(derived_root, "derived_root")
    eligible = _ordered_eligible_source_paths(source_root)
    selected = eligible[:limit]
    relative_paths = [path.relative_to(source_root).as_posix() for path in selected]

    items: list[dict[str, Any]] = []
    for source_path, relative_path in zip(selected, relative_paths, strict=True):
        derived_dir = derived_root / source_path.stem
        artifact_names = CORE_ARTIFACTS + LATER_CHAIN_ARTIFACTS + LEGACY_CHAIN_ARTIFACTS
        artifacts = {
            artifact_name: _probe_artifact(artifact_name, derived_dir)
            for artifact_name in artifact_names
        }
        core_complete = _chain_complete(artifacts, CORE_ARTIFACTS)
        later_complete = _chain_complete(artifacts, LATER_CHAIN_ARTIFACTS)
        legacy_complete = _chain_complete(artifacts, LEGACY_CHAIN_ARTIFACTS)
        pose_detection_count = _pose_detection_count(artifacts["pose2.npy"])
        items.append(
            {
                "source_relative_path": relative_path,
                "image_id": source_path.stem,
                "artifacts": artifacts,
                "core_complete": core_complete,
                "later_chain_complete": later_complete,
                "legacy_chain_complete": legacy_complete,
                "coverage": {"aspect_bucket": _aspect_bucket(artifacts["matting.npy"])},
                "quality_status": {
                    "pose2_detection_count": pose_detection_count,
                    "detector_disagreement": (
                        pose_detection_count is not None and pose_detection_count != 1
                    ),
                    "caption_semantics": "excluded",
                },
            }
        )

    artifact_counts = {
        artifact_name: sum(_readable(item["artifacts"][artifact_name]) for item in items)
        for artifact_name in CORE_ARTIFACTS + LATER_CHAIN_ARTIFACTS + LEGACY_CHAIN_ARTIFACTS
    }
    aspect_buckets = Counter(item["coverage"]["aspect_bucket"] for item in items)
    pose_detection_counts = Counter(
        str(item["quality_status"]["pose2_detection_count"])
        for item in items
        if item["quality_status"]["pose2_detection_count"] is not None
    )
    later_paths = [item["source_relative_path"] for item in items if item["later_chain_complete"]]

    return {
        "schema_version": 1,
        "kind": "core-artifact-coverage-audit",
        "status": "PRE_COMPUTE_READ_ONLY",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "implementation": {
            "module": "research_harness.core_coverage",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "source_root": str(source_root),
        "derived_root": str(derived_root),
        "source_content_read_count": 0,
        "artifact_sets": {
            "core": list(CORE_ARTIFACTS),
            "later_chain": list(LATER_CHAIN_ARTIFACTS),
            "legacy_chain": list(LEGACY_CHAIN_ARTIFACTS),
        },
        "cohort": {
            "selection_rule": "first limit eligible flat source filenames in bytewise POSIX relative-path order",
            "requested_limit": limit,
            "eligible_source_count": len(eligible),
            "selected_count": len(selected),
            "source_relative_paths": relative_paths,
            "membership_sha256": _cohort_membership_sha256(relative_paths),
        },
        "summary": {
            "core_artifact_readable_counts": {
                artifact_name: artifact_counts[artifact_name] for artifact_name in CORE_ARTIFACTS
            },
            "later_artifact_readable_counts": {
                artifact_name: artifact_counts[artifact_name] for artifact_name in LATER_CHAIN_ARTIFACTS
            },
            "legacy_artifact_readable_counts": {
                artifact_name: artifact_counts[artifact_name] for artifact_name in LEGACY_CHAIN_ARTIFACTS
            },
            "core_complete_count": sum(item["core_complete"] for item in items),
            "later_chain_complete_count": len(later_paths),
            "legacy_chain_complete_count": sum(item["legacy_chain_complete"] for item in items),
            "aspect_buckets": dict(sorted(aspect_buckets.items())),
            "pose2_detection_counts": dict(sorted(pose_detection_counts.items())),
            "detector_disagreement_count": sum(
                item["quality_status"]["detector_disagreement"] for item in items
            ),
            "later_chain_complete_source_relative_paths": later_paths,
        },
        "limitations": [
            "Source images were never opened, decoded, hashed, or dimension-read; the cohort is bound here only by filename membership and digest.",
            "Array readability means NPY header parsing plus bounded endpoint access, not full-tensor integrity or semantic validation.",
            "Detector disagreement is a quality status and is excluded from caption semantics.",
            "No model invocation, GPU/scheduler action, artifact generation, corpus mutation, or empirical comparison occurred.",
        ],
        "items": items,
    }


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compact_coverage_report(report: dict[str, Any]) -> dict[str, Any]:
    """Retain a reproducible aggregate audit without committing every probe row.

    The compact form keeps cohort membership plus a digest of omitted per-item
    details. It also preserves the exceptional rows needed to reason about core
    incompleteness and detector-quality abstentions.
    """
    raw_items = report.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("coverage report must contain an item list before compaction")
    items = [item for item in raw_items if isinstance(item, dict)]
    if len(items) != len(raw_items):
        raise ValueError("coverage report item list must contain objects")

    def core_exceptions(item: dict[str, Any]) -> dict[str, Any] | None:
        if item.get("core_complete") is True:
            return None
        artifacts = item.get("artifacts", {})
        missing = [
            name
            for name in CORE_ARTIFACTS
            if not isinstance(artifacts.get(name), dict) or artifacts[name].get("status") != "readable"
        ]
        return {
            "source_relative_path": item.get("source_relative_path"),
            "unreadable_or_missing_core_artifacts": missing,
        }

    def legacy_exceptions(item: dict[str, Any]) -> dict[str, Any] | None:
        if item.get("legacy_chain_complete") is True:
            return None
        artifacts = item.get("artifacts", {})
        missing = [
            name
            for name in LEGACY_CHAIN_ARTIFACTS
            if not isinstance(artifacts.get(name), dict) or artifacts[name].get("status") != "readable"
        ]
        return {
            "source_relative_path": item.get("source_relative_path"),
            "unreadable_or_missing_legacy_artifacts": missing,
        }

    detector_exceptions = [
        {
            "source_relative_path": item.get("source_relative_path"),
            "pose2_detection_count": item.get("quality_status", {}).get("pose2_detection_count"),
        }
        for item in items
        if item.get("quality_status", {}).get("detector_disagreement") is True
    ]
    compact = {key: value for key, value in report.items() if key != "items"}
    compact["detail_provenance"] = {
        "item_details_included": False,
        "item_details_sha256": _canonical_json_sha256(items),
        "core_incomplete_items": [
            exception for item in items if (exception := core_exceptions(item)) is not None
        ],
        "legacy_chain_incomplete_items": [
            exception for item in items if (exception := legacy_exceptions(item)) is not None
        ],
        "detector_disagreement_items": detector_exceptions,
    }
    return compact


def write_coverage_audit(
    report: dict[str, Any],
    output_path: Path,
    *,
    protected_roots: tuple[Path, ...],
    overwrite: bool = False,
) -> None:
    """Write a report atomically without implicit replacement or corpus writes."""
    protected = tuple(_resolved_directory(root, "protected_root") for root in protected_roots)
    output = output_path.resolve(strict=False)
    if any(output.is_relative_to(root) for root in protected):
        raise ValueError("coverage-audit output must not be inside a protected corpus root")
    if output.exists() and not overwrite:
        raise ValueError("coverage-audit output already exists; choose a new versioned path")
    if output.is_dir():
        raise ValueError("coverage-audit output must be a file path")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only audit of existing Stratum core coverage")
    parser.add_argument("source_root", type=Path)
    parser.add_argument("derived_root", type=Path)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-item-details",
        action="store_true",
        help="persist all per-item probe details instead of the compact provenance-bound summary",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace an existing non-corpus report path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        detailed_report = audit_core_coverage(args.source_root, args.derived_root, limit=args.limit)
        report = detailed_report if args.include_item_details else compact_coverage_report(detailed_report)
        write_coverage_audit(
            report,
            args.output,
            protected_roots=(args.source_root, args.derived_root),
            overwrite=args.overwrite,
        )
    except ValueError as exc:
        print(f"core-coverage-audit: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
