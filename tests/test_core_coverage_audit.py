"""Synthetic-only tests for the read-only core-artifact coverage audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from research_harness.core_coverage import (
    CORE_ARTIFACTS,
    LATER_CHAIN_ARTIFACTS,
    audit_core_coverage,
    compact_coverage_report,
    write_coverage_audit,
)


def _write_core_artifacts(derived: Path, *, pose_detections: int, height: int, width: int) -> None:
    derived.mkdir(parents=True)
    np.save(derived / "pose2.npy", np.ones((pose_detections, 308, 3), dtype=np.float32))
    np.save(derived / "seg2.npy", np.ones((height, width), dtype=np.uint8))
    np.save(derived / "normal2.npy", np.ones((height, width, 3), dtype=np.float16))
    np.save(derived / "pointmap.npy", np.ones((height, width, 3), dtype=np.float16))
    np.save(derived / "matting.npy", np.ones((height, width), dtype=np.float16))


def _write_legacy_chain(derived: Path) -> None:
    (derived / "caption.txt").write_text("legacy caption\n", encoding="utf-8")
    np.save(derived / "t5_hidden.npy", np.ones((2, 3), dtype=np.float16))
    np.save(derived / "t5_mask.npy", np.ones((2,), dtype=np.uint8))


def _write_later_chain(derived: Path) -> None:
    (derived / "determinations.json").write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    (derived / "caption2.txt").write_text("context caption\n", encoding="utf-8")
    np.save(derived / "t52_hidden.npy", np.ones((2, 3), dtype=np.float16))
    np.save(derived / "t52_mask.npy", np.ones((2,), dtype=np.uint8))


def test_audit_reports_bytewise_first_cohort_and_chain_limitations_without_source_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "approved"
    derived_root = tmp_path / "stratum"
    source_root.mkdir()
    for name in ("b.jpg", "A.webp", "c.png", "ignore.txt"):
        (source_root / name).write_bytes(b"not decoded by this audit")

    _write_core_artifacts(derived_root / "A", pose_detections=1, height=120, width=80)
    _write_later_chain(derived_root / "A")
    _write_legacy_chain(derived_root / "A")

    _write_core_artifacts(derived_root / "b", pose_detections=2, height=80, width=160)
    _write_legacy_chain(derived_root / "b")

    (derived_root / "c").mkdir(parents=True)
    np.save(derived_root / "c" / "pose2.npy", np.ones((1, 308, 3), dtype=np.float32))
    np.save(derived_root / "c" / "seg2.npy", np.ones((80, 80), dtype=np.uint8))
    (derived_root / "c" / "normal2.npy").write_bytes(b"not an npy file")
    np.save(derived_root / "c" / "pointmap.npy", np.ones((80, 80, 3), dtype=np.float16))
    np.save(derived_root / "c" / "matting.npy", np.ones((80, 80), dtype=np.float16))

    original_open = Path.open

    def source_read_guard(path: Path, *args: object, **kwargs: object):
        if path.parent == source_root:
            raise AssertionError(f"audit must not open source content: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", source_read_guard)
    report = audit_core_coverage(source_root, derived_root, limit=3)

    assert report["source_content_read_count"] == 0
    assert report["implementation"]["module"] == "research_harness.core_coverage"
    assert len(report["implementation"]["source_sha256"]) == 64
    assert report["cohort"]["source_relative_paths"] == ["A.webp", "b.jpg", "c.png"]
    assert report["summary"]["core_complete_count"] == 2
    assert report["summary"]["later_chain_complete_count"] == 1
    assert report["summary"]["legacy_chain_complete_count"] == 2
    assert report["summary"]["aspect_buckets"] == {
        "landscape": 1,
        "portrait": 1,
        "squareish": 1,
    }
    assert report["summary"]["pose2_detection_counts"] == {"1": 2, "2": 1}
    assert report["summary"]["detector_disagreement_count"] == 1
    assert report["summary"]["later_chain_complete_source_relative_paths"] == ["A.webp"]

    assert tuple(report["artifact_sets"]["core"]) == CORE_ARTIFACTS
    assert tuple(report["artifact_sets"]["later_chain"]) == LATER_CHAIN_ARTIFACTS
    c_record = report["items"][2]
    assert c_record["artifacts"]["normal2.npy"]["status"] == "unreadable"
    assert c_record["core_complete"] is False

    expected_digest = hashlib.sha256(b"A.webp\nb.jpg\nc.png\n").hexdigest()
    assert report["cohort"]["membership_sha256"] == expected_digest


def test_audit_rejects_non_positive_limit_and_output_inside_protected_corpus_roots(tmp_path: Path) -> None:
    source_root = tmp_path / "approved"
    derived_root = tmp_path / "stratum"
    source_root.mkdir()
    derived_root.mkdir()

    with pytest.raises(ValueError, match="limit"):
        audit_core_coverage(source_root, derived_root, limit=0)

    report = {
        "schema_version": 1,
        "kind": "core-artifact-coverage-audit",
    }
    with pytest.raises(ValueError, match="protected"):
        write_coverage_audit(report, source_root / "report.json", protected_roots=(source_root, derived_root))

    output = tmp_path / "research" / "report.json"
    write_coverage_audit(report, output, protected_roots=(source_root, derived_root))
    assert json.loads(output.read_text(encoding="utf-8")) == report

    with pytest.raises(ValueError, match="already exists"):
        write_coverage_audit(report, output, protected_roots=(source_root, derived_root))

    write_coverage_audit(
        report,
        output,
        protected_roots=(source_root, derived_root),
        overwrite=True,
    )


def test_audit_keeps_detector_disagreement_as_quality_status_not_caption_content(tmp_path: Path) -> None:
    source_root = tmp_path / "approved"
    derived_root = tmp_path / "stratum"
    source_root.mkdir()
    (source_root / "sample.jpg").write_bytes(b"not decoded")
    _write_core_artifacts(derived_root / "sample", pose_detections=3, height=80, width=80)

    report = audit_core_coverage(source_root, derived_root, limit=1)

    item = report["items"][0]
    assert item["quality_status"] == {
        "pose2_detection_count": 3,
        "detector_disagreement": True,
        "caption_semantics": "excluded",
    }
    assert "caption" not in item
    assert "subject_count" not in item


def test_missing_pose_artifact_is_not_misreported_as_detector_disagreement(tmp_path: Path) -> None:
    source_root = tmp_path / "approved"
    derived_root = tmp_path / "stratum"
    source_root.mkdir()
    (source_root / "sample.jpg").write_bytes(b"not decoded")
    (derived_root / "sample").mkdir(parents=True)

    report = audit_core_coverage(source_root, derived_root, limit=1)

    assert report["items"][0]["quality_status"]["pose2_detection_count"] is None
    assert report["items"][0]["quality_status"]["detector_disagreement"] is False


def test_compact_report_binds_hidden_item_details_and_keeps_quality_exceptions(tmp_path: Path) -> None:
    source_root = tmp_path / "approved"
    derived_root = tmp_path / "stratum"
    source_root.mkdir()
    for name in ("a.jpg", "b.jpg"):
        (source_root / name).write_bytes(b"not decoded")
    _write_core_artifacts(derived_root / "a", pose_detections=1, height=80, width=80)
    _write_core_artifacts(derived_root / "b", pose_detections=2, height=80, width=80)

    compact = compact_coverage_report(audit_core_coverage(source_root, derived_root, limit=2))

    assert "items" not in compact
    assert compact["detail_provenance"]["item_details_included"] is False
    assert compact["detail_provenance"]["detector_disagreement_items"] == [
        {"source_relative_path": "b.jpg", "pose2_detection_count": 2}
    ]
    assert len(compact["detail_provenance"]["item_details_sha256"]) == 64
