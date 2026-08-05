"""Synthetic-only coverage-balanced first-500 candidate-freeze tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from research_harness.core_coverage import audit_core_coverage, compact_coverage_report
from research_harness.coverage_freeze import (
    COVERAGE_BALANCED_SELECTION_SALT,
    freeze_coverage_balanced_manifest,
    rank_source_relative_path,
    write_coverage_balanced_manifest,
)


def _write_source(path: Path, *, size: tuple[int, int]) -> None:
    Image.new("RGB", size, color=(10, 20, 30)).save(path)


def _write_core_artifacts(
    derived_dir: Path,
    *,
    pose_detections: int = 1,
    height: int,
    width: int,
) -> None:
    derived_dir.mkdir(parents=True)
    np.save(derived_dir / "pose2.npy", np.ones((pose_detections, 308, 3), dtype=np.float32))
    np.save(derived_dir / "seg2.npy", np.ones((height, width), dtype=np.uint8))
    np.save(derived_dir / "normal2.npy", np.ones((height, width, 3), dtype=np.float16))
    np.save(derived_dir / "pointmap.npy", np.ones((height, width, 3), dtype=np.float16))
    np.save(derived_dir / "matting.npy", np.ones((height, width), dtype=np.float16))


def _make_coverage_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "approved"
    derived_root = tmp_path / "stratum"
    source_root.mkdir()

    # Selection uses matting dimensions only; source images are not opened until
    # a candidate has been chosen by the deterministic path hash.
    fixture = {
        "p1.jpg": ((80, 120), 1),
        "p2.jpg": ((80, 120), 1),
        "p3.jpg": ((80, 120), 1),
        "s1.jpg": ((100, 100), 1),
        "s2.jpg": ((100, 100), 1),
        "l1.jpg": ((140, 80), 1),
        "l2.jpg": ((140, 80), 1),
        # This row has the right framing but is a detector-quality abstention,
        # never a caption candidate.
        "z-anomaly.jpg": ((80, 120), 2),
    }
    for name, (size, pose_detections) in fixture.items():
        _write_source(source_root / name, size=size)
        _write_core_artifacts(
            derived_root / Path(name).stem,
            pose_detections=pose_detections,
            height=size[1],
            width=size[0],
        )
    return source_root, derived_root


def test_freeze_selects_coverage_quotas_before_reading_only_selected_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, derived_root = _make_coverage_fixture(tmp_path)
    detailed = audit_core_coverage(source_root, derived_root, limit=8)
    compact = compact_coverage_report(detailed)
    quotas = {"portrait": 2, "squareish": 1, "landscape": 1}

    source_reads: list[Path] = []
    original_read_bytes = Path.read_bytes

    def track_source_reads(path: Path) -> bytes:
        if path.parent == source_root:
            source_reads.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", track_source_reads)
    manifest = freeze_coverage_balanced_manifest(
        source_root,
        derived_root,
        expected_membership_sha256=compact["cohort"]["membership_sha256"],
        expected_detail_sha256=compact["detail_provenance"]["item_details_sha256"],
        quotas=quotas,
    )

    eligible_by_aspect = {
        aspect: sorted(
            (
                item["source_relative_path"]
                for item in detailed["items"]
                if item["core_complete"]
                and item["quality_status"]["pose2_detection_count"] == 1
                and item["coverage"]["aspect_bucket"] == aspect
            ),
            key=rank_source_relative_path,
        )[:count]
        for aspect, count in quotas.items()
    }
    expected_paths = [
        path
        for aspect in ("portrait", "squareish", "landscape")
        for path in eligible_by_aspect[aspect]
    ]

    assert COVERAGE_BALANCED_SELECTION_SALT == "stratum-first500-coverage-design-v1"
    expected_rank = hashlib.sha256(
        COVERAGE_BALANCED_SELECTION_SALT.encode("utf-8")
        + bytes(bytearray(1))
        + b"p1.jpg"
    ).hexdigest()
    assert rank_source_relative_path("p1.jpg") == expected_rank
    assert [item["source_relative_path"] for item in manifest["items"]] == expected_paths
    assert source_reads == [source_root / path for path in expected_paths]
    assert manifest["selection"]["primary_pool_count"] == 7
    assert manifest["selection"]["detector_anomaly_holdout_count"] == 1
    assert manifest["selection"]["quotas"] == quotas
    assert manifest["freeze_implementation"]["module"] == "research_harness.coverage_freeze"
    assert len(manifest["freeze_implementation"]["source_sha256"]) == 64
    assert manifest["source_byte_read_count"] == len(expected_paths)
    assert all(item["source_byte_read_count"] == 1 for item in manifest["items"])
    assert all("z-anomaly.jpg" != item["source_relative_path"] for item in manifest["items"])
    assert all(len(item["source_sha256"]) == 64 for item in manifest["items"])
    canonical_payload = json.dumps(
        {key: value for key, value in manifest.items() if key != "manifest_fingerprint"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert manifest["manifest_fingerprint"] == hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


def test_freeze_rejects_audit_identity_drift_before_any_source_byte_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, derived_root = _make_coverage_fixture(tmp_path)
    source_reads: list[Path] = []
    original_read_bytes = Path.read_bytes

    def track_source_reads(path: Path) -> bytes:
        if path.parent == source_root:
            source_reads.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", track_source_reads)

    with pytest.raises(ValueError, match="membership"):
        freeze_coverage_balanced_manifest(
            source_root,
            derived_root,
            expected_membership_sha256="0" * 64,
            expected_detail_sha256="1" * 64,
            quotas={"portrait": 1, "squareish": 1, "landscape": 1},
        )

    assert source_reads == []


def test_freeze_rejects_quota_shortfall_before_any_source_byte_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, derived_root = _make_coverage_fixture(tmp_path)
    detailed = audit_core_coverage(source_root, derived_root, limit=8)
    compact = compact_coverage_report(detailed)
    source_reads: list[Path] = []
    original_read_bytes = Path.read_bytes

    def track_source_reads(path: Path) -> bytes:
        if path.parent == source_root:
            source_reads.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", track_source_reads)

    with pytest.raises(ValueError, match="quota"):
        freeze_coverage_balanced_manifest(
            source_root,
            derived_root,
            expected_membership_sha256=compact["cohort"]["membership_sha256"],
            expected_detail_sha256=compact["detail_provenance"]["item_details_sha256"],
            quotas={"portrait": 4, "squareish": 1, "landscape": 1},
        )

    assert source_reads == []


def test_writer_requires_allowed_noncorpus_path_and_refuses_implicit_overwrite(tmp_path: Path) -> None:
    source_root = tmp_path / "approved"
    derived_root = tmp_path / "stratum"
    allowed_root = tmp_path / "research"
    source_root.mkdir()
    derived_root.mkdir()
    allowed_root.mkdir()
    manifest = {"schema_version": 1, "kind": "synthetic-freeze"}
    output = allowed_root / "manifest.json"

    write_coverage_balanced_manifest(
        manifest,
        output,
        allowed_output_roots=(allowed_root,),
        protected_roots=(source_root, derived_root),
    )
    assert output.exists()

    with pytest.raises(ValueError, match="already exists"):
        write_coverage_balanced_manifest(
            manifest,
            output,
            allowed_output_roots=(allowed_root,),
            protected_roots=(source_root, derived_root),
        )

    with pytest.raises(ValueError, match="allowed output root"):
        write_coverage_balanced_manifest(
            manifest,
            tmp_path / "outside.json",
            allowed_output_roots=(allowed_root,),
            protected_roots=(source_root, derived_root),
        )

    with pytest.raises(ValueError, match="protected corpus root"):
        write_coverage_balanced_manifest(
            manifest,
            source_root / "manifest.json",
            allowed_output_roots=(source_root,),
            protected_roots=(source_root, derived_root),
        )
