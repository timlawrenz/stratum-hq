"""Frozen-cohort source verification tests (no canonical-corpus access).

These run entirely on tmp_path fixtures so they stay green on the neutral CI
runner (no /mnt/nas-ai-models mount), exercising the same schema the frozen
plans use: ``pilot_manifest.items`` with ``image_id`` / ``source_relative_path``
/ ``source_sha256`` plus ``source_root``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest

from research_harness.cohort_verify import CohortVerifyError, verify_plan_sources


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_plan(root, source_root, items) -> dict:
    plan = {
        "schema_version": 1,
        "pilot_manifest": {
            "source_root": str(source_root),
            "items": items,
        },
    }
    path = root / "plan.json"
    path.write_text(json.dumps(plan))
    return path


def test_all_intact_when_present_and_matching(tmp_path) -> None:
    source_root = tmp_path / "approved"
    source_root.mkdir()
    blob = b"image-bytes-a"
    (source_root / "a.jpg").write_bytes(blob)
    items = [{
        "image_id": "aaa",
        "source_relative_path": "a.jpg",
        "source_sha256": _sha(blob),
    }]
    _write_plan(tmp_path, source_root, items)

    report = verify_plan_sources(tmp_path / "plan.json")

    assert report["checked"] == 1
    assert report["present"] == 1
    assert report["missing"] == 0
    assert report["sha_mismatch"] == 0
    assert report["all_intact"] is True
    assert report["items"][0]["status"] == "present-match"


def test_missing_source_reported_not_intact(tmp_path) -> None:
    source_root = tmp_path / "approved"
    source_root.mkdir()
    blob = b"image-bytes-b"
    (source_root / "b.jpg").write_bytes(blob)
    items = [
        {"image_id": "bbb", "source_relative_path": "b.jpg",
         "source_sha256": _sha(blob)},
        # present in manifest but absent from the tree -> the #132 failure mode
        {"image_id": "g1", "source_relative_path": "g1.jpg",
         "source_sha256": _sha(b"whatever")},
    ]
    _write_plan(tmp_path, source_root, items)

    report = verify_plan_sources(tmp_path / "plan.json")

    assert report["checked"] == 2
    assert report["present"] == 1
    assert report["missing"] == 1
    assert report["all_intact"] is False
    statuses = {e["image_id"]: e["status"] for e in report["items"]}
    assert statuses == {"bbb": "present-match", "g1": "missing"}


def test_sha_mismatch_reported(tmp_path) -> None:
    source_root = tmp_path / "approved"
    source_root.mkdir()
    (source_root / "c.jpg").write_bytes(b"actual-bytes-c")
    items = [{
        "image_id": "ccc",
        "source_relative_path": "c.jpg",
        "source_sha256": _sha(b"different-expected-bytes"),
    }]
    _write_plan(tmp_path, source_root, items)

    report = verify_plan_sources(tmp_path / "plan.json")

    assert report["all_intact"] is False
    assert report["sha_mismatch"] == 1
    assert report["items"][0]["status"] == "present-mismatch"


def test_no_expected_sha_still_ok_when_present(tmp_path) -> None:
    source_root = tmp_path / "approved"
    source_root.mkdir()
    (source_root / "d.jpg").write_bytes(b"d-bytes")
    items = [{"image_id": "ddd", "source_relative_path": "d.jpg"}]
    _write_plan(tmp_path, source_root, items)

    assert verify_plan_sources(tmp_path / "plan.json")["all_intact"] is True


def test_path_traversal_rejected(tmp_path) -> None:
    source_root = tmp_path / "approved"
    source_root.mkdir()
    items = [{
        "image_id": "evil",
        "source_relative_path": "../outside.jpg",
        "source_sha256": None,
    }]
    _write_plan(tmp_path, source_root, items)

    with pytest.raises(CohortVerifyError, match="escapes the source root"):
        verify_plan_sources(tmp_path / "plan.json")


def test_missing_plan_fails(tmp_path) -> None:
    with pytest.raises(CohortVerifyError, match="unable to read plan"):
        verify_plan_sources(tmp_path / "nope.json")


def test_malformed_plan_fails(tmp_path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"pilot_manifest": {"source_root": str(tmp_path)}}))
    with pytest.raises(CohortVerifyError, match="items must be a JSON array"):
        verify_plan_sources(path)


def test_cli_exit_codes(tmp_path) -> None:
    source_root = tmp_path / "approved"
    source_root.mkdir()
    (source_root / "ok.jpg").write_bytes(b"ok")
    good = _write_plan(tmp_path, source_root, [
        {"image_id": "ok", "source_relative_path": "ok.jpg", "source_sha256": _sha(b"ok")},
    ])
    bad_plan = tmp_path / "bad.json"
    bad_plan.write_text(json.dumps({
        "pilot_manifest": {
            "source_root": str(source_root),
            "items": [
                {"image_id": "ok", "source_relative_path": "ok.jpg",
                 "source_sha256": _sha(b"ok")},
                {"image_id": "gone", "source_relative_path": "gone.jpg",
                 "source_sha256": _sha(b"gone")},
            ],
        }
    }))

    intact = subprocess.run(
        [sys.executable, "-m", "research_harness.cli", "verify-cohort-sources",
         str(good), "--json"],
        capture_output=True, text=True,
    )
    assert intact.returncode == 0
    assert json.loads(intact.stdout)["all_intact"] is True

    broken = subprocess.run(
        [sys.executable, "-m", "research_harness.cli", "verify-cohort-sources",
         str(bad_plan), "--json"],
        capture_output=True, text=True,
    )
    assert broken.returncode == 1
    assert json.loads(broken.stdout)["all_intact"] is False
