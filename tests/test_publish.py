"""Tests for publish and reconcile functionality."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from stratum.cli import parse_args
from stratum.publish import (
    _is_rate_limited,
    _parse_range_label,
    _retry_on_429,
    reconcile_hub_manifest,
)


# --- _parse_range_label ---

def test_parse_range_label_parquet():
    assert _parse_range_label("00000-09999.parquet") == (0, 9999)


def test_parse_range_label_tar():
    assert _parse_range_label("00000-00999.tar") == (0, 999)


def test_parse_range_label_high_range():
    assert _parse_range_label("70000-79999.parquet") == (70000, 79999)


def test_parse_range_label_invalid():
    assert _parse_range_label("manifest.json") is None
    assert _parse_range_label("README.md") is None
    assert _parse_range_label("") is None


def test_parse_range_label_no_extension():
    assert _parse_range_label("00000-09999") is None


# --- _is_rate_limited ---

def test_is_rate_limited_with_response_attr():
    exc = Exception("rate limited")
    exc.response = MagicMock(status_code=429)
    assert _is_rate_limited(exc) is True


def test_is_rate_limited_other_status():
    exc = Exception("server error")
    exc.response = MagicMock(status_code=500)
    assert _is_rate_limited(exc) is False


def test_is_rate_limited_string_fallback():
    exc = Exception("HTTP 429 Too Many Requests")
    assert _is_rate_limited(exc) is True


def test_is_rate_limited_unrelated():
    exc = Exception("connection timed out")
    assert _is_rate_limited(exc) is False


# --- _retry_on_429 ---

def test_retry_on_429_succeeds_immediately():
    fn = MagicMock(return_value="ok")
    result = _retry_on_429(fn, "a", key="b")
    assert result == "ok"
    fn.assert_called_once_with("a", key="b")


@patch("stratum.publish.time.sleep")
def test_retry_on_429_retries_then_succeeds(mock_sleep):
    exc = Exception("429")
    exc.response = MagicMock(status_code=429)
    fn = MagicMock(side_effect=[exc, exc, "ok"])

    result = _retry_on_429(fn, "arg")
    assert result == "ok"
    assert fn.call_count == 3
    assert mock_sleep.call_count == 2


@patch("stratum.publish.time.sleep")
def test_retry_on_429_exhausted(mock_sleep):
    exc = Exception("429")
    exc.response = MagicMock(status_code=429)
    fn = MagicMock(side_effect=exc)

    raised = False
    try:
        _retry_on_429(fn, "arg")
    except Exception:
        raised = True
    assert raised
    assert fn.call_count == 5  # MAX_RETRIES


def test_retry_on_429_non_429_raises_immediately():
    exc = ValueError("bad input")
    fn = MagicMock(side_effect=exc)

    raised = False
    try:
        _retry_on_429(fn, "arg")
    except ValueError:
        raised = True
    assert raised
    assert fn.call_count == 1


# --- CLI parsing ---

def test_parse_args_reconcile():
    args = parse_args(["reconcile", "--hub-repo", "user/dataset"])
    assert args.command == "reconcile"
    assert args.hub_repo == "user/dataset"
    assert args.dry_run is False


def test_parse_args_reconcile_dry_run():
    args = parse_args(["reconcile", "--hub-repo", "user/dataset", "--dry-run"])
    assert args.dry_run is True


def test_parse_args_reconcile_with_license():
    args = parse_args(["reconcile", "--hub-repo", "u/d", "--license", "mit"])
    assert args.license == "mit"


# --- reconcile_hub_manifest ---

@patch("stratum.publish._load_manifest")
def test_reconcile_fixes_caption_and_t5_counts(mock_load_manifest):
    """Simulates the user's exact scenario: caption=80k→70k, t5=10k→8k."""
    mock_load_manifest.return_value = {
        "version": "0.0.5",
        "total_images": 80000,
        "layers": {
            "caption": {"format": "parquet", "chunks": {}, "count": 80000},
            "t5": {"format": "npy_tar", "chunks": {}, "count": 10000},
        },
        "created_with": "stratum-hq v0.1.0",
    }

    mock_api = MagicMock()
    mock_api.list_repo_files.return_value = [
        # 7 data parquets = 70,000 images
        "data/00000-09999.parquet",
        "data/10000-19999.parquet",
        "data/20000-29999.parquet",
        "data/30000-39999.parquet",
        "data/40000-49999.parquet",
        "data/50000-59999.parquet",
        "data/60000-69999.parquet",
        # 8 t5 tars = 8,000 images
        "t5/00000-00999.tar",
        "t5/01000-01999.tar",
        "t5/02000-02999.tar",
        "t5/03000-03999.tar",
        "t5/04000-04999.tar",
        "t5/05000-05999.tar",
        "t5/06000-06999.tar",
        "t5/07000-07999.tar",
        # Metadata files
        "manifest.json",
        "README.md",
    ]

    uploaded = {}

    def capture_upload(path_or_fileobj, path_in_repo, repo_id, repo_type):
        uploaded[path_in_repo] = Path(path_or_fileobj).read_text(encoding="utf-8")

    mock_api.upload_file.side_effect = capture_upload

    result = reconcile_hub_manifest(
        hub_repo="user/stratum-ffhq",
        _api=mock_api,
    )

    assert result == 0
    assert "manifest.json" in uploaded
    assert "README.md" in uploaded

    manifest = json.loads(uploaded["manifest.json"])
    assert manifest["total_images"] == 70000
    assert manifest["layers"]["caption"]["count"] == 70000
    assert len(manifest["layers"]["caption"]["chunks"]) == 7
    assert manifest["layers"]["t5"]["count"] == 8000
    assert len(manifest["layers"]["t5"]["chunks"]) == 8
    assert manifest["version"] == "0.0.6"


@patch("stratum.publish._load_manifest")
def test_reconcile_dry_run_does_not_upload(mock_load_manifest):
    mock_load_manifest.return_value = {
        "version": "0.0.1",
        "total_images": 0,
        "layers": {},
        "created_with": "stratum-hq",
    }

    mock_api = MagicMock()
    mock_api.list_repo_files.return_value = [
        "data/00000-04999.parquet",
        "manifest.json",
    ]

    result = reconcile_hub_manifest(
        hub_repo="user/data",
        dry_run=True,
        _api=mock_api,
    )

    assert result == 0
    mock_api.upload_file.assert_not_called()


@patch("stratum.publish._load_manifest")
def test_reconcile_removes_stale_layers(mock_load_manifest):
    """If a layer no longer has files on HF, it should be removed."""
    mock_load_manifest.return_value = {
        "version": "0.0.3",
        "total_images": 10000,
        "layers": {
            "caption": {"format": "parquet", "chunks": {}, "count": 10000},
            "dinov3": {"format": "npy_tar", "chunks": {}, "count": 10000},
            "t5": {"format": "npy_tar", "chunks": {}, "count": 10000},
        },
        "created_with": "stratum-hq",
    }

    mock_api = MagicMock()
    # Only caption and t5 files exist; dinov3 was deleted
    mock_api.list_repo_files.return_value = [
        "data/00000-09999.parquet",
        "t5/00000-09999.tar",
        "manifest.json",
        "README.md",
    ]

    uploaded = {}

    def capture_upload(path_or_fileobj, path_in_repo, repo_id, repo_type):
        uploaded[path_in_repo] = Path(path_or_fileobj).read_text(encoding="utf-8")

    mock_api.upload_file.side_effect = capture_upload

    result = reconcile_hub_manifest(hub_repo="user/data", _api=mock_api)
    assert result == 0

    manifest = json.loads(uploaded["manifest.json"])
    assert "dinov3" not in manifest["layers"]
    assert "caption" in manifest["layers"]
    assert "t5" in manifest["layers"]


@patch("stratum.publish._load_manifest")
def test_reconcile_empty_repo(mock_load_manifest):
    mock_load_manifest.return_value = {
        "version": "0.0.0",
        "total_images": 0,
        "layers": {},
        "created_with": "stratum-hq",
    }

    mock_api = MagicMock()
    mock_api.list_repo_files.return_value = ["manifest.json", "README.md"]

    uploaded = {}

    def capture_upload(path_or_fileobj, path_in_repo, repo_id, repo_type):
        uploaded[path_in_repo] = Path(path_or_fileobj).read_text(encoding="utf-8")

    mock_api.upload_file.side_effect = capture_upload

    result = reconcile_hub_manifest(hub_repo="user/data", _api=mock_api)
    assert result == 0

    manifest = json.loads(uploaded["manifest.json"])
    assert manifest["total_images"] == 0
    assert manifest["layers"] == {}


@patch("stratum.publish._load_manifest")
@patch("stratum.publish.time.sleep")
def test_reconcile_retries_on_429(mock_sleep, mock_load_manifest):
    mock_load_manifest.return_value = {
        "version": "0.0.0",
        "total_images": 0,
        "layers": {},
        "created_with": "stratum-hq",
    }

    rate_exc = Exception("429")
    rate_exc.response = MagicMock(status_code=429)

    mock_api = MagicMock()
    mock_api.list_repo_files.return_value = ["data/00000-00099.parquet"]
    # First upload_file call: fail twice with 429, then succeed; second call: succeed
    mock_api.upload_file.side_effect = [rate_exc, rate_exc, None, None]

    result = reconcile_hub_manifest(hub_repo="user/data", _api=mock_api)
    assert result == 0
    assert mock_sleep.call_count == 2
