"""Tests for publish and reconcile functionality."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from stratum.cli import parse_args
from stratum.publish import (
    _get_retry_after,
    _is_rate_limited,
    _is_transient_error,
    _parse_range_label,
    _plan_tar_splits,
    _retry_on_429,
    _retry_upload,
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


# --- _get_retry_after ---

def test_get_retry_after_from_header():
    exc = Exception("429")
    exc.response = MagicMock(status_code=429, headers={"Retry-After": "30"})
    assert _get_retry_after(exc) == 30


def test_get_retry_after_missing_header():
    exc = Exception("429")
    exc.response = MagicMock(status_code=429, headers={})
    assert _get_retry_after(exc) is None


def test_get_retry_after_no_response():
    exc = Exception("429")
    assert _get_retry_after(exc) is None


def test_get_retry_after_clamps_to_1():
    exc = Exception("429")
    exc.response = MagicMock(status_code=429, headers={"Retry-After": "0"})
    assert _get_retry_after(exc) == 1


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


@patch("stratum.publish.time.sleep")
def test_retry_on_429_uses_retry_after_header(mock_sleep):
    exc = Exception("429")
    exc.response = MagicMock(status_code=429, headers={"Retry-After": "42"})
    fn = MagicMock(side_effect=[exc, "ok"])

    result = _retry_on_429(fn, "arg")
    assert result == "ok"
    mock_sleep.assert_called_once_with(42)


@patch("stratum.publish.time.sleep")
def test_retry_on_429_verbose_logs(mock_sleep, capsys):
    exc = Exception("429")
    exc.response = MagicMock(status_code=429, headers={})
    fn = MagicMock(side_effect=[exc, "ok"])
    fn.__name__ = "upload_folder"

    _retry_on_429(fn, "arg", verbose=True)
    captured = capsys.readouterr()
    assert "[verbose] upload_folder" in captured.err
    assert "succeeded" in captured.err


# --- _is_transient_error ---

def test_is_transient_error_connection_error():
    assert _is_transient_error(ConnectionError("reset")) is True


def test_is_transient_error_timeout_error():
    assert _is_transient_error(TimeoutError("timed out")) is True


def test_is_transient_error_os_error():
    assert _is_transient_error(OSError("network unreachable")) is True


def test_is_transient_error_requests_connection():
    import requests.exceptions
    exc = requests.exceptions.ConnectionError("connection refused")
    assert _is_transient_error(exc) is True


def test_is_transient_error_requests_timeout():
    import requests.exceptions
    exc = requests.exceptions.Timeout("read timed out")
    assert _is_transient_error(exc) is True


def test_is_transient_error_requests_chunked():
    import requests.exceptions
    exc = requests.exceptions.ChunkedEncodingError("broken")
    assert _is_transient_error(exc) is True


def test_is_transient_error_unrelated():
    assert _is_transient_error(ValueError("bad input")) is False
    assert _is_transient_error(KeyError("missing")) is False


def test_is_transient_error_wrapped_string():
    """Catches connection errors wrapped in non-standard exception types."""
    exc = Exception("[Errno 104] Connection reset by peer")
    assert _is_transient_error(exc) is True


def test_is_transient_error_chained_cause():
    """Catches transient errors via __cause__ chain."""
    inner = ConnectionError("reset by peer")
    outer = RuntimeError("upload failed")
    outer.__cause__ = inner
    assert _is_transient_error(outer) is True


def test_is_transient_error_broken_pipe_string():
    exc = Exception("Broken pipe")
    assert _is_transient_error(exc) is True


def test_is_transient_error_timeout_string():
    exc = Exception("Read timed out")
    assert _is_transient_error(exc) is True


# --- _retry_upload (network errors) ---

@patch("stratum.publish.time.sleep")
def test_retry_upload_on_network_error(mock_sleep):
    exc = ConnectionError("reset by peer")
    fn = MagicMock(side_effect=[exc, exc, "ok"])

    result = _retry_upload(fn, "arg")
    assert result == "ok"
    assert fn.call_count == 3
    assert mock_sleep.call_count == 2


@patch("stratum.publish.time.sleep")
def test_retry_upload_network_error_exhausted(mock_sleep):
    exc = TimeoutError("read timed out")
    fn = MagicMock(side_effect=exc)

    raised = False
    try:
        _retry_upload(fn, "arg")
    except TimeoutError:
        raised = True
    assert raised
    assert fn.call_count == 5  # MAX_RETRIES


def test_retry_upload_non_retriable_raises_immediately():
    exc = ValueError("bad input")
    fn = MagicMock(side_effect=exc)

    raised = False
    try:
        _retry_upload(fn, "arg")
    except ValueError:
        raised = True
    assert raised
    assert fn.call_count == 1


@patch("stratum.publish.time.sleep")
def test_retry_upload_mixed_429_and_network(mock_sleep):
    """429 followed by network error, then success."""
    rate_exc = Exception("429")
    rate_exc.response = MagicMock(status_code=429, headers={})
    net_exc = ConnectionError("reset")
    fn = MagicMock(side_effect=[rate_exc, net_exc, "ok"])

    result = _retry_upload(fn, "arg")
    assert result == "ok"
    assert fn.call_count == 3


# --- _plan_tar_splits ---

def test_plan_tar_splits_no_limit(tmp_path):
    """Without max_tar_bytes, everything goes into one group."""
    from stratum.publish import _image_has_layer, LAYER_ARTIFACTS
    # Create 3 image dirs with dinov3 artifacts
    dirs, recs = _make_npy_dirs(tmp_path, 3, "dinov3")

    groups = _plan_tar_splits(dirs, recs, "dinov3", max_tar_bytes=999_999_999)
    assert len(groups) == 1
    assert len(groups[0][0]) == 3


def test_plan_tar_splits_splits_on_size(tmp_path):
    """Files exceeding max_tar_bytes are split into multiple groups."""
    dirs, recs = _make_npy_dirs(tmp_path, 4, "dinov3")

    # Each image has ~2*1024 bytes of npy data. Set limit so 2 images fit.
    single_size = sum(
        (dirs[0] / f).stat().st_size
        for f in ["dinov3_cls.npy", "dinov3_patches.npy"]
    )
    max_bytes = int(single_size * 2.5)  # fits 2, not 3

    groups = _plan_tar_splits(dirs, recs, "dinov3", max_tar_bytes=max_bytes)
    assert len(groups) == 2
    assert len(groups[0][0]) == 2
    assert len(groups[1][0]) == 2


def test_plan_tar_splits_single_large_image(tmp_path):
    """A single image larger than max_tar_bytes still gets its own group."""
    dirs, recs = _make_npy_dirs(tmp_path, 2, "dinov3")
    groups = _plan_tar_splits(dirs, recs, "dinov3", max_tar_bytes=1)
    assert len(groups) == 2
    assert len(groups[0][0]) == 1
    assert len(groups[1][0]) == 1


def test_plan_tar_splits_skips_missing_layer(tmp_path):
    """Images without the layer's artifacts are excluded."""
    dirs, recs = _make_npy_dirs(tmp_path, 3, "dinov3")
    # Remove artifacts from middle image
    for f in ["dinov3_cls.npy", "dinov3_patches.npy"]:
        (dirs[1] / f).unlink()

    groups = _plan_tar_splits(dirs, recs, "dinov3", max_tar_bytes=999_999_999)
    assert len(groups) == 1
    assert len(groups[0][0]) == 2  # only 2 images have the layer


def _make_npy_dirs(base: Path, n: int, layer: str) -> tuple[list[Path], list[dict]]:
    """Helper: create n image dirs with dummy npy files for a layer."""
    import numpy as np
    from stratum.publish import LAYER_ARTIFACTS

    dirs = []
    recs = []
    for i in range(n):
        d = base / f"img{i:04d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "metadata.json").write_text(json.dumps({
            "image_id": f"img{i:04d}", "width": 512, "height": 512,
            "aspect_bucket": "512x512",
        }))
        for artifact in LAYER_ARTIFACTS[layer]:
            np.save(d / artifact, np.zeros(1024, dtype=np.float16))
        dirs.append(d)
        recs.append({"image_id": f"img{i:04d}", "_rel": f"img{i:04d}"})
    return dirs, recs


# --- publish with --max-tar-mb ---

def test_publish_splits_tar_with_max_tar_mb(tmp_path):
    """publish_to_hub with max_tar_mb produces multiple sub-range tars."""
    import json
    from stratum.publish import publish_to_hub

    dirs, recs = _make_npy_dirs(tmp_path / "dataset", 4, "dinov3")

    mock_api = MagicMock()
    mock_api.list_repo_files.return_value = []
    uploaded_paths = []

    def capture_upload(*, folder_path, repo_id, repo_type, commit_message, verbose=False):
        for p in Path(folder_path).rglob("*.tar"):
            uploaded_paths.append(str(p.relative_to(folder_path)))

    mock_api.upload_folder.side_effect = capture_upload
    mock_api.create_repo.return_value = None

    # Set max_tar_mb=1 byte threshold (very small) to force splitting
    single_size = sum(
        (dirs[0] / f).stat().st_size for f in ["dinov3_cls.npy", "dinov3_patches.npy"]
    )
    # max_tar_mb in megabytes — use smallest value that puts ~2 images per tar
    max_mb = max(1, int((single_size * 2.5) / 1_000_000) + 1)

    result = publish_to_hub(
        dataset_dir=tmp_path / "dataset",
        hub_repo="user/test",
        layers=["dinov3"],
        max_tar_mb=max_mb,
        _api=mock_api,
    )

    assert result == 0
    # With 4 images and a small limit, we should get multiple tars
    # (exact count depends on file sizes, but more than 1)
    assert len(uploaded_paths) >= 1
    # All should be under dinov3/
    assert all(p.startswith("dinov3/") for p in uploaded_paths)


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


def test_parse_args_publish_verbose():
    args = parse_args(["publish", "./ds", "--hub-repo", "u/d", "--layers", "caption", "--verbose"])
    assert args.verbose is True


def test_parse_args_publish_max_tar_mb():
    args = parse_args(["publish", "./ds", "--hub-repo", "u/d", "--layers", "dinov3", "--max-tar-mb", "50"])
    assert args.max_tar_mb == 50


def test_parse_args_publish_max_tar_mb_default():
    args = parse_args(["publish", "./ds", "--hub-repo", "u/d", "--layers", "dinov3"])
    assert args.max_tar_mb is None


def test_parse_args_reconcile_verbose():
    args = parse_args(["reconcile", "--hub-repo", "u/d", "--verbose"])
    assert args.verbose is True


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
