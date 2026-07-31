"""Tests for stratum2 package — config, loader, and pipeline components."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

# Ensure src/ is on the path
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


class TestStratum2Config:
    """Tests for stratum2.config module."""

    def test_config_module_importable(self):
        """stratum2.config module exists and imports without error."""
        from stratum2 import config

        assert config is not None

    def test_sapiens2_size_default_is_5b(self):
        """SAPIENS2_SIZE defaults to '5b'."""
        from stratum2 import config

        assert config.SAPIENS2_SIZE == "5b"

    def test_sapiens2_size_from_env(self, monkeypatch):
        """SAPIENS2_SIZE reads from environment variable."""
        monkeypatch.setenv("SAPIENS2_SIZE", "0.4b")
        # Re-import to pick up env var (modules use os.environ.get at import time)
        import stratum2.config

        importlib.reload(stratum2.config)
        assert stratum2.config.SAPIENS2_SIZE == "0.4b"
        # Restore
        monkeypatch.delenv("SAPIENS2_SIZE")
        importlib.reload(stratum2.config)

    def test_artifact_filenames_defined(self):
        """All stratum2 artifact filenames are defined as non-empty strings."""
        from stratum2 import config

        assert isinstance(config.SEG2_FILE, str) and len(config.SEG2_FILE) > 0
        assert isinstance(config.NORMAL2_FILE, str) and len(config.NORMAL2_FILE) > 0
        assert isinstance(config.POINTMAP_FILE, str) and len(config.POINTMAP_FILE) > 0
        assert isinstance(config.POSE2_FILE, str) and len(config.POSE2_FILE) > 0
        assert isinstance(config.MATTING_FILE, str) and len(config.MATTING_FILE) > 0

    def test_sapiens2_repos_contain_expected_tasks(self):
        """SAPIENS2_REPOS has entries for seg, normal, pointmap, pose, matting."""
        from stratum2 import config

        for task in ["seg", "normal", "pointmap", "pose", "matting"]:
            assert task in config.SAPIENS2_REPOS
            assert "facebook/sapiens2" in config.SAPIENS2_REPOS[task]

    def test_cache_dir_default_is_nas(self):
        """SAPIENS2_CACHE_DIR defaults to /mnt/nas-ai-models/sapiens2."""
        from stratum2 import config

        assert str(config.SAPIENS2_CACHE_DIR) == "/mnt/nas-ai-models/sapiens2"

    def test_pose_detector_repo_defined(self):
        """POSE_DETECTOR_REPO is set to the DETR model."""
        from stratum2 import config

        assert "detr-resnet-101" in config.POSE_DETECTOR_REPO

    def test_sapiens2_filenames_use_size(self):
        """SAPIENS2_FILENAMES incorporate SAPIENS2_SIZE in filenames."""
        from stratum2 import config

        size = config.SAPIENS2_SIZE
        for task in ["seg", "normal", "pointmap", "pose"]:
            assert size in config.SAPIENS2_FILENAMES[task]
        # matting is always 1b
        assert "1b" in config.SAPIENS2_FILENAMES["matting"]

    def test_stratum2_package_importable(self):
        """stratum2 package itself is importable."""
        import stratum2

        assert stratum2 is not None


class TestStratum2Loader:
    """Tests for stratum2.loader module."""

    def test_loader_module_importable(self):
        """stratum2.loader module exists and imports."""
        from stratum2 import loader

        assert loader is not None

    def test_download_checkpoint_creates_cache_dir(self, tmp_path, monkeypatch):
        """_download_checkpoint creates cache directory structure."""
        from stratum2 import loader

        monkeypatch.setattr(loader, "SAPIENS2_CACHE_DIR", tmp_path)
        # hf_hub_download is imported inside _download_checkpoint — mock it in huggingface_hub
        with mock.patch("huggingface_hub.hf_hub_download") as mock_dl:
            # First call: file doesn't exist yet, should trigger download
            expected_path = tmp_path / "test--repo" / "test.safetensors"
            mock_dl.return_value = str(expected_path)
            mock_dl.side_effect = lambda *a, **kw: expected_path.parent.mkdir(
                parents=True, exist_ok=True
            ) or expected_path.touch() or str(expected_path)
            path = loader._download_checkpoint("test/repo", "test.safetensors")
            assert path == expected_path
            assert mock_dl.call_count == 1

            # Second call: file exists, should skip download
            path2 = loader._download_checkpoint("test/repo", "test.safetensors")
            assert path2 == expected_path
            assert mock_dl.call_count == 1  # no additional download

    def test_get_config_path_returns_existing_file(self):
        """get_config_path returns a path that exists for valid task/size."""
        from stratum2 import loader

        path = loader.get_config_path("seg", "1b")
        assert path is not None
        # The config file should exist on disk
        assert Path(path).exists(), f"Config not found at {path}"

    def test_get_config_path_unknown_task_raises(self):
        """get_config_path raises ValueError for unknown task."""
        from stratum2 import loader

        with pytest.raises(ValueError, match="Unknown task"):
            loader.get_config_path("nonexistent", "1b")
