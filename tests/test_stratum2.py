"""Tests for stratum2 package — config, loader, and pipeline components."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
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
        import stratum2.config

        importlib.reload(stratum2.config)
        assert stratum2.config.SAPIENS2_SIZE == "0.4b"
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
        with mock.patch("huggingface_hub.hf_hub_download") as mock_dl:
            expected_path = tmp_path / "test--repo" / "test.safetensors"
            mock_dl.return_value = str(expected_path)
            mock_dl.side_effect = lambda *a, **kw: expected_path.parent.mkdir(
                parents=True, exist_ok=True
            ) or expected_path.touch() or str(expected_path)
            path = loader._download_checkpoint("test/repo", "test.safetensors")
            assert path == expected_path
            assert mock_dl.call_count == 1

            path2 = loader._download_checkpoint("test/repo", "test.safetensors")
            assert path2 == expected_path
            assert mock_dl.call_count == 1

    def test_get_config_path_returns_existing_file(self):
        """get_config_path returns a path that exists for valid task/size."""
        from stratum2 import loader

        path = loader.get_config_path("seg", "1b")
        assert path is not None
        assert Path(path).exists(), f"Config not found at {path}"

    def test_get_config_path_unknown_task_raises(self):
        """get_config_path raises ValueError for unknown task."""
        from stratum2 import loader

        with pytest.raises(ValueError, match="Unknown task"):
            loader.get_config_path("nonexistent", "1b")


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


def _make_fake_image(h: int = 256, w: int = 256) -> np.ndarray:
    """Create a synthetic BGR image for testing."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_fake_seg_model(output_h: int = 256, output_w: int = 256):
    """Create a mock Sapiens2 seg model that returns predictable logits."""
    fake = mock.MagicMock()

    # model.pipeline(dict(img=image)) → data dict
    fake.pipeline.return_value = {"img_meta": {}}

    # model.data_preprocessor(data) → data with "inputs" tensor
    import torch

    fake.data_preprocessor.return_value = {
        "inputs": torch.randn(1, 3, 1024, 768),
        "data_samples": {"meta": {"padding_size": (0, 0, 0, 0)}},
    }

    # model(inputs) → seg_logits (1×29×H×W)
    fake.return_value = torch.randn(1, 29, output_h, output_w)
    return fake


class TestSeg2Pipeline:
    """Tests for stratum2.pipeline.seg — Sapiens2 segmentation."""

    def test_seg_module_importable(self):
        """stratum2.pipeline.seg module exists."""
        from stratum2.pipeline import seg

        assert seg is not None

    def test_process_saves_seg2_file(self, tmp_path):
        """process() writes seg2.npy to the output directory."""
        from stratum2.pipeline.seg import process

        img_path = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img_path), _make_fake_image())
        out_dir = tmp_path / "output"
        fake_model = _make_fake_seg_model()

        result = process(
            image_path=img_path,
            output_dir=out_dir,
            seg_model=fake_model,
            device="cpu",
        )
        assert result is True
        seg_file = out_dir / "seg2.npy"
        assert seg_file.exists(), f"Expected {seg_file} to exist"

    def test_process_output_is_uint8_hw(self, tmp_path):
        """seg2.npy has shape (H, W) and dtype uint8."""
        from stratum2.pipeline.seg import process

        img_path = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img_path), _make_fake_image(128, 128))
        out_dir = tmp_path / "output"
        fake_model = _make_fake_seg_model()

        process(
            image_path=img_path,
            output_dir=out_dir,
            seg_model=fake_model,
            device="cpu",
        )
        seg = np.load(out_dir / "seg2.npy")
        assert seg.ndim == 2, f"Expected 2D, got {seg.ndim}D"
        assert seg.dtype == np.uint8

    def test_process_output_values_in_class_range(self, tmp_path):
        """seg2.npy values are in [0, 28] (29 classes)."""
        from stratum2.pipeline.seg import process

        img_path = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img_path), _make_fake_image(128, 128))
        out_dir = tmp_path / "output"
        fake_model = _make_fake_seg_model()

        process(
            image_path=img_path,
            output_dir=out_dir,
            seg_model=fake_model,
            device="cpu",
        )
        seg = np.load(out_dir / "seg2.npy")
        assert seg.min() >= 0
        assert seg.max() <= 28

    def test_process_uses_model_pipeline_with_bgr_image(self, tmp_path):
        """process() calls model.pipeline() with a BGR image from cv2.imread."""
        from stratum2.pipeline.seg import process

        img_path = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img_path), _make_fake_image())
        out_dir = tmp_path / "output"
        fake_model = _make_fake_seg_model()

        process(
            image_path=img_path,
            output_dir=out_dir,
            seg_model=fake_model,
            device="cpu",
        )
        # model.pipeline was called with dict(img=image_bgr)
        fake_model.pipeline.assert_called_once()
        call_args = fake_model.pipeline.call_args[0][0]
        assert "img" in call_args

    def test_process_resizes_output_to_image_size(self, tmp_path):
        """seg2.npy dimensions match the input image size after resize."""
        from stratum2.pipeline.seg import process

        h, w = 200, 300
        img_path = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img_path), _make_fake_image(h, w))
        out_dir = tmp_path / "output"
        fake_model = _make_fake_seg_model()

        process(
            image_path=img_path,
            output_dir=out_dir,
            seg_model=fake_model,
            device="cpu",
        )
        seg = np.load(out_dir / "seg2.npy")
        assert seg.shape == (h, w), f"Expected {(h, w)}, got {seg.shape}"
