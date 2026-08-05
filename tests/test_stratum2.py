"""Tests for stratum2 package — config, loader, and pipeline components."""
from __future__ import annotations

import importlib
import sys
import types
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
        """_download_checkpoint creates cache structure through the hub boundary."""
        from stratum2 import loader

        monkeypatch.setattr(loader, "SAPIENS2_CACHE_DIR", tmp_path)
        expected_path = tmp_path / "test--repo" / "test.safetensors"
        fake_hub = types.ModuleType("huggingface_hub")
        mock_download = mock.Mock(
            side_effect=lambda *args, **kwargs: expected_path.parent.mkdir(
                parents=True, exist_ok=True
            ) or expected_path.touch() or str(expected_path)
        )
        fake_hub.__dict__["hf_hub_download"] = mock_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

        path = loader._download_checkpoint("test/repo", "test.safetensors")
        assert path == expected_path
        assert mock_download.call_count == 1

        path2 = loader._download_checkpoint("test/repo", "test.safetensors")
        assert path2 == expected_path
        assert mock_download.call_count == 1

    def test_get_config_path_returns_existing_file(self, tmp_path, monkeypatch):
        """get_config_path resolves a config shipped by the installed package."""
        from stratum2 import loader

        config = tmp_path / "dense" / "configs" / "seg" / "dome"
        config.mkdir(parents=True)
        expected = config / "sapiens2_1b_seg_dome-1024x768.py"
        expected.write_text("# synthetic config\n", encoding="utf-8")
        fake_sapiens = types.ModuleType("sapiens")
        fake_sapiens.__file__ = str(tmp_path / "__init__.py")
        monkeypatch.setitem(sys.modules, "sapiens", fake_sapiens)

        path = loader.get_config_path("seg", "1b")
        assert path == expected
        assert path.exists()

    def test_get_config_path_unknown_task_raises(self):
        """get_config_path raises ValueError for unknown task."""
        from stratum2 import loader

        with pytest.raises(ValueError, match="Unknown task"):
            loader.get_config_path("nonexistent", "1b")


# ---------------------------------------------------------------------------
# CLI forwarding tests
# ---------------------------------------------------------------------------


def test_cmd_process_forwards_caption_max_tokens_to_orchestrator(monkeypatch, tmp_path):
    """A non-default caption budget reaches orchestration without loading a model."""
    from stratum import discovery
    from stratum2 import orchestrator
    from stratum2.cli import cmd_process, parse_args

    source = tmp_path / "source"
    image = source / "candidate.jpg"
    output = tmp_path / "output"
    captured: dict = {}

    def fake_discover_images(input_dir, image_list_path=None):
        assert input_dir == source
        assert image_list_path is None
        return [image]

    def fake_run_passes(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(discovery, "discover_images", fake_discover_images)
    monkeypatch.setattr(orchestrator, "run_passes", fake_run_passes)

    args = parse_args(
        [
            "process",
            str(source),
            "--output",
            str(output),
            "--passes",
            "caption",
            "--caption-max-tokens",
            "731",
            "--device",
            "cpu",
        ]
    )

    assert cmd_process(args) == 0
    assert captured["caption_max_tokens"] == 731
    assert captured["passes"] == ["caption"]
    assert captured["images"] == [image]


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


# ---------------------------------------------------------------------------
# Normal2 pipeline tests
# ---------------------------------------------------------------------------


def _make_fake_normal_model(output_h: int = 64, output_w: int = 48):
    """Create a mock Sapiens2 normal model."""
    fake = mock.MagicMock()
    fake.pipeline.return_value = {}
    import torch

    fake.data_preprocessor.return_value = {
        "inputs": torch.randn(1, 3, output_h, output_w),
        "data_samples": {"meta": {"padding_size": (0, 0, 0, 0)}},
    }
    # Return unit-ish normals
    normals = torch.randn(1, 3, output_h, output_w)
    normals = normals / torch.norm(normals, dim=1, keepdim=True).clamp(min=1e-8)
    fake.return_value = normals
    return fake


class TestNormal2Pipeline:
    """Tests for stratum2.pipeline.normal — Sapiens2 surface normals."""

    def test_normal_module_importable(self):
        """stratum2.pipeline.normal module exists."""
        from stratum2.pipeline import normal

        assert normal is not None

    def test_process_saves_normal2_file(self, tmp_path):
        """process() writes normal2.npy to the output directory."""
        from stratum2.pipeline.normal import process

        h, w = 128, 128
        img_path = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img_path), _make_fake_image(h, w))
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        # Need seg2.npy for foreground mask
        seg = np.ones((h, w), dtype=np.uint8)
        np.save(str(out_dir / "seg2.npy"), seg)

        fake_model = _make_fake_normal_model()
        result = process(
            image_path=img_path,
            output_dir=out_dir,
            normal_model=fake_model,
            device="cpu",
        )
        assert result is True
        assert (out_dir / "normal2.npy").exists()

    def test_process_skips_when_seg2_missing(self, tmp_path):
        """process() returns False when seg2.npy doesn't exist."""
        from stratum2.pipeline.normal import process

        img_path = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img_path), _make_fake_image())
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        fake_model = _make_fake_normal_model()

        result = process(
            image_path=img_path,
            output_dir=out_dir,
            normal_model=fake_model,
            device="cpu",
        )
        assert result is False
        assert not (out_dir / "normal2.npy").exists()

    def test_process_output_is_float16_hw3(self, tmp_path):
        """normal2.npy has shape (H, W, 3) and dtype float16."""
        from stratum2.pipeline.normal import process

        h, w = 128, 128
        img_path = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img_path), _make_fake_image(h, w))
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        seg = np.ones((h, w), dtype=np.uint8)
        np.save(str(out_dir / "seg2.npy"), seg)

        fake_model = _make_fake_normal_model()
        process(
            image_path=img_path,
            output_dir=out_dir,
            normal_model=fake_model,
            device="cpu",
        )
        normal_map = np.load(out_dir / "normal2.npy")
        assert normal_map.ndim == 3
        assert normal_map.shape[2] == 3
        assert normal_map.dtype == np.float16

    def test_process_normals_are_unit_vectors(self, tmp_path):
        """Non-zero normal vectors are L2-normalized (unit length)."""
        from stratum2.pipeline.normal import process

        h, w = 64, 48  # Match fake model output to avoid interpolation
        img_path = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img_path), _make_fake_image(h, w))
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        seg = np.ones((h, w), dtype=np.uint8)
        np.save(str(out_dir / "seg2.npy"), seg)

        fake_model = _make_fake_normal_model(output_h=h, output_w=w)
        process(
            image_path=img_path,
            output_dir=out_dir,
            normal_model=fake_model,
            device="cpu",
        )
        normal_map = np.load(out_dir / "normal2.npy").astype(np.float32)
        norms = np.linalg.norm(normal_map, axis=-1)
        fg_norms = norms[seg > 0]
        np.testing.assert_allclose(fg_norms, 1.0, atol=0.01)

    def test_process_background_is_zero(self, tmp_path):
        """Background pixels (seg==0) have zero normals."""
        from stratum2.pipeline.normal import process

        h, w = 128, 128
        img_path = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img_path), _make_fake_image(h, w))
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        # Half foreground, half background
        seg = np.zeros((h, w), dtype=np.uint8)
        seg[:64, :] = 1
        np.save(str(out_dir / "seg2.npy"), seg)

        fake_model = _make_fake_normal_model()
        process(
            image_path=img_path,
            output_dir=out_dir,
            normal_model=fake_model,
            device="cpu",
        )
        normal_map = np.load(out_dir / "normal2.npy")
        bg_normals = normal_map[seg == 0]
        assert np.all(bg_normals == 0), "Background normals should be zero"

    def test_process_output_size_matches_image(self, tmp_path):
        """normal2.npy spatial dimensions match the input image."""
        from stratum2.pipeline.normal import process

        h, w = 200, 300
        img_path = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img_path), _make_fake_image(h, w))
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        seg = np.ones((h, w), dtype=np.uint8)
        np.save(str(out_dir / "seg2.npy"), seg)

        fake_model = _make_fake_normal_model()
        process(
            image_path=img_path,
            output_dir=out_dir,
            normal_model=fake_model,
            device="cpu",
        )
        normal_map = np.load(out_dir / "normal2.npy")
        assert normal_map.shape[:2] == (h, w)


# ---------------------------------------------------------------------------
# Pointmap pipeline tests
# ---------------------------------------------------------------------------


def _make_fake_pointmap_model(output_h: int = 64, output_w: int = 48):
    """Create a mock Sapiens2 pointmap model that returns (pointmap, scale)."""
    fake = mock.MagicMock()
    fake.pipeline.return_value = {}
    import torch

    fake.data_preprocessor.return_value = {
        "inputs": torch.randn(1, 3, output_h, output_w),
        "data_samples": {"meta": {"padding_size": (0, 0, 0, 0)}},
    }
    pointmap = torch.randn(1, 3, output_h, output_w)
    scale = torch.tensor([[1.0]])
    fake.return_value = (pointmap, scale)
    return fake


class TestPointmapPipeline:
    """Tests for stratum2.pipeline.pointmap — Sapiens2 pointmap."""

    def test_pointmap_module_importable(self):
        from stratum2.pipeline import pointmap

        assert pointmap is not None

    def test_process_saves_pointmap_file(self, tmp_path):
        from stratum2.pipeline.pointmap import process

        h, w = 128, 128
        img_path = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img_path), _make_fake_image(h, w))
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        seg = np.ones((h, w), dtype=np.uint8)
        np.save(str(out_dir / "seg2.npy"), seg)

        fake_model = _make_fake_pointmap_model()
        result = process(
            image_path=img_path,
            output_dir=out_dir,
            pointmap_model=fake_model,
            device="cpu",
        )
        assert result is True
        assert (out_dir / "pointmap.npy").exists()

    def test_process_skips_when_seg2_missing(self, tmp_path):
        from stratum2.pipeline.pointmap import process

        img_path = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img_path), _make_fake_image())
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        fake_model = _make_fake_pointmap_model()

        result = process(
            image_path=img_path,
            output_dir=out_dir,
            pointmap_model=fake_model,
            device="cpu",
        )
        assert result is False

    def test_process_output_is_float16_hw3(self, tmp_path):
        from stratum2.pipeline.pointmap import process

        h, w = 128, 128
        img_path = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img_path), _make_fake_image(h, w))
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        seg = np.ones((h, w), dtype=np.uint8)
        np.save(str(out_dir / "seg2.npy"), seg)

        fake_model = _make_fake_pointmap_model()
        process(
            image_path=img_path,
            output_dir=out_dir,
            pointmap_model=fake_model,
            device="cpu",
        )
        pm = np.load(out_dir / "pointmap.npy")
        assert pm.ndim == 3
        assert pm.shape[2] == 3
        assert pm.dtype == np.float16

    def test_process_background_is_zero(self, tmp_path):
        from stratum2.pipeline.pointmap import process

        h, w = 128, 128
        img_path = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img_path), _make_fake_image(h, w))
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        seg = np.zeros((h, w), dtype=np.uint8)
        seg[:64, :] = 1
        np.save(str(out_dir / "seg2.npy"), seg)

        fake_model = _make_fake_pointmap_model()
        process(
            image_path=img_path,
            output_dir=out_dir,
            pointmap_model=fake_model,
            device="cpu",
        )
        pm = np.load(out_dir / "pointmap.npy")
        bg_points = pm[seg == 0]
        assert np.all(bg_points == 0)


# ---------------------------------------------------------------------------
# Matting pipeline tests
# ---------------------------------------------------------------------------


def _make_fake_matting_model(output_h: int = 64, output_w: int = 48):
    """Create a mock Sapiens2 matting model returning [fgr_rgb(3), alpha(1)]."""
    fake = mock.MagicMock()
    fake.pipeline.return_value = {}
    import torch

    fake.data_preprocessor.return_value = {
        "inputs": torch.randn(1, 3, output_h, output_w),
        "data_samples": {"meta": {"padding_size": (0, 0, 0, 0)}},
    }
    # 4-channel output: fgr_rgb(3) + alpha(1)
    out = torch.zeros(1, 4, output_h, output_w)
    out[:, 3] = 0.5  # alpha = 0.5 everywhere
    fake.return_value = out
    return fake


class TestMattingPipeline:
    """Tests for stratum2.pipeline.matting — Sapiens2 human matting."""

    def test_matting_module_importable(self):
        from stratum2.pipeline import matting

        assert matting is not None

    def test_process_saves_matting_file(self, tmp_path):
        from stratum2.pipeline.matting import process

        h, w = 128, 128
        img_path = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img_path), _make_fake_image(h, w))
        out_dir = tmp_path / "output"

        fake_model = _make_fake_matting_model()
        result = process(
            image_path=img_path,
            output_dir=out_dir,
            matting_model=fake_model,
            device="cpu",
        )
        assert result is True
        assert (out_dir / "matting.npy").exists()

    def test_process_output_is_float16_hw(self, tmp_path):
        from stratum2.pipeline.matting import process

        h, w = 128, 128
        img_path = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img_path), _make_fake_image(h, w))
        out_dir = tmp_path / "output"

        fake_model = _make_fake_matting_model()
        process(
            image_path=img_path,
            output_dir=out_dir,
            matting_model=fake_model,
            device="cpu",
        )
        alpha = np.load(out_dir / "matting.npy")
        assert alpha.ndim == 2
        assert alpha.dtype == np.float16

    def test_process_alpha_in_range_01(self, tmp_path):
        from stratum2.pipeline.matting import process

        h, w = 128, 128
        img_path = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img_path), _make_fake_image(h, w))
        out_dir = tmp_path / "output"

        fake_model = _make_fake_matting_model()
        process(
            image_path=img_path,
            output_dir=out_dir,
            matting_model=fake_model,
            device="cpu",
        )
        alpha = np.load(out_dir / "matting.npy").astype(np.float32)
        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0

    def test_process_does_not_require_seg(self, tmp_path):
        """Matting should work without seg2.npy (standalone)."""
        from stratum2.pipeline.matting import process

        h, w = 128, 128
        img_path = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img_path), _make_fake_image(h, w))
        out_dir = tmp_path / "output"
        # No seg2.npy created

        fake_model = _make_fake_matting_model()
        result = process(
            image_path=img_path,
            output_dir=out_dir,
            matting_model=fake_model,
            device="cpu",
        )
        assert result is True
        assert (out_dir / "matting.npy").exists()


# ---------------------------------------------------------------------------
# Pose2 pipeline tests
# ---------------------------------------------------------------------------


def _make_fake_pose_model():
    """Create a mock Sapiens2 pose model with codec."""
    fake = mock.MagicMock()
    fake.pipeline.return_value = {
        "inputs": mock.MagicMock(),
        "data_samples": {
            "meta": {
                "input_size": np.array([768, 1024], dtype=np.float32),
                "bbox_center": np.array([512.0, 384.0], dtype=np.float32),
                "bbox_scale": np.array([768.0, 1024.0], dtype=np.float32),
            }
        },
    }
    import torch

    fake.data_preprocessor.return_value = {
        "inputs": torch.randn(1, 3, 1024, 768),
        "data_samples": fake.pipeline.return_value["data_samples"],
    }
    # Return heatmaps: B × K × H × W
    fake.return_value = torch.randn(1, 308, 64, 48)

    # Mock codec.decode
    fake.codec = mock.MagicMock()
    fake.codec.decode.return_value = (
        np.random.randn(1, 308, 2).astype(np.float32),  # keypoints
        np.random.rand(1, 308).astype(np.float32),  # scores
    )
    return fake


class TestPose2Pipeline:
    """Tests for stratum2.pipeline.pose — Sapiens2 308-keypoint pose."""

    def test_pose_module_importable(self):
        from stratum2.pipeline import pose

        assert pose is not None

    def test_process_saves_pose2_file(self, tmp_path, monkeypatch):
        """process() writes pose2.npy."""
        from stratum2.pipeline.pose import process

        h, w = 128, 128
        img_path = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img_path), _make_fake_image(h, w))
        out_dir = tmp_path / "output"

        fake_pose_model = _make_fake_pose_model()

        # Mock DETR detection to return a simple bbox
        with mock.patch(
            "stratum2.pipeline.pose._get_detector"
        ) as mock_det, mock.patch(
            "stratum2.pipeline.pose._detect_persons"
        ) as mock_detect:
            mock_det.return_value = (mock.MagicMock(), mock.MagicMock())
            mock_detect.return_value = np.array(
                [[10, 10, 100, 200]], dtype=np.float32
            )

            result = process(
                image_path=img_path,
                output_dir=out_dir,
                pose_model=fake_pose_model,
                device="cpu",
            )
            assert result is True
            assert (out_dir / "pose2.npy").exists()

    def test_process_output_is_n_k_3(self, tmp_path, monkeypatch):
        """pose2.npy has shape (N, 308, 3) for N persons."""
        from stratum2.pipeline.pose import process

        h, w = 128, 128
        img_path = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img_path), _make_fake_image(h, w))
        out_dir = tmp_path / "output"

        fake_pose_model = _make_fake_pose_model()

        with mock.patch(
            "stratum2.pipeline.pose._get_detector"
        ) as mock_det, mock.patch(
            "stratum2.pipeline.pose._detect_persons"
        ) as mock_detect:
            mock_det.return_value = (mock.MagicMock(), mock.MagicMock())
            mock_detect.return_value = np.array(
                [[10, 10, 100, 200]], dtype=np.float32
            )

            process(
                image_path=img_path,
                output_dir=out_dir,
                pose_model=fake_pose_model,
                device="cpu",
            )
            pose = np.load(out_dir / "pose2.npy")
            assert pose.ndim == 3
            assert pose.shape[1] == 308
            assert pose.shape[2] == 3  # (x, y, confidence)

    def test_process_no_persons_gives_empty(self, tmp_path, monkeypatch):
        """Zero detections produces (0, 308, 3) array."""
        from stratum2.pipeline.pose import process

        h, w = 128, 128
        img_path = tmp_path / "test.png"
        import cv2

        cv2.imwrite(str(img_path), _make_fake_image(h, w))
        out_dir = tmp_path / "output"

        fake_pose_model = _make_fake_pose_model()

        with mock.patch(
            "stratum2.pipeline.pose._get_detector"
        ) as mock_det, mock.patch(
            "stratum2.pipeline.pose._detect_persons"
        ) as mock_detect:
            mock_det.return_value = (mock.MagicMock(), mock.MagicMock())
            mock_detect.return_value = np.array([], dtype=np.float32).reshape(
                0, 4
            )

            process(
                image_path=img_path,
                output_dir=out_dir,
                pose_model=fake_pose_model,
                device="cpu",
            )
            pose = np.load(out_dir / "pose2.npy")
            assert pose.shape == (0, 308, 3)
