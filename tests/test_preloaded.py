"""Tests for preloaded-image support in stratum2 pipeline process() functions."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import cv2  # noqa: E402
import torch  # noqa: E402


def _make_bgr(h: int = 64, w: int = 48) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _fake_dense_model(output_h: int = 64, output_w: int = 48, channels: int = 3, pointmap=False):
    fake = mock.MagicMock()
    fake.pipeline.return_value = {}
    fake.data_preprocessor.return_value = {
        "inputs": torch.randn(1, 3, output_h, output_w),
        "data_samples": {"meta": {"padding_size": (0, 0, 0, 0)}},
    }
    if pointmap:
        fake.return_value = (torch.randn(1, 3, output_h, output_w), torch.tensor([[1.0]]))
    else:
        fake.return_value = torch.randn(1, channels, output_h, output_w)
    return fake


class TestPreloadedImageSupport:
    """process() accepts a preloaded BGR image and skips cv2.imread."""

    def _make_img(self, tmp_path) -> Path:
        p = tmp_path / "test.png"
        cv2.imwrite(str(p), _make_bgr())
        return p

    def test_seg_uses_preloaded_image(self, tmp_path):
        from stratum2.pipeline.seg import process

        img_path = self._make_img(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        preloaded = _make_bgr()

        with mock.patch("cv2.imread", return_value=None) as m:
            result = process(
                img_path, out_dir, _fake_dense_model(64, 48, 29), "cpu",
                image=preloaded,
            )
        assert result is True
        assert not m.called, "cv2.imread should not be called when image is preloaded"
        assert (out_dir / "seg2.npy").exists()

    def test_seg_falls_back_to_imread_without_image(self, tmp_path):
        from stratum2.pipeline.seg import process

        img_path = self._make_img(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with mock.patch("cv2.imread", return_value=_make_bgr()) as m:
            result = process(
                img_path, out_dir, _fake_dense_model(64, 48, 29), "cpu",
            )
        assert result is True
        assert m.called, "cv2.imread should be called when no image is preloaded"

    def test_normal_uses_preloaded_image(self, tmp_path):
        from stratum2.pipeline.normal import process

        img_path = self._make_img(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        seg = np.ones((64, 48), dtype=np.uint8)
        np.save(str(out_dir / "seg2.npy"), seg)

        with mock.patch("cv2.imread", return_value=None) as m:
            result = process(
                img_path, out_dir, _fake_dense_model(64, 48, 3), "cpu",
                image=_make_bgr(),
            )
        assert result is True
        assert not m.called
        assert (out_dir / "normal2.npy").exists()

    def test_pointmap_uses_preloaded_image(self, tmp_path):
        from stratum2.pipeline.pointmap import process

        img_path = self._make_img(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        seg = np.ones((64, 48), dtype=np.uint8)
        np.save(str(out_dir / "seg2.npy"), seg)

        with mock.patch("cv2.imread", return_value=None) as m:
            result = process(
                img_path, out_dir, _fake_dense_model(64, 48, 3, pointmap=True), "cpu",
                image=_make_bgr(),
            )
        assert result is True
        assert not m.called
        assert (out_dir / "pointmap.npy").exists()

    def test_matting_uses_preloaded_image(self, tmp_path):
        from stratum2.pipeline.matting import process

        img_path = self._make_img(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with mock.patch("cv2.imread", return_value=None) as m:
            result = process(
                img_path, out_dir, _fake_dense_model(64, 48, 4), "cpu",
                image=_make_bgr(),
            )
        assert result is True
        assert not m.called
        assert (out_dir / "matting.npy").exists()

    def test_pose_uses_preloaded_image(self, tmp_path):
        from stratum2.pipeline.pose import process

        img_path = self._make_img(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        fake_pose = mock.MagicMock()
        fake_pose.pipeline.return_value = {
            "inputs": mock.MagicMock(),
            "data_samples": {
                "meta": {
                    "input_size": np.array([48, 64], dtype=np.float32),
                    "bbox_center": np.array([32.0, 24.0], dtype=np.float32),
                    "bbox_scale": np.array([48.0, 64.0], dtype=np.float32),
                }
            },
        }
        fake_pose.data_preprocessor.return_value = {
            "inputs": torch.randn(1, 3, 48, 64),
            "data_samples": fake_pose.pipeline.return_value["data_samples"],
        }
        fake_pose.return_value = torch.randn(1, 308, 16, 12)
        fake_pose.codec = mock.MagicMock()
        fake_pose.codec.decode.return_value = (
            np.random.randn(1, 308, 2).astype(np.float32),
            np.random.rand(1, 308).astype(np.float32),
        )

        with mock.patch("cv2.imread", return_value=None) as m, mock.patch(
            "stratum2.pipeline.pose._get_detector"
        ) as det, mock.patch("stratum2.pipeline.pose._detect_persons") as detect:
            det.return_value = (mock.MagicMock(), mock.MagicMock())
            detect.return_value = np.array([[0, 0, 63, 47]], dtype=np.float32)
            result = process(
                img_path, out_dir, fake_pose, "cpu",
                image=_make_bgr(),
            )
        assert result is True
        assert not m.called
        assert (out_dir / "pose2.npy").exists()


class TestDetectorCache:
    """The DETR person detector must be loaded once and reused.

    Loading a 785-tensor model from disk per image costs ~2s and dominated
    the pose2 pass. The upstream sapiens2 vis_pose.py caches it; stratum2
    must too.
    """

    def test_get_detector_caches_across_calls(self, tmp_path):
        from stratum2.pipeline import pose

        ckpt = str(tmp_path / "detr")
        fake_proc = mock.MagicMock()
        fake_model = mock.MagicMock()

        with mock.patch.object(
            pose, "_get_detector", side_effect=[(fake_proc, fake_model)]
        ) as loader:
            pose._detector_cache.clear()
            # Call through a fresh accessor twice — only one load should happen
            p1, m1 = pose._get_cached_detector("cuda:0", ckpt)
            p2, m2 = pose._get_cached_detector("cuda:0", ckpt)
            assert p1 is p2 and m1 is m2
            assert loader.call_count == 1

    def test_process_calls_get_detector_once_for_two_images(self, tmp_path):
        """Two pose process() calls must load the DETR detector only once."""
        from stratum2.pipeline import pose
        from stratum2.pipeline.pose import process

        def _make_fake_pose_model():
            fake = mock.MagicMock()
            fake.pipeline.return_value = {
                "inputs": mock.MagicMock(),
                "data_samples": {
                    "meta": {
                        "input_size": np.array([48, 64], dtype=np.float32),
                        "bbox_center": np.array([32.0, 24.0], dtype=np.float32),
                        "bbox_scale": np.array([48.0, 64.0], dtype=np.float32),
                    }
                },
            }
            fake.data_preprocessor.return_value = {
                "inputs": torch.randn(1, 3, 48, 64),
                "data_samples": fake.pipeline.return_value["data_samples"],
            }
            fake.return_value = torch.randn(1, 308, 16, 12)
            fake.codec = mock.MagicMock()
            fake.codec.decode.return_value = (
                np.random.randn(1, 308, 2).astype(np.float32),
                np.random.rand(1, 308).astype(np.float32),
            )
            return fake

        img1 = tmp_path / "a.png"
        img2 = tmp_path / "b.png"
        cv2.imwrite(str(img1), _make_bgr())
        cv2.imwrite(str(img2), _make_bgr())
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"

        fake_proc = mock.MagicMock()
        fake_det = mock.MagicMock()
        pose._detector_cache.clear()
        with mock.patch("cv2.imread", side_effect=lambda p: _make_bgr()), mock.patch(
            "stratum2.pipeline.pose._get_detector",
            return_value=(fake_proc, fake_det),
        ) as slow_loader, mock.patch(
            "stratum2.pipeline.pose._detect_persons",
            return_value=np.array([[0, 0, 63, 47]], dtype=np.float32),
        ):
            process(img1, out1, _make_fake_pose_model(), "cpu")
            process(img2, out2, _make_fake_pose_model(), "cpu")
            # The slow loader must have been hit exactly once across 2 images
            assert slow_loader.call_count == 1
