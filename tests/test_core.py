"""Tests for core stratum modules (no GPU required)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from stratum.config import (
    DEFAULT_ASPECT_BUCKETS,
    METADATA_FILE,
    CAPTION_FILE,
    DINOV3_CLS_FILE,
    POSE_FILE,
)
from stratum.discovery import (
    discover_images,
    image_id_from_path,
    output_dir_for_image,
    shard_image_list,
    scan_dataset_status,
)
from stratum.pipeline.bucket import (
    assign_aspect_bucket,
    compute_aspect_ratio,
    load_bucketed_image,
    parse_bucket_dims,
)
from stratum.cli import parse_args, resolve_passes, parse_shard


# --- bucket tests ---

def test_compute_aspect_ratio():
    assert compute_aspect_ratio(1024, 1024) == 1.0
    assert compute_aspect_ratio(1920, 1080) == 1920 / 1080


def test_assign_aspect_bucket_square():
    assert assign_aspect_bucket(1024, 1024) == "1024x1024"


def test_assign_aspect_bucket_portrait():
    bucket = assign_aspect_bucket(800, 1200)
    assert bucket == "832x1216"


def test_parse_bucket_dims():
    assert parse_bucket_dims("832x1216") == (832, 1216)
    assert parse_bucket_dims("bucket_832x1216") == (832, 1216)
    assert parse_bucket_dims("invalid") is None
    assert parse_bucket_dims("") is None


def test_load_bucketed_image():
    """Create a small test image and bucket it."""
    from PIL import Image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (200, 300), color=(128, 64, 32))
        img.save(f.name)
        result = load_bucketed_image(Path(f.name), 1024, 1024)
        assert result.size == (1024, 1024)
        Path(f.name).unlink()


# --- discovery tests ---

def test_image_id_from_path():
    input_dir = Path("/data/images")
    img = Path("/data/images/ffhq/batch1/00001.png")
    assert image_id_from_path(img, input_dir) == "ffhq/batch1/00001"


def test_image_id_flat():
    input_dir = Path("/data/images")
    img = Path("/data/images/photo.jpg")
    assert image_id_from_path(img, input_dir) == "photo"


def test_output_dir_for_image():
    input_dir = Path("/data/images")
    img = Path("/data/images/ffhq/batch1/00001.png")
    out_base = Path("/data/dataset")
    result = output_dir_for_image(img, input_dir, out_base)
    assert result == Path("/data/dataset/ffhq/batch1/00001")


def test_shard_image_list():
    images = [Path(f"img_{i}.jpg") for i in range(10)]
    shard_0 = shard_image_list(images, 0, 3)
    shard_1 = shard_image_list(images, 1, 3)
    shard_2 = shard_image_list(images, 2, 3)
    # All images covered, no overlaps
    all_shards = set(str(p) for s in [shard_0, shard_1, shard_2] for p in s)
    assert len(all_shards) == 10


def test_discover_images():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "sub").mkdir()
        (tmp / "img1.jpg").write_bytes(b"fake")
        (tmp / "sub" / "img2.png").write_bytes(b"fake")
        (tmp / "readme.txt").write_bytes(b"not an image")
        images = discover_images(tmp)
        assert len(images) == 2
        assert all(p.suffix in {".jpg", ".png"} for p in images)


def test_scan_dataset_status():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Create two image dirs
        d1 = tmp / "img1"
        d1.mkdir()
        (d1 / METADATA_FILE).write_text('{"image_id": "img1"}')
        (d1 / CAPTION_FILE).write_text("A test caption")

        d2 = tmp / "img2"
        d2.mkdir()
        (d2 / METADATA_FILE).write_text('{"image_id": "img2"}')

        status = scan_dataset_status(tmp)
        assert status["total"] == 2
        assert status["caption"] == 1
        assert status["metadata"] == 2


# --- CLI tests ---

def test_resolve_passes_all():
    passes = resolve_passes("all")
    assert "caption" in passes
    assert "dinov3" in passes
    assert "pixel" not in passes  # pixel is opt-in


def test_resolve_passes_specific():
    passes = resolve_passes("dinov3,t5")
    assert passes == ["dinov3", "t5"]


def test_parse_shard():
    assert parse_shard(None) is None
    assert parse_shard("0/4") == (0, 4)
    assert parse_shard("3/4") == (3, 4)


def test_parse_args_process():
    args = parse_args(["process", "./images", "--output", "./out", "--passes", "caption"])
    assert args.command == "process"
    assert args.input_dir == Path("./images")
    assert args.output == Path("./out")
    assert args.passes == "caption"


def test_parse_args_status():
    args = parse_args(["status", "./dataset"])
    assert args.command == "status"
    assert args.dataset_dir == Path("./dataset")
