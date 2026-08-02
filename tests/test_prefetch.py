"""Tests for stratum2.prefetch — background image prefetch reader."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest


class TestPrefetchReader:
    """Tests for stratum2.prefetch.PrefetchReader."""

    def test_module_importable(self):
        from stratum2 import prefetch

        assert prefetch is not None

    def test_yields_images_in_order(self, tmp_path):
        """Reader yields (path, image) in the same order as the input list."""
        from stratum2.prefetch import PrefetchReader

        paths = []
        for i in range(5):
            p = tmp_path / f"img{i}.png"
            p.write_bytes(b"x")
            paths.append(p)

        # Deterministic fake read: return an array derived from the path index
        def fake_read(path: Path) -> np.ndarray:
            return np.array([int(path.stem[-1])], dtype=np.uint8)

        with PrefetchReader(paths, depth=2, read_fn=fake_read) as reader:
            got = list(reader)

        assert [p.name for p, _ in got] == [p.name for p in paths]
        assert [img[0] for _, img in got] == [0, 1, 2, 3, 4]

    def test_unreadable_image_yields_none(self, tmp_path):
        """A failed read yields (path, None) and iteration continues."""
        from stratum2.prefetch import PrefetchReader

        paths = [tmp_path / "a.png", tmp_path / "b.png"]
        for p in paths:
            p.write_bytes(b"x")

        def fake_read(path: Path):
            if path.name == "a.png":
                return None  # simulate cv2.imread failure
            return np.zeros((2, 2), dtype=np.uint8)

        with PrefetchReader(paths, depth=2, read_fn=fake_read) as reader:
            got = list(reader)

        assert got[0][1] is None
        assert got[1][1] is not None

    def test_bounded_prefetch_never_reads_too_far_ahead(self, tmp_path):
        """Reader must not read far ahead of the consumer.

        Invariant: reads <= consumed + depth + 1. The +1 is the in-flight
        read — the thread reads an image *before* putting it in the queue,
        so one extra read can be in flight beyond the queue capacity.
        """
        from stratum2.prefetch import PrefetchReader

        paths = [tmp_path / f"img{i}.png" for i in range(10)]
        for p in paths:
            p.write_bytes(b"x")

        reads: list[str] = []

        def fake_read(path: Path) -> np.ndarray:
            reads.append(path.name)
            return np.zeros((2, 2), dtype=np.uint8)

        depth = 3
        with PrefetchReader(paths, depth=depth, read_fn=fake_read) as reader:
            it = iter(reader)
            # Consume one item, then check how many were read
            next(it)
            time.sleep(0.2)  # give the background thread time to fill
            assert len(reads) <= 1 + depth + 1, f"Read {len(reads)} ahead: {reads}"

    def test_close_stops_thread(self, tmp_path):
        """close() terminates the background thread (no leaked threads)."""
        from stratum2.prefetch import PrefetchReader

        paths = [tmp_path / f"img{i}.png" for i in range(5)]
        for p in paths:
            p.write_bytes(b"x")

        import threading

        def fake_read(path: Path) -> np.ndarray:
            return np.zeros((2, 2), dtype=np.uint8)

        reader = PrefetchReader(paths, depth=2, read_fn=fake_read)
        reader.start()
        time.sleep(0.1)
        reader.close()

        alive = [t for t in threading.enumerate() if t.name == "prefetch-reader"]
        assert not alive, "Prefetch thread still alive after close"

    def test_empty_input_iterates_zero(self, tmp_path):
        """Empty image list yields nothing."""
        from stratum2.prefetch import PrefetchReader

        with PrefetchReader([], depth=2, read_fn=lambda p: None) as reader:
            assert list(reader) == []
