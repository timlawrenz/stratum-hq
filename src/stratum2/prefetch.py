"""Background image prefetch — overlap CIFS reads with GPU compute.

The orchestrator reads images one at a time, leaving the GPU idle while it
waits on the network filesystem. ``PrefetchReader`` moves the read+decode
into a background thread with a bounded queue, so the next ``depth`` images
are already in memory when the consumer needs them.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np


def _default_read_fn(image_path: Path) -> np.ndarray | None:
    """Read an image as BGR via OpenCV (matches pipeline convention)."""
    import cv2

    return cv2.imread(str(image_path))


class PrefetchReader:
    """Reads images ahead of the consumer in a background thread.

    Args:
        images: Iterable of image paths, consumed in order.
        depth: How many images to read ahead (queue capacity). Larger
            values hide more I/O latency but waste reads if the run is
            interrupted.
        read_fn: Callable(path) -> np.ndarray | None. Defaults to
            ``cv2.imread`` (BGR). Returning None signals an unreadable
            image; the consumer should fall back to its own read.

    Yields ``(image_path, image_or_none)`` tuples in input order.
    """

    def __init__(
        self,
        images: list[Path],
        depth: int = 8,
        read_fn=None,
    ) -> None:
        self._images = list(images)
        self._depth = max(1, depth)
        self._read_fn = read_fn or _default_read_fn
        self._queue: queue.Queue = queue.Queue(maxsize=self._depth)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> "PrefetchReader":
        """Start the background prefetch thread."""
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._fill,
            name="prefetch-reader",
            daemon=True,
        )
        self._thread.start()
        return self

    def _fill(self) -> None:
        for path in self._images:
            if self._stop.is_set():
                break
            try:
                image = self._read_fn(path)
            except Exception:
                image = None
            # Blocks if the queue is full — that's the backpressure.
            self._queue.put((path, image))
        # Sentinel: signal end of input.
        self._queue.put((None, None))

    def close(self) -> None:
        """Stop the background thread and drain pending items.

        The thread may be blocked on ``put()`` with a full queue — setting
        the stop event alone won't unblock it. Draining items gives the
        blocked put() room to complete, after which the thread exits.
        """
        self._stop.set()
        if self._thread is not None:
            deadline = time.time() + 5.0
            while self._thread.is_alive() and time.time() < deadline:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    time.sleep(0.005)
            self._thread.join(timeout=1)
            self._thread = None

    def __enter__(self) -> "PrefetchReader":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    def __iter__(self) -> Iterator[tuple[Path | None, np.ndarray | None]]:
        while True:
            item = self._queue.get()
            if item == (None, None):  # sentinel
                break
            yield item
