"""CPU-only probe: inspect normal2.npy + seg2.npy + source bytes format on the frozen cohort."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from PIL import Image

CAND = Path("/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json")
DER = Path("/mnt/nas-ai-models/training-data/crawlr/stratum")
SRC = Path("/mnt/nas-ai-models/training-data/crawlr/approved")


def main() -> int:
    cand = json.loads(CAND.read_text(encoding="utf-8"))
    items = cand["items"]
    print("n items:", len(items))
    for it in items[:4]:
        iid = it["image_id"]
        rel = it["source_relative_path"]
        n = np.load(DER / iid / "normal2.npy", allow_pickle=False)
        s = np.load(DER / iid / "seg2.npy", allow_pickle=False)
        with Image.open(SRC / rel) as im:
            w, h = im.size
        print(
            iid,
            "normal2", n.shape, n.dtype,
            "finite", bool(np.isfinite(n).all()),
            "min/max", round(float(n.min()), 4), round(float(n.max()), 4),
            "seg2", s.shape, s.dtype,
            "src", w, "x", h,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
