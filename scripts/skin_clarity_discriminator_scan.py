"""Exploratory discriminator scan for skin-clarity (arm #125), CPU-only.

The raw mean-Laplacian density collapsed (24/24 blemished, max_share 1.00):
global skin-pore texture + focus dominate the absolute Laplacian, drowning the
blemish/unevenness signal. This scan evaluates alternative scale-invariant
discriminators on the SAME frozen cohort so the band can be re-cut honestly
(not threshold-fitted):

D1  hp_density_luma  = mean|Lap| / skin_luma_std      (contrast-normalized)
D2  tail_ratio       = p95|Lap| / median|Lap|          (heavy-tail: freckle/spot
                        events are sparse outliers vs the region's own texture)
D3  spot_fraction    = frac(skin px where |Lap| > region median + 3*robust MAD)
D4  p90_density      = p90|Lap| over skin interior
D5  mean_delta       = mean area-normalized |2nd-diff| (endpoint-free)

Bands are 3-way low/mid/high tercile cuts (clear/even/blemished) on whichever
metric discriminates (max_share < 0.75, coverage floor >= 8/24).

Read-only, in-memory, no corpus write.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

from research_harness.skin_clarity import CANONICAL_SIDE, SKIN_CLASSES, _luminance, _resample  # noqa: E402

MANIFEST = "/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json"
DERIVED = "/mnt/nas-ai-models/training-data/crawlr/stratum"


def _laplacian_magnitude(lum: np.ndarray) -> np.ndarray:
    lap = np.zeros_like(lum)
    lap[1:-1, 1:-1] = (
        4.0 * lum[1:-1, 1:-1]
        - lum[1:-1, :-2] - lum[1:-1, 2:]
        - lum[:-2, 1:-1] - lum[2:, 1:-1]
    )
    return np.abs(lap)


def _mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def main() -> int:
    manifest = json.loads(Path(MANIFEST).read_text())
    rows = []
    metrics: dict[str, list] = {}
    for item in manifest["items"]:
        image_id = item["image_id"]
        seg2 = np.load(Path(DERIVED) / image_id / "seg2.npy")
        src = Path(DERIVED) / image_id
        payload_path = None
        for cand in (src / item["source_relative_path"].split("/")[-1],
                     Path("/mnt/nas-ai-models/training-data/crawlr/approved") / item["source_relative_path"]):
            if cand.exists():
                payload_path = cand
                break
        if payload_path is None:
            rows.append({"image_id": image_id, "abstained": True, "abstention_reason": "source not found"})
            continue
        import io
        with open(payload_path, "rb") as fh:
            rgb = np.asarray(Image.open(io.BytesIO(fh.read())).convert("RGB"), dtype=np.uint8)
        if seg2.shape != rgb.shape[:2]:
            seg2 = np.asarray(Image.fromarray(seg2.astype(np.uint8)).resize(
                (rgb.shape[1], rgb.shape[0]), Image.NEAREST))
        resized_rgb, seg_small = _resample(rgb, seg2)
        skin = np.isin(seg_small, list(SKIN_CLASSES))
        interior = binary_erosion(skin, iterations=2).astype(bool)
        if int(interior.sum()) < 200:
            rows.append({"image_id": image_id, "abstained": True,
                         "abstention_reason": f"interior too small {int(interior.sum())}"})
            continue
        lum = _luminance(resized_rgb)
        lap = _laplacian_magnitude(lum)
        px = lap[interior]
        luma_px = lum[interior]
        luma_std = float(luma_px.std())
        med = float(np.median(px))
        d = {
            "D1_hp_density_luma": round(float(px.mean()) / max(luma_std, 1e-6), 4),
            "D2_tail_ratio": round(float(np.percentile(px, 95)) / max(med, 1e-6), 4),
            "D3_spot_fraction": round(float((px > med + 3.0 * _mad(px)).mean()), 4),
            "D4_p90_density": round(float(np.percentile(px, 90)) / max(luma_std, 1e-6), 4),
        }
        d["image_id"] = image_id
        rows.append(d)
        for k in ("D1_hp_density_luma", "D2_tail_ratio", "D3_spot_fraction", "D4_p90_density"):
            metrics.setdefault(k, []).append(d[k])
        print(f"{image_id[:12]}  " + "  ".join(f"{k}={v}" for k, v in d.items() if k != "image_id"))

    print("\n=== SUMMARY (3-way tercile cuts on 24 items) ===")
    results = {}
    for name, vals in metrics.items():
        arr = np.array(vals)
        lo, hi = np.percentile(arr, 33.3), np.percentile(arr, 66.6)
        bands = []
        for v in arr:
            bands.append("clear" if v <= lo else ("even" if v <= hi else "blemished"))
        cnt = Counter(bands)
        max_share = max(cnt.values()) / len(bands)
        results[name] = {
            "min": round(float(arr.min()), 4), "p33": round(float(lo), 4),
            "med": round(float(np.median(arr)), 4), "p66": round(float(hi), 4),
            "max": round(float(arr.max()), 4), "counts": dict(cnt),
            "max_share": round(max_share, 4),
        }
        print(f"{name}: {json.dumps(results[name])}")
    Path("/mnt/nas-ai-models/research/stratum/skin-clarity-discriminator-scan.json").write_text(
        json.dumps({"rows": rows, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
