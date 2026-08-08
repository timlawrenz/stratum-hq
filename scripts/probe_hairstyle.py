"""CPU probe: run the arm-#82 hairstyle measurement over the frozen
24-item cohort BEFORE the plan is frozen. Reports two scale-invariant bands
(hair-length, hair-arrangement) + raw geometry so the thresholds are
CALIBRATED from the real distribution (band-degeneracy rule):
if any band takes >=75% of measured items it is not discriminating and must be
re-probed. Read-only, no GPU claim, no corpus write, no new model.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from research_harness.hairstyle import (  # noqa: E402
    HairstyleError,
    compute_hairstyle,
)

MANIFEST = "/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json"
DERIVED = "/mnt/nas-ai-models/training-data/crawlr/stratum"


def main() -> int:
    manifest = json.loads(Path(MANIFEST).read_text())
    items = manifest["items"]
    rows = []
    for item in items:
        image_id = item["image_id"]
        segp = Path(DERIVED) / image_id / "seg2.npy"
        posep = Path(DERIVED) / image_id / "pose2.npy"
        try:
            seg2 = np.load(segp, allow_pickle=False)
            pose2 = np.load(posep, allow_pickle=False)
        except FileNotFoundError as exc:
            rows.append({
                "image_id": image_id, "abstained": True,
                "abstention_reason": f"artifact missing: {exc.filename}",
            })
            print(f"{image_id[:12]}  ABSTAIN (artifact missing)")
            continue
        try:
            cfg = compute_hairstyle(seg2, pose2)
        except HairstyleError as exc:
            print(f"FAIL {image_id[:12]}: {exc}")
            return 2
        rows.append({
            "image_id": image_id,
            "abstained": cfg.get("abstained"),
            "abstention_reason": cfg.get("abstention_reason"),
            "hair_present": cfg.get("hair_present"),
            "hair_length_band": cfg.get("hair_length_band"),
            "hair_arrangement_band": cfg.get("hair_arrangement_band"),
            "hair_below_shoulder_ratio": cfg.get("hair_below_shoulder_ratio"),
            "hair_below_shoulder_fraction": cfg.get("hair_below_shoulder_fraction"),
            "hair_span_ratio": cfg.get("hair_span_ratio"),
            "hair_centroid_row_fraction": cfg.get("hair_centroid_row_fraction"),
        })
        print(
            f"{image_id[:12]}  length={str(cfg.get('hair_length_band')):<16} "
            f"arrg={str(cfg.get('hair_arrangement_band')):<10}  "
            f"bsr={cfg.get('hair_below_shoulder_ratio')}  "
            f"bsf={cfg.get('hair_below_shoulder_fraction')}  "
            f"span={cfg.get('hair_span_ratio')}"
        )

    det = [r for r in rows if not r.get("abstained")]
    n = len(det)
    print("\n=== CALIBRATION SUMMARY ===")
    print(f"measured: {n}/{len(rows)}")

    for band_ax in ("hair_length_band", "hair_arrangement_band"):
        vals = [r.get(band_ax) for r in det if r.get(band_ax) is not None]
        c = Counter(vals)
        max_share = max(c.values()) / len(vals) if vals else 0
        print(f"{band_ax}: {dict(c)}  max_share={max_share:.2f}")

    print("\n--- continuous discriminators (for re-cutting degenerate bands) ---")
    for ax in ("hair_below_shoulder_ratio", "hair_below_shoulder_fraction",
               "hair_span_ratio", "hair_centroid_row_fraction"):
        vals = sorted(r.get(ax) for r in det if r.get(ax) is not None)
        if vals:
            q = statistics.quantiles(vals, n=4)
            print(f"{ax}: n={len(vals)} min={vals[0]:.3f} p25={q[0]:.3f} "
                  f"median={statistics.median(vals):.3f} p75={q[2]:.3f} max={vals[-1]:.3f}")

    for r in rows:
        if r.get("abstained"):
            print(f"  ABSTAIN {r['image_id'][:12]}: {r.get('abstention_reason')}")

    Path("/mnt/nas-ai-models/research/stratum/hairstyle-calibration-probe.json").write_text(
        json.dumps({"rows": rows, "summary": {
            "measured": n, "items": len(rows),
            "hair_length_band": dict(Counter(r.get("hair_length_band") for r in det)),
            "hair_arrangement_band": dict(Counter(r.get("hair_arrangement_band") for r in det)),
        }}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
