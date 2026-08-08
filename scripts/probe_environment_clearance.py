"""CPU probe: run the arm-#85 environment-clearance measurement over the frozen
24-item cohort BEFORE the plan is frozen. Reports the normalized clearance
ratios + clearance band so the thresholds are CALIBRATED from the real
distribution (band-degeneracy rule): if any band takes >=75% of measured items
it is not discriminating and must be re-probed. Read-only, no GPU claim, no
corpus write, no new model.
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

from research_harness.environment_clearance import (  # noqa: E402
    EnvironmentClearanceError,
    compute_environment_clearance,
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
        try:
            seg2 = np.load(segp, allow_pickle=False)
        except FileNotFoundError:
            rows.append({
                "image_id": image_id, "abstained": True,
                "abstention_reason": "seg2 missing",
            })
            print(f"{image_id[:12]}  ABSTAIN (seg2 missing)")
            continue
        try:
            cfg = compute_environment_clearance(seg2)
        except EnvironmentClearanceError as exc:
            print(f"FAIL {image_id[:12]}: {exc}")
            return 2
        rows.append({
            "image_id": image_id,
            "abstained": cfg.get("abstained"),
            "abstention_reason": cfg.get("abstention_reason"),
            "subject_present": cfg.get("subject_present"),
            "clearance_band": cfg.get("clearance_band"),
            "clearance_ratio": cfg.get("clearance_ratio"),
            "clearance_top": cfg.get("clearance_top"),
            "clearance_bottom": cfg.get("clearance_bottom"),
            "clearance_left": cfg.get("clearance_left"),
            "clearance_right": cfg.get("clearance_right"),
            "subject_frame_coverage": cfg.get("subject_frame_coverage"),
        })
        print(
            f"{image_id[:12]}  band={str(cfg.get('clearance_band')):<10}  "
            f"ratio={cfg.get('clearance_ratio')}  "
            f"T/B/L/R={cfg.get('clearance_top')}/{cfg.get('clearance_bottom')}/"
            f"{cfg.get('clearance_left')}/{cfg.get('clearance_right')}"
        )

    det = [r for r in rows if not r.get("abstained")]
    n = len(det)
    print("\n=== CALIBRATION SUMMARY ===")
    print(f"measured: {n}/{len(rows)}")
    vals = [r.get("clearance_band") for r in det]
    c = Counter(vals)
    max_share = max(c.values()) / len(vals) if vals else 0
    print(f"clearance_band: {dict(c)}  max_share={max_share:.2f}")

    for ax in ("clearance_ratio", "clearance_top", "clearance_bottom",
               "clearance_left", "clearance_right"):
        vals2 = sorted(r.get(ax) for r in det if r.get(ax) is not None)
        if vals2:
            q = statistics.quantiles(vals2, n=4)
            print(f"{ax}: n={len(vals2)} min={vals2[0]:.3f} p25={q[0]:.3f} "
                  f"median={statistics.median(vals2):.3f} p75={q[2]:.3f} max={vals2[-1]:.3f}")

    for r in rows:
        if r.get("abstained"):
            print(f"  ABSTAIN {r['image_id'][:12]}: {r.get('abstention_reason')}")

    Path("/mnt/nas-ai-models/research/stratum/environment-clearance-calibration-probe.json").write_text(
        json.dumps({"rows": rows, "summary": {
            "measured": n, "items": len(rows),
            "clearance_band": dict(c),
        }}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
