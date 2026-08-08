"""CPU probe: run the arm-#81 facial-expression measurement over the frozen
24-item cohort BEFORE the plan is frozen. Reports the expression band + raw
scale-invariant ratios so the thresholds are CALIBRATED from the real
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

from research_harness.facial_expression import (  # noqa: E402
    FacialExpressionError,
    compute_facial_expression,
)

MANIFEST = "/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json"
DERIVED = "/mnt/nas-ai-models/training-data/crawlr/stratum"


def main() -> int:
    manifest = json.loads(Path(MANIFEST).read_text())
    items = manifest["items"]
    rows = []
    for item in items:
        image_id = item["image_id"]
        posep = Path(DERIVED) / image_id / "pose2.npy"
        try:
            pose2 = np.load(posep, allow_pickle=False)
        except FileNotFoundError:
            rows.append({"image_id": image_id, "abstained": True,
                         "abstention_reason": "pose2 missing"})
            print(f"{image_id[:12]}  ABSTAIN (pose2 missing)")
            continue
        try:
            cfg = compute_facial_expression(pose2)
        except FacialExpressionError as exc:
            print(f"FAIL {image_id[:12]}: {exc}")
            return 2
        rows.append({
            "image_id": image_id,
            "abstained": cfg.get("abstained"),
            "abstention_reason": cfg.get("abstention_reason"),
            "expression_band": cfg.get("expression_band"),
            "spread_ratio": cfg.get("spread_ratio"),
            "openness_ratio": cfg.get("openness_ratio"),
            "corner_elevation_ratio": cfg.get("corner_elevation_ratio"),
            "reference_fallback": cfg.get("reference_fallback"),
        })
        print(
            f"{image_id[:12]}  expr={str(cfg.get('expression_band')):<14}  "
            f"spread={cfg.get('spread_ratio')}  open={cfg.get('openness_ratio')}  "
            f"elev={cfg.get('corner_elevation_ratio')}"
        )

    det = [r for r in rows if not r.get("abstained")]
    n = len(det)
    print("\n=== CALIBRATION SUMMARY ===")
    print(f"measured: {n}/{len(rows)}")
    vals = [r.get("expression_band") for r in det]
    c = Counter(vals)
    max_share = max(c.values()) / len(vals) if vals else 0
    print(f"expression_band: {dict(c)}  max_share={max_share:.2f}")

    for ax in ("spread_ratio", "openness_ratio", "corner_elevation_ratio"):
        vals2 = sorted(r.get(ax) for r in det if r.get(ax) is not None)
        if vals2:
            q = statistics.quantiles(vals2, n=4)
            print(f"{ax}: n={len(vals2)} min={vals2[0]:.3f} p25={q[0]:.3f} "
                  f"median={statistics.median(vals2):.3f} p75={q[2]:.3f} max={vals2[-1]:.3f}")

    nfb = sum(1 for r in det if r.get("reference_fallback"))
    print(f"reference_fallback: {nfb}/{n}")

    for r in rows:
        if r.get("abstained"):
            print(f"  ABSTAIN {r['image_id'][:12]}: {r.get('abstention_reason')}")

    Path("/mnt/nas-ai-models/research/stratum/facial-expression-calibration-probe.json").write_text(
        json.dumps({"rows": rows, "summary": {
            "measured": n, "items": len(rows),
            "expression_band": dict(c),
        }}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
