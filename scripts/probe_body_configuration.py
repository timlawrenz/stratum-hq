"""CPU probe: run the arm-#83 body-configuration measurement over the frozen
24-item cohort BEFORE the plan is frozen. Reports the per-item posture class
+ raw scale-invariant features so the classification thresholds are
CALIBRATED from the real distribution (band-degeneracy rule arm #34/#35/#59/etc.):
if any band takes >=75% of measured items it is not discriminating and must be
re-probed. Read-only, no GPU claim, no corpus write, no new model.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from research_harness.body_configuration import (  # noqa: E402
    BodyConfigurationError,
    compute_body_configuration,
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
        segp = Path(DERIVED) / image_id / "seg2.npy"
        try:
            pose2 = np.load(posep, allow_pickle=False)
        except FileNotFoundError:
            rows.append({
                "image_id": image_id, "abstained": True,
                "abstention_reason": f"pose2 missing at {posep}",
            })
            print(f"{image_id[:12]}  ABSTAIN (pose2 missing)")
            continue
        try:
            seg2 = np.load(segp, allow_pickle=False)
        except FileNotFoundError:
            rows.append({
                "image_id": image_id, "abstained": True,
                "abstention_reason": f"seg2 missing at {segp}",
            })
            print(f"{image_id[:12]}  ABSTAIN (seg2 missing)")
            continue
        try:
            cfg = compute_body_configuration(pose2, seg2)
        except BodyConfigurationError as exc:
            print(f"FAIL {image_id[:12]}: {exc}")
            return 2
        rows.append({
            "image_id": image_id,
            "abstained": cfg.get("abstained"),
            "abstention_reason": cfg.get("abstention_reason"),
            "posture_class": cfg.get("posture_class"),
            "pelvis_height_fraction": cfg.get("pelvis_height_fraction"),
            "torso_leg_extent_ratio": cfg.get("torso_leg_extent_ratio"),
            "median_knee_flexion_deg": cfg.get("median_knee_flexion_deg"),
            "torso_lean_deg": cfg.get("torso_lean_deg"),
        })
        pc = cfg.get("posture_class")
        print(f"{image_id[:12]}  class={str(pc):<9}  "
              f"pelvis={cfg.get('pelvis_height_fraction')}  "
              f"torso_leg={cfg.get('torso_leg_extent_ratio')}  "
              f"knee={cfg.get('median_knee_flexion_deg')}  "
              f"lean={cfg.get('torso_lean_deg')}")

    det = [r for r in rows if not r.get("abstained")]
    n = len(det)
    print("\n=== CALIBRATION SUMMARY ===")
    print(f"measured: {n}/{len(rows)}")
    c = Counter(r.get("posture_class") for r in det)
    max_share = max(c.values()) / n if n else 0
    print(f"posture_class: {dict(c)}  max_share={max_share:.2f}")

    # Distributions of the underlying continuous features (for re-cutting
    # bands if the class is degenerate).
    for ax in ("pelvis_height_fraction", "torso_leg_extent_ratio",
               "median_knee_flexion_deg", "torso_lean_deg"):
        vals = sorted(r.get(ax) for r in det if r.get(ax) is not None)
        if vals:
            import statistics
            print(f"{ax}: n={len(vals)} min={vals[0]:.3f} "
                  f"p25={statistics.quantiles(vals, n=4)[0]:.3f} "
                  f"median={statistics.median(vals):.3f} "
                  f"p75={statistics.quantiles(vals, n=4)[2]:.3f} "
                  f"max={vals[-1]:.3f}")
    for r in rows:
        if r.get("abstained"):
            print(f"  ABSTAIN {r['image_id'][:12]}: {r.get('abstention_reason')}")

    Path("/mnt/nas-ai-models/research/stratum/body-configuration-calibration-probe.json").write_text(
        json.dumps({"rows": rows, "summary": {
            "measured": n, "items": len(rows),
            "posture_class": dict(c),
            "bands": {ax: Counter(str(r.get(ax)) for r in det) for ax in
                      ("posture_class",)},
        }}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
