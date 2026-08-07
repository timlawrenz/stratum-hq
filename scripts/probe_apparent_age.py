#!/usr/bin/env python
"""CPU band-calibration probe for arm #73 apparent-age (read-only).

Runs MiVOLO-V2 over the frozen 24-item cohort BEFORE freezing the plan,
reporting the raw age distribution and band spread so the age-band thresholds
are CALIBRATED from the real cohort (band-degeneracy rule arm #34/#35/#58/#59/
#60): if any single band takes >= 75% of measured items the axis is not
discriminating and the thresholds must be re-probed. Read-only; no GPU claim;
no corpus write; sensitive canonical images run on owned hardware only.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from research_harness.apparent_age import (  # noqa: E402
    AGE_EARLY_TWENTIES_MAX,
    AGE_LATE_TEENS_MAX,
    AGE_MID_TWENTIES_MAX,
    _age_band,
    compute_apparent_age,
)

PLAN = ROOT / "research/stage-b-plans/stage-b-face-geometry-v1.json"
SOURCE = Path("/mnt/nas-ai-models/training-data/crawlr/approved")
DERIVED = Path("/mnt/nas-ai-models/training-data/crawlr/stratum")


def main() -> int:
    plan = json.loads(PLAN.read_text())
    items = plan["pilot_manifest"]["items"]
    rows = []
    ages = []
    for item in items:
        rel = item["source_relative_path"]
        src = Path(SOURCE) / rel
        rgb = np.ascontiguousarray(np.asarray(Image.open(src).convert("RGB"), dtype=np.uint8)).copy()
        seg2 = np.load(DERIVED / item["image_id"] / "seg2.npy")
        m = compute_apparent_age(seg2, rgb)
        if not m.get("abstained"):
            ages.append(m["age_years"])
        rows.append({
            "image_id": item["image_id"],
            "age_years": m.get("age_years"),
            "age_band": m.get("age_band"),
            "via": m.get("via"),
            "gender_probe": m.get("gender_probe"),
            "abstained": m.get("abstained", False),
            "abstention_reason": m.get("abstention_reason"),
            "seg2_face_neck_px": m.get("seg2_face_neck_px"),
        })
        print(f"{item['image_id'][:12]}  age={str(m.get('age_years')):<7} "
              f"band={str(m.get('age_band')):<28} via={m.get('via')}")

    n = len(rows)
    n_meas = sum(1 for r in rows if not r["abstained"])
    n_abs = n - n_meas
    bands = Counter(r["age_band"] for r in rows if r["age_band"])
    max_share = max(bands.values()) / n_meas if n_meas else 0
    ages_sorted = sorted(ages)
    print("\n=== CALIBRATION SUMMARY ===")
    print(f"measured: {n_meas}/{n}  abstained: {n_abs}")
    print(f"age band counts: {dict(bands)}  (crops: <{AGE_LATE_TEENS_MAX} / <{AGE_EARLY_TWENTIES_MAX} / "
          f"<{AGE_MID_TWENTIES_MAX} / else)")
    print(f"max band share: {max_share:.2%} (>=75% means degenerate -> re-probe)")
    if ages_sorted:
        print(f"age p10/p50/p90: {ages_sorted[n_meas//10]:.1f} / {ages_sorted[n_meas//2]:.1f} / "
              f"{ages_sorted[9*n_meas//10]:.1f}   min/max: {ages_sorted[0]:.1f}/{ages_sorted[-1]:.1f}")
    for r in rows:
        if r["abstained"]:
            print(f"  ABSTAIN {r['image_id'][:12]}: {r['abstention_reason']}")
    Path("/mnt/nas-ai-models/research/stratum/apparent-age-calibration-probe.json").write_text(
        json.dumps({"rows": rows, "summary": {
            "items": n, "measured": n_meas, "abstained": n_abs,
            "band_counts": dict(bands), "max_share": max_share,
            "age_late_teens_max": AGE_LATE_TEENS_MAX,
            "age_early_twenties_max": AGE_EARLY_TWENTIES_MAX,
            "age_mid_twenties_max": AGE_MID_TWENTIES_MAX,
            "ages": ages_sorted,
        }}, indent=2)
    )
    print("probe saved to /mnt/nas-ai-models/research/stratum/apparent-age-calibration-probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
