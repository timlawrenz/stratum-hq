#!/usr/bin/env python
"""CPU band-calibration probe for arm #58 pointmap-depth (read-only)."""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

from research_harness.pointmap_depth import compute_pointmap_depth  # noqa: E402

DERIVED = Path("/mnt/nas-ai-models/training-data/crawlr/stratum")
ids = json.load(open("/tmp/frozen_ids.json"))

rows = []
for i in ids:
    item = DERIVED / i
    pm = np.load(item / "pointmap.npy")
    seg = np.load(item / "seg2.npy")
    m = compute_pointmap_depth(pm, seg)
    rows.append({"id": i, **m})

measurable = [r for r in rows if not r["abstained"]]
n = len(rows)
print(f"items: {n}  measurable: {len(measurable)}  abstained: {n - len(measurable)}")
for r in rows:
    if r["abstained"]:
        print(f"  ABSTAIN {r['id']}: {r['abstention_reason']}")

if measurable:
    bands = {}
    for key in ("relief_band",):
        from collections import Counter
        c = Counter(r[key] for r in measurable)
        print(f"{key} distribution: {dict(c)}  -> max share {(max(c.values()) / len(measurable)):.2%}")
    # hand ordering distribution
    ho = sorted(r["hand_ordering"] for r in measurable if r["hand_ordering"])
    print(f"hand_ordering fired: {len(ho)}/{len(measurable)} -> {dict(__import__('collections').Counter(ho))}")
    hfront = sum(1 for r in measurable if r["left_hand_in_front"] or r["right_hand_in_front"])
    print(f"hand_in_front fired: {hfront}/{len(measurable)}")
    reliefs = sorted(r["depth_relief_ratio"] for r in measurable)
    print(f"relief p10/p50/p90: {reliefs[len(reliefs)//10]:.3f} / {reliefs[len(reliefs)//2]:.3f} / {reliefs[9*len(reliefs)//10]:.3f}")
    # nearest/farthest region variety
    from collections import Counter
    print("nearest_region:", dict(Counter(r["nearest_region"] for r in measurable)))
    print("farthest_region:", dict(Counter(r["farthest_region"] for r in measurable)))

json.dump(rows, open("/tmp/pointmap_depth_probe.json", "w"), indent=1, default=str)
print("probe saved to /tmp/pointmap_depth_probe.json")
