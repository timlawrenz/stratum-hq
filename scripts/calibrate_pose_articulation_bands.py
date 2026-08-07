"""CPU-only elbow/knee flexion band distribution on the frozen 24-item cohort.

Calibration evidence for arm #62 band thresholds: confirms the <135deg bent
band is not degenerate (>=75% in one band would mean the threshold does not
discriminate).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np  # noqa: E402

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

from research_harness.pose_articulation import compute_pose_articulation  # noqa: E402

CAND = Path("/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json")
DER = Path("/mnt/nas-ai-models/training-data/crawlr/stratum")


def main() -> int:
    items = json.loads(CAND.read_text(encoding="utf-8"))["items"]
    bent = ext = na = 0
    els: list[float] = []
    kns: list[float] = []
    for it in items:
        pose = np.load(DER / it["image_id"] / "pose2.npy", allow_pickle=False)
        seg = np.load(DER / it["image_id"] / "seg2.npy", allow_pickle=False)
        m = compute_pose_articulation(pose, seg)
        for k in ("elbow_flexion_left", "elbow_flexion_right"):
            v = m.get(k)
            if v is None:
                na += 1
            elif float(v) < 135.0:
                bent += 1
            else:
                ext += 1
            if v is not None:
                els.append(float(v))
        for k in ("knee_flexion_left", "knee_flexion_right"):
            v = m.get(k)
            if v is not None:
                kns.append(float(v))
    els_arr = np.array(els)
    kns_arr = np.array(kns)
    print(f"elbow: bent(<135)={bent} ext(>=135)={ext} n/a={na}  total={bent + ext + na}")
    if len(els):
        print(f"  elbow mean={els_arr.mean():.1f} min={els_arr.min():.1f} max={els_arr.max():.1f} "
              f"p25={np.percentile(els_arr, 25):.1f} p50={np.percentile(els_arr, 50):.1f} p75={np.percentile(els_arr, 75):.1f}")
    if len(kns):
        print(f"knee: n={len(kns)} mean={kns_arr.mean():.1f} "
              f"p25={np.percentile(kns_arr, 25):.1f} p50={np.percentile(kns_arr, 50):.1f} p75={np.percentile(kns_arr, 75):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
