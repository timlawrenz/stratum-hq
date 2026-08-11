"""CPU probe: run the arm-#74 camera-viewing-angle framing measurement over the
frozen 24-item cohort BEFORE the plan is frozen. Reports per-item shot-scale /
headroom / camera-height bands so the thresholds are CALIBRATED from the real
distribution (band-degeneracy rule arm #34/#35/#59): if any band takes >=75%
of measured items it is not discriminating and must be re-probed. Read-only,
no GPU claim, no corpus write, no new model.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from research_harness.camera_viewing_angle import (  # noqa: E402
    compute_camera_viewing_angle,
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
                "abstention_reason": f"seg2 missing at {segp}",
            })
            print(f"{image_id[:12]}  ABSTAIN (seg2 missing)")
            continue
        img_h, img_w = seg2.shape
        framing = compute_camera_viewing_angle(seg2, img_h, img_w)
        rows.append({
            "image_id": image_id,
            **{k: framing.get(k) for k in
               ("abstained", "abstention_reason", "shot_scale_band",
                "headroom_band", "camera_height_band",
                "subject_frame_height_share", "headroom_frame_share")},
        })
        print(f"{image_id[:12]}  shot={framing.get('shot_scale_band')}  "
              f"headroom={framing.get('headroom_band')}  "
              f"camera={framing.get('camera_height_band')}  "
              f"share={framing.get('subject_frame_height_share')}")

    det = [r for r in rows if not r.get("abstained")]
    n = len(det)
    print("\n=== CALIBRATION SUMMARY ===")
    print(f"measured: {n}/{len(rows)}")
    for ax in ("shot_scale_band", "headroom_band", "camera_height_band"):
        c = Counter(r.get(ax) for r in det)
        max_share = max(c.values()) / n if n else 0
        print(f"{ax}: {dict(c)}  max_share={max_share:.2f}")
    for r in rows:
        if r.get("abstained"):
            print(f"  ABSTAIN {r['image_id'][:12]}: {r.get('abstention_reason')}")
    Path("/mnt/nas-ai-models/research/stratum/camera-viewing-angle-calibration-probe.json").write_text(
        json.dumps({"rows": rows, "summary": {
            "measured": n, "items": len(rows),
            "bands": {ax: dict(Counter(r.get(ax) for r in det)) for ax in
                      ("shot_scale_band", "headroom_band", "camera_height_band")},
        }}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
