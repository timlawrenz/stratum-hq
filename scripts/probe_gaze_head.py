"""CPU probe: run the arm-#68 gaze/head-orientation measurement (reusing the
arm-#60 MediaPipe FaceLandmarker mesh) over the frozen 24-item cohort BEFORE
the plan is frozen. Reports per-item yaw/pitch/roll + bands so the band
thresholds are CALIBRATED from the real distribution (band-degeneracy rule arm
#34/#35/#59): if any band takes >=75% of detected items it is not
discriminating and must be re-probed. Read-only, no GPU claim, no corpus write.
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

from research_harness.gaze_head import (  # noqa: E402
    GAZE_HEAD_MODEL_ASSET,
    compute_gaze_head,
)

MANIFEST = "/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json"
DERIVED = "/mnt/nas-ai-models/training-data/crawlr/stratum"
SOURCE = "/mnt/nas-ai-models/training-data/crawlr/approved"
FACE_NECK = 3  # DOME-29 class index (Face_Neck)


def main() -> int:
    manifest = json.loads(Path(MANIFEST).read_text())
    items = manifest["items"]
    rows = []
    for item in items:
        rel = item["source_relative_path"]
        src = Path(SOURCE) / rel
        image_id = item["image_id"]
        segp = Path(DERIVED) / image_id / "seg2.npy"
        try:
            seg2 = np.load(segp, allow_pickle=False)
        except FileNotFoundError:
            seg2 = None
        rgb = np.ascontiguousarray(np.asarray(Image.open(src).convert("RGB"), dtype=np.uint8)).copy()
        if seg2 is None or seg2.shape != rgb.shape[:2]:
            rows.append({
                "image_id": image_id,
                "abstained": True,
                "abstention_reason": f"seg2 missing/misaligned at {segp}",
            })
            print(f"{image_id[:12]}  ABSTAIN (seg2 missing)")
            continue
        gaze = compute_gaze_head(seg2, rgb, model_asset_path=GAZE_HEAD_MODEL_ASSET)
        rows.append({
            "image_id": image_id,
            **{k: gaze.get(k) for k in
               ("abstained", "abstention_reason", "yaw", "pitch", "roll",
                "yaw_band", "pitch_band", "roll_band", "via")},
        })
        print(f"{item['image_id'][:12]}  via={gaze.get('via')}  "
              f"yaw={gaze.get('yaw')}  pitch={gaze.get('pitch')}  roll={gaze.get('roll')}  "
              f"[{gaze.get('yaw_band')} | {gaze.get('pitch_band')} | {gaze.get('roll_band')}]")

    det = [r for r in rows if not r.get("abstained")]
    n = len(det)
    print("\n=== CALIBRATION SUMMARY ===")
    print(f"detected: {n}/{len(rows)}")
    for ax in ("yaw_band", "pitch_band", "roll_band"):
        c = Counter(r.get(ax) for r in det)
        max_share = max(c.values()) / n if n else 0
        print(f"{ax}: {dict(c)}  max_share={max_share:.2f}")
    for r in rows:
        if r.get("abstained"):
            print(f"  ABSTAIN {r['image_id'][:12]}: {r.get('abstention_reason')}")
    Path("/mnt/nas-ai-models/research/stratum/gaze-head-calibration-probe.json").write_text(
        json.dumps({"rows": rows, "summary": {
            "detected": n, "items": len(rows),
            "bands": {ax: dict(Counter(r.get(ax) for r in det)) for ax in ("yaw_band", "pitch_band", "roll_band")},
        }}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
