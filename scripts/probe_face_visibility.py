"""CPU probe: run the arm-#84 face-visibility measurement over the frozen
24-item cohort BEFORE the plan is frozen. Reports the occlusion fraction +
face-visibility band so the thresholds are CALIBRATED from the real
distribution (band-degeneracy rule): if any band takes >=75% of measured
items it is not discriminating and must be re-probed. Read-only, no GPU claim,
no corpus write, no new model.
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

from research_harness.face_visibility import (  # noqa: E402
    FaceVisibilityError,
    compute_face_visibility,
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
            cfg = compute_face_visibility(seg2)
        except FaceVisibilityError as exc:
            print(f"FAIL {image_id[:12]}: {exc}")
            return 2
        rows.append({
            "image_id": image_id,
            "abstained": cfg.get("abstained"),
            "abstention_reason": cfg.get("abstention_reason"),
            "face_present": cfg.get("face_present"),
            "face_visibility_band": cfg.get("face_visibility_band"),
            "face_share_of_head": cfg.get("face_share_of_head"),
            "face_px": cfg.get("face_px"),
            "face_frame_coverage": cfg.get("face_frame_coverage"),
        })
        print(
            f"{image_id[:12]}  band={str(cfg.get('face_visibility_band')):<18}  "
            f"share_head={cfg.get('face_share_of_head')}  "
            f"face_px={cfg.get('face_px')}"
        )

    det = [r for r in rows if not r.get("abstained")]
    n = len(det)
    print("\n=== CALIBRATION SUMMARY ===")
    print(f"measured: {n}/{len(rows)}")
    vals = [r.get("face_visibility_band") for r in det]
    c = Counter(vals)
    max_share = max(c.values()) / len(vals) if vals else 0
    print(f"face_visibility_band: {dict(c)}  max_share={max_share:.2f}")

    occ = sorted(r.get("face_share_of_head") for r in det if r.get("face_share_of_head") is not None)
    if occ:
        q = statistics.quantiles(occ, n=4)
        print(f"face_share_of_head: n={len(occ)} min={occ[0]:.3f} p25={q[0]:.3f} "
              f"median={statistics.median(occ):.3f} p75={q[2]:.3f} max={occ[-1]:.3f}")

    for r in rows:
        if r.get("abstained"):
            print(f"  ABSTAIN {r['image_id'][:12]}: {r.get('abstention_reason')}")

    Path("/mnt/nas-ai-models/research/stratum/face-visibility-calibration-probe.json").write_text(
        json.dumps({"rows": rows, "summary": {
            "measured": n, "items": len(rows),
            "face_visibility_band": dict(c),
        }}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
