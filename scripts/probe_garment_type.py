"""CPU probe: run the arm-#97 garment-type measurement over the frozen
24-item cohort BEFORE the plan is frozen. Reports the garment-type band + raw
coverage ratios so presence floors are CALIBRATED from the real distribution
(band-degeneracy rule): if any band takes >=75% of measured items it is not
discriminating and must be re-probed. Read-only, no GPU claim, no corpus write,
no new model.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from research_harness.garment_type import (  # noqa: E402
    GarmentTypeError,
    compute_garment_type,
)

MANIFEST = "/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json"
DERIVED = "/mnt/nas-ai-models/training-data/crawlr/stratum"

MAX_BAND_SHARE = 0.75


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
            rows.append({"image_id": image_id, "abstained": True,
                         "abstention_reason": "seg2 missing"})
            print(f"{image_id[:12]}  ABSTAIN (seg2 missing)")
            continue
        try:
            cfg = compute_garment_type(seg2)
        except GarmentTypeError as exc:
            print(f"FAIL {image_id[:12]}: {exc}")
            return 2
        rows.append({
            "image_id": image_id,
            "abstained": cfg.get("abstained"),
            "abstention_reason": cfg.get("abstention_reason"),
            "garment_type_band": cfg.get("garment_type_band"),
            "upper_garment_present": cfg.get("upper_garment_present"),
            "lower_garment_present": cfg.get("lower_garment_present"),
            "skin_dominant": cfg.get("skin_dominant"),
            "upper_garment_coverage": cfg.get("upper_garment_coverage"),
            "lower_garment_coverage": cfg.get("lower_garment_coverage"),
            "upper_skin_coverage": cfg.get("upper_skin_coverage"),
            "lower_skin_coverage": cfg.get("lower_skin_coverage"),
        })
        print(
            f"{image_id[:12]}  band={str(cfg.get('garment_type_band')):<22}  "
            f"up={cfg.get('upper_garment_present')}  lo={cfg.get('lower_garment_present')}  "
            f"skin={cfg.get('skin_dominant')}  upcov={cfg.get('upper_garment_coverage')}  "
            f"locov={cfg.get('lower_garment_coverage')}"
        )

    det = [r for r in rows if not r.get("abstained")]
    n = len(det)
    print("\n=== CALIBRATION SUMMARY ===")
    print(f"measured: {n}/{len(rows)}")
    vals = [r.get("garment_type_band") for r in det]
    c = Counter(vals)
    max_share = max(c.values()) / len(vals) if vals else 0
    print(f"garment_type_band: {dict(c)}  max_share={max_share:.2f}")
    print(f"  {'DEGENERATE (>75%)' if max_share > MAX_BAND_SHARE else 'OK (<=75%)'}")

    for r in rows:
        if r.get("abstained"):
            print(f"  ABSTAIN {r['image_id'][:12]}: {r.get('abstention_reason')}")

    Path("/mnt/nas-ai-models/research/stratum/garment-type-calibration-probe.json").write_text(
        json.dumps({"rows": rows, "summary": {
            "measured": n, "items": len(rows),
            "garment_type_band": dict(c), "max_share": max_share,
        }}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
