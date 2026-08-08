"""CPU probe: run the arm-#80 eye-color measurement over the frozen 24-item
cohort BEFORE the plan is frozen. Reports the eye-color band + raw HSV stats
so the thresholds are CALIBRATED from the real distribution (band-degeneracy
rule): if any band takes >=75% of measured items it is not discriminating and
must be re-probed. Read-only, no GPU claim, no corpus write, no new model.
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

from research_harness.eye_color import (  # noqa: E402
    EyeColorError,
    compute_eye_color,
)

MANIFEST = "/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json"
DERIVED = "/mnt/nas-ai-models/training-data/crawlr/stratum"
APPROVED = "/mnt/nas-ai-models/training-data/crawlr/approved"


def main() -> int:
    manifest = json.loads(Path(MANIFEST).read_text())
    items = manifest["items"]
    rows = []
    for item in items:
        image_id = item["image_id"]
        posep = Path(DERIVED) / image_id / "pose2.npy"
        rel = item.get("source_relative_path")
        srcp = Path(APPROVED) / rel
        try:
            pose2 = np.load(posep, allow_pickle=False)
        except FileNotFoundError:
            rows.append({"image_id": image_id, "abstained": True,
                         "abstention_reason": "pose2 missing"})
            print(f"{image_id[:12]}  ABSTAIN (pose2 missing)")
            continue
        try:
            rgb = np.asarray(Image.open(srcp).convert("RGB"), dtype=np.uint8)
        except FileNotFoundError:
            rows.append({"image_id": image_id, "abstained": True,
                         "abstention_reason": "source missing"})
            print(f"{image_id[:12]}  ABSTAIN (source missing)")
            continue
        try:
            cfg = compute_eye_color(pose2, rgb)
        except EyeColorError as exc:
            print(f"FAIL {image_id[:12]}: {exc}")
            return 2
        rows.append({
            "image_id": image_id,
            "abstained": cfg.get("abstained"),
            "abstention_reason": cfg.get("abstention_reason"),
            "eye_color_band": cfg.get("eye_color_band"),
            "sample_count": cfg.get("sample_count"),
            "hue_deg": cfg.get("hue_deg"),
            "saturation": cfg.get("saturation"),
            "value": cfg.get("value"),
            "per_eye": cfg.get("per_eye"),
        })
        print(
            f"{image_id[:12]}  color={str(cfg.get('eye_color_band')):<12}  "
            f"samples={cfg.get('sample_count')}  "
            f"hue={cfg.get('hue_deg')}  sat={cfg.get('saturation')}  val={cfg.get('value')}"
        )

    det = [r for r in rows if not r.get("abstained")]
    n = len(det)
    print("\n=== CALIBRATION SUMMARY ===")
    print(f"measured: {n}/{len(rows)}")
    vals = [r.get("eye_color_band") for r in det]
    c = Counter(vals)
    max_share = max(c.values()) / len(vals) if vals else 0
    print(f"eye_color_band: {dict(c)}  max_share={max_share:.2f}")

    for r in rows:
        if r.get("abstained"):
            print(f"  ABSTAIN {r['image_id'][:12]}: {r.get('abstention_reason')}")

    Path("/mnt/nas-ai-models/research/stratum/eye-color-calibration-probe.json").write_text(
        json.dumps({"rows": rows, "summary": {
            "measured": n, "items": len(rows),
            "eye_color_band": dict(c),
        }}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
