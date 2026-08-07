"""CPU probe: run the arm-#69 CLIP zero-shot scene classifier over the frozen
24-item cohort BEFORE the plan is frozen. Reports per-item category + softmax
confidence so the abstention floor and any band thresholds are CALIBRATED from
the real distribution (band-degeneracy rule arm #34/#35/#59): if any category
takes >=75% of items the closed set is not discriminating and must be re-probed.
Read-only, no GPU claim, no corpus write.
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

from research_harness.scene_category import (  # noqa: E402
    ABSTAIN_CONFIDENCE,
    SCENE_CATEGORIES,
    SCENE_CATEGORY_MODEL_ASSET,
    compute_scene_category,
)

MANIFEST = "/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json"
SOURCE = "/mnt/nas-ai-models/training-data/crawlr/approved"


def main() -> int:
    manifest = json.loads(Path(MANIFEST).read_text())
    items = manifest["items"]
    rows = []
    confs = []
    for item in items:
        rel = item["source_relative_path"]
        src = Path(SOURCE) / rel
        rgb = Image.open(src).convert("RGB")
        scene = compute_scene_category(
            np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)).copy(),
            model_asset_dir=SCENE_CATEGORY_MODEL_ASSET,
        )
        if not scene.get("abstained"):
            confs.append(scene["confidence"])
        rows.append({
            "image_id": item["image_id"],
            "category": scene.get("category"),
            "confidence": scene.get("confidence"),
            "abstained": scene.get("abstained", False),
            "abstention_reason": scene.get("abstention_reason"),
        })
        print(f"{item['image_id'][:12]}  {str(scene.get('category')):<18}  conf={scene.get('confidence')}")

    cats = Counter(r["category"] for r in rows if r["category"])
    n_class = sum(1 for r in rows if not r["abstained"])
    max_share = max(cats.values()) / n_class if n_class else 0
    confs_sorted = sorted(confs)
    p50 = confs_sorted[len(confs_sorted) // 2] if confs_sorted else None
    print("\n=== CALIBRATION SUMMARY ===")
    print(f"classified: {n_class}/{len(rows)}")
    print(f"distinct categories: {len(cats)} of {len(SCENE_CATEGORIES)}")
    print(f"category counts: {dict(cats)}")
    print(f"max top-1 share: {max_share:.2f} (floor ABSTAIN={ABSTAIN_CONFIDENCE})")
    print(f"p50 confidence: {p50:.3f}")
    print(f"min confidence (classified): {min(confs) if confs else None:.3f}")
    print(f"n abstained below floor: {sum(1 for r in rows if r['abstained'])}")
    for r in rows:
        if r["abstained"]:
            print(f"  ABSTAIN {r['image_id'][:12]}: {r['abstention_reason']}")
    Path("/mnt/nas-ai-models/research/stratum/scene-category-calibration-probe.json").write_text(
        json.dumps({"rows": rows, "summary": {
            "classified": n_class, "items": len(rows), "distinct": len(cats),
            "counts": dict(cats), "max_share": max_share, "p50_conf": p50,
            "abstain_floor": ABSTAIN_CONFIDENCE,
        }}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
