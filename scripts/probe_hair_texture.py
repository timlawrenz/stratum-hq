"""CPU probe: run the arm-#94 hair-texture measurement over the frozen
24-item cohort BEFORE the plan is frozen. Reports the scale-invariant
texture band (straight / wavy / curly) + raw R2 statistics so the thresholds
are CALIBRATED from the real distribution (band-degeneracy rule):
if any band takes >=75% of measured items it is not discriminating and must
be re-cut. Read-only, no GPU claim, no corpus write, no new model.
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

from research_harness.hair_texture import (  # noqa: E402
    HairTextureError,
    compute_hair_texture,
)

MANIFEST = "/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json"
DERIVED = "/mnt/nas-ai-models/training-data/crawlr/stratum"
SOURCE = "/mnt/nas-ai-models/training-data/crawlr/approved"


def main() -> int:
    manifest = json.loads(Path(MANIFEST).read_text())
    items = manifest["items"]
    rows = []
    for item in items:
        image_id = item["image_id"]
        segp = Path(DERIVED) / image_id / "seg2.npy"
        srcp = Path(SOURCE) / item["source_relative_path"]
        try:
            seg2 = np.load(segp, allow_pickle=False)
        except FileNotFoundError as exc:
            rows.append({
                "image_id": image_id, "abstained": True,
                "abstention_reason": f"seg2 missing: {exc.filename}",
            })
            print(f"{image_id[:12]}  ABSTAIN (seg2 missing)")
            continue
        try:
            from PIL import Image
            img = np.asarray(Image.open(srcp).convert("RGB"), dtype=np.uint8)
            if img.shape[0] != seg2.shape[0] or img.shape[1] != seg2.shape[1]:
                img = np.asarray(
                    Image.fromarray(img).resize(
                        (seg2.shape[1], seg2.shape[0]), Image.BILINEAR
                    ),
                    dtype=np.uint8,
                )
        except Exception as exc:  # noqa: BLE001 - probe should not die on one item
            rows.append({
                "image_id": image_id, "abstained": True,
                "abstention_reason": f"source read failed: {exc}",
            })
            print(f"{image_id[:12]}  ABSTAIN (source read failed: {exc})")
            continue
        try:
            cfg = compute_hair_texture(seg2, img)
        except HairTextureError as exc:
            print(f"FAIL {image_id[:12]}: {exc}")
            return 2
        rows.append({
            "image_id": image_id,
            "abstained": cfg.get("abstained"),
            "abstention_reason": cfg.get("abstention_reason"),
            "hair_present": cfg.get("hair_present"),
            "hair_texture_band": cfg.get("hair_texture_band"),
            "r2_concentration": cfg.get("r2_concentration"),
            "orientation_entropy": cfg.get("orientation_entropy"),
            "mean_gradient": cfg.get("mean_gradient"),
        })
        print(
            f"{image_id[:12]}  band={str(cfg.get('hair_texture_band')):<9} "
            f"r2={cfg.get('r2_concentration')}  entr={cfg.get('orientation_entropy')}  "
            f"grad={cfg.get('mean_gradient')}"
        )

    det = [r for r in rows if not r.get("abstained")]
    n = len(det)
    print("\n=== CALIBRATION SUMMARY ===")
    print(f"measured: {n}/{len(rows)}")

    vals = [r.get("hair_texture_band") for r in det if r.get("hair_texture_band") is not None]
    c = Counter(vals)
    max_share = max(c.values()) / len(vals) if vals else 0
    print(f"hair_texture_band: {dict(c)}  max_share={max_share:.2f}")

    print("\n--- continuous discriminators (for re-cutting degenerate bands) ---")
    for ax in ("r2_concentration", "orientation_entropy", "mean_gradient"):
        v = sorted(r.get(ax) for r in det if r.get(ax) is not None)
        if v:
            q = statistics.quantiles(v, n=4)
            print(f"{ax}: n={len(v)} min={v[0]:.3f} p25={q[0]:.3f} "
                  f"median={statistics.median(v):.3f} p75={q[2]:.3f} max={v[-1]:.3f}")

    for r in rows:
        if r.get("abstained"):
            print(f"  ABSTAIN {r['image_id'][:12]}: {r.get('abstention_reason')}")

    Path("/mnt/nas-ai-models/research/stratum/hair-texture-calibration-probe.json").write_text(
        json.dumps({"rows": rows, "summary": {
            "measured": n, "items": len(rows),
            "hair_texture_band": dict(Counter(r.get("hair_texture_band") for r in det)),
        }}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())