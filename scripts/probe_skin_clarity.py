"""CPU probe: skin-clarity HP-energy density over the frozen 24-item cohort.

Arm #125 (NEW deterministic part skin-clarity, selected by the SELECTOR
EXPLOIT slot, currently ACTIVE). Band-calibration probe: report the
clear/even/blemished distribution so band floors are CALIBRATED from the real
cohort (band-degeneracy rule: a single band >=75% of measured items means the
axis does not discriminate and must be re-cut or kept payload-only).

Read-only, CPU, no new model, no corpus write. The sensitive cohort runs on
owned hardware only.

Output: /mnt/nas-ai-models/research/stratum/skin-clarity-calibration-probe.json
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

from research_harness.skin_clarity import compute_skin_clarity  # noqa: E402

MANIFEST = "/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json"
DERIVED = "/mnt/nas-ai-models/training-data/crawlr/stratum"
OUT = "/mnt/nas-ai-models/research/stratum/skin-clarity-calibration-probe.json"


def main() -> int:
    manifest = json.loads(Path(MANIFEST).read_text())
    rows = []
    densities = []
    for item in manifest["items"]:
        image_id = item["image_id"]
        seg2 = np.load(Path(DERIVED) / image_id / "seg2.npy")
        src = Path(DERIVED) / image_id
        import io

        payload_path = None
        for cand in (src / item["source_relative_path"].split("/")[-1],
                     Path("/mnt/nas-ai-models/training-data/crawlr/approved") / item["source_relative_path"]):
            if cand.exists():
                payload_path = cand
                break
        if payload_path is None:
            r = {"abstained": True, "abstention_reason": "source not found"}
            r["image_id"] = image_id
            rows.append(r)
            print(f"{image_id[:12]}  ABSTAIN: source not found")
            continue
        with open(payload_path, "rb") as fh:
            payload = fh.read()
        im = Image.open(io.BytesIO(payload)).convert("RGB")
        rgb = np.ascontiguousarray(np.asarray(im, dtype=np.uint8))
        if seg2.shape != rgb.shape[:2]:
            seg2 = np.asarray(Image.fromarray(seg2.astype(np.uint8)).resize(
                (rgb.shape[1], rgb.shape[0]), Image.NEAREST))
        try:
            r = compute_skin_clarity(seg2, rgb)
        except Exception as exc:  # noqa: BLE001
            r = {"abstained": True, "abstention_reason": f"error: {exc!r}"}
        r["image_id"] = image_id
        rows.append(r)
        d = r.get("skin_hp_density_luma")
        if d is not None:
            densities.append(d)
        print(
            f"{image_id[:12]}  band={str(r.get('skin_clarity_band')):<10} "
            f"D1={r.get('skin_hp_density_luma')} density={r.get('skin_hp_density')}  "
            f"interior={r.get('skin_interior_px')} luma_std={r.get('skin_luma_std')}  "
            + (f"ABSTAIN: {r.get('abstention_reason')}" if r.get("abstained") else "")
        )

    bands = Counter(r["skin_clarity_band"] for r in rows if r.get("skin_clarity_band"))
    n_banded = sum(bands.values())
    max_share = max(bands.values()) / n_banded if n_banded else 0
    basis = {}
    if densities:
        basis = {
            "min": round(float(min(densities)), 4),
            "p25": round(float(np.percentile(densities, 25)), 4),
            "median": round(float(np.median(densities)), 4),
            "p75": round(float(np.percentile(densities, 75)), 4),
            "max": round(float(max(densities)), 4),
            "p33": round(float(np.percentile(densities, 33.3)), 4),
            "p66": round(float(np.percentile(densities, 66.6)), 4),
        }
    summary = {
        "items": len(rows),
        "n_banded": n_banded,
        "n_abstained": sum(1 for r in rows if r.get("abstained")),
        "band_counts": dict(bands),
        "max_share": round(max_share, 4),
        "coverage_floor": 8,
        "coverage_floor_met": n_banded >= 8,
        "density_distribution": basis,
    }
    Path(OUT).write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    print("\n=== PROBE RESULT ===")
    print(json.dumps(summary, indent=2))
    if n_banded and max_share >= 0.75:
        print("BAND DEGENERACY: a single band >=75% — axis does not discriminate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
