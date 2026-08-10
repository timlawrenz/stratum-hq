"""CPU probe: waist-shape v2 (limb-excluded torso profile) over the frozen 24-item cohort.

Arm #106 decision (a) — better waist estimator (recorded on the issue,
2026-08-10). The v1 estimator measured 10/24 with only 4/24 plausible and
max_share 1.00 (all 'hourglass', median ratio 2.86) because row-width minima
were corrupted by arm/hand pixels in the torso span. v2 pre-registered gates
(strictly harder than v1's outcome on the discriminator axis):

- measured   >= 10/24   (v1 measured exactly 10/24)
- plausible  >= 6/24    (v1: 4/24)
- max_share  <  0.75 across {straight, moderate, hourglass, wider-hips}
                        among plausible items (v1: 1.00)

Plane gate (<=45 deg from horizontal on both segments) and the human-plausible
hip:waist band [0.7, 2.4] follow research_harness.proportions semantics
(owner directive — not weakened). Read-only, CPU, no new model, no corpus
write. Output: /mnt/nas-ai-models/research/stratum/waist-shape-v2-calibration-probe.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from research_harness.waist_shape import compute_waist_shape  # noqa: E402

MANIFEST = "/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json"
DERIVED = "/mnt/nas-ai-models/training-data/crawlr/stratum"
OUT = "/mnt/nas-ai-models/research/stratum/waist-shape-v2-calibration-probe.json"

# Pre-registered capability gates (see module docstring).
GATE_MEASURED = 10
GATE_PLAUSIBLE = 6
GATE_MAX_SHARE = 0.75


def main() -> int:
    manifest = json.loads(Path(MANIFEST).read_text())
    rows = []
    for item in manifest["items"]:
        image_id = item["image_id"]
        seg2 = np.load(Path(DERIVED) / image_id / "seg2.npy")
        pose2 = np.load(Path(DERIVED) / image_id / "pose2.npy")
        try:
            r = compute_waist_shape(seg2, pose2)
        except Exception as exc:  # noqa: BLE001
            r = {"abstained": True, "abstention_reason": f"error: {exc!r}"}
        r["image_id"] = image_id
        rows.append(r)
        print(f"{image_id[:12]}  band={str(r.get('waist_shape_band')):<11} "
              f"ratio={r.get('hip_waist_ratio')}  "
              f"hip/waist px={r.get('hip_w_px')}/{r.get('waist_w_px')}  "
              f"{'ABSTAIN: ' + str(r.get('abstention_reason')) if r.get('abstained') else ''}")

    measured = [r for r in rows if r.get("hip_waist_ratio") is not None]
    plausible = [r for r in measured
                 if r["hip_waist_ratio"] is not None
                 and 0.7 <= float(r["hip_waist_ratio"]) <= 2.4]
    bands = Counter(r["waist_shape_band"] for r in plausible if r.get("waist_shape_band"))
    n_banded = sum(bands.values())
    max_share = max(bands.values()) / n_banded if n_banded else 0.0
    ratios = sorted(float(r["hip_waist_ratio"]) for r in plausible)
    abst = [r["abstention_reason"] for r in rows if r.get("abstained")]
    abst_counts = Counter(abst)
    summary = {
        "items": len(rows),
        "measured": len(measured),
        "abstained": len(rows) - len(measured),
        "plausible": len(plausible),
        "implausible_measured": len(measured) - len(plausible),
        "ratio_stats": {
            "n": len(ratios),
            "min": round(ratios[0], 4) if ratios else None,
            "median": round(ratios[len(ratios) // 2], 4) if ratios else None,
            "max": round(ratios[-1], 4) if ratios else None,
        },
        "band_counts": dict(bands),
        "max_share": round(max_share, 4),
        "abstention_reasons": dict(abst_counts),
        # Pre-registered gates (recorded before measurement)
        "gates": {
            "measured_ge": GATE_MEASURED,
            "plausible_ge": GATE_PLAUSIBLE,
            "max_share_lt": GATE_MAX_SHARE,
            "plane_gate": "segments within 45 deg of horizontal (proportions semantics)",
            "plausibility_band": [0.7, 2.4],
        },
        "verdict": (
            "PASS" if (len(measured) >= GATE_MEASURED
                       and len(plausible) >= GATE_PLAUSIBLE
                       and max_share < GATE_MAX_SHARE)
            else "FAIL"
        ),
    }
    Path(OUT).write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    print("\n=== PROBE RESULT (v2) ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())