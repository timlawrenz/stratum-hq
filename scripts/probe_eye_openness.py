"""CPU probe: eye-openness / eyelid-state over the frozen 24-item cohort.

Arm #113 (NEW deterministic part eye-openness, selected via the EXPLORE slot).
Band-calibration probe: report the per-eye openness-ratio distribution plus
the iris-state closed-eye signature so band floors are CALIBRATED from the
real cohort (band-degeneracy rule: a single band >=75% of measured items
means the axis does not discriminate and must be re-cut or kept payload-only).

Read-only, CPU, no new model, no corpus write. The sensitive cohort runs on
owned hardware only.

Output: /mnt/nas-ai-models/research/stratum/eye-openness-calibration-probe.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from research_harness.eye_openness import compute_eye_openness  # noqa: E402

MANIFEST = "/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json"
DERIVED = "/mnt/nas-ai-models/training-data/crawlr/stratum"
OUT = "/mnt/nas-ai-models/research/stratum/eye-openness-calibration-probe.json"


def main() -> int:
    manifest = json.loads(Path(MANIFEST).read_text())
    rows = []
    ratios: list[float] = []
    for item in manifest["items"]:
        image_id = item["image_id"]
        pose2 = np.load(Path(DERIVED) / image_id / "pose2.npy")
        try:
            r = compute_eye_openness(pose2)
        except Exception as exc:  # noqa: BLE001
            r = {"abstained": True, "abstention_reason": f"error: {exc!r}"}
        r["image_id"] = image_id
        rows.append(r)
        if not r.get("abstained") and r.get("openness_ratio") is not None:
            ratios.append(float(r["openness_ratio"]))
        per = r.get("per_eye") or {}
        states = {s: (e.get("eye_state"), e.get("openness_ratio"), e.get("iris_state"))
                  for s, e in per.items()}
        print(f"{image_id[:12]}  band={str(r.get('eye_openness_band')):<7} "
              f"ratio={r.get('openness_ratio')}  "
              f"L={states.get('l')}  R={states.get('r')}  "
              f"{'ABSTAIN: ' + str(r.get('abstention_reason')) if r.get('abstained') else ''}")

    bands = Counter(r["eye_openness_band"] for r in rows if r.get("eye_openness_band"))
    n_banded = sum(bands.values())
    max_share = max(bands.values()) / n_banded if n_banded else 0
    sv = sorted(ratios)
    summary = {
        "items": len(rows),
        "n_banded": n_banded,
        "n_abstained": sum(1 for r in rows if r.get("abstained")),
        "band_counts": dict(bands),
        "max_share": round(max_share, 4),
        "openness_ratio": {
            "n": len(sv),
            "min": round(sv[0], 4) if sv else None,
            "p25": round(sv[len(sv) // 4], 4) if sv else None,
            "median": round(sv[len(sv) // 2], 4) if sv else None,
            "p75": round(sv[3 * len(sv) // 4], 4) if sv else None,
            "max": round(sv[-1], 4) if sv else None,
        },
        "closed_max": 0.025,
        "lidded_max": 0.10,
    }
    Path(OUT).write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    print("\n=== PROBE RESULT ===")
    print(json.dumps(summary, indent=2))
    if n_banded and max_share >= 0.75:
        print("BAND DEGENERACY: a single band >=75% — axis does not discriminate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())