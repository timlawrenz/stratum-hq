"""GPU probe: Multi-HMR-Anny body-volume measurement over the frozen 24-item
cohort BEFORE the plan is frozen (arm #96).

Gate steps:
(a) MODEL-QUALIFICATION / CAPABILITY probe — verify the Multi-HMR-Anny stack
    actually produces plausible whole-body meshes on the real frozen cohort
    coverage floor pre-registered >=18/24 plausible meshes) and that the
    scale-invariant geometric body-volume metric is sane. This is the
    verify-BEFORE-trust gate for a NEW MODEL CLASS.
(b) BAND calibration — report the normalized-volume distribution so band
    floors are CALIBRATED from the real cohort (band-degeneracy rule: if a
    single band would take >=75% of measured items the scheme is not
    discriminating and must be re-cut or downgraded to payload-only).

Read-only, scheduler-managed GPU on owned hardware (local 4090 via the GPU
scheduler — the poll performs the atomic claim), no corpus write. The
sensitive cohort never leaves owned hardware.

Output: /mnt/nas-ai-models/research/stratum/body-volume-calibration-probe.json
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

from research_harness.body_volume import (  # noqa: E402
    BodyVolumeError,
    CHECKPOINT,
    compute_body_volume,
)

MANIFEST = "/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json"
SOURCE = "/mnt/nas-ai-models/training-data/crawlr/approved"
DERIVED = "/mnt/nas-ai-models/training-data/crawlr/stratum"
OUT = "/mnt/nas-ai-models/research/stratum/body-volume-calibration-probe.json"

# Pre-registered coverage floor (from the arm declaration).
COVERAGE_FLOOR = 18


def _candidate_band_cuts(sorted_vals: list[float]) -> list[tuple[tuple[float, float], dict]]:
    """Enumerate honest 3-band cuts (slim/average/fuller) over the measured
    normalized-volume values and return (floors, band_counts) candidates with
    max_share < 0.75. Floors are either explicit candidate values or
    percentile cuts of the measured distribution (never threshold-fitted on a
    single test item — all cuts are stated in advance from the cohort stats).
    """
    n = len(sorted_vals)
    if n == 0:
        return []
    candidates: list[tuple[tuple[float, float], dict]] = []

    def _eval(slim_hi: float, avg_hi: float):
        bands = []
        for v in sorted_vals:
            if v < slim_hi:
                bands.append("slim")
            elif v < avg_hi:
                bands.append("average")
            else:
                bands.append("fuller")
        counts = Counter(bands)
        max_share = max(counts.values()) / n
        return counts, max_share

    # Candidate 1: equal-ish percentiles.
    p33 = sorted_vals[int(0.33 * n)]
    p66 = sorted_vals[int(0.66 * n)]
    if p33 < p66:
        c, m = _eval(p33, p66)
        candidates.append(((float(p33), float(p66)), {"counts": dict(c), "max_share": m}))
    # Candidate 2: natural mid split at the median gap neighbourhood.
    med = sorted_vals[n // 2]
    c23, m23 = _eval(med * 0.98, med * 1.02)
    candidates.append(((float(med * 0.98), float(med * 1.02)), {"counts": dict(c23), "max_share": m23}))
    # Candidate 3: quartiles.
    q1 = sorted_vals[int(0.25 * n)]
    q3 = sorted_vals[int(0.75 * n)]
    if q1 < q3:
        c, m = _eval(q1, q3)
        candidates.append(((float(q1), float(q3)), {"counts": dict(c), "max_share": m}))
    return candidates


def main() -> int:
    manifest = json.loads(Path(MANIFEST).read_text())
    items = manifest["items"]
    rows: list[dict] = []
    raw_norm: list[float] = []
    weights: list[float] = []
    n_detected = 0
    n_abstained = 0
    for item in items:
        image_id = item["image_id"]
        rel = item["source_relative_path"]
        rgb = np.ascontiguousarray(np.asarray(
            Image.open(Path(SOURCE) / rel).convert("RGB"), dtype=np.uint8).copy())
        seg2_path = Path(DERIVED) / image_id / "seg2.npy"
        seg2 = np.load(seg2_path) if seg2_path.exists() else None
        try:
            r = compute_body_volume(seg2, rgb, checkpoint=CHECKPOINT)
        except BodyVolumeError as exc:
            r = {"abstained": True, "abstention_reason": f"BodyVolumeError: {exc}"}
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            r = {"abstained": True, "abstention_reason": f"unexpected error: {exc!r}"}
        r["image_id"] = image_id
        rows.append(r)
        if not r.get("abstained") or r.get("normalized_volume") is not None:
            if r.get("normalized_volume") is not None:
                raw_norm.append(float(r["normalized_volume"]))
            n_detected += 1
            if r.get("shape_weight") is not None:
                weights.append(float(r["shape_weight"]))
            print(f"{image_id[:12]}  mesh_verts={r.get('mesh_vertices')}  "
                  f"norm_vol={r.get('normalized_volume')}  "
                  f"weight={r.get('shape_weight')}  abstained={r.get('abstained')} "
                  f"({r.get('abstention_reason') or ''})")
        else:
            n_abstained += 1
            print(f"{image_id[:12]}  ABSTAIN: {r.get('abstention_reason')}")

    print("\n=== COVERAGE (pre-registered floor >= %d/24) ===" % COVERAGE_FLOOR)
    print(f"plausible meshes: {n_detected}/{len(items)}  (abstained {n_abstained})")
    coverage_ok = n_detected >= COVERAGE_FLOOR

    print("\n=== NORMALIZED VOLUME (volume/height^3) ===")
    sv = sorted(raw_norm)
    if sv:
        print(f"n={len(sv)}  min={sv[0]:.5f}  p25={sv[int(0.25*len(sv))]:.5f}  "
              f"median={sv[len(sv)//2]:.5f}  p75={sv[int(0.75*len(sv))]:.5f}  max={sv[-1]:.5f}")
    else:
        print("none measured")

    print("\n=== WEIGHT PHENOTYPE (payload-only cross-check) ===")
    sw = sorted(weights)
    if sw:
        print(f"n={len(sw)}  min={sw[0]:.4f}  median={sw[len(sw)//2]:.4f}  max={sw[-1]:.4f}")

    # Band calibration (honest, from the cohort stats).
    band_candidates = _candidate_band_cuts(sv) if sv else []
    chosen: dict | None = None
    for floors, info in band_candidates:
        print(f"band cut floors={floors[0]:.5f}/{floors[1]:.5f} -> "
              f"counts={info['counts']} max_share={info['max_share']:.2f} "
              f"({'OK' if info['max_share'] < 0.75 else 'DEGENERATE'})")
        if chosen is None and info["max_share"] < 0.75:
            chosen = {"floors": [round(floors[0], 6), round(floors[1], 6)], "info": info}

    # The module's CALIBRATED band distribution (what the frozen round-trip
    # will actually verbalize).
    module_bands = Counter(r["body_volume_band"] for r in rows if r.get("body_volume_band"))
    n_banded = sum(module_bands.values())

    summary = {
        "items": len(items),
        "n_detected_plausible_mesh": n_detected,
        "n_abstained": n_abstained,
        "coverage_floor": COVERAGE_FLOOR,
        "coverage_ok": coverage_ok,
        "normalized_volume": {
            "n": len(sv),
            "min": round(sv[0], 6) if sv else None,
            "p25": round(sv[int(0.25 * len(sv))], 6) if sv else None,
            "median": round(sv[len(sv) // 2], 6) if sv else None,
            "p75": round(sv[int(0.75 * len(sv))], 6) if sv else None,
            "max": round(sv[-1], 6) if sv else None,
        },
        "weight_phenotype": {
            "n": len(sw),
            "min": round(sw[0], 4) if sw else None,
            "median": round(sw[len(sw) // 2], 4) if sw else None,
            "max": round(sw[-1], 4) if sw else None,
        },
        "band_candidates": [
            {"floors": [round(f[0], 6), round(f[1], 6)], "counts": i["counts"],
             "max_share": round(i["max_share"], 4)}
            for f, i in band_candidates
        ],
        "chosen_bands": chosen,
        "nondegenerate_bands_available": chosen is not None,
        "module_band_counts": dict(module_bands),
        "module_n_banded": n_banded,
        "module_max_share": round(max(module_bands.values()) / n_banded, 4) if n_banded else None,
    }
    Path(OUT).write_text(json.dumps({
        "rows": rows,
        "summary": summary,
    }, indent=2))
    print(f"\n=== PROBE RESULT ===")
    print(json.dumps(summary, indent=2))
    if not coverage_ok:
        print(f"COVERAGE FAIL: {n_detected}/{len(items)} < floor {COVERAGE_FLOOR}")
    if chosen is None and sv:
        print("BAND DEGENERACY: no non-degenerate 3-band cut found (max_share >= 0.75 on all cuts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
