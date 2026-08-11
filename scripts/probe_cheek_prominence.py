"""CPU capability + band-calibration probe for cheek-prominence (arm #128).

NEW evidence part `cheek-prominence` (reuses the already-qualified MediaPipe
FaceLandmarker 478-point mesh — same model as face-geometry #60, gaze-head #68,
eyebrow-position #111, nose-geometry #121; no new model). Band-calibration
probe on the frozen 24-item cohort: per-item cheek_bone_ratio (zygomatic-arch
span 116-345 / lower-jaw span 172-397) so band floors are calibrated from the
real cohort (band-degeneracy rule: a single band >= 75% of measured items means
the axis does not discriminate). The primary span pair deliberately avoids
face-geometry #60's face-width cheek pair 234/454 (anti-redundancy disclosure
in the module); jaw pair 172/397 is shared with #60 so the axes compose on a
common anchor.

Phase 1 (synthetic, non-sensitive): module must load + run + clean-abstain on a
blank non-face frame. Phase 2 (frozen cohort, read-only, local-first): sources
prefer the canonical approved/ export and fall back to the raw/ sharded byte
tree (identical bytes, SHA-verified by the frozen manifest) — the hold-#132
documented pattern, mirroring probe_nose_geometry.py.

Read-only, CPU, no new model, no corpus write.
Output: /mnt/nas-ai-models/research/stratum/cheek-prominence-calibration-probe.json
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

from research_harness.cheek_prominence import (  # noqa: E402
    PROMINENT_MIN,
    RATIO_PLAUSIBLE,
    SUBTLE_MAX,
    compute_cheek_prominence,
    render_cheek_prominence,
)

MODEL = "/mnt/nas-ai-models/research/stratum/models/face-geometry/face_landmarker.task"
MANIFEST = "/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json"
SOURCE_ROOT = Path("/mnt/nas-ai-models/training-data/crawlr/approved")
DERIVED = Path("/mnt/nas-ai-models/training-data/crawlr/stratum")
OUT = "/mnt/nas-ai-models/research/stratum/cheek-prominence-calibration-probe.json"


def _synthetic(model_path: str) -> dict:
    """Non-sensitive smoke: blank non-face must abstain cleanly (no crash)."""
    W, H = 320, 320
    blank = np.zeros((H, W, 3), dtype=np.uint8) + 210
    seg2 = np.zeros((H, W), dtype=np.uint8)
    try:
        res = compute_cheek_prominence(seg2, blank, model_asset_path=model_path)
    except Exception as exc:  # noqa: BLE001
        return {"loads": True, "synthetic_run": False, "error": repr(exc)}
    return {
        "loads": True,
        "synthetic_run": True,
        "abstained_cleanly": bool(res.get("abstained")),
        "reason": res.get("abstention_reason"),
        "render": render_cheek_prominence(res),
    }


def _load_source_rgb(item: dict) -> np.ndarray:
    """Decode the source RGB read-only. Prefers the canonical approved/ export;
    falls back to the raw/ sharded byte tree (identical bytes, SHA-verified by
    the caller) when the approved entry was purged — READ-ONLY, no mutation."""
    from pathlib import Path as _P

    src = SOURCE_ROOT / item["source_relative_path"]
    if src.exists():
        return np.asarray(Image.open(src).convert("RGB"), dtype=np.uint8)
    key = item["image_id"]
    raw_cand = _P("/mnt/nas-ai-models/training-data/crawlr/raw") / key[0:2] / key[2:4] / key
    if raw_cand.exists():
        return np.asarray(Image.open(raw_cand).convert("RGB"), dtype=np.uint8)
    raise FileNotFoundError(f"neither approved nor raw source found for {key}")


def main() -> int:
    manifest = json.loads(Path(MANIFEST).read_text())
    synthetic = _synthetic(MODEL)
    print("=== SYNTHETIC ===\n" + json.dumps(synthetic, indent=1), flush=True)
    if not synthetic.get("abstained_cleanly", False):
        print("PROBE FAIL: module did not abstain cleanly on a blank non-face frame",
              flush=True)
        sys.exit(1)

    rows = []
    for item in manifest["items"]:
        image_id = item["image_id"]
        try:
            rgb = _load_source_rgb(item)
            seg2 = np.load(DERIVED / image_id / "seg2.npy")
            r = compute_cheek_prominence(seg2, rgb, model_asset_path=MODEL)
        except Exception as exc:  # noqa: BLE001
            r = {"abstained": True, "abstention_reason": f"error: {exc!r}"}
        r["image_id"] = image_id
        rows.append(r)
        print(
            f"{image_id[:12]}  band={str(r.get('cheek_prominence_band')):<8} "
            f"ratio={r.get('cheek_bone_ratio')} "
            f"{'ABSTAIN: ' + str(r.get('abstention_reason')) if r.get('abstained') else ''}",
            flush=True,
        )

    measured = [r for r in rows if not r.get("abstained") and r.get("cheek_bone_ratio") is not None]
    vals = sorted(float(r["cheek_bone_ratio"]) for r in measured)

    summary = {
        "items": len(rows),
        "n_measured": len(measured),
        "n_abstained": sum(1 for r in rows if r.get("abstained")),
        "plausible_band": list(RATIO_PLAUSIBLE),
        "cheek_bone_ratio": {
            "n": len(vals), "min": round(vals[0], 4) if vals else None,
            "p25": round(vals[len(vals) // 4], 4) if vals else None,
            "median": round(vals[len(vals) // 2], 4) if vals else None,
            "p75": round(vals[3 * len(vals) // 4], 4) if vals else None,
            "max": round(vals[-1], 4) if vals else None,
            "p33": round(float(np.percentile(vals, 33.3)), 4) if vals else None,
            "p66": round(float(np.percentile(vals, 66.6)), 4) if vals else None,
        },
    }

    band_candidates = []
    cuts = [
        (round(float(np.percentile(vals, 33.3)), 3), round(float(np.percentile(vals, 66.6)), 3)),
        (1.15, 1.35),
        (SUBTLE_MAX, PROMINENT_MIN),  # current module constants (provisional)
        (1.30, 1.50),
    ]
    for ci, (cn, cx) in enumerate(cuts):
        wb = Counter()
        for v in vals:
            wb["subtle" if v < cn else ("prominent" if v > cx else "moderate")] += 1
        mx = max(wb.values()) / len(vals) if vals else 0
        band_candidates.append({
            "cand": f"cut-{ci}",
            "cuts": (cn, cx), "bands": dict(wb),
            "max_share": round(mx, 4),
            "degenerate": mx >= 0.75,
        })
    summary["band_candidates"] = band_candidates

    wb = Counter()
    for v in vals:
        wb["subtle" if v < SUBTLE_MAX else ("prominent" if v > PROMINENT_MIN else "moderate")] += 1
    n = len(vals)
    summary["final_band"] = {
        "subtle_max": SUBTLE_MAX, "prominent_min": PROMINENT_MIN, "bands": dict(wb),
        "max_share": round(max(wb.values()) / n, 4) if n else None,
    }

    Path(OUT).write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    print("\n=== FINAL BANDS (module constants) ===", flush=True)
    print(json.dumps(summary["final_band"], indent=2), flush=True)
    print("\n=== BAND CANDIDATES ===", flush=True)
    for c in band_candidates:
        tag = "OK" if not c["degenerate"] else "DEGENERATE"
        print(f"  {c['cand']}: cuts {c['cuts']} -> {c['bands']} "
              f"mx={c['max_share']} ({tag})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())