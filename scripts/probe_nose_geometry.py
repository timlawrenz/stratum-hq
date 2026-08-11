"""CPU capability + band-calibration probe for nose-geometry (arm #121).

NEW evidence part `nose-geometry` (reuses the already-qualified MediaPipe
FaceLandmarker 478-point mesh — the same model as face-geometry #60, gaze-head
#68, eyebrow-position #111; no new model). Band-calibration probe on the frozen
24-item cohort: report the per-item nose_width_ratio and nose_length_ratio
distributions so band floors are calibrated from the real cohort
(band-degeneracy rule: a single band >= 75% of measured items means the axis
does not discriminate and must be re-cut or kept payload-only).

Phase 1 (synthetic, non-sensitive): module must load + run + clean-abstain on a
blank non-face frame. Phase 2 (frozen cohort, read-only, local-first).

Read-only, CPU, no new model, no corpus write.
Output: /mnt/nas-ai-models/research/stratum/nose-geometry-calibration-probe.json
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

from research_harness.nose_geometry import (  # noqa: E402
    LENGTH_LONG_MIN,
    LENGTH_SHORT_MAX,
    WIDTH_NARROW_MAX,
    WIDTH_WIDE_MIN,
    compute_nose_geometry,
    render_nose_geometry,
    set_band_floors,
)

MODEL = "/mnt/nas-ai-models/research/stratum/models/face-geometry/face_landmarker.task"
MANIFEST = "/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json"
SOURCE_ROOT = Path("/mnt/nas-ai-models/training-data/crawlr/approved")
DERIVED = Path("/mnt/nas-ai-models/training-data/crawlr/stratum")
OUT = "/mnt/nas-ai-models/research/stratum/nose-geometry-calibration-probe.json"


def _synthetic(model_path: str) -> dict:
    """Non-sensitive smoke: blank non-face must abstain cleanly (no crash)."""
    W, H = 320, 320
    blank = np.zeros((H, W, 3), dtype=np.uint8) + 210
    seg2 = np.zeros((H, W), dtype=np.uint8)
    try:
        res = compute_nose_geometry(seg2, blank, model_asset_path=model_path)
    except Exception as exc:  # noqa: BLE001
        return {"loads": True, "synthetic_run": False, "error": repr(exc)}
    return {
        "loads": True,
        "synthetic_run": True,
        "abstained_cleanly": bool(res.get("abstained")),
        "reason": res.get("abstention_reason"),
        "render": render_nose_geometry(res),
    }


def _load_source_rgb(item: dict) -> np.ndarray:
    """Decode the source RGB read-only. Prefers the canonical approved/ export;
    falls back to the raw/ sharded byte tree (identical bytes, SHA-verified by
    the caller) when the approved entry was purged — READ-ONLY, no mutation."""
    from pathlib import Path as _P

    src = SOURCE_ROOT / item["source_relative_path"]
    if src.exists():
        return np.asarray(Image.open(src).convert("RGB"), dtype=np.uint8)
    # Fallback: raw/{k[0:2]}/{k[2:4]}/{k} (the approved export is built from
    # these bytes; the file content is identical).
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
            r = compute_nose_geometry(seg2, rgb, model_asset_path=MODEL)
        except Exception as exc:  # noqa: BLE001
            r = {"abstained": True, "abstention_reason": f"error: {exc!r}"}
        r["image_id"] = image_id
        rows.append(r)
        print(
            f"{image_id[:12]}  w_band?={str(r.get('nose_width_band')):<8} "
            f"w_ratio={r.get('nose_width_ratio')} l_ratio={r.get('nose_length_ratio')} "
            f"{'ABSTAIN: ' + str(r.get('abstention_reason')) if r.get('abstained') else ''}",
            flush=True,
        )

    measured = [r for r in rows if not r.get("abstained") and r.get("nose_width_ratio") is not None]
    wr = sorted(float(r["nose_width_ratio"]) for r in measured)
    lr = sorted(float(r["nose_length_ratio"]) for r in measured)

    summary = {
        "items": len(rows),
        "n_measured": len(measured),
        "n_abstained": sum(1 for r in rows if r.get("abstained")),
        "nose_width_ratio": {
            "n": len(wr), "min": round(wr[0], 4) if wr else None,
            "p25": round(wr[len(wr) // 4], 4) if wr else None,
            "median": round(wr[len(wr) // 2], 4) if wr else None,
            "p75": round(wr[3 * len(wr) // 4], 4) if wr else None,
            "max": round(wr[-1], 4) if wr else None,
            "p33": round(float(np.percentile(wr, 33.3)), 4) if wr else None,
            "p66": round(float(np.percentile(wr, 66.6)), 4) if wr else None,
        },
        "nose_length_ratio": {
            "n": len(lr), "min": round(lr[0], 4) if lr else None,
            "p25": round(lr[len(lr) // 4], 4) if lr else None,
            "median": round(lr[len(lr) // 2], 4) if lr else None,
            "p75": round(lr[3 * len(lr) // 4], 4) if lr else None,
            "max": round(lr[-1], 4) if lr else None,
            "p33": round(float(np.percentile(lr, 33.3)), 4) if lr else None,
            "p66": round(float(np.percentile(lr, 66.6)), 4) if lr else None,
        },
    }

    # Candidate 3-way cuts: report the degeneracy consequence of several cuts
    # so the calibration is transparent (band-degeneracy rule enforced).
    band_candidates = []
    width_vals = [float(r["nose_width_ratio"]) for r in measured]
    length_vals = [float(r["nose_length_ratio"]) for r in measured]
    for wi, (wn, ww) in enumerate([
        (round(np.percentile(width_vals, 33.3), 3), round(np.percentile(width_vals, 66.6), 3)),
        (0.90, 1.10),
        (0.88, 1.12),
    ]):
        for li, (ls, ll) in enumerate([
            (round(np.percentile(length_vals, 33.3), 3), round(np.percentile(length_vals, 66.6), 3)),
            (1.10, 1.30),
            (1.05, 1.35),
        ]):
            wb = Counter()
            lb = Counter()
            for wvv, lvv in zip(width_vals, length_vals):
                if wvv < wn:
                    wb["narrow"] += 1
                elif wvv > ww:
                    wb["wide"] += 1
                else:
                    wb["average"] += 1
                if lvv < ls:
                    lb["short"] += 1
                elif lvv > ll:
                    lb["long"] += 1
                else:
                    lb["average"] += 1
            wmx = max(wb.values()) / len(width_vals) if width_vals else 0
            lmx = max(lb.values()) / len(length_vals) if length_vals else 0
            band_candidates.append({
                "cand": f"w{wi}-l{li}",
                "width_cuts": (wn, ww), "width_bands": dict(wb),
                "width_max_share": round(wmx, 4),
                "length_cuts": (ls, ll), "length_bands": dict(lb),
                "length_max_share": round(lmx, 4),
                "degenerate": wmx >= 0.75 or lmx >= 0.75,
            })
    summary["band_candidates"] = band_candidates

    # Final authoritative bands from the module's calibrated constants.
    from research_harness.nose_geometry import (
        LENGTH_LONG_MIN as _LL, LENGTH_SHORT_MAX as _LS,
        WIDTH_NARROW_MAX as _WN, WIDTH_WIDE_MIN as _WW,
    )
    wb = Counter()
    lb = Counter()
    for wvv, lvv in zip(width_vals, length_vals):
        wb["narrow" if wvv < _WN else ("wide" if wvv > _WW else "average")] += 1
        lb["short" if lvv < _LS else ("long" if lvv > _LL else "average")] += 1
    n = len(width_vals)
    summary["final_band_width"] = {
        "narrow_max": _WN, "wide_min": _WW, "bands": dict(wb),
        "max_share": round(max(wb.values()) / n, 4) if n else None,
    }
    summary["final_band_length"] = {
        "short_max": _LS, "long_min": _LL, "bands": dict(lb),
        "max_share": round(max(lb.values()) / n, 4) if n else None,
    }

    Path(OUT).write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    print("\n=== FINAL BANDS (module constants) ===", flush=True)
    print(json.dumps({"width": summary["final_band_width"], "length": summary["final_band_length"]}, indent=2), flush=True)
    print("\n=== BAND CANDIDATES ===", flush=True)
    for c in band_candidates:
        tag = "OK" if not c["degenerate"] else "DEGENERATE"
        print(f"  {c['cand']}: w cuts {c['width_cuts']} -> {c['width_bands']} "
              f"mx={c['width_max_share']}; l cuts {c['length_cuts']} -> {c['length_bands']} "
              f"mx={c['length_max_share']} ({tag})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
