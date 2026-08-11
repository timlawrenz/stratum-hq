"""CPU capability + band-calibration probe for eyebrow-position (arm #111).

NEW evidence part `eyebrow-position` (reuses the already-qualified MediaPipe
FaceLandmarker 478-point mesh — the SAME model as face-geometry #60 / gaze-head
#68; no new model). Band-calibration probe on the frozen 24-item cohort:
report the per-item arch_elevation and inner_brow_elevation distributions so
band floors are calibrated from the real cohort (band-degeneracy rule: a
single band >= 75% of measured items means the axis does not discriminate and
must be re-cut or kept payload-only).

Phase 1 (synthetic, non-sensitive): module must load + run + clean-abstain on a
blank non-face frame. Phase 2 (frozen cohort, read-only, local-first).

Read-only, CPU, no new model, no corpus write.
Output: /mnt/nas-ai-models/research/stratum/eyebrow-position-calibration-probe.json
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

from research_harness.eyebrow_position import (  # noqa: E402
    compute_eyebrow_position,
    render_eyebrow_position,
    set_band_floors,
)

MODEL = "/mnt/nas-ai-models/research/stratum/models/face-geometry/face_landmarker.task"
MANIFEST = "/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json"
SOURCE_ROOT = Path("/mnt/nas-ai-models/training-data/crawlr/approved")
DERIVED = Path("/mnt/nas-ai-models/training-data/crawlr/stratum")
OUT = "/mnt/nas-ai-models/research/stratum/eyebrow-position-calibration-probe.json"


def _synthetic(model_path: str) -> dict:
    """Non-sensitive smoke: blank non-face must abstain cleanly (no crash)."""
    W, H = 320, 320
    blank = np.zeros((H, W, 3), dtype=np.uint8) + 210
    seg2 = np.zeros((H, W), dtype=np.uint8)
    try:
        res = compute_eyebrow_position(seg2, blank, model_asset_path=model_path)
    except Exception as exc:  # noqa: BLE001
        return {"loads": True, "synthetic_run": False, "error": repr(exc)}
    return {
        "loads": True,
        "synthetic_run": True,
        "abstained_cleanly": bool(res.get("abstained")),
        "reason": res.get("abstention_reason"),
        "render": render_eyebrow_position(res),
    }


def _load_source_rgb(item: dict) -> np.ndarray:
    src = SOURCE_ROOT / item["source_relative_path"]
    return np.asarray(Image.open(src).convert("RGB"), dtype=np.uint8)


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
            r = compute_eyebrow_position(seg2, rgb, model_asset_path=MODEL)
        except Exception as exc:  # noqa: BLE001
            r = {"abstained": True, "abstention_reason": f"error: {exc!r}"}
        r["image_id"] = image_id
        rows.append(r)
        print(
            f"{image_id[:12]}  band?={str(r.get('eyebrow_position_band')):<8} "
            f"arch={r.get('arch_elevation')} inner={r.get('inner_brow_elevation')} "
            f"{'ABSTAIN: ' + str(r.get('abstention_reason')) if r.get('abstained') else ''}",
            flush=True,
        )

    measured = [r for r in rows if not r.get("abstained") and r.get("arch_elevation") is not None]
    arch = sorted(float(r["arch_elevation"]) for r in measured)
    inner = sorted(float(r["inner_brow_elevation"]) for r in measured)
    delta = sorted(float(r["inner_brow_elevation"]) - float(r["arch_elevation"]) for r in measured)

    # Candidate band floors from the measured distribution (report honestly,
    # and print the degeneracy check for several candidate cuts).
    summary = {
        "items": len(rows),
        "n_measured": len(measured),
        "n_abstained": sum(1 for r in rows if r.get("abstained")),
        "arch_elevation": {
            "n": len(arch), "min": round(arch[0], 4) if arch else None,
            "p25": round(arch[len(arch) // 4], 4) if arch else None,
            "median": round(arch[len(arch) // 2], 4) if arch else None,
            "p75": round(arch[3 * len(arch) // 4], 4) if arch else None,
            "max": round(arch[-1], 4) if arch else None,
        },
        "inner_brow_elevation": {
            "n": len(inner), "min": round(inner[0], 4) if inner else None,
            "p25": round(inner[len(inner) // 4], 4) if inner else None,
            "median": round(inner[len(inner) // 2], 4) if inner else None,
            "p75": round(inner[3 * len(inner) // 4], 4) if inner else None,
            "max": round(inner[-1], 4) if inner else None,
        },
        "inner_minus_arch": {
            "n": len(delta), "min": round(delta[0], 4) if delta else None,
            "median": round(delta[len(delta) // 2], 4) if delta else None,
            "max": round(delta[-1], 4) if delta else None,
        },
    }

    # Candidate cuts to inspect: raised when arch is above a high percentile,
    # furrowed when inner drops notably below arch. Print the degeneracy
    # consequence of each so the calibration is transparent.
    band_candidates = []
    measured_arr = [(float(r["arch_elevation"]), float(r["inner_brow_elevation"])) for r in measured]
    for raised_low in (round(np.percentile(arch, 60), 3), round(np.percentile(arch, 55), 3)):
        for furrow in (-0.03, -0.02, -0.01, 0.0):
            bands = Counter()
            for a, i in measured_arr:
                if a >= raised_low:
                    bands["raised"] += 1
                elif i - a <= furrow:
                    bands["furrowed"] += 1
                else:
                    bands["neutral"] += 1
            n = sum(bands.values())
            max_share = max(bands.values()) / n if n else 0
            band_candidates.append({
                "raised_low": raised_low, "furrow_delta": furrow,
                "bands": dict(bands), "max_share": round(max_share, 4),
                "degenerate": max_share >= 0.75,
            })
    summary["band_candidates"] = band_candidates

    # Final authoritative 3-way band from the module's calibrated constants,
    # plus the two boundary alternatives for transparency.
    from research_harness.eyebrow_position import FURROWED_HIGH as _FH, RAISED_LOW as _RL
    arch_only = Counter()
    for a, _i in measured_arr:
        if a >= _RL:
            arch_only["raised"] += 1
        elif a < _FH:
            arch_only["furrowed"] += 1
        else:
            arch_only["neutral"] += 1
    n = sum(arch_only.values())
    summary["final_band"] = {
        "raised_low": _RL, "furrowed_high": _FH,
        "bands": dict(arch_only),
        "max_share": round(max(arch_only.values()) / n, 4) if n else None,
    }

    Path(OUT).write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    print("\n=== FINAL BAND (module constants) ===", flush=True)
    print(json.dumps(summary["final_band"], indent=2), flush=True)
    print("\n=== BAND CANDIDATES ===", flush=True)
    for c in band_candidates:
        tag = "OK" if not c["degenerate"] else "DEGENERATE"
        print(f"  raised_low={c['raised_low']} furrow={c['furrow_delta']} "
              f"-> {c['bands']} max_share={c['max_share']} ({tag})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
