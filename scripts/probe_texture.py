"""CPU-only deterministic probe: run compute_texture on the frozen 24-item
cohort and report measurable coverage + band histograms (band-calibration
discipline: a band holding >=75% of items is not discriminating)."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from research_harness.texture import compute_texture  # noqa: E402

CANDIDATE = Path("/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json")
SOURCE_ROOT = Path("/mnt/nas-ai-models/training-data/crawlr/approved")
DERIVED_ROOT = Path("/mnt/nas-ai-models/training-data/crawlr/stratum")


def main() -> int:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    items = candidate["items"]
    print(f"candidate items: {len(items)}")

    n_subject = 0
    n_measurable = 0
    n_fabric = 0
    n_skin = 0
    fab_tex_hist: dict[str, int] = {}
    fab_pat_hist: dict[str, int] = {}
    skin_tex_hist: dict[str, int] = {}
    fab_edge: list[float] = []
    fab_dev: list[float] = []
    skin_edge: list[float] = []
    for item in items:
        image_id = item["image_id"]
        rel = item["source_relative_path"]
        seg = np.load(DERIVED_ROOT / image_id / "seg2.npy", allow_pickle=False)
        with Image.open(SOURCE_ROOT / rel) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        m = compute_texture(seg, rgb)
        if m["subject_present"]:
            n_subject += 1
        if m["texture_measurable"]:
            n_measurable += 1
        if m.get("fabric_class"):
            n_fabric += 1
            fab_edge.append(float(m["fabric_edge_fraction"]))
            fab_dev.append(float(m["fabric_deviant_fraction"]))
            fab_tex_hist[m["fabric_texture_band"]] = fab_tex_hist.get(m["fabric_texture_band"], 0) + 1
            fab_pat_hist[m["fabric_pattern_band"]] = fab_pat_hist.get(m["fabric_pattern_band"], 0) + 1
        if m.get("skin_class"):
            n_skin += 1
            skin_edge.append(float(m["skin_edge_fraction"]))
            skin_tex_hist[m["skin_texture_band"]] = skin_tex_hist.get(m["skin_texture_band"], 0) + 1
        flag = "OK" if m["texture_measurable"] else "ABSTAIN"
        print(
            f"{image_id}: {flag}  fabric={m.get('fabric_class')} "
            f"({m.get('fabric_texture_band')}/{m.get('fabric_pattern_band')}, "
            f"edge={m.get('fabric_edge_fraction')}, dev={m.get('fabric_deviant_fraction')})  "
            f"skin={m.get('skin_class')} ({m.get('skin_texture_band')}, edge={m.get('skin_edge_fraction')})"
        )

    print("\n--- aggregate ---")
    print(f"subject_present: {n_subject}/{len(items)}")
    print(f"texture_measurable: {n_measurable}/{len(items)}")
    print(f"fabric measurable: {n_fabric}/{len(items)}   skin measurable: {n_skin}/{len(items)}")
    if fab_edge:
        print(f"fabric edge_fraction: min {min(fab_edge):.4f} p50 {statistics.median(fab_edge):.4f} "
              f"p90 {sorted(fab_edge)[int(len(fab_edge) * 0.9) - 1]:.4f} max {max(fab_edge):.4f}")
        print(f"fabric deviant_fraction: min {min(fab_dev):.4f} p50 {statistics.median(fab_dev):.4f} "
              f"p90 {sorted(fab_dev)[int(len(fab_dev) * 0.9) - 1]:.4f} max {max(fab_dev):.4f}")
    if skin_edge:
        print(f"skin edge_fraction: min {min(skin_edge):.4f} p50 {statistics.median(skin_edge):.4f} "
              f"p90 {sorted(skin_edge)[int(len(skin_edge) * 0.9) - 1]:.4f} max {max(skin_edge):.4f}")
        print("skin edge values sorted:", [round(v, 4) for v in sorted(skin_edge)])
    if fab_edge:
        print("fabric edge values sorted:", [round(v, 4) for v in sorted(fab_edge)])
        print("fabric deviant values sorted:", [round(v, 4) for v in sorted(fab_dev)])
    print("fabric texture band histogram:", json.dumps(dict(sorted(fab_tex_hist.items()))))
    print("fabric pattern band histogram:", json.dumps(dict(sorted(fab_pat_hist.items()))))
    print("skin texture band histogram:", json.dumps(dict(sorted(skin_tex_hist.items()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())