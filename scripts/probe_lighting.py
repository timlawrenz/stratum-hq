"""CPU-only deterministic probe: run compute_lighting on the frozen 24-item
cohort and report measurable coverage (mirrors the skin-color pre-run probe)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from research_harness.lighting import compute_lighting  # noqa: E402

CANDIDATE = Path("/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json")
SOURCE_ROOT = Path("/mnt/nas-ai-models/training-data/crawlr/approved")
DERIVED_ROOT = Path("/mnt/nas-ai-models/training-data/crawlr/stratum")


def main() -> int:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    items = candidate["items"]
    print(f"candidate items: {len(items)}")

    n_subject = 0
    n_measurable = 0
    n_direction = 0
    luma_hist: dict[str, int] = {}
    dr_hist: dict[str, int] = {}
    shadow_hist: dict[str, int] = {}
    dir_hist: dict[str, int] = {}
    for item in items:
        image_id = item["image_id"]
        rel = item["source_relative_path"]
        normal2 = np.load(DERIVED_ROOT / image_id / "normal2.npy", allow_pickle=False)
        seg = np.load(DERIVED_ROOT / image_id / "seg2.npy", allow_pickle=False)
        with Image.open(SOURCE_ROOT / rel) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        m = compute_lighting(normal2, seg, rgb)
        if m["subject_present"]:
            n_subject += 1
        if m["lighting_measurable"]:
            n_measurable += 1
            luma_hist[m["luma_band"]] = luma_hist.get(m["luma_band"], 0) + 1
            dr_hist[m["dynamic_range_band"]] = dr_hist.get(m["dynamic_range_band"], 0) + 1
            shadow_hist[m["shadow_band"]] = shadow_hist.get(m["shadow_band"], 0) + 1
        if m["light_direction"] and m["light_direction"] != "undetermined":
            n_direction += 1
            dir_hist[m["light_direction"]] = dir_hist.get(m["light_direction"], 0) + 1
        flag = "OK" if m["lighting_measurable"] else "ABSTAIN"
        print(
            f"{image_id}: {flag}  luma={m['luma_band']}  dr={m['dynamic_range_band']}  "
            f"shadow={m['shadow_band']}  dir={m['light_direction']}"
        )

    print("\n--- aggregate ---")
    print(f"subject_present: {n_subject}/{len(items)}")
    print(f"lighting_measurable: {n_measurable}/{len(items)}")
    print(f"light direction resolved: {n_direction}/{len(items)}")
    print("luma band histogram:", json.dumps(dict(sorted(luma_hist.items()))))
    print("dynamic-range band histogram:", json.dumps(dict(sorted(dr_hist.items()))))
    print("shadow band histogram:", json.dumps(dict(sorted(shadow_hist.items()))))
    print("direction histogram:", json.dumps(dict(sorted(dir_hist.items()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
