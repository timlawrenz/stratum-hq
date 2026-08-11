"""CPU-only deterministic probe: run compute_setting on the frozen 24-item
cohort and report measurable coverage (mirrors the lighting pre-run probe)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from research_harness.setting import compute_setting  # noqa: E402

CANDIDATE = Path("/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json")
SOURCE_ROOT = Path("/mnt/nas-ai-models/training-data/crawlr/approved")
DERIVED_ROOT = Path("/mnt/nas-ai-models/training-data/crawlr/stratum")


def main() -> int:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    items = candidate["items"]
    print(f"candidate items: {len(items)}")

    n_subject = 0
    n_measurable = 0
    color_hist: dict[str, int] = {}
    tone_hist: dict[str, int] = {}
    vibrancy_hist: dict[str, int] = {}
    pattern_hist: dict[str, int] = {}
    coverage_min = 1.0
    coverage_max = 0.0
    for item in items:
        image_id = item["image_id"]
        rel = item["source_relative_path"]
        seg = np.load(DERIVED_ROOT / image_id / "seg2.npy", allow_pickle=False)
        with Image.open(SOURCE_ROOT / rel) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        m = compute_setting(seg, rgb)
        if m["subject_present"]:
            n_subject += 1
        if m["setting_measurable"]:
            n_measurable += 1
            cov = float(m["background_coverage"])
            coverage_min = min(coverage_min, cov)
            coverage_max = max(coverage_max, cov)
            color_hist[m["dominant_background_color"]] = color_hist.get(m["dominant_background_color"], 0) + 1
            tone_hist[m["background_tone_band"]] = tone_hist.get(m["background_tone_band"], 0) + 1
            vibrancy_hist[m["background_vibrancy_band"]] = vibrancy_hist.get(m["background_vibrancy_band"], 0) + 1
            pattern_hist[m["background_pattern_band"]] = pattern_hist.get(m["background_pattern_band"], 0) + 1
        flag = "OK" if m["setting_measurable"] else "ABSTAIN"
        print(
            f"{image_id}: {flag}  cov={m['background_coverage']}  "
            f"color={m['dominant_background_color']}  tone={m['background_tone_band']}  "
            f"vibrancy={m['background_vibrancy_band']}  pattern={m['background_pattern_band']}"
        )

    print("\n--- aggregate ---")
    print(f"subject_present: {n_subject}/{len(items)}")
    print(f"setting_measurable: {n_measurable}/{len(items)}")
    print(f"background coverage range: {coverage_min:.3f}..{coverage_max:.3f}")
    print("dominant color histogram:", json.dumps(dict(sorted(color_hist.items()))))
    print("tone band histogram:", json.dumps(dict(sorted(tone_hist.items()))))
    print("vibrancy band histogram:", json.dumps(dict(sorted(vibrancy_hist.items()))))
    print("pattern band histogram:", json.dumps(dict(sorted(pattern_hist.items()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())