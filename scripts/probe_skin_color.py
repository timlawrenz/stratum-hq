"""CPU-only deterministic probe: run compute_skin_tone on the frozen 24-item
cohort and report measurable coverage (mirrors the hair arm's pre-run probe)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from research_harness.skin_color import compute_skin_tone  # noqa: E402

CANDIDATE = Path("/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json")
SOURCE_ROOT = Path("/mnt/nas-ai-models/training-data/crawlr/approved")
DERIVED_ROOT = Path("/mnt/nas-ai-models/training-data/crawlr/stratum")

from PIL import Image  # noqa: E402


def main() -> int:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    items = candidate["items"]
    print(f"candidate items: {len(items)}")

    tones: dict[str, int] = {}
    n_subject = 0
    n_exposed = 0
    n_tone = 0
    n_both_regions = 0
    agree = 0
    for item in items:
        image_id = item["image_id"]
        rel = item["source_relative_path"]
        seg = np.load(DERIVED_ROOT / image_id / "seg2.npy")
        src = SOURCE_ROOT / rel
        with Image.open(src) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        m = compute_skin_tone(seg, rgb)
        if m["subject_present"]:
            n_subject += 1
        if m["exposed_skin_present"]:
            n_exposed += 1
            n_tone += 1
            tones[m["skin_tone_name"]] = tones.get(m["skin_tone_name"], 0) + 1
        if m["face_tone_name"] is not None and m["body_tone_name"] is not None:
            n_both_regions += 1
            if m["face_body_agree"]:
                agree += 1
        flag = "OK" if m["exposed_skin_present"] else "ABSTAIN"
        print(f"{image_id}: {flag}  tone={m['skin_tone_name']}  coverage={m['skin_coverage']}  face={m['face_tone_name']} body={m['body_tone_name']}")

    print("\n--- aggregate ---")
    print(f"subject_present: {n_subject}/{len(items)}")
    print(f"exposed_skin_present (tone measurable): {n_exposed}/{len(items)}")
    print(f"face+body both measurable: {n_both_regions}/{len(items)} (agree {agree})")
    print("tone histogram:", json.dumps(dict(sorted(tones.items()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
