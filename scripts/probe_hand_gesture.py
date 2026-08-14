"""CPU probe: MediaPipe Hands hand-gesture over the frozen 24-item cohort.

Arm #109 (NEW model class MediaPipe Hands + NEW evidence part hand-gesture,
registered 2026-08-10 via the gated propose-dimensions channel, selected via
EXPLOIT at selection_progress 33). Capability + band-calibration probe:

- coverage floor (pre-registered on issue #109): >= 14/24 items with at
  least one visible hand;
- band-degeneracy rule (arm #34/#35/#59): no gesture band >= 75% of
  measured hands; the one-vs-two-hands split must also be non-degenerate
  before it is verbalized;
- hand-raised flags gated on the pose2 GOLIATH-308 wrist/shoulder reference
  (reported per item, honest abstention when keypoints are unreliable).

Read-only, CPU, local owned hardware, no corpus write.

Output: /mnt/nas-ai-models/research/stratum/hand-gesture-calibration-probe.json
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

from research_harness.hand_gesture import (  # noqa: E402
    MODEL_SHA256,
    compute_hand_gesture,
)

MANIFEST = "/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json"
SOURCE = "/mnt/nas-ai-models/training-data/crawlr/approved"
DERIVED = "/mnt/nas-ai-models/training-data/crawlr/stratum"
MODEL = "/mnt/nas-ai-models/research/stratum/models/hand-gesture/hand_landmarker.task"
OUT = "/mnt/nas-ai-models/research/stratum/hand-gesture-calibration-probe.json"

import hashlib  # noqa: E402


def main() -> int:
    model_sha = hashlib.sha256(Path(MODEL).read_bytes()).hexdigest()
    print(f"model sha256 {model_sha} (declared {MODEL_SHA256}) match={model_sha == MODEL_SHA256}")
    manifest = json.loads(Path(MANIFEST).read_text())
    rows = []
    for item in manifest["items"]:
        image_id = item["image_id"]
        rel = item["source_relative_path"]
        src = Path(SOURCE) / rel
        rgb = np.ascontiguousarray(np.asarray(Image.open(src).convert("RGB"), dtype=np.uint8))
        pose2 = np.load(Path(DERIVED) / image_id / "pose2.npy")
        try:
            r = compute_hand_gesture(rgb, pose2, model_asset_path=MODEL)
        except Exception as exc:  # noqa: BLE001
            r = {"abstained": True, "abstention_reason": f"error: {exc!r}"}
        r["image_id"] = image_id
        rows.append(r)
        hands = r.get("per_hand") or []
        gest = [f"{h.get('side', '?')}:{h['gesture_class']}" for h in hands]
        raised = [f"{h.get('side', '?')}:{h['hand_raised'].get('hand_raised')}"
                  for h in hands if h.get("hand_raised", {}).get("measured")]
        print(f"{image_id[:12]}  n={r.get('hands_detected')}  via={r.get('via')}  "
              f"claim={r.get('gesture_claim')!r}  raised={raised}  "
              f"{'ABSTAIN: ' + str(r.get('abstention_reason')) if r.get('abstained') else ''}")

    n_items = len(rows)
    n_with_hand = sum(1 for r in rows if not r.get("abstained") and r.get("hands_detected", 0) >= 1)
    n_banded_items = sum(1 for r in rows if r.get("gesture_band"))
    gestures = Counter()
    for r in rows:
        for h in r.get("per_hand") or []:
            gestures[h["gesture_class"]] += 1
    n_hands = sum(gestures.values())
    max_gesture_share = max(gestures.values()) / n_hands if n_hands else 0
    count_bands = Counter(r.get("hand_count_band") for r in rows if not r.get("abstained"))
    n_count_banded = sum(count_bands.values())
    max_count_share = max(count_bands.values()) / n_count_banded if n_count_banded else 0
    raised_measured = [f.get("hand_raised")
                       for r in rows for f in (r.get("raised_flags") or []) if f.get("measured")]
    raised_true = sum(1 for v in raised_measured if v)

    summary = {
        "items": n_items,
        "coverage_floor": 14,
        "n_items_with_hand": n_with_hand,
        "coverage_pass": n_with_hand >= 14,
        "n_hands_measured": n_hands,
        "gesture_counts": dict(gestures),
        "max_gesture_share": round(max_gesture_share, 4),
        "band_degenerate": max_gesture_share >= 0.75,
        "hand_count_bands": dict(count_bands),
        "max_count_share": round(max_count_share, 4),
        "count_band_degenerate": max_count_share >= 0.75,
        "raised_measured": len(raised_measured),
        "raised_true": raised_true,
        "model_sha256": model_sha,
        "declared_model_sha256": MODEL_SHA256,
    }
    Path(OUT).write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    print("\n=== PROBE RESULT ===")
    print(json.dumps(summary, indent=2))
    if not summary["coverage_pass"]:
        print(f"COVERAGE FAIL: {n_with_hand}/24 items with a visible hand < 14/24 floor")
    if summary["band_degenerate"]:
        print("BAND DEGENERACY: a gesture band >=75% — axis does not discriminate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())