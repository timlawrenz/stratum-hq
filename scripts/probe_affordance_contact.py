"""CPU probe: run the arm-#76 affordance-contact measurement over the frozen
24-item cohort BEFORE the plan is frozen. Reports per-item hand-contact count,
hand-elevation count, and grounding so the thresholds are CALIBRATED from the
real distribution (band-degeneracy rule arm #34/#35/#59/#74): if any band takes
>=75% of measured items it is not discriminating and must be re-probed.
Read-only, no GPU claim, no corpus write, no new model.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from research_harness.affordance_contact import (  # noqa: E402
    AffordanceContactError,
    compute_affordance_contact,
)

MANIFEST = "/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json"
DERIVED = "/mnt/nas-ai-models/training-data/crawlr/stratum"


def main() -> int:
    manifest = json.loads(Path(MANIFEST).read_text())
    items = manifest["items"]
    rows = []
    for item in items:
        image_id = item["image_id"]
        segp = Path(DERIVED) / image_id / "seg2.npy"
        posep = Path(DERIVED) / image_id / "pose2.npy"
        try:
            seg2 = np.load(segp, allow_pickle=False)
        except FileNotFoundError:
            rows.append({
                "image_id": image_id, "abstained": True,
                "abstention_reason": f"seg2 missing at {segp}",
            })
            print(f"{image_id[:12]}  ABSTAIN (seg2 missing)")
            continue
        try:
            pose2 = np.load(posep, allow_pickle=False)
        except FileNotFoundError:
            rows.append({
                "image_id": image_id, "abstained": True,
                "abstention_reason": f"pose2 missing at {posep}",
            })
            print(f"{image_id[:12]}  ABSTAIN (pose2 missing)")
            continue
        try:
            contact = compute_affordance_contact(pose2, seg2)
        except AffordanceContactError as exc:
            print(f"FAIL {image_id[:12]}: {exc}")
            return 2
        rows.append({
            "image_id": image_id,
            "hand_contact_count": contact.get("hand_contact_count"),
            "hand_elevation_count": contact.get("hand_elevation_count"),
            "grounded": contact.get("grounded"),
            "left_hand_visible": contact.get("left_hand_visible"),
            "right_hand_visible": contact.get("right_hand_visible"),
        })
        print(f"{image_id[:12]}  contact={contact.get('hand_contact_count')}  "
              f"elevation={contact.get('hand_elevation_count')}  "
              f"grounded={int(bool(contact.get('grounded')))}  "
              f"l_vis={int(bool(contact.get('left_hand_visible')))}  "
              f"r_vis={int(bool(contact.get('right_hand_visible')))}")

    det = [r for r in rows if not r.get("abstained")]
    n = len(det)
    print("\n=== CALIBRATION SUMMARY ===")
    print(f"measured: {n}/{len(rows)}")
    for ax in ("hand_contact_count", "hand_elevation_count", "grounded"):
        c = Counter(r.get(ax) for r in det)
        max_share = max(c.values()) / n if n else 0
        print(f"{ax}: {dict(c)}  max_share={max_share:.2f}")
    for r in rows:
        if r.get("abstained"):
            print(f"  ABSTAIN {r['image_id'][:12]}: {r.get('abstention_reason')}")
    Path("/mnt/nas-ai-models/research/stratum/affordance-contact-calibration-probe.json").write_text(
        json.dumps({"rows": rows, "summary": {
            "measured": n, "items": len(rows),
            "bands": {ax: dict(Counter(r.get(ax) for r in det)) for ax in
                      ("hand_contact_count", "hand_elevation_count", "grounded")},
        }}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
