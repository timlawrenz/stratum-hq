"""CPU-only deterministic probe: run compute_pose_articulation on the frozen
24-item cohort and report measurable coverage + band histograms (mirrors the
setting/texture pre-run probes). Calibration evidence for arm #62."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np  # noqa: E402

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

from research_harness.pose_articulation import (  # noqa: E402
    compute_pose_articulation,
)

CANDIDATE = Path("/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json")
DERIVED_ROOT = Path("/mnt/nas-ai-models/training-data/crawlr/stratum")


def main() -> int:
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    items = candidate["items"]
    print(f"candidate items: {len(items)}")

    n_subject = 0
    n_elbow_l = n_elbow_r = 0
    n_knee_l = n_knee_r = 0
    n_stance = 0
    n_twist = 0
    n_cross = 0
    n_legs_cross = 0
    contrap_posturas = 0
    stance_hist: dict[str, int] = {}
    arm_near_hist = {"both": 0, "left": 0, "right": 0, "none": 0}
    for item in items:
        image_id = item["image_id"]
        pose = np.load(DERIVED_ROOT / image_id / "pose2.npy", allow_pickle=False)
        seg = np.load(DERIVED_ROOT / image_id / "seg2.npy", allow_pickle=False)
        m = compute_pose_articulation(pose, seg)

        if m["subject_present"]:
            n_subject += 1
        efl, efr = m["elbow_flexion_left"], m["elbow_flexion_right"]
        kfl, kfr = m["knee_flexion_left"], m["knee_flexion_right"]
        n_elbow_l += efl is not None
        n_elbow_r += efr is not None
        n_knee_l += kfl is not None
        n_knee_r += kfr is not None
        if m["stance_class"]:
            n_stance += 1
            stance_hist[m["stance_class"]] = stance_hist.get(m["stance_class"], 0) + 1
        if m["torso_twist_deg"] is not None:
            n_twist += 1
        if m["arm_crossing_count"] is not None:
            n_cross += int(m["arm_crossing_count"] > 0)
        if m["legs_crossed"]:
            n_legs_cross += 1
        if m["contrapposto"]:
            contrap_posturas += 1

        ln, rn = m["left_arm_near_torso_fraction"], m["right_arm_near_torso_fraction"]
        if ln is None and rn is None:
            arm_near_hist["none"] += 1
        elif ln is not None and rn is not None and ln > 0.5 and rn > 0.5:
            arm_near_hist["both"] += 1
        elif ln is not None and ln > 0.5:
            arm_near_hist["left"] += 1
        elif rn is not None and rn > 0.5:
            arm_near_hist["right"] += 1
        else:
            arm_near_hist["none"] += 1

        print(
            f"{image_id}: subject={m['subject_present']} stance={m['stance_class']} "
            f"twist={m['torso_twist_deg']} lean={m['torso_lean_deg']} "
            f"pelv={m['pelvis_tilt_deg']} efl={efl} efr={efr} kfl={kfl} kfr={kfr} "
            f"cross={m['arm_crossing_count']} legsx={m['legs_crossed']} "
            f"nearL={ln} nearR={rn} contrapposto={m['contrapposto']}"
        )

    print("\n--- aggregate ---")
    print(f"subject_present: {n_subject}/{len(items)}")
    print(f"elbow left/right measurable: {n_elbow_l}/{n_elbow_r}/24")
    print(f"knee left/right measurable: {n_knee_l}/{n_knee_r}/24")
    print(f"stance_class resolved: {n_stance}/24  hist={json.dumps(dict(sorted(stance_hist.items())))}")
    print(f"torso twist measurable: {n_twist}/24")
    print(f"arm_crossing_count>0: {n_cross}/24 ; legs_crossed: {n_legs_cross}/24")
    print(f"contrapposto true: {contrap_posturas}/24")
    print(f"arm-near-torso (>0.5 both/side): {json.dumps(arm_near_hist)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
