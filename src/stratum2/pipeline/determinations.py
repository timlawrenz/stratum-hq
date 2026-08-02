"""Extract geometry determinations from pose2 and seg2.

Produces ``determinations.json``.
"""

import json
import math
from pathlib import Path
import numpy as np

from stratum2.config import GOLIATH_308, DOME_29


def eprint(*args, **kwargs):
    import sys

    print(*args, file=sys.stderr, **kwargs)


def get_body_parts_visible(seg2: np.ndarray, pose2_person: np.ndarray | None):
    # Coarse mapping
    # 3: Face_Neck, 22: Torso, 6/7/11/31/32/33: Arms, 10/11/14... Legs

    # We will compute based on DOME_29 for pixel frac
    parts = []
    total_pixels = seg2.shape[0] * seg2.shape[1]
    if total_pixels == 0:
        return parts

    # Face
    face_px = (seg2 == 3).sum()
    if face_px > 0:
        parts.append(
            {
                "part": "face",
                "pixel_frac": float(face_px / total_pixels),
                "kp_conf": float(np.mean(pose2_person[21:263, 2]))
                if pose2_person is not None
                else 0.0,
            }
        )

    # Torso
    torso_px = (seg2 == 22).sum()
    if torso_px > 0:
        parts.append(
            {
                "part": "torso",
                "pixel_frac": float(torso_px / total_pixels),
                "kp_conf": float(pose2_person[[5, 6, 9, 10], 2].mean())
                if pose2_person is not None
                else 0.0,
            }
        )

    return parts


def process(image_path: Path, output_dir: Path, **kwargs) -> bool:
    out_path = output_dir / "determinations.json"
    if out_path.exists():
        return True

    pose2_path = output_dir / "pose2.npy"
    seg2_path = output_dir / "seg2.npy"

    if not pose2_path.exists() or not seg2_path.exists():
        eprint(
            f"warning: determinations skipped for {image_path}: pose2/seg2 not found"
        )
        return False

    pose2 = np.load(pose2_path)
    seg2 = np.load(seg2_path)

    # 1. Subject N and Anomaly
    n = pose2.shape[0]
    anomaly = "none"
    if n == 0:
        anomaly = "no_detection"
    elif n > 1:
        anomaly = f"extra_detections({n})"

    doc = {
        "schema_version": 2,
        "subject": {
            "n_detections": n,
            "detector_anomaly": anomaly,
            "note": "exactly one real subject guaranteed by curation; N!=1 is a quality flag, not content",
        },
        "subject_extent": {},
        "body_parts_visible": [],
        "orientation": {},
        "relations": [],
    }

    if n > 0:
        p = pose2[0]  # The primary subject

        # extent
        vis = p[p[:, 2] > 0.3]
        if len(vis) > 0:
            doc["subject_extent"] = {
                "bbox_px": [
                    float(vis[:, 0].min()),
                    float(vis[:, 1].min()),
                    float(vis[:, 0].max()),
                    float(vis[:, 1].max()),
                ]
            }

        # upright_deg
        neck_idx = GOLIATH_308.index("neck")
        lhip_idx = GOLIATH_308.index("left_hip")
        rhip_idx = GOLIATH_308.index("right_hip")

        neck = p[neck_idx]
        hip_y = (p[lhip_idx, 1] + p[rhip_idx, 1]) / 2.0
        hip_x = (p[lhip_idx, 0] + p[rhip_idx, 0]) / 2.0

        dy = hip_y - neck[1]
        dx = hip_x - neck[0]

        # 0 = upright (hip directly below neck, +y is down)
        # atan2(y, x) -> atan2(dy, dx) = pi/2 for straight down
        angle_rad = math.atan2(dy, dx)
        deg = math.degrees(angle_rad)

        # Map [90] (down) -> 0 upright
        upright = abs(deg - 90.0)
        doc["orientation"]["upright_deg"] = round(upright, 1)

        # body parts
        doc["body_parts_visible"] = get_body_parts_visible(seg2, p)

        # relations
        rels = []

        # Arm relation: wrist above shoulder?
        lwri_idx = GOLIATH_308.index("left_wrist")
        lsho_idx = GOLIATH_308.index("left_shoulder")
        if p[lwri_idx, 2] > 0.3 and p[lsho_idx, 2] > 0.3:
            if p[lwri_idx, 1] < p[lsho_idx, 1]:  # wrist y < shoulder y
                rels.append("left arm extended upward")

        # If face is the ONLY thing visible, wipe relations
        parts = [bp["part"] for bp in doc["body_parts_visible"]]
        if parts == ["face"]:
            rels = []

        doc["relations"] = rels

    out_path.write_text(json.dumps(doc, indent=2))
    return True
