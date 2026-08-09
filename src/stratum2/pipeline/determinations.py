"""Extract geometry determinations from pose2 and seg2.

Produces ``determinations.json``.

Design rule (user-directed): per-region seg<->pose corroboration, no
cross-part inference. A region's keypoints are trusted for relations anchored
on that region iff seg2 shows that region's pixels above a minimal fraction.
pose2 is read-only inside this pass — no confidence mutation, no ordering
sensitivity.
"""

import json
import math
from pathlib import Path

import numpy as np

from stratum2.config import GOLIATH_308

# Minimal fraction for a region to count as corroborated, measured against the
# subject's OWN foreground pixels (seg2 > 0), not the whole frame. Frame-
# normalized fractions penalize environmental full-body shots where the subject
# is a small part of the scene (a slender arm can be clearly visible yet <1% of
# a 1.7Mpx outdoor frame).
MIN_REGION_FRAC = 0.01

# Absolute pixel floor: a region must also clear this raw count so a tiny seg2
# smear can't pass a 1%-of-tiny-foreground gate on near-empty crops.
MIN_REGION_PX = 200

# Region -> seg2 classes (DOME_29 indices). Clothing classes count toward the
# underlying body region (a shirt means a torso is present).
REGION_SEG_CLASSES = {
    "face": [3, 4, 24, 25, 26, 27, 28],  # Face_Neck, Hair, lips/teeth/tongue
    "torso": [22, 23, 13],  # Torso, Upper_Clothing, Lower_Clothing
    "left_arm": [11, 7],  # Left_Upper_Arm, Left_Lower_Arm
    "right_arm": [20, 16],  # Right_Upper_Arm, Right_Lower_Arm
    "left_hand": [6],  # Left_Hand
    "right_hand": [15],  # Right_Hand
    "left_leg": [12, 8, 9, 10],  # Left_Upper_Leg, Left_Lower_Leg, Left_Shoe, Left_Sock
    "right_leg": [
        21,
        17,
        18,
        19,
    ],  # Right_Upper_Leg, Right_Lower_Leg, Right_Shoe, Right_Sock
}


# Region -> pose2 keypoint indices used for that region's confidence.
def _kp_indices_for_region(region: str) -> list[int]:
    if region == "face":
        return list(range(70, 308))
    if region == "torso":
        return [5, 6, 9, 10]  # shoulders, hips
    if region == "left_arm":
        return [5, 7, 62]  # l_shoulder, l_elbow, l_wrist
    if region == "right_arm":
        return [6, 8, 41]  # r_shoulder, r_elbow, r_wrist
    if region == "left_hand":
        return [62] + list(range(42, 62))  # l_wrist + left fingers
    if region == "right_hand":
        return [41] + list(range(21, 41))  # r_wrist + right fingers
    if region == "left_leg":
        return [9, 11, 13]  # l_hip, l_knee, l_ankle
    if region == "right_leg":
        return [10, 12, 14]  # r_hip, r_knee, r_ankle
    return []


def eprint(*args, **kwargs):
    import sys

    print(*args, file=sys.stderr, **kwargs)


def get_body_parts_visible(seg2: np.ndarray, pose2_person: np.ndarray | None):
    """Report all regions with pixel fractions + kp confidence.

    ``pixel_frac`` is normalized against the subject's own foreground pixels
    (seg2 > 0) — this is what the corroboration gate reads. ``frame_frac`` is
    the frame-normalized fraction, kept for reference only.
    """
    parts = []
    total_pixels = seg2.shape[0] * seg2.shape[1]
    if total_pixels == 0:
        return parts

    fg_pixels = int((seg2 > 0).sum())
    # Guard against degenerate near-empty seg: fall back to frame denominator so
    # fractions stay sane instead of exploding on a 2-pixel foreground.
    denom = fg_pixels if fg_pixels > 0 else total_pixels

    for region, classes in REGION_SEG_CLASSES.items():
        px = int(sum((seg2 == c).sum() for c in classes))
        if px <= 0:
            continue
        idxs = _kp_indices_for_region(region)
        kp_conf = (
            float(pose2_person[idxs, 2].mean())
            if (pose2_person is not None and idxs)
            else 0.0
        )
        parts.append(
            {
                "part": region,
                "pixel_frac": float(px / denom),
                "frame_frac": float(px / total_pixels),
                "pixel_count": px,
                "kp_conf": kp_conf,
            }
        )
    return parts


def _corroborated_regions(body_parts: list[dict]) -> set[str]:
    """Regions seg2 corroborates: foreground fraction AND absolute pixel floor."""
    return {
        bp["part"]
        for bp in body_parts
        if bp["pixel_frac"] > MIN_REGION_FRAC
        and bp.get("pixel_count", 0) >= MIN_REGION_PX
    }


def _kp(p, name):
    return p[GOLIATH_308.index(name)]


def _confident(kp, thresh=0.3):
    return kp[2] > thresh


def _get_limb_relation(p, side, joint1, joint2):
    idx1 = GOLIATH_308.index(f"{side}_{joint1}")
    idx2 = GOLIATH_308.index(f"{side}_{joint2}")
    if p[idx1, 2] > 0.3 and p[idx2, 2] > 0.3:
        if p[idx1, 1] < p[idx2, 1]:  # joint1 above joint2
            return f"{side} {joint1} extended upward"
        else:
            return f"{side} {joint1} extended downward"
    return None


def derive_determinations(
    pose2: np.ndarray,
    seg2: np.ndarray,
    *,
    pointmap: np.ndarray | None = None,
) -> dict:
    """Derive one determinations document from already-loaded artifact arrays.

    The function is intentionally pure: it neither mutates its NumPy inputs nor
    reads/writes a corpus path. This lets bounded research runs reuse the exact
    determination semantics while keeping output outside ``crawlr/stratum``.
    """
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

    if n == 0:
        return doc

    p = pose2[0]  # read-only view; never mutated

    # Per-region corroboration from seg2 (the independent witness).
    body_parts = get_body_parts_visible(seg2, p)
    doc["body_parts_visible"] = body_parts
    corroborated = _corroborated_regions(body_parts)

    # Pointmap (camera). Requires torso corroboration for shoulder height.
    if pointmap is not None:
        fg_mask = seg2 > 0
        if fg_mask.any():
            zs = pointmap[..., 2][fg_mask]
            median_z = float(np.median(zs))
            cam_doc = {"distance_m": round(median_z, 2)}

            if "torso" in corroborated:
                lsho = _kp(p, "left_shoulder")
                rsho = _kp(p, "right_shoulder")
                sho = lsho if lsho[2] > rsho[2] else rsho
                if _confident(sho):
                    x, y = int(sho[0]), int(sho[1])
                    if 0 <= y < pointmap.shape[0] and 0 <= x < pointmap.shape[1]:
                        # Pointmap is camera frame, +Y down, camera at origin.
                        cam_doc["shoulder_height_rel_camera_m"] = round(
                            float(pointmap[y, x, 1]), 2
                        )
            doc["camera"] = cam_doc

    # Extent (from raw pose2, unmodified).
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

    # Upright orientation. Anchored on torso (neck + hips).
    if "torso" in corroborated:
        neck = _kp(p, "neck")
        lhip = _kp(p, "left_hip")
        rhip = _kp(p, "right_hip")
        if _confident(neck) and (_confident(lhip) or _confident(rhip)):
            if _confident(lhip) and _confident(rhip):
                hip_y = (lhip[1] + rhip[1]) / 2.0
                hip_x = (lhip[0] + rhip[0]) / 2.0
            elif _confident(lhip):
                hip_y, hip_x = lhip[1], lhip[0]
            else:
                hip_y, hip_x = rhip[1], rhip[0]
            dy = hip_y - neck[1]
            dx = hip_x - neck[0]
            deg = math.degrees(math.atan2(dy, dx))
            doc["orientation"]["upright_deg"] = round(abs(deg - 90.0), 1)

    # Relations — each declares the regions it needs.
    rels = []

    # Arms (anchored on the corresponding arm region).
    if "left_arm" in corroborated:
        r = _get_limb_relation(p, "left", "wrist", "shoulder")
        if r:
            rels.append(r.replace("wrist", "arm"))
    if "right_arm" in corroborated:
        r = _get_limb_relation(p, "right", "wrist", "shoulder")
        if r:
            rels.append(r.replace("wrist", "arm"))

    # Legs (anchored on the corresponding leg region).
    if "left_leg" in corroborated:
        r = _get_limb_relation(p, "left", "ankle", "hip")
        if r:
            rels.append(r.replace("ankle", "leg"))
    if "right_leg" in corroborated:
        r = _get_limb_relation(p, "right", "ankle", "hip")
        if r:
            rels.append(r.replace("ankle", "leg"))

    # Hands together & held object (anchored on both hand regions).
    if "left_hand" in corroborated and "right_hand" in corroborated:
        lwri = _kp(p, "left_wrist")
        rwri = _kp(p, "right_wrist")
        if _confident(lwri) and _confident(rwri):
            dist = math.hypot(lwri[0] - rwri[0], lwri[1] - rwri[1])
            lsho = _kp(p, "left_shoulder")
            rsho = _kp(p, "right_shoulder")
            sho_width = (
                math.hypot(lsho[0] - rsho[0], lsho[1] - rsho[1])
                if (_confident(lsho) and _confident(rsho))
                else 200.0
            )
            if dist < sho_width * 0.5:
                rels.append("hands together")

                # Held object: background pixels between the wrists.
                x1, x2 = min(lwri[0], rwri[0]), max(lwri[0], rwri[0])
                y1, y2 = min(lwri[1], rwri[1]), max(lwri[1], rwri[1])
                x1, x2 = max(0, int(x1 - 50)), min(seg2.shape[1], int(x2 + 50))
                y1, y2 = max(0, int(y1 - 50)), min(seg2.shape[0], int(y2 + 50))
                if x2 > x1 and y2 > y1:
                    zone = seg2[y1:y2, x1:x2]
                    if (zone == 0).sum() > 50:
                        y_center = (y1 + y2) / 2
                        hip_y_ref = 0.0
                        lhip = _kp(p, "left_hip")
                        rhip = _kp(p, "right_hip")
                        if _confident(lhip) and _confident(rhip):
                            hip_y_ref = (lhip[1] + rhip[1]) / 2.0
                        elif _confident(lhip):
                            hip_y_ref = lhip[1]
                        elif _confident(rhip):
                            hip_y_ref = rhip[1]
                        level = (
                            "pelvis level"
                            if (hip_y_ref > 0 and abs(y_center - hip_y_ref) < 200)
                            else "waist level"
                        )
                        rels.append(f"hands gripping an object at {level}")

    # Facing (anchored on face region).
    if "face" in corroborated:
        lear = _kp(p, "left_ear")
        rear = _kp(p, "right_ear")
        if _confident(lear) and _confident(rear):
            rels.append("face turned toward camera")
        elif _confident(lear) or _confident(rear):
            rels.append("face in profile")

    doc["relations"] = rels
    return doc


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
    pointmap_path = output_dir / "pointmap.npy"
    pointmap = np.load(pointmap_path) if pointmap_path.exists() else None
    doc = derive_determinations(pose2, seg2, pointmap=pointmap)
    out_path.write_text(json.dumps(doc, indent=2))
    return True
