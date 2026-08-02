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
                "kp_conf": float(np.mean(pose2_person[70:308, 2]))
                if pose2_person is not None
                else 0.0,
            }
        )

    # Torso (include clothing classes that cover the torso: Upper_Clothing=23, Lower_Clothing=13)
    torso_px = (seg2 == 22).sum() + (seg2 == 23).sum() + (seg2 == 13).sum()
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


def _get_limb_relation(p, side, joint1, joint2):
    idx1 = GOLIATH_308.index(f"{side}_{joint1}")
    idx2 = GOLIATH_308.index(f"{side}_{joint2}")
    if p[idx1, 2] > 0.3 and p[idx2, 2] > 0.3:
        if p[idx1, 1] < p[idx2, 1]:  # joint1 above joint2
            return f"{side} {joint1} extended upward"
        else:
            return f"{side} {joint1} extended downward"
    return None


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
        p = pose2[0]
        # ---------------------------------------------------------
        # PRE-PROCESSING: Wipe keypoint confidences if the body part 
        # isn't actually in the image to prevent DETR hallucinations.
        # ---------------------------------------------------------
        body_parts = get_body_parts_visible(seg2, p)
        doc["body_parts_visible"] = body_parts

        # Determine which broad regions are actually present in segmentation
        parts_present = {bp["part"] for bp in body_parts if bp["pixel_frac"] > 0.01}

        # Wipe keypoint confidences if the body part isn't actually in the image
        # This prevents Sapiens2 from hallucinating tiny bodies inside face crops
        if "torso" not in parts_present:
            p[GOLIATH_308.index("left_shoulder"), 2] = 0.0
            p[GOLIATH_308.index("right_shoulder"), 2] = 0.0
            p[GOLIATH_308.index("left_hip"), 2] = 0.0
            p[GOLIATH_308.index("right_hip"), 2] = 0.0


        # Pointmap (camera)
        pointmap_path = output_dir / "pointmap.npy"
        if pointmap_path.exists():
            pm = np.load(pointmap_path)
            fg_mask = seg2 > 0
            if fg_mask.any():
                zs = pm[..., 2][fg_mask]
                median_z = float(np.median(zs))

                # Height relative to shoulder
                # Get Y coordinate from pointmap at the shoulder pixel
                lsho = p[GOLIATH_308.index("left_shoulder")]
                rsho = p[GOLIATH_308.index("right_shoulder")]

                # Pick the more confident shoulder
                sho = lsho if lsho[2] > rsho[2] else rsho
                height_rel = None

                if sho[2] > 0.3:
                    x, y = int(sho[0]), int(sho[1])
                    if 0 <= y < pm.shape[0] and 0 <= x < pm.shape[1]:
                        # PM is camera frame: +Y is down.
                        # If shoulder Y is negative, shoulder is above camera.
                        # Meaning camera is below shoulder (negative relative height).
                        # If shoulder Y is positive, shoulder is below camera.
                        # Meaning camera is above shoulder (positive relative height).
                        # So val_y is exactly camera_height_rel_shoulder!
                        val_y = float(pm[y, x, 1])
                        height_rel = val_y

                cam_doc = {"distance_m": round(median_z, 2)}
                if height_rel is not None:
                    cam_doc["height_rel_shoulder_m"] = round(height_rel, 2)

                doc["camera"] = cam_doc

        # Extent
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

        # Upright

        # We need a robust check before computing upright_deg:
        # Only compute upright_deg if BOTH neck and hips exist in the cleaned keypoints
        neck = p[GOLIATH_308.index("neck")]
        lhip = p[GOLIATH_308.index("left_hip")]
        rhip = p[GOLIATH_308.index("right_hip")]

        if neck[2] > 0.3 and (lhip[2] > 0.3 or rhip[2] > 0.3):
            hip_y = 0.0
            hip_x = 0.0
            if lhip[2] > 0.3 and rhip[2] > 0.3:
                hip_y = (lhip[1] + rhip[1]) / 2.0
                hip_x = (lhip[0] + rhip[0]) / 2.0
            elif lhip[2] > 0.3:
                hip_y, hip_x = lhip[1], lhip[0]
            else:
                hip_y, hip_x = rhip[1], rhip[0]

            dy = hip_y - neck[1]
            dx = hip_x - neck[0]

            deg = math.degrees(math.atan2(dy, dx))
            upright = abs(deg - 90.0)
            doc["orientation"]["upright_deg"] = round(upright, 1)

        # Relations
        rels = []

        # Arms
        l_arm = _get_limb_relation(p, "left", "wrist", "shoulder")
        if l_arm:
            rels.append(l_arm.replace("wrist", "arm"))
        r_arm = _get_limb_relation(p, "right", "wrist", "shoulder")
        if r_arm:
            rels.append(r_arm.replace("wrist", "arm"))

        # Legs
        l_leg = _get_limb_relation(p, "left", "ankle", "hip")
        if l_leg:
            rels.append(l_leg.replace("ankle", "leg"))
        r_leg = _get_limb_relation(p, "right", "ankle", "hip")
        if r_leg:
            rels.append(r_leg.replace("ankle", "leg"))

        # Hands together & Held object
        lwri = p[GOLIATH_308.index("left_wrist")]
        rwri = p[GOLIATH_308.index("right_wrist")]
        hands_together = False

        if lwri[2] > 0.3 and rwri[2] > 0.3:
            dist = math.hypot(lwri[0] - rwri[0], lwri[1] - rwri[1])

            # Dynamic threshold based on shoulder width
            lsho = p[GOLIATH_308.index("left_shoulder")]
            rsho = p[GOLIATH_308.index("right_shoulder")]
            sho_width = (
                math.hypot(lsho[0] - rsho[0], lsho[1] - rsho[1])
                if (lsho[2] > 0.3 and rsho[2] > 0.3)
                else 200.0
            )

            # "Together" if closer than ~half shoulder width
            if dist < sho_width * 0.5:
                hands_together = True
                rels.append("hands together")

                # Check for held object: are there background pixels between wrists?
                # Bbox around wrists
                x1, x2 = min(lwri[0], rwri[0]), max(lwri[0], rwri[0])
                y1, y2 = min(lwri[1], rwri[1]), max(lwri[1], rwri[1])
                # Expand box
                x1, x2 = max(0, int(x1 - 50)), min(seg2.shape[1], int(x2 + 50))
                y1, y2 = max(0, int(y1 - 50)), min(seg2.shape[0], int(y2 + 50))

                if x2 > x1 and y2 > y1:
                    zone = seg2[y1:y2, x1:x2]
                    bg_count = (zone == 0).sum()
                    if bg_count > 50:  # Threshold for held object visible between hands
                        # Level relative to joints
                        # Synthetic hips are at 600, wrists at 500
                        y_center = (y1 + y2) / 2

                        hip_y_ref = 0.0
                        lhip = p[GOLIATH_308.index("left_hip")]
                        rhip = p[GOLIATH_308.index("right_hip")]
                        if lhip[2] > 0.3 and rhip[2] > 0.3:
                            hip_y_ref = (lhip[1] + rhip[1]) / 2.0
                        elif lhip[2] > 0.3:
                            hip_y_ref = lhip[1]
                        elif rhip[2] > 0.3:
                            hip_y_ref = rhip[1]

                        if hip_y_ref > 0 and abs(y_center - hip_y_ref) < 200:
                            level = "pelvis level"
                        else:
                            level = "waist level"

                        rels.append(f"hands gripping an object at {level}")

        # Facing
        lear = p[GOLIATH_308.index("left_ear")]
        rear = p[GOLIATH_308.index("right_ear")]

        if lear[2] > 0.3 and rear[2] > 0.3:
            rels.append("face turned toward camera")
        elif lear[2] > 0.3 or rear[2] > 0.3:
            rels.append("face in profile")

        # Crop Silence
        parts = [bp["part"] for bp in doc["body_parts_visible"]]
        if parts == ["face"]:
            rels = []

        doc["relations"] = rels

    out_path.write_text(json.dumps(doc, indent=2))
    return True