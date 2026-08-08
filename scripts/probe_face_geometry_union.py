#!/usr/bin/env python3
"""Union capability probe for arm #60: per item try (a) full frame, (b) seg2
Face_Neck crop, (c) tight face-region crops at a couple of scales; take the
first valid 478-landmark mesh. Reports union coverage + per-path abstain
reasons + ratio distributions. Deterministic, read-only, CPU-only."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))
MODEL = "/mnt/nas-ai-models/research/stratum/models/face-geometry/face_landmarker.task"
DERIVED = Path("/mnt/nas-ai-models/training-data/crawlr/stratum")
PLAN = "/mnt/nas-ai-models/research/stratum/stage-b-matting-alpha-v1/stage-b-plan.json"
plan = json.load(open(PLAN))
SOURCE_ROOT = plan["pilot_manifest"]["source_root"]

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision.core.image import ImageFormat
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from mediapipe import Image as MpImage

LM_CACHE = {}

def landmarker():
    if "lm" not in LM_CACHE:
        LM_CACHE["lm"] = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL),
            running_mode=RunningMode.IMAGE, num_faces=1,
        ))
    return LM_CACHE["lm"]

def detect(arr):
    try:
        return landmarker().detect(MpImage(image_format=ImageFormat.SRGB, data=arr))
    except Exception as exc:  # noqa: BLE001
        return {"_error": repr(exc)}

L_EYE_IN, R_EYE_IN = 133, 263
L_EYE_OUT, R_EYE_OUT = 33, 362
MOUTH_L, MOUTH_R = 61, 291
CHEEK_L, CHEEK_R = 234, 454
JAW_L, JAW_R = 172, 397
NOSE_TIP, CHIN = 1, 152
FACE_NECK = 3

def euclid(a, b): return float(np.linalg.norm(a - b))

def mesh_to_facts(res, img):
    if not res or not getattr(res, "face_landmarks", None) or not res.face_landmarks:
        return None
    mesh = res.face_landmarks[0]
    pts = np.array([(p.x, p.y, p.z) for p in mesh])
    xs = pts[:, 0] * img.width; ys = pts[:, 1] * img.height
    if xs.max() - xs.min() < 15 or ys.max() - ys.min() < 15:
        return {"_too_small": True}
    face_w = euclid(pts[CHEEK_L, :2], pts[CHEEK_R, :2])
    eye_dist = euclid(pts[L_EYE_IN, :2], pts[R_EYE_IN, :2])
    interpup = euclid((pts[L_EYE_OUT, :2] + pts[L_EYE_IN, :2]) / 2,
                      (pts[R_EYE_OUT, :2] + pts[R_EYE_IN, :2]) / 2)
    mouth_w = euclid(pts[MOUTH_L, :2], pts[MOUTH_R, :2])
    jaw_w = euclid(pts[JAW_L, :2], pts[JAW_R, :2])
    nose_low = (pts[1][1] + pts[2][1] + pts[98][1]) / 3
    brow_y = (pts[105][1] + pts[334][1]) / 2
    chin_y = pts[152][1]
    f = {
        "n_landmarks": int(len(pts)),
        "z_span_rel": float(pts[:, 2].max() - pts[:, 2].min()),
        "face_w": face_w, "eye_dist": eye_dist, "interpupillary": interpup,
        "mouth_w": mouth_w, "jaw_w": jaw_w,
        "face_bbox_px": [round(xs.min()), round(ys.min()), round(xs.max()), round(ys.max())],
    }
    if face_w > 1e-6:
        f["eye_spacing_face_width"] = eye_dist / face_w
        f["mouth_face_width"] = mouth_w / face_w
        f["jaw_face_width"] = jaw_w / face_w
    if chin_y > brow_y + 1e-9:
        f["midface_share"] = (nose_low - brow_y) / (chin_y - brow_y)
    return f

rows = []
for item in plan["pilot_manifest"]["items"]:
    iid = item["image_id"]
    img = Image.open(Path(SOURCE_ROOT) / item["source_relative_path"]).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    seg = np.load(DERIVED / iid / "seg2.npy")
    fn_px = int((seg == FACE_NECK).sum())

    paths = {}
    # (a) full frame
    paths["full_frame"] = detect(arr)
    # (b) seg2 Face_Neck crop, margin ~ 1.0 max-side, and a tighter 0.5 variant
    masks = []
    if fn_px > 200:
        ms = seg == FACE_NECK
        ys, xs = np.where(ms)
        for tag, mf in (("fncrop_1.0", 1.0), ("fncrop_0.5", 0.5)):
            h, w = ys.max() - ys.min(), xs.max() - xs.min()
            mm = int(mf * max(h, w))
            cy0, cy1 = max(0, ys.min() - mm), min(seg.shape[0] - 1, ys.max() + mm)
            cx0, cx1 = max(0, xs.min() - mm), min(seg.shape[1] - 1, xs.max() + mm)
            cr = np.asarray(img.crop((cx0, cy0, cx1, cy1)), dtype=np.uint8)
            paths[tag] = detect(cr)

    winner = None
    for tag, res in paths.items():
        facts = mesh_to_facts(res, img) if not (isinstance(res, dict) and "_error" in res) else None
        if facts and "_too_small" not in facts:
            winner = (tag, facts)
            break
    if winner:
        tag, facts = winner
        facts["via"] = tag
        facts["id"] = iid
        facts["fn_px"] = fn_px
        facts["detection"] = "DETECTED"
        rows.append(facts)
    else:
        reasons = []
        for tag, res in paths.items():
            if isinstance(res, dict) and "_error" in res:
                reasons.append(f"{tag}:ERR")
            elif res and getattr(res, "face_landmarks", None):
                reasons.append(f"{tag}:too_small")
            else:
                reasons.append(f"{tag}:none")
        rows.append({
            "id": iid, "fn_px": fn_px, "detection": "ABSTAIN",
            "abstention_reason": " | ".join(reasons),
        })

det = [r for r in rows if r["detection"] == "DETECTED"]
abst = [r for r in rows if r["detection"] != "DETECTED"]
print(f"UNION: {len(det)}/24 detected, {len(abst)} abstain")
from collections import Counter
print("  via:", dict(Counter(r.get("via") for r in det)))
for r in abst:
    print(f"  ABSTAIN {r['id'][:16]} fn_px={r['fn_px']}: {r['abstention_reason']}")
if det:
    for k in ("eye_spacing_face_width", "mouth_face_width", "jaw_face_width", "midface_share"):
        vals = sorted(r[k] for r in det if r.get(k) is not None)
        if vals:
            print(f"  {k}: n={len(vals)} min={vals[0]:.3f} p10={vals[len(vals)//10]:.3f} p25={vals[len(vals)//4]:.3f} p50={vals[len(vals)//2]:.3f} p75={vals[3*len(vals)//4]:.3f} p90={vals[9*len(vals)//10]:.3f} max={vals[-1]:.3f}")
json.dump({"rows": rows, "n_detected": len(det)}, open("/tmp/face_geometry_union_probe.json", "w"), indent=1, default=str)
print("saved /tmp/face_geometry_union_probe.json")
