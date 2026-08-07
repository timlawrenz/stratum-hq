#!/usr/bin/env python3
"""Full-frame FaceLandmarker sweep across the 24 frozen items — the ARM-policy
candidate. Reports detection, landmark count, ratio distributions, and the
seg2 Face_Neck vs full-frame agreement (exactly-one-face invariant)."""
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

lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL),
    running_mode=RunningMode.IMAGE, num_faces=1,
))

L_EYE_IN, R_EYE_IN = 133, 263
L_EYE_OUT, R_EYE_OUT = 33, 362
MOUTH_L, MOUTH_R = 61, 291
CHEEK_L, CHEEK_R = 234, 454
JAW_L, JAW_R = 172, 397
NOSE_TIP, CHIN = 1, 152
FACE_NECK = 3

def euclid(a, b): return float(np.linalg.norm(a - b))

rows = []
for item in plan["pilot_manifest"]["items"]:
    iid = item["image_id"]
    img = Image.open(Path(SOURCE_ROOT) / item["source_relative_path"]).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    seg = np.load(DERIVED / iid / "seg2.npy")
    fn_px = int((seg == FACE_NECK).sum())
    r = {"id": iid, "seg_shape": list(seg.shape), "img": img.size, "fn_px": fn_px}
    try:
        res = lm.detect(MpImage(image_format=ImageFormat.SRGB, data=arr))
    except Exception as exc:  # noqa: BLE001
        r.update({"detection": "ERROR", "error": repr(exc)})
        rows.append(r); continue
    n = len(res.face_landmarks) if res and res.face_landmarks else 0
    if n == 0:
        r.update({"detection": "ABSTAIN", "abstention_reason": "no face on full frame",
                  "fn_px": fn_px})
        rows.append(r); continue
    if n > 1:
        # cannot happen with num_faces=1, but record honestly
        r.update({"detection": "MULTI", "n_faces_on_frame": len(res.face_landmarks)})
        rows.append(r); continue
    lm_pts = [(p.x, p.y, p.z) for p in res.face_landmarks[0]]
    pts = np.array(lm_pts)
    # exactly-one-body invariant: subject is single woman; face = her face.
    # Cross-check: face bbox inside the frame and reasonably sized.
    xs = pts[:, 0] * img.width; ys = pts[:, 1] * img.height
    fw = xs.max() - xs.min(); fh = ys.max() - ys.min()
    if fw < 20 or fh < 20:
        r.update({"detection": "ABSTAIN", "abstention_reason": f"face too small fw={fw:.0f} fh={fh:.0f}"})
        rows.append(r); continue
    face_w = euclid(pts[CHEEK_L, :2], pts[CHEEK_R, :2])
    eye_dist = euclid(pts[L_EYE_IN, :2], pts[R_EYE_IN, :2])
    interpup = euclid((pts[L_EYE_OUT, :2] + pts[L_EYE_IN, :2]) / 2,
                      (pts[R_EYE_OUT, :2] + pts[R_EYE_IN, :2]) / 2)
    mouth_w = euclid(pts[MOUTH_L, :2], pts[MOUTH_R, :2])
    jaw_w = euclid(pts[JAW_L, :2], pts[JAW_R, :2])
    face_h = euclid(pts[NOSE_TIP, :2], pts[CHIN, :2]) + 1e-9
    r.update({
        "detection": "DETECTED", "n_landmarks": len(pts),
        "z_span_rel": float(pts[:, 2].max() - pts[:, 2].min()),
        "face_w": face_w, "eye_dist": eye_dist, "interpupillary": interpup,
        "mouth_w": mouth_w, "jaw_w": jaw_w, "face_h": face_h,
        "face_bbox_px": [round(xs.min()), round(ys.min()), round(xs.max()), round(ys.max())],
    })
    if face_w > 1e-6 and face_h > 0:
        r["eye_spacing_face_width"] = eye_dist / face_w
        r["mouth_face_width"] = mouth_w / face_w
        r["jaw_face_width"] = jaw_w / face_w
        # facial third proxies (canonical indices)
        # brow line ~ 105 (left brow) / 334 (right brow); nose base ~ 2/98
        nose_low = (pts[1][1] + pts[2][1] + pts[98][1]) / 3  # nose base y
        brow_y = (pts[105][1] + pts[334][1]) / 2
        chin_y = pts[152][1]
        if (chin_y - brow_y) > 1e-9:
            r["midface_share"] = (nose_low - brow_y) / (chin_y - brow_y)
    rows.append(r)

det = [x for x in rows if x["detection"] == "DETECTED"]
abst = [x for x in rows if x["detection"] != "DETECTED"]
print(f"full-frame: {len(det)}/24 detected, {len(abst)} abstain")
for x in abst:
    print(f"  ABSTAIN {x['id'][:16]} fn_px={x.get('fn_px')}: {x.get('abstention_reason', x.get('detection'))}")
if det:
    import collections
    for k in ("eye_spacing_face_width", "mouth_face_width", "jaw_face_width", "midface_share"):
        vals = sorted(x[k] for x in det if x.get(k) is not None)
        if vals:
            print(f"  {k}: n={len(vals)} min={vals[0]:.3f} p10={vals[len(vals)//10]:.3f} p25={vals[len(vals)//4]:.3f} p50={vals[len(vals)//2]:.3f} p75={vals[3*len(vals)//4]:.3f} p90={vals[9*len(vals)//10]:.3f} max={vals[-1]:.3f}")
    # fn_px distribution for detectable items (face region sanity)
    fns = [x["fn_px"] for x in det]
    print(f"  seg2 Face_Neck px: min={min(fns)} p50={sorted(fns)[len(fns)//2]} max={max(fns)}")
json.dump({"rows": rows, "n_detected": len(det)}, open("/tmp/face_geometry_fullframe_probe.json", "w"), indent=1, default=str)
print("saved /tmp/face_geometry_fullframe_probe.json")
