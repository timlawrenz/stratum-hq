#!/usr/bin/env python3
"""CPU capability probe for the face-geometry arm (#60) NEW-model-class producer.

Verifies MediaPipe FaceLandmarker (tasks API, 478-point mesh) on the local
owned-hardware stack BEFORE the arm's round-trip trusts it (qualification gate
step 2). Two phases:

1. SYNTHETIC (non-sensitive) — confirm the .task model loads, the tasks API
   runs, and the module returns a usable result or abstains cleanly on a
   non-face image (no crash, no nonsense).
2. FROZEN COHORT (read-only, local-first) — for each of the 24 frozen items:
   derive the face crop from the seg2 Face_Neck bbox (margin-expanded), run
   FaceLandmarker (num_faces=1), and report detection rate, 478-landmark
   yield, and a candidate scale-invariant ratio set:
     eye_spacing/face_width (interpupillary over cheek width)
     mouth/face_width
     jaw/face_width
   plus face crop aspect / margin stats. Only scale-invariant ratios leave the
   payload; no absolute px in caption prose (measurement-semantics directive).

Output is JSON at /tmp/face_geometry_probe.json + a human-readable summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

MODEL = "/mnt/nas-ai-models/research/stratum/models/face-geometry/face_landmarker.task"
DERIVED = Path("/mnt/nas-ai-models/training-data/crawlr/stratum")
PLAN = "/mnt/nas-ai-models/research/stratum/stage-b-matting-alpha-v1/stage-b-plan.json"
_PLAN = json.load(open(PLAN))
SOURCE_ROOT = _PLAN["pilot_manifest"]["source_root"]

# Canonical MediaPipe FaceMesh landmark indices (468-point face, plus 10 iris).
L_EYE_OUT, L_EYE_IN = 33, 133
R_EYE_OUT, R_EYE_IN = 362, 263
MOUTH_L, MOUTH_R = 61, 291
CHEEK_L, CHEEK_R = 234, 454   # widest cheek points
JAW_L, JAW_R = 172, 397       # lower chin/jaw points
NOSE_TIP = 1
CHIN = 152

FACE_NECK = 3  # DOME-29 class index (Face_Neck)

CROP_MARGIN_FRAC = 1.0  # expand bbox by ~100% of its own max side each dim


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def run_synthetic(landmarker) -> dict:
    """Phase 1: non-sensitive smoke + clean-abstain check."""
    from PIL import Image, ImageDraw

    W, H = 320, 320
    img = Image.new("RGB", (W, H), (210, 210, 210))  # blank non-face
    d = ImageDraw.Draw(img)
    d.ellipse([80, 120, 240, 210], fill=(120, 120, 120))  # blob, not a face
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)

    from mediapipe import Image as MpImage
    from mediapipe.tasks.python.vision.core.image import ImageFormat

    res = None
    try:
        res = landmarker.detect(MpImage(image_format=ImageFormat.SRGB, data=arr))
    except Exception as exc:  # noqa: BLE001
        return {"loads": True, "synthetic_run": False, "error": repr(exc)}
    n_faces = len(res.face_landmarks) if res and res.face_landmarks else 0
    return {
        "loads": True,
        "synthetic_run": True,
        "faces_detected_on_synthetic_nonface": n_faces,
        "clean_abstain": n_faces == 0,  # a blank blob must NOT produce landmarks
    }


def face_crop_and_ratios(item) -> dict:
    iid = item["image_id"]
    d = DERIVED / iid
    seg2 = np.load(d / "seg2.npy")
    mask = seg2 == FACE_NECK
    out = {"id": iid, "abs_seg_shape": list(seg2.shape)}
    if mask.sum() < 200:
        out.update({"detection": "ABSTAIN_no_face_region",
                    "abstention_reason": f"seg2 Face_Neck px={int(mask.sum())}"})
        return out

    ys, xs = np.where(mask)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    h, w = y1 - y0, x1 - x0
    m = int(CROP_MARGIN_FRAC * max(h, w))
    # clamp to frame
    cy0, cy1 = max(0, y0 - m), min(seg2.shape[0] - 1, y1 + m)
    cx0, cx1 = max(0, x0 - m), min(seg2.shape[1] - 1, x1 + m)
    out.update({"crop_h": cy1 - cy0, "crop_w": cx1 - cx0})

    # decode source RGB for the crop (source_root + source_relative_path)
    from PIL import Image
    src = Path(SOURCE_ROOT) / item["source_relative_path"]
    image = Image.open(src).convert("RGB")
    crop = np.asarray(image.crop((cx0, cy0, cx1, cy1)), dtype=np.uint8)
    if crop.size == 0:
        out.update({"detection": "ABSTAIN_empty_crop"})
        return out

    from mediapipe import Image as MpImage
    from mediapipe.tasks.python.vision.core.image import ImageFormat
    res = landmarker.detect(
        MpImage(image_format=ImageFormat.SRGB, data=crop)
    )
    if not res or not res.face_landmarks or len(res.face_landmarks) == 0:
        out.update({
            "detection": "ABSTAIN_no_landmarks",
            "abstention_reason": "FaceLandmarker found no face in the seg2 Face_Neck crop",
        })
        return out

    lm = res.face_landmarks[0]
    pts = np.array([(p.x, p.y, p.z) for p in lm])  # normalized x,y; relative z
    out.update({
        "detection": "DETECTED",
        "n_landmarks": int(len(pts)),
        "z_span_rel": float(pts[:, 2].max() - pts[:, 2].min()),
        "face_w": float(_euclid(pts[CHEEK_L, :2], pts[CHEEK_R, :2])),
        "eye_dist": float(_euclid(pts[L_EYE_IN, :2], pts[R_EYE_IN, :2])),
        "interpupillary": float(_euclid(
            (pts[L_EYE_OUT, :2] + pts[L_EYE_IN, :2]) / 2,
            (pts[R_EYE_OUT, :2] + pts[R_EYE_IN, :2]) / 2,
        )),
        "mouth_w": float(_euclid(pts[MOUTH_L, :2], pts[MOUTH_R, :2])),
        "jaw_w": float(_euclid(pts[JAW_L, :2], pts[JAW_R, :2])),
        "face_h": float(_euclid(pts[NOSE_TIP], pts[CHIN]) + 1e-9),
    })
    if out["face_w"] > 1e-6:
        out["eye_spacing_face_width"] = out["eye_dist"] / out["face_w"]
        out["mouth_face_width"] = out["mouth_w"] / out["face_w"]
        out["jaw_face_width"] = out["jaw_w"] / out["face_w"]
    return out


if __name__ == "__main__":
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (FaceLandmarker,
                                               FaceLandmarkerOptions,
                                               RunningMode)

    print(f"model: {MODEL}", flush=True)
    landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
    ))
    synth = run_synthetic(landmarker)
    print("=== SYNTHETIC (non-sensitive) ===", json.dumps(synth, indent=1), flush=True)
    if not synth.get("clean_abstain", False):
        print("PROBE FAIL: model produced landmarks on a blank blob", flush=True)
        sys.exit(1)

    items = _PLAN["pilot_manifest"]["items"]
    rows = [face_crop_and_ratios(it) for it in items]
    detected = [r for r in rows if r["detection"] == "DETECTED"]
    json.dump(
        {"synthetic": synth, "rows": rows, "n_detected": len(detected), "n_items": len(rows)},
        open("/tmp/face_geometry_probe.json", "w"), indent=1, default=str,
    )
    print(f"\n=== FROZEN COHORT: {len(detected)}/{len(rows)} face detected ===", flush=True)
    for r in rows:
        if r["detection"] != "DETECTED":
            print(f"  {r['id'][:16]} ABSTAIN: {r.get('abstention_reason')}", flush=True)
    if detected:
        import numpy as _np
        for k in ("eye_spacing_face_width", "mouth_face_width", "jaw_face_width"):
            vals = sorted(r[k] for r in detected if r.get(k) is not None)
            if vals:
                print(
                    f"  {k}: n={len(vals)} "
                    f"min={min(vals):.3f} p10={vals[len(vals)//10]:.3f} "
                    f"p50={vals[len(vals)//2]:.3f} p90={vals[9*len(vals)//10]:.3f} "
                    f"max={max(vals):.3f}", flush=True
                )
        zs = sorted(r["z_span_rel"] for r in detected)
        print(f"  z_span_rel p10/p50/p90: {zs[len(zs)//10]:.4f}/{zs[len(zs)//2]:.4f}/{zs[9*len(zs)//10]:.4f}", flush=True)
    print("probe saved to /tmp/face_geometry_probe.json", flush=True)
