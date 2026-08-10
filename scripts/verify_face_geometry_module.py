#!/usr/bin/env python3
"""Verify the face_geometry module against the frozen cohort (mirror of the
probe, but calling the module's compute_face_geometry + render_face_geometry).
CPU-only, read-only."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))
from research_harness.face_geometry import compute_face_geometry, render_face_geometry

MODEL = "/mnt/nas-ai-models/research/stratum/models/face-geometry/face_landmarker.task"
DERIVED = Path("/mnt/nas-ai-models/training-data/crawlr/stratum")
plan = json.load(open('/mnt/nas-ai-models/research/stratum/stage-b-matting-alpha-v1/stage-b-plan.json'))
SOURCE_ROOT = plan["pilot_manifest"]["source_root"]

det = 0
abst = 0
for item in plan["pilot_manifest"]["items"]:
    iid = item["image_id"]
    img = Image.open(Path(SOURCE_ROOT) / item["source_relative_path"]).convert("RGB")
    rgb = np.asarray(img, dtype=np.uint8)
    seg = np.load(DERIVED / iid / "seg2.npy")
    f = compute_face_geometry(seg, rgb, model_asset_path=MODEL)
    if f.get("abstained"):
        abst += 1
        print(f"  ABSTAIN {iid[:16]}: {f.get('abstention_reason')}")
        continue
    det += 1
    lines = render_face_geometry(f)
    print(f"  DETECT {iid[:16]} via={f['via']} eye={f.get('eye_spacing_band')} "
          f"mouth={f.get('mouth_band')} jaw={f.get('jaw_band')} mid={f.get('midface_band')} "
          f"zspan={f.get('z_span_rel'):.3f}")
    print(f"      {lines}")

print(f"\nmodule verify: {det}/24 detected, {abst} abstained")
