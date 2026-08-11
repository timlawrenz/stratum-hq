#!/usr/bin/env python3
"""DINOv3 reconstruction-fidelity scoring for the LOO specialist-signal attempt.

Scores every generated image against its item's SOURCE DINOv3 CLS (saved corpus
embedding = C_base) and patch tokens. Signals:
  d_cls   = 1 - cos(C_src, C_gen)                       (CLS distance)
  d_patch = 1 - mean over src patches of max cos(p_src, p'_gen)  (nearest-match patch)
Conditions:
  C_after = full deterministic caption (recon-prose.png)
  C_before(spec) = full minus that specialist's claims  (recon-loo-*.png)
  plus context: arm37-degraded-baseline, arm37-ctx4k, prose-agg, vlm, null.
Per-specialist delta = mean_i[d(C_before) - d(C_after)]  (>0 => specialist adds fidelity).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/tim/source/activity/stratum-hq/src")
from stratum.config import DINO_MODEL_ID  # noqa: E402
from stratum.pipeline.dinov3 import compute_dinov3_both, load_dinov3  # noqa: E402

RUN_ROOT = Path("/mnt/nas-ai-models/research/stratum/prose-caption2-demo-v1")
STRATUM_ROOT = Path("/mnt/nas-ai-models/training-data/crawlr/stratum")
ARM37 = Path("/mnt/nas-ai-models/research/stratum/stage-b-reconstruction-v1/outputs")
BUCKET = (832, 1216)

PILOTS = ["03r1r5psuxprfkhomitcz5vh9w1c", "07hyx5wjk5rc339v2s3e2orcoorr",
          "0f3fdmh6iackl6049wmrm9n2p4i5", "058v0mg1bbxthvatdw324k8j4b0s",
          "08v25q5524t2u0zl0xtnzi6bd22f"]


def source_cls_patches(image_id: str):
    d = STRATUM_ROOT / image_id
    return (np.load(d / "dinov3_cls.npy").astype(np.float32),
            np.load(d / "dinov3_patches.npy").astype(np.float32))


def embed(image_path: Path, dino, device):
    from PIL import Image
    with Image.open(image_path) as im:
        image = im.convert("RGB")
    cls_list, patches_np = compute_dinov3_both(
        dino, device, image, target_width=BUCKET[0], target_height=BUCKET[1])
    return np.asarray(cls_list, np.float32), (patches_np.astype(np.float32) if patches_np is not None else None)


def cosine(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    return float(a @ b)


def patch_score(sp, gp):
    """mean over source patches of nearest (max-cos) match in generated patches."""
    sp = sp / np.linalg.norm(sp, axis=-1, keepdims=True)
    gp = gp / np.linalg.norm(gp, axis=-1, keepdims=True)
    return float(np.mean(np.max(sp @ gp.T, axis=-1)))


def build_registry() -> dict[str, list[dict]]:
    reg = {i: [] for i in PILOTS}
    # C_after (full deterministic) + agg + vlm + null from the recon manifest
    manifest = json.loads((RUN_ROOT / "_recon-manifest.json").read_text()) if (RUN_ROOT / "_recon-manifest.json").exists() else []
    for e in manifest:
        i = e.get("image_id")
        if i in reg and Path(e["png"]).exists():
            reg[i].append({"cond": e["condition"], "png": Path(e["png"])})
    # arm-37 baselines (degraded baseline + ctx4k) same seed
    import hashlib
    for i in reg:
        seed = int(hashlib.sha256(i.encode()).hexdigest()[:8], 16) or 1
        for cond, sub in [("arm37-baseline", "recon-baseline"), ("arm37-ctx4k", "recon-ctx4k")]:
            p = ARM37 / sub / f"{i}_{seed}.png"
            if p.exists():
                reg[i].append({"cond": cond, "png": p})
    # LOO before images (if generated)
    loo = json.loads((RUN_ROOT / "_loo-run.json").read_text()) if (RUN_ROOT / "_loo-run.json").exists() else []
    for e in loo:
        i = e["image_id"]
        if i in reg and Path(e["png"]).exists():
            reg[i].append({"cond": f"loo-{e['specialist']}", "png": Path(e["png"])})
    return reg


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = __import__("torch").device("cpu")
    dino = load_dinov3(device, DINO_MODEL_ID)
    reg = build_registry()
    results = []
    for i in PILOTS:
        src_cls, src_pat = source_cls_patches(i)
        row = {"image_id": i, "conds": {}}
        for c in reg[i]:
            gen_cls, gen_pat = embed(c["png"], dino, device)
            d_cls = 1 - cosine(src_cls, gen_cls)
            d_patch = (1 - patch_score(src_pat, gen_pat)) if gen_pat is not None else None
            row["conds"][c["cond"]] = {"d_cls": round(d_cls, 4),
                                       "sim_cls": round(1 - d_cls, 4),
                                       "d_patch": round(d_patch, 4) if d_patch is not None else None}
        results.append(row)
        print(f"== {i} ==")
        for cond, v in sorted(row["conds"].items()):
            print(f"   {cond:34s} sim_cls={v['sim_cls']:.4f} d_cls={v['d_cls']:.4f} d_patch={v['d_patch']}")
    (RUN_ROOT / "_dinov3-scores.json").write_text(json.dumps(results, indent=2))
    print("saved", RUN_ROOT / "_dinov3-scores.json")


if __name__ == "__main__":
    main()