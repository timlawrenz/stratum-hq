"""CPU probe (pass 2): discriminators for arm-#75 image-focus / depth-of-field.

Pass 1 showed the DOF ratio (bg/subject interior acutance) is well spread
(8/8/7 at 0.45/0.75 cuts) but subject acutance is NOT absolute-band-able
(min 3.08, all >=3) and one item has a flat (gradient ~0) background. This
pass tests RELATIVE subject-focus discriminators and background-content
guards so bands stay scale-invariant and the flat-background case is honest.

Candidate relative discriminators (all scale-invariant, within-image):
- subj_focus_ratio  = subject interior median acutance / frame top-percentile
                       acutance (p_aggile edge energy: the sharpest texture any-
                       where in the image). Near 1 => subject carries the
                       sharpest detail (crisp focus); << 1 => subject soft.
- subj_vs_global     = subject interior median / full-frame median acutance.
- bg_content         = background interior median acutance (absolute at the
                       canonical scale — used ONLY as an abstention/content
                       guard, never verbalized).
- bg_p99, bg_std     = background interior texture spread (guards: flat vs
                       blurred-but-textured).
- bg_ratio           = bg/subject interior acutance (the DOF signal, pass 1).
Read-only, no GPU, no corpus write.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import binary_erosion  # noqa: E402

MANIFEST = "/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json"
SOURCE = "/mnt/nas-ai-models/training-data/crawlr/approved"
DERIVED = "/mnt/nas-ai-models/training-data/crawlr/stratum"
CANONICAL_SIDE = 512


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _erode(mask: np.ndarray, iters: int) -> np.ndarray:
    return binary_erosion(mask, iterations=iters).astype(bool)


def main() -> int:
    manifest = json.loads(Path(MANIFEST).read_text())
    rows: list[dict] = []
    all_subj = []
    all_bg_ratio = []
    for item in manifest["items"]:
        image_id = item["image_id"]
        src = Path(SOURCE) / item["source_relative_path"]
        payload = src.read_bytes()
        if _sha256(payload) != item["source_sha256"]:
            rows.append({"image_id": image_id, "abstained": True, "abstention_reason": "source sha drifted"})
            continue
        seg2 = np.load(Path(DERIVED) / image_id / "seg2.npy", allow_pickle=False)
        with Image.open(__import__("io").BytesIO(payload)) as opened:
            width, height = opened.size
            if item["source_dimensions"]["width"] != width or item["source_dimensions"]["height"] != height:
                rows.append({"image_id": image_id, "abstained": True, "abstention_reason": "dims drifted"})
                continue
            rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        scale = CANONICAL_SIDE / max(width, height)
        new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
        resized = np.asarray(Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.LANCZOS), dtype=np.uint8)
        lum = (
            0.299 * resized[:, :, 0].astype(np.float64)
            + 0.587 * resized[:, :, 1].astype(np.float64)
            + 0.114 * resized[:, :, 2].astype(np.float64)
        )
        gy, gx = np.gradient(lum)
        grad = np.sqrt(gx * gx + gy * gy)
        seg_small = np.asarray(Image.fromarray(seg2).resize((new_w, new_h), Image.Resampling.NEAREST), dtype=np.uint8)
        subject = seg_small != 0
        background = seg_small == 0
        subject_i = _erode(subject, 2)
        background_i = _erode(background, 3)

        subj_int = grad[subject_i]
        bg_int = grad[background_i]
        if subj_int.size < 100 or bg_int.size < 100:
            rows.append({
                "image_id": image_id, "abstained": True,
                "abstention_reason": f"regions too small (subj {subj_int.size}, bg {bg_int.size})",
            })
            continue

        frame_top = np.percentile(grad, 99)
        subj_acut = float(np.median(subj_int))
        bg_acut = float(np.median(bg_int))
        global_acut = float(np.median(grad))
        row = {
            "image_id": image_id,
            "abstained": False,
            "native_dims": [width, height],
            "subject_share": round(float(subject.mean()), 4),
            "subj_acutance": round(subj_acut, 4),
            "subj_focus_ratio": round(subj_acut / frame_top, 4) if frame_top > 0 else None,
            "subj_vs_global": round(subj_acut / global_acut, 4) if global_acut > 0 else None,
            "bg_acutance": round(bg_acut, 4),
            "bg_ratio": round(bg_acut / subj_acut, 4) if subj_acut > 0 else None,
            "bg_p99": round(float(np.percentile(bg_int, 99)), 4),
            "bg_std": round(float(bg_int.std()), 4),
        }
        rows.append(row)
        all_subj.append(subj_acut)
        all_bg_ratio.append(row["bg_ratio"])
        print(
            f"{image_id[:12]}  subj={row['subj_acutance']} ratio={row['subj_focus_ratio']} "
            f"vsg={row['subj_vs_global']} bg={row['bg_acutance']} bgratio={row['bg_ratio']} "
            f"bgp99={row['bg_p99']} bgstd={row['bg_std']} share={row['subject_share']}"
        )

    det = [r for r in rows if not r.get("abstained")]
    n = len(det)
    print("\n=== PASS-2 SUMMARY ===")
    for key, label in (
        ("subj_focus_ratio", "subject focus ratio (subj/frame-p99)"),
        ("subj_vs_global", "subject vs global"),
        ("bg_ratio", "DOF bg/subject ratio"),
        ("bg_p99", "background p99 acutance"),
    ):
        vals = np.array([r[key] for r in det if r.get(key) is not None])
        if vals.size:
            print(f"{label}: min={vals.min():.3f} p25={np.percentile(vals, 25):.3f} "
                  f"med={np.median(vals):.3f} p75={np.percentile(vals, 75):.3f} max={vals.max():.3f} n={vals.size}")
    for lo, hi, name in ((None, 0.35, "subj_focus <=0.35"), (0.35, 0.55, "0.35-0.55"),
                         (0.55, None, ">0.55")):
        vals = np.array([r["subj_focus_ratio"] for r in det if r["subj_focus_ratio"] is not None])
        sel = ((vals >= lo) if hi is None else ((vals >= lo) & (vals < hi))) if lo is not None else \
              ((vals < hi) if hi is not None else vals >= 0)
        print(f"  {name}: {int(sel.sum())}/{vals.size} ({sel.mean():.2f})")
    flat = [r for r in det if r["bg_p99"] < 4.0]
    print(f"flat-ish backgrounds (bg_p99 < 4.0): {len(flat)}/ {n}")
    for r in flat:
        print(f"    {r['image_id'][:12]}  bg_p99={r['bg_p99']} bg_ratio={r['bg_ratio']}")
    Path("/mnt/nas-ai-models/research/stratum/image-focus-calibration-probe2.json").write_text(
        json.dumps({"rows": rows}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())