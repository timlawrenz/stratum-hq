"""CPU capability probe: proposal #112 jewelry (earrings / necklace) zero-shot.

Follows the established pre-activation pattern (eyewear #105, lip-color #107,
waist-shape #106): measure whether the proposed axis is DEGENERATE on the frozen
24-item cohort BEFORE any GPU round-trip is frozen/queued.

DECLARED input gap surfaced by this probe: the proposal declares inputs
"seg2-defined ear/neck crop regions", but DOME-29 has NO Ear or Neck class
(Face_Neck = 3 is the only head/neck-adjacent region). So the honest proxy crop
is the FACE_NECK region, mirroring the eyewear #105 probe (Face_Neck crop +
CLIP ViT-L/14 zero-shot). This is recorded in the probe JSON as a scope note.

Bands: earrings present-absent and necklace present-absent, from CLIP ViT-L/14
zero-shot over the Face_Neck crop with a closed vocabulary. Degeneracy gate =
max_share >= 0.75 on either band, or coverage floor not met (pre-registered
>=8/24 with ANY jewelry so the axis isn't all-null).

Read-only (no corpus write), CPU-only (staged CLIP ViT-L/14 model), no GPU
claim, no registry mutation — probe results only.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from stratum2.config import DOME_29  # noqa: E402

MANIFEST = "/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json"
DERIVED = "/mnt/nas-ai-models/training-data/crawlr/stratum"
SOURCE = "/mnt/nas-ai-models/training-data/crawlr/approved"
MODEL_ASSET = "/mnt/nas-ai-models/research/stratum/models/scene-category"  # openai/clip-vit-large-patch14 staged
MODEL_SHA256 = "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb"

# DOME-29 class indices.
FACE_NECK = DOME_29.index("Face_Neck")

# Raw pixel floor for the crop region (mirror hair.py presence floor).
MIN_CLASS_PX = 200

# Closed vocabulary — the full set is scored every time.
EARRINGS_LABELS = ("wearing earrings", "not wearing earrings")
NECKLACE_LABELS = ("wearing a necklace", "not wearing a necklace")
PROMPT_PREFIX = "a photo of a woman"

# Coverage floor pre-registered in the declaration: axis must not be all-null.
MIN_ANY_JEWELRY = 8
DEGENERACY_GATE = 0.75


class _ClipRuntime:
    _processor = None
    _model = None

    @classmethod
    def get(cls):
        if cls._model is None:
            from transformers import CLIPModel, CLIPProcessor

            cls._processor = CLIPProcessor.from_pretrained(MODEL_ASSET)
            cls._model = CLIPModel.from_pretrained(MODEL_ASSET)
            cls._model.eval()
        return cls._processor, cls._model


def zero_shot(crop: np.ndarray, labels: tuple[str, ...]) -> tuple[float, float, list[float], list[float]]:
    """Return (argmax_prob, argmax_index, probs, logits) for the crop."""
    processor, model = _ClipRuntime.get()
    image = Image.fromarray(crop)
    texts = [f"{PROMPT_PREFIX} {lab}" for lab in labels]
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.get_image_features(pixel_values=inputs["pixel_values"])
        if not torch.is_tensor(out):
            out = out.pooler_output
        image_feat = out / out.norm(dim=-1, keepdim=True)
        text_out = model.get_text_features(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        if not torch.is_tensor(text_out):
            text_out = text_out.pooler_output
        text_feat = text_out / text_out.norm(dim=-1, keepdim=True)
        logit_scale = float(model.logit_scale.exp()) if hasattr(model, "logit_scale") else 100.0
        logits = logit_scale * (image_feat @ text_feat.T)
    logits = logits[0]
    probs = torch.softmax(logits, dim=-1).tolist()
    return max(probs), int(probs.index(max(probs))), probs, [float(v) for v in logits.tolist()]


def face_neck_crop(rgb: np.ndarray, seg2: np.ndarray) -> np.ndarray | None:
    mask = seg2 == FACE_NECK
    rows, cols = np.nonzero(mask)
    if rows.size < MIN_CLASS_PX:
        return None
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    h, w, _ = rgb.shape
    mh = max(1, int((r1 - r0) * 0.05))
    mw = max(1, int((c1 - c0) * 0.05))
    return rgb[max(0, r0 - mh):min(h, r1 + mh), max(0, c0 - mw):min(w, c1 + mw)]


def main() -> int:
    manifest = json.loads(Path(MANIFEST).read_text())
    rows = []
    for item in manifest["items"]:
        image_id = item["image_id"]
        segp = Path(DERIVED) / image_id / "seg2.npy"
        srcp = Path(SOURCE) / item["source_relative_path"]
        try:
            seg2 = np.load(segp, allow_pickle=False)
        except FileNotFoundError as exc:
            rows.append({"image_id": image_id, "abstained": True, "abstention_reason": f"seg2 missing: {exc.filename}"})
            print(f"{image_id[:12]}  ABSTAIN (seg2 missing)")
            continue
        try:
            img = np.asarray(Image.open(srcp).convert("RGB"), dtype=np.uint8)
            if img.shape[:2] != seg2.shape[:2]:
                img = np.asarray(Image.fromarray(img).resize((seg2.shape[1], seg2.shape[0]), Image.BILINEAR), dtype=np.uint8)
        except Exception as exc:  # noqa: BLE001
            rows.append({"image_id": image_id, "abstained": True, "abstention_reason": f"source read failed: {exc}"})
            print(f"{image_id[:12]}  ABSTAIN (source read failed: {exc})")
            continue
        crop = face_neck_crop(img, seg2)
        if crop is None:
            rows.append({"image_id": image_id, "abstained": True, "abstention_reason": "Face_Neck region below raw-pixel floor"})
            print(f"{image_id[:12]}  ABSTAIN (Face_Neck below px floor)")
            continue
        ear_prob, ear_idx, ear_probs, _ = zero_shot(crop, EARRINGS_LABELS)
        neck_prob, neck_idx, neck_probs, _ = zero_shot(crop, NECKLACE_LABELS)
        earrings = EARRINGS_LABELS[ear_idx] == "wearing earrings"
        necklace = NECKLACE_LABELS[neck_idx] == "wearing a necklace"
        rows.append({
            "image_id": image_id,
            "abstained": False,
            "face_neck_px": int((seg2 == FACE_NECK).sum()),
            "earrings_band": "earrings" if earrings else "none",
            "earrings_conf": round(ear_prob, 4),
            "earrings_probs": {k: round(v, 4) for k, v in zip(EARRINGS_LABELS, ear_probs)},
            "necklace_band": "necklace" if necklace else "none",
            "necklace_conf": round(neck_prob, 4),
            "necklace_probs": {k: round(v, 4) for k, v in zip(NECKLACE_LABELS, neck_probs)},
        })
        print(f"{image_id[:12]}  ear={'earrings' if earrings else 'none':<9}({ear_prob:.3f})  neck={'necklace' if necklace else 'none':<9}({neck_prob:.3f})")

    det = [r for r in rows if not r.get("abstained")]
    n = len(det)
    ear_bands = Counter(r["earrings_band"] for r in det)
    neck_bands = Counter(r["necklace_band"] for r in det)
    any_jewelry = sum(1 for r in det if r["earrings_band"] == "earrings" or r["necklace_band"] == "necklace")
    ear_max_share = max(ear_bands.values()) / n if n else 1.0
    neck_max_share = max(neck_bands.values()) / n if n else 1.0

    summary = {
        "measured": n,
        "abstained_or_error": len(rows) - n,
        "face_neck_region_available": sum(1 for r in det),
        "earrings_band_distribution": dict(ear_bands),
        "necklace_band_distribution": dict(neck_bands),
        "any_jewelry_count": any_jewelry,
        "any_jewelry_floor": MIN_ANY_JEWELRY,
        "any_jewelry_floor_met": any_jewelry >= MIN_ANY_JEWELRY,
        "earrings_max_share": round(ear_max_share, 4),
        "necklace_max_share": round(neck_max_share, 4),
        "degeneracy_gate": DEGENERACY_GATE,
    }
    print("\n=== JEWELRY CAPABILITY SUMMARY ===")
    print(json.dumps(summary, indent=1))
    out = {
        "dimension": "jewelry",
        "arm_issue": 112,
        "probe_type": "capability",
        "date": "2026-08-10",
        "model": "openai/clip-vit-large-patch14 (staged, sha256 a2bf730a0c...)",
        "crop_policy": "Face_Neck (DOME-29 class 3) — DOME-29 has NO Ear/Neck class, "
                       "so the declared 'seg2-defined ear/neck crop regions' are proxied by Face_Neck "
                       "(scope note: ear/neck regions not independently segmentable from DOME-29)",
        "summary": summary,
        "rows": rows,
    }
    outpath = Path("/mnt/nas-ai-models/research/stratum/jewelry-calibration-probe.json")
    outpath.write_text(json.dumps(out, indent=1))
    print(f"\nprobe artifact: {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
