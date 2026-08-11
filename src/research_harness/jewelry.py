"""Learned jewelry-presence measurement (CLIP ViT-L/14 zero-shot).

Arm #112 — registered 2026-08-10 via the gated propose-dimensions channel
(brainstorm-new-data — registry terminal at 30 validated). NEW evidence part
`jewelry` + NEW MODEL CLASS CLIP ViT-L/14 zero-shot over the seg2-defined
ear/neck crop region.

SCOPE NOTE (measured 2026-08-10): DOME-29 has NO Ear or Neck class, so the
declared "seg2-defined ear/neck crop regions" are proxied by the Face_Neck
region (DOME-29 class 3) — the only head/neck-adjacent region. This is
recorded in the capability probe artifact
(/mnt/nas-ai-models/research/stratum/jewelry-calibration-probe.json) and the
plan coverage notes.

Capability probe (2026-08-10, frozen 24-item cohort, CPU):
- 23/24 measured, 1 honest abstention (Face_Neck below the 200 px floor).
- earrings sub-band: 6 earrings / 17 none, max_share 0.739 < 0.75
  (NON-degenerate at the gate).
- necklace sub-band: 5 necklace / 18 none, max_share 0.783 >= 0.75
  (DEGENERATE) -> silenced to payload-only (arm #74 precedent).
- HONEST RE-SCOPE: the VERBALIZED axis is the merged jewelry-presence band
  (earrings OR necklace) 8 present / 15 absent of 23 measured, max_share
  0.652 NON-degenerate; coverage floor 8/24 MET (pre-registered >=8/24 with
  any jewelry so the axis isn't all-null).

ONLY the scale-invariant merged band is verbalized. Raw CLIP probabilities /
logits and the per-sub-axis bands stay in the machine-readable
`evidence_payload` and are never caption claims (measurement-semantics
directive).

Abstention: abstains when the Face_Neck region is absent/tiny/cropped, the
crop is empty, or the argmax softmax confidence falls below the calibrated
floor (0.45 — below the probe-observed honest minimum 0.5058 so no honest
item is dropped, but an ambiguous near-tie never becomes a confident-looking
guess). Never fabricate jewelry presence; detector disagreement is a quality
anomaly, never caption content. CPU-only (CLIP on CPU, no VRAM contention
with the caption model), in-memory, no corpus write.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from stratum2.config import DOME_29

# DOME-29 class indices (authoritative in stratum2.config.DOME_29).
FACE_NECK = DOME_29.index("Face_Neck")

# Presence floor: a face/neck region must clear a raw pixel count (mirror
# hair.py / the capability probe).
MIN_CLASS_PX = 200

# Closed vocabularies (order fixed; the full set is scored every time).
EARRINGS_LABELS: tuple[str, ...] = ("wearing earrings", "not wearing earrings")
NECKLACE_LABELS: tuple[str, ...] = ("wearing a necklace", "not wearing a necklace")

# Prompt template (consistent prefix keeps prompt-parity across labels).
PROMPT_PREFIX = "a photo of a woman"

# Abstention floor for the argmax softmax confidence. CALIBRATED from the
# frozen-cohort CLIP probe (2026-08-10): argmax confidence min 0.5058
# (necklace sub-band) / 0.5232 (earrings) — the 0.45 floor sits below the
# honest minimum so no measured item is dropped, while a genuine near-tie
# still abstains honestly.
ABSTAIN_CONFIDENCE = 0.45

# Crop margin fraction (frame edges excluded by the seg2 mask; a small margin
# keeps CLIP grounded on the face/neck region).
CROP_MARGIN_FRACTION = 0.05

# Frozen model asset dir (same staged CLIP ViT-L/14 as scene-category #69).
JEWELRY_MODEL_ASSET = "/mnt/nas-ai-models/research/stratum/models/scene-category"

# Model sha256 (bound in the declaration; same checkpoint as scene-category).
MODEL_SHA256 = "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb"


class JewelryError(RuntimeError):
    pass


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise JewelryError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise JewelryError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise JewelryError(f"seg2 must be integer class labels, got dtype {seg2.dtype}")


def _face_neck_crop(rgb: np.ndarray, seg2: np.ndarray) -> np.ndarray | None:
    """Crop the seg2 Face_Neck region from the source RGB (with a small margin)."""
    mask = seg2 == FACE_NECK
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    h, w, _ = rgb.shape
    mh = max(1, int((r1 - r0) * CROP_MARGIN_FRACTION))
    mw = max(1, int((c1 - c0) * CROP_MARGIN_FRACTION))
    return rgb[max(0, r0 - mh):min(h, r1 + mh), max(0, c0 - mw):min(w, c1 + mw)]


class _ClipRuntime:
    """Lazy, process-wide CLIP model + processor (CPU)."""

    _processor = None
    _model = None

    @classmethod
    def get(cls, model_asset_dir: str):
        if cls._model is None:
            from transformers import CLIPModel, CLIPProcessor

            cls._processor = CLIPProcessor.from_pretrained(model_asset_dir)
            cls._model = CLIPModel.from_pretrained(model_asset_dir)
            cls._model.eval()
        return cls._processor, cls._model

    @classmethod
    def reset(cls) -> None:
        cls._processor = None
        cls._model = None


def _zero_shot_probabilities(
    rgb_crop: np.ndarray,
    labels: tuple[str, ...],
    *,
    model_asset_dir: str,
) -> tuple[list[float], list[float]]:
    """Return (softmax probabilities over `labels`, similarity logits)."""
    processor, model = _ClipRuntime.get(model_asset_dir)
    import torch
    from PIL import Image

    image = Image.fromarray(rgb_crop)
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
    return probs, [float(v) for v in logits.tolist()]


def compute_jewelry(
    seg2: np.ndarray,
    image_rgb: np.ndarray,
    *,
    min_px: int = MIN_CLASS_PX,
    model_asset_dir: str = JEWELRY_MODEL_ASSET,
) -> dict[str, Any]:
    """Compute the learned jewelry-presence band with honest abstention.

    Args:
        seg2: (H, W) DOME-29 class labels at full source resolution.
        image_rgb: (H, W, 3) uint8 RGB source pixels, pixel-aligned to seg2.
        min_px: raw-pixel floor for the Face_Neck region.
        model_asset_dir: absolute path to the frozen CLIP model asset dir.

    Returns a dict with scale-invariant facts only:
    - subject_present / abstained / abstention_reason
    - jewelry_band (jewelry-present / no-jewelry) or None — the ONLY
      verbalized axis (merged earrings-OR-necklace, honest re-scope)
    - raw payload: earrings/necklace sub-bands (payload-only, never prose),
      confidences, probabilities, logits, face_neck_px
    """
    validate_seg2_array(seg2)
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise JewelryError("image_rgb must be an (H, W, 3) numpy array")
    if image_rgb.shape[0] != seg2.shape[0] or image_rgb.shape[1] != seg2.shape[1]:
        raise JewelryError(
            f"image_rgb {image_rgb.shape} must be pixel-aligned with seg2 {seg2.shape}"
        )

    out: dict[str, Any] = {
        "subject_present": True,
        "abstained": False,
        "abstention_reason": None,
        "jewelry_band": None,
        "earrings_band": None,
        "necklace_band": None,
        "confidence": None,
        "earrings_confidence": None,
        "necklace_confidence": None,
        "earrings_probabilities": None,
        "necklace_probabilities": None,
        "earrings_logits": None,
        "necklace_logits": None,
        "labels": {"earrings": list(EARRINGS_LABELS), "necklace": list(NECKLACE_LABELS)},
        "abstain_confidence": ABSTAIN_CONFIDENCE,
        "face_neck_px": 0,
    }

    fg_pixels = int((seg2 > 0).sum())
    if fg_pixels <= 0:
        out.update({
            "subject_present": False,
            "abstained": True,
            "abstention_reason": "no foreground subject present",
        })
        return out

    face_neck_mask = seg2 == FACE_NECK
    face_neck_px = int(face_neck_mask.sum())
    out["face_neck_px"] = face_neck_px
    if face_neck_px < min_px:
        out.update({
            "abstained": True,
            "abstention_reason": "Face_Neck region absent or below the raw-pixel floor",
        })
        return out

    crop = _face_neck_crop(image_rgb, seg2)
    if crop is None or crop.size == 0:
        out.update({
            "abstained": True,
            "abstention_reason": "Face_Neck crop is empty",
        })
        return out

    earrings_probs, earrings_logits = _zero_shot_probabilities(
        crop, EARRINGS_LABELS, model_asset_dir=model_asset_dir
    )
    necklace_probs, necklace_logits = _zero_shot_probabilities(
        crop, NECKLACE_LABELS, model_asset_dir=model_asset_dir
    )

    earrings_idx = int(max(range(len(earrings_probs)), key=lambda i: earrings_probs[i]))
    necklace_idx = int(max(range(len(necklace_probs)), key=lambda i: necklace_probs[i]))
    earrings_conf = float(earrings_probs[earrings_idx])
    necklace_conf = float(necklace_probs[necklace_idx])
    earrings_band = "earrings" if earrings_idx == 0 else "none"
    necklace_band = "necklace" if necklace_idx == 0 else "none"

    out.update({
        "earrings_band": earrings_band,
        "necklace_band": necklace_band,
        "earrings_confidence": earrings_conf,
        "necklace_confidence": necklace_conf,
        "earrings_probabilities": earrings_probs,
        "necklace_probabilities": necklace_probs,
        "earrings_logits": earrings_logits,
        "necklace_logits": necklace_logits,
    })

    # Sub-bands are payload-only. The verbalized axis is the MERGED
    # jewelry-presence band (earrings OR necklace) — the honest re-scope from
    # the capability probe (necklace sub-band 0.783 >= 0.75 degenerate;
    # merged 0.652 non-degenerate; coverage floor 8/24 met).
    if earrings_conf < ABSTAIN_CONFIDENCE and necklace_conf < ABSTAIN_CONFIDENCE:
        out.update({
            "abstained": True,
            "abstention_reason": (
                "both sub-band argmax confidences below the calibrated floor"
            ),
        })
        return out

    if earrings_band == "earrings" or necklace_band == "necklace":
        out["jewelry_band"] = "jewelry-present"
        out["confidence"] = max(earrings_conf, necklace_conf)
    else:
        out["jewelry_band"] = "no-jewelry"
        out["confidence"] = max(earrings_conf, necklace_conf)
    return out


def render_jewelry(config: Mapping[str, Any] | None) -> str:
    """Deterministic natural-language rendering of a jewelry dict.

    Verbalizes ONLY the coarse merged scale-invariant band. Raw CLIP
    probabilities / logits and the per-sub-axis bands stay in the
    machine-readable evidence_payload JSON and are not caption claims.
    """
    lines = [
        "JEWELRY-PRESENCE (CLIP ViT-L/14 zero-shot over the face/neck region, scale-invariant):"
    ]
    if not config:
        lines.append("- jewelry presence not measured for this item")
        return "\n".join(lines)
    if config.get("abstained"):
        reason = config.get("abstention_reason") or "jewelry presence not confident"
        lines.append(f"- jewelry presence abstained ({reason})")
        return "\n".join(lines)
    if not config.get("subject_present"):
        lines.append("- jewelry presence abstained (no foreground subject present)")
        return "\n".join(lines)
    band = config.get("jewelry_band")
    if band == "jewelry-present":
        lines.append(
            "- subject IS wearing jewelry (earrings and/or a necklace detected); "
            "only describe earrings/necklace the evidence names, never specific "
            "materials or brands"
        )
    elif band == "no-jewelry":
        lines.append(
            "- NO jewelry detected (no earrings and no necklace visible); "
            "do NOT describe earrings, a necklace, or any other jewelry"
        )
    return "\n".join(lines)