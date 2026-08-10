"""Learned hair-texture / curl-waviness measurement (CLIP ViT-L/14 zero-shot).

Arm #94 — RE-SCOPED 2026-08-10 under the open-world sourcing directive
(2026-08-05). The deterministic gradient-orientation axis measured DEGENERATE
on the frozen cohort (2026-08-10 probe: 11/24 all one band, max_share 1.00;
13/24 honest abstains; five alternative deterministic discriminators
non-separating). The re-scope replaces the deterministic signal with a NEW
MODEL CLASS: the open-weight CLIP ViT-L/14 zero-shot classifier
(openai/clip-vit-large-patch14, MIT, local CPU on owned hardware) run over the
seg2 DOME-29 Hair-region crop with a closed straight/wavy/curly/coily
vocabulary.

Capability probe (hair-texture-clip-probe-20260810.json, 2026-08-10):
24/24 measured, bands straight 3 / wavy 13 / coily 6 / curly 2, max_share
0.542 (under the 0.75 degeneracy gate), argmax confidence median 0.632 /
min 0.373. The learned specialist resolves the axis the deterministic signal
could not.

ONLY the scale-invariant coarse band is verbalized. Raw CLIP probabilities /
logits stay in the machine-readable `evidence_payload` and are never caption
claims (measurement-semantics directive).

Abstention: abstains when the Hair region is absent/tiny/cropped, the crop is
empty, or the argmax softmax confidence falls below the calibrated floor
(0.35 — below the cohort min 0.373 so no honest item is dropped, but an
ambiguous near-tie never becomes a confident-looking guess). Never fabricate a
curl state; detector disagreement is a quality anomaly, never caption content.
CPU-only (CLIP on CPU, no VRAM contention with the caption model), in-memory,
no corpus write.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from stratum2.config import DOME_29

# DOME-29 class indices (authoritative in stratum2.config.DOME_29).
HAIR = DOME_29.index("Hair")

# Presence floor: a hair region must clear a raw pixel count (mirror hair.py).
MIN_CLASS_PX = 200

# Closed curl vocabulary (order fixed; the full set is scored every time).
TEXTURE_LABELS: tuple[str, ...] = ("straight hair", "wavy hair", "curly hair", "coily hair")
BAND_BY_LABEL: dict[str, str] = {
    "straight hair": "straight",
    "wavy hair": "wavy",
    "curly hair": "curly",
    "coily hair": "coily",
}

# Prompt template (consistent prefix keeps prompt-parity across labels).
PROMPT_PREFIX = "a photo of a woman with"

# Abstention floor for the argmax softmax confidence. CALIBRATED from the
# frozen-cohort CLIP probe (2026-08-10): argmax confidence min 0.373 /
# median 0.632 — the 0.35 floor sits below the honest minimum so no measured
# item is dropped, while a near-tie still abstains honestly.
ABSTAIN_CONFIDENCE = 0.35

# Crop margin fraction (frame edges of glasses/hairline excluded by the seg2
# mask; a small margin keeps CLIP grounded on the hair region).
CROP_MARGIN_FRACTION = 0.05

# Frozen model asset dir (same staged CLIP ViT-L/14 as scene-category #69).
HAIR_TEXTURE_MODEL_ASSET = "/mnt/nas-ai-models/research/stratum/models/scene-category"

# Model sha256 (bound in the declaration; same checkpoint as scene-category).
MODEL_SHA256 = "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb"


class HairTextureError(RuntimeError):
    pass


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise HairTextureError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise HairTextureError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise HairTextureError(f"seg2 must be integer class labels, got dtype {seg2.dtype}")


def _hair_crop(rgb: np.ndarray, seg2: np.ndarray) -> np.ndarray | None:
    """Crop the seg2 Hair region from the source RGB (with a small margin)."""
    mask = seg2 == HAIR
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
    *,
    model_asset_dir: str,
) -> tuple[list[float], list[float]]:
    """Return (softmax probabilities over TEXTURE_LABELS, similarity logits)."""
    processor, model = _ClipRuntime.get(model_asset_dir)
    import torch
    from PIL import Image

    image = Image.fromarray(rgb_crop)
    texts = [f"{PROMPT_PREFIX} {lab}" for lab in TEXTURE_LABELS]
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


def compute_hair_texture(
    seg2: np.ndarray,
    image_rgb: np.ndarray,
    *,
    min_px: int = MIN_CLASS_PX,
    model_asset_dir: str = HAIR_TEXTURE_MODEL_ASSET,
) -> dict[str, Any]:
    """Compute the learned hair-texture band with honest abstention.

    Args:
        seg2: (H, W) DOME-29 class labels at full source resolution.
        image_rgb: (H, W, 3) uint8 RGB source pixels, pixel-aligned to seg2.
        min_px: raw-pixel floor for the Hair region.
        model_asset_dir: absolute path to the frozen CLIP model asset dir.

    Returns a dict with scale-invariant facts only:
    - subject_present / hair_present / abstained / abstention_reason
    - hair_texture_band (straight / wavy / curly / coily) or None
    - raw payload: confidence, probabilities, logits, hair_px
    """
    validate_seg2_array(seg2)
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise HairTextureError("image_rgb must be an (H, W, 3) numpy array")
    if image_rgb.shape[0] != seg2.shape[0] or image_rgb.shape[1] != seg2.shape[1]:
        raise HairTextureError(
            f"image_rgb {image_rgb.shape} must be pixel-aligned with seg2 {seg2.shape}"
        )

    out: dict[str, Any] = {
        "subject_present": True,
        "abstained": False,
        "abstention_reason": None,
        "hair_present": False,
        "hair_texture_band": None,
        "confidence": None,
        "probabilities": None,
        "logits": None,
        "labels": list(TEXTURE_LABELS),
        "abstain_confidence": ABSTAIN_CONFIDENCE,
        "hair_px": 0,
    }

    fg_pixels = int((seg2 > 0).sum())
    if fg_pixels <= 0:
        out.update({
            "subject_present": False,
            "abstained": True,
            "abstention_reason": "no foreground subject present",
        })
        return out

    hair_mask = seg2 == HAIR
    hair_px = int(hair_mask.sum())
    out["hair_px"] = hair_px
    if hair_px < min_px:
        out.update({
            "abstained": True,
            "abstention_reason": "Hair region absent or below the raw-pixel floor",
        })
        return out
    out["hair_present"] = True

    crop = _hair_crop(image_rgb, seg2)
    if crop is None or crop.size == 0:
        out.update({
            "abstained": True,
            "abstention_reason": "hair crop empty after region extraction",
        })
        return out

    try:
        probs, logits = _zero_shot_probabilities(crop, model_asset_dir=model_asset_dir)
    except Exception as exc:  # noqa: BLE001
        raise HairTextureError(f"CLIP inference failed: {exc!r}") from exc

    max_idx = int(np.argmax(probs))
    max_conf = float(probs[max_idx])
    out["confidence"] = round(max_conf, 4)
    out["probabilities"] = [round(p, 4) for p in probs]
    out["logits"] = [round(v, 4) for v in logits]
    if max_conf < ABSTAIN_CONFIDENCE:
        out.update({
            "abstained": True,
            "abstention_reason": (
                f"hair-texture confidence {max_conf:.3f} below the calibrated "
                f"floor {ABSTAIN_CONFIDENCE:.2f}"
            ),
        })
        return out
    out["hair_texture_band"] = BAND_BY_LABEL[TEXTURE_LABELS[max_idx]]
    return out


def render_hair_texture(config: Mapping[str, Any]) -> list[str]:
    """Scale-invariant hair-texture claims for the dossier (arm #94).

    Verbalizes ONLY the coarse texture band. Raw CLIP probabilities / logits
    stay in the machine-readable payload.
    """
    if not config:
        # Dimension not measured for this item — emit no claim, never
        # fabricate a curl state.
        return []
    if config.get("abstained"):
        reason = config.get("abstention_reason") or "hair texture not measurable"
        return [f"hair-texture: abstain ({reason})"]
    if not config.get("hair_present"):
        return ["hair-texture: abstain (no hair region present)"]
    band = config.get("hair_texture_band")
    if band == "straight":
        return ["hair-texture: hair is straight (no curl wave pattern)"]
    if band == "wavy":
        return ["hair-texture: hair is wavy (gentle wave pattern)"]
    if band == "curly":
        return ["hair-texture: hair is curly (distinct curl pattern)"]
    if band == "coily":
        return ["hair-texture: hair is coily (tightly coiled texture)"]
    return []