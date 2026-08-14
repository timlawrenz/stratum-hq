"""Deterministic semantic scene-category measurement (CLIP ViT-L/14 zero-shot).

Arm #69. NEW-model-class specialist: runs the open-weight CLIP ViT-L/14
zero-shot text-image similarity model (`openai/clip-vit-large-patch14`, MIT
license, local CPU on owned hardware) over the full-frame decoded source RGB
and maps it to one frozen closed scene-category label:

- indoor studio / plain wall backdrop / bedroom / living room / outdoor beach /
  outdoor garden / outdoor field / body of water / urban street / poolside.

The frozen label set is COHORT-DERIVED from the arm-#47 VLM dense-description
scene vocabulary, so it covers the actual scene-dominant cohort rather than a
paper-typical "living room / office" set. Only the SCALE-INVARIANT category
label (or a surfaced abstention) is verbalized — a semantic scene category is
camera-frame-invariant. The CLIP similarity logits / softmax probabilities stay
in the machine-readable ``evidence_payload`` JSON and are never caption claims.

Abstention policy: emit the argmax category only when its softmax confidence
clears the calibrated threshold; below it the item abstains with a surfaced
reason (the classifier cannot confidently resolve the scene, so a label would
be a guess). Calibration measured on the frozen 24-item cohort 2026-08-07:
24/24 classified, 8 distinct categories (…), top-1 max share 29% (no band ≥
75%), p50 confidence ~0.543. The abstention floor is calibrated from the
cohort confidence distribution (see ``ABSTAIN_CONFIDENCE``) so a confident-
looking guess is never overwritten.

Provenance: open-weight model (openai/clip-vit-large-patch14, model.safetensors
sha256 pinned in the declaration; staged at the frozen model asset dir) run on
owned hardware only; no hosted third-party inference of the sensitive corpus;
no corpus write. model_asset_dir is dependency-injected so unit tests can point
at a fixture directory and the runner at the frozen model asset.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# ---------------------------------------------------------------------------
# Frozen closed scene-category set (cohort-derived from arm-#47 VLM
# dense-description scene vocabulary, 2026-08-07). Order does not matter; the
# full set is scored every time (closed-set argmax).
# ---------------------------------------------------------------------------
SCENE_CATEGORIES: tuple[str, ...] = (
    "indoor studio",
    "plain wall backdrop",
    "bedroom",
    "living room",
    "outdoor beach",
    "outdoor garden",
    "outdoor field",
    "body of water",
    "urban street",
    "poolside",
)

# CLIP zero-shot text prompt template. The blank is filled with the category.
# A consistent prefix keeps prompt-parity across all categories.
PROMPT_PREFIX = "a photo of a"

# Softmax temperature for the similarity logits (higher = sharper). Kept at 1.0
# — the raw softmax of logit_scale * cos-sim is the canonical CLIP zero-shot.
TEMPERATURE = 1.0

# Abstention floor for the argmax softmax confidence. CALIBRATED from the
# frozen-cohort probe (2026-08-07): cohort p50 confidence 0.543, min observed
# …; below this floor the classifier is not confident enough to commit to a
# scene label and the item abstains (honest, never a guess).
ABSTAIN_CONFIDENCE = 0.25

# Model asset (bind the sha256 in the declaration; dir injected by caller).
MODEL_SHA256 = "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb"

# Frozen model asset dir (arm #69).
SCENE_CATEGORY_MODEL_ASSET = "/mnt/nas-ai-models/research/stratum/models/scene-category"


class SceneCategoryError(RuntimeError):
    pass


def validate_rgb_array(rgb: np.ndarray) -> None:
    if not isinstance(rgb, np.ndarray):
        raise SceneCategoryError("rgb must be a numpy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise SceneCategoryError(f"rgb must be (H, W, 3), got shape {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise SceneCategoryError(f"rgb must be uint8, got dtype {rgb.dtype}")


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
    rgb: np.ndarray,
    *,
    model_asset_dir: str,
) -> tuple[list[float], list[float]]:
    """Return (softmax probabilities over SCENE_CATEGORIES, similarity logits).

    CLIP zero-shot: cosine sim between the image embedding and each frozen
    category text embedding, scaled by the model's learned logit_scale, then
    softmax over the closed set at the configured temperature.
    """
    processor, model = _ClipRuntime.get(model_asset_dir)
    import torch
    from PIL import Image

    image = Image.fromarray(rgb)
    texts = [f"{PROMPT_PREFIX} {cat}" for cat in SCENE_CATEGORIES]
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.get_image_features(pixel_values=inputs["pixel_values"])
        if not torch.is_tensor(out):
            out = out.pooler_output
        image_feat = out / out.norm(dim=-1, keepdim=True)
        text_out = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        if not torch.is_tensor(text_out):
            text_out = text_out.pooler_output
        text_feat = text_out / text_out.norm(dim=-1, keepdim=True)
        logit_scale = float(model.logit_scale.exp()) if hasattr(model, "logit_scale") else 100.0
        logits = logit_scale * (image_feat @ text_feat.T)
    logits = logits[0] / TEMPERATURE
    probs = torch.softmax(logits, dim=-1).tolist()
    return probs, [float(v) for v in logits.tolist()]


def compute_scene_category(
    rgb: np.ndarray,
    *,
    model_asset_dir: str = SCENE_CATEGORY_MODEL_ASSET,
) -> dict[str, Any]:
    """Compute the semantic scene category for a full-frame item.

    Args:
        rgb: (H, W, 3) uint8 decoded source pixels (full frame).
        model_asset_dir: absolute path to the frozen CLIP model asset dir.

    Returns a dict with ``abstained``, ``category`` (or None), argmax
    confidence, the softmax probability vector, and similarity logits.
    """
    validate_rgb_array(rgb)
    try:
        probs, logits = _zero_shot_probabilities(rgb, model_asset_dir=model_asset_dir)
    except Exception as exc:  # noqa: BLE001
        raise SceneCategoryError(f"CLIP inference failed: {exc!r}") from exc

    max_idx = int(np.argmax(probs))
    max_conf = float(probs[max_idx])
    if max_conf < ABSTAIN_CONFIDENCE:
        return {
            "abstained": True,
            "abstention_reason": (
                f"scene classification confidence {max_conf:.3f} below the "
                f"calibrated floor {ABSTAIN_CONFIDENCE:.2f}"
            ),
            "category": None,
            "confidence": round(max_conf, 4),
            "probabilities": [round(p, 4) for p in probs],
            "logits": [round(v, 4) for v in logits],
            "categories": list(SCENE_CATEGORIES),
            "abstain_confidence": ABSTAIN_CONFIDENCE,
        }

    return {
        "abstained": False,
        "detection": "DETECTED",
        "category": SCENE_CATEGORIES[max_idx],
        "confidence": round(max_conf, 4),
        "probabilities": [round(p, 4) for p in probs],
        "logits": [round(v, 4) for v in logits],
        "categories": list(SCENE_CATEGORIES),
        "abstain_confidence": ABSTAIN_CONFIDENCE,
    }


def render_scene_category(scene: Mapping[str, Any]) -> list[str]:
    """Scale-invariant scene-category claim (arm #69).

    Verbalizes ONLY the semantic category label (or a surfaced abstention).
    Logits / probabilities stay in the machine-readable payload and are never
    caption claims.
    """
    if scene.get("abstained"):
        reason = scene.get("abstention_reason") or "scene classification not confident"
        return [f"scene-category: abstain ({reason})"]
    if not scene or not scene.get("category"):
        # Dimension not measured for this item — never fabricate a claim.
        return []
    cat = scene["category"]
    conf = scene.get("confidence")
    if isinstance(conf, (int, float)) and conf < scene.get("abstain_confidence", 0.0):
        return [f"scene-category: abstain (confidence {conf:.3f} below floor)"]
    return [f"scene-category: the setting is a {cat}"]
