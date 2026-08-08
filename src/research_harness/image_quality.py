"""Deterministic no-reference image-quality measurement (zero-shot CLIP-IQA).

Arm #95. NEW-model-class specialist: runs the open-weight CLIP ViT-L/14
text-image similarity model (``openai/clip-vit-large-patch14``, same frozen
asset family already qualified for arm #69, MIT license, local CPU on owned
hardware) over the full-frame decoded source RGB and implements the
**zero-shot CLIP-IQA** scoring method (Wang et al., AAAI 2023,
arXiv:2207.10896): for each of the three frozen quality-aspect prompt pairs
(revision-2, the degenerate "good/bad" pair excluded per the band-degeneracy
rule) the softmax probability of the positive descriptor is computed from the
image-text cosine similarities, and the aspect scores are averaged into one
scale-invariant perceptual quality score in [0, 1].

The score is mapped to one coarse scale-invariant quality band:

- sharp (score >= SHARP_FLOOR)
- moderate (score >= MODERATE_FLOOR)
- degraded (score < MODERATE_FLOOR)

Only the coarse band (or a surfaced abstention) is verbalized. The raw score,
per-aspect probabilities, and similarity logits stay in the machine-readable
``evidence_payload`` JSON and are never caption claims (measurement-semantics
directive). Unlike arm #75 image-focus (deterministic gradient acutance / DOF),
CLIP-IQA grounds subjective perceptual quality ('crisp', 'grainy', 'high
quality') that acutance alone cannot express.

Abstention policy: abstain on model/input failure or an implausible / degenerate
score, never fabricate a quality class. Band floors are CALIBRATED from the
frozen 24-item cohort probe (2026-08-08) with the no-single-band->=75% rule
(band-degeneracy rule arm #34/#35/#59).

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
# Frozen CLIP-IQA quality-aspect prompt pairs (Wang et al. AAAI 2023,
# zero-shot variant). Each pair is (positive descriptor, negative descriptor);
# the per-aspect score is softmax over the two image-text similarities.
#
# REVISION-2 (2026-08-08, aspect-level band-degeneracy recovery after the
# strike-1 round-trip): the revision-1 aggregate averaged a ("Good photo.",
# "Bad photo.") pair that measured 22/24 items in the "good" bucket on the
# frozen cohort — 91.7% max share, over the pre-registered 0.75 band-degeneracy
# line. A near-constant component compresses the dynamic range of the aggregate
# score and dilutes the genuinely-discriminating aspects. Per the standing
# band-degeneracy rule (arm #34/#35/#59; uniform axes silenced — arm #74), the
# degenerate aspect is EXCLUDED and the aggregate is the mean of the three
# aspects below (measured max shares from the strike-1 run payloads on the SAME
# cohort: sharp/blurry 0.417, colorful/pale 0.375, bright/dim 0.708).
#
# Qualification re-gate (capability) for revision-2: the re-cut is probed on
# the photo-content degradation ladder and passes ONCE the band floors are
# re-calibrated to the 3-aspect score's lower absolute scale (the excluded
# "good" aspect contributed a ~0.9 constant offset). SHARP_FLOOR 0.60 → 0.55,
# MODERATE_FLOOR stays 0.35; with these floors the origin rungs land "sharp"
# and the worst rungs "degraded", restoring ladder monotonicity (verified by
# the revision-2 calibration probe). The revision-1 4-pair aggregate stays as
# history; the 3-aspect aggregate is the honest revision-2 measurement.
# ---------------------------------------------------------------------------
QUALITY_PROMPT_PAIRS: tuple[tuple[str, str], ...] = (
    ("Sharp photo.", "Blurry photo."),
    ("Colorful photo.", "Pale photo."),
    ("Bright photo.", "Dim photo."),
)

# Aspects measured in revision-1 but excluded from the revision-2 aggregate as
# degenerate (>75% single-bucket share on the frozen cohort). Documented so the
# reviewer sees the re-cut via the payload's excluded_degenerate_aspects field.
DEGENERATE_ASPECTS_EXCLUDED: tuple[tuple[str, str], ...] = (
    ("Good photo.", "Bad photo."),
)

# Softmax temperature for similarity logits (higher = sharper). Kept at 1.0 —
# raw softmax of logit_scale * cos-sim is the canonical CLIP zero-shot.
TEMPERATURE = 1.0

# Band floors, CALIBRATED from the frozen-cohort probe (2026-08-08, revision-2)
# with the no-band->=75% rule. REVISION-2: floors re-calibrated to the 3-aspect
# score's lower absolute scale (the excluded "good" aspect contributed a ~0.9
# constant offset to the revision-1 4-pair aggregate) — SHARP_FLOOR 0.60 → 0.55
# keeps the capability degradation ladder monotonic (origin rungs "sharp",
# worst rungs "degraded"); MODERATE_FLOOR stays 0.35.
SHARP_FLOOR = 0.55
MODERATE_FLOOR = 0.35

# Model asset (bind the sha256 in the declaration; dir injected by caller).
MODEL_SHA256 = "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb"

# Frozen model asset dir (arm #95).
IMAGE_QUALITY_MODEL_ASSET = "/mnt/nas-ai-models/research/stratum/models/image-quality"


class ImageQualityError(RuntimeError):
    pass


def validate_rgb_array(rgb: np.ndarray) -> None:
    if not isinstance(rgb, np.ndarray):
        raise ImageQualityError("rgb must be a numpy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ImageQualityError(f"rgb must be (H, W, 3), got shape {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise ImageQualityError(f"rgb must be uint8, got dtype {rgb.dtype}")


class _ClipRuntime:
    """Lazy, process-wide CLIP model + processor (CPU).

    Keyed on the requested model_asset_dir: a different asset dir reloads the
    model instead of silently reusing a previously-loaded one (each arm's
    frozen asset is bound by sha256, so cross-asset reuse must never happen).
    """

    _processor = None
    _model = None
    _loaded_dir: str | None = None

    @classmethod
    def get(cls, model_asset_dir: str):
        if cls._model is None or cls._loaded_dir != model_asset_dir:
            from transformers import CLIPModel, CLIPProcessor

            cls._processor = CLIPProcessor.from_pretrained(model_asset_dir)
            cls._model = CLIPModel.from_pretrained(model_asset_dir)
            cls._model.eval()
            cls._loaded_dir = model_asset_dir
        return cls._processor, cls._model

    @classmethod
    def reset(cls) -> None:
        cls._processor = None
        cls._model = None
        cls._loaded_dir = None


def _clip_iqa_score(
    rgb: np.ndarray,
    *,
    model_asset_dir: str,
) -> dict[str, Any]:
    """Return the zero-shot CLIP-IQA score dict (aspects, score, logits)."""
    processor, model = _ClipRuntime.get(model_asset_dir)
    import torch
    from PIL import Image

    image = Image.fromarray(rgb)
    aspect_scores: list[float] = []
    aspect_logits: list[list[float]] = []
    with torch.no_grad():
        image_inputs = processor(text=None, images=image, return_tensors="pt")
        image_feat = model.get_image_features(pixel_values=image_inputs["pixel_values"])
        if not torch.is_tensor(image_feat):
            image_feat = image_feat.pooler_output
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        logit_scale = float(model.logit_scale.exp()) if hasattr(model, "logit_scale") else 100.0
        for pos, neg in QUALITY_PROMPT_PAIRS:
            texts = [pos, neg]
            text_inputs = processor(text=texts, images=None, return_tensors="pt", padding=True)
            text_feat = model.get_text_features(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            if not torch.is_tensor(text_feat):
                text_feat = text_feat.pooler_output
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            logits = (logit_scale * (image_feat @ text_feat.T))[0] / TEMPERATURE
            probs = torch.softmax(logits, dim=-1)
            aspect_scores.append(float(probs[0]))
            aspect_logits.append([float(v) for v in logits.tolist()])
    score = float(np.mean(aspect_scores))
    return {
        "score": round(score, 4),
        "aspect_scores": [round(s, 4) for s in aspect_scores],
        "aspect_logits": [[round(v, 4) for v in pair] for pair in aspect_logits],
        "prompt_pairs": list(QUALITY_PROMPT_PAIRS),
    }


def _quality_band(score: float) -> str:
    if score >= SHARP_FLOOR:
        return "sharp"
    if score >= MODERATE_FLOOR:
        return "moderate"
    return "degraded"


def compute_image_quality(
    rgb: np.ndarray,
    *,
    model_asset_dir: str = IMAGE_QUALITY_MODEL_ASSET,
) -> dict[str, Any]:
    """Compute the no-reference perceptual-quality band for a full-frame item.

    Args:
        rgb: (H, W, 3) uint8 decoded source pixels (full frame).
        model_asset_dir: absolute path to the frozen CLIP model asset dir.

    Returns a dict with ``abstained``, ``quality_band`` (or None), the raw
    CLIP-IQA score, per-aspect probabilities, and similarity logits.
    """
    validate_rgb_array(rgb)
    try:
        result = _clip_iqa_score(rgb, model_asset_dir=model_asset_dir)
    except Exception as exc:  # noqa: BLE001
        raise ImageQualityError(f"CLIP-IQA inference failed: {exc!r}") from exc

    score = result["score"]
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        return {
            "abstained": True,
            "abstention_reason": (
                f"CLIP-IQA score {score!r} outside the plausible [0, 1] band"
            ),
            "quality_band": None,
            **result,
        }

    return {
        "abstained": False,
        "detection": "DETECTED",
        "quality_band": _quality_band(score),
        "sharp_floor": SHARP_FLOOR,
        "moderate_floor": MODERATE_FLOOR,
        "excluded_degenerate_aspects": list(DEGENERATE_ASPECTS_EXCLUDED),
        **result,
    }


def render_image_quality(quality: Mapping[str, Any]) -> list[str]:
    """Scale-invariant perceptual-quality claim (arm #95).

    Verbalizes ONLY the coarse quality band (or a surfaced abstention). The
    raw CLIP-IQA score and logits stay in the machine-readable payload and are
    never caption claims.
    """
    if quality.get("abstained"):
        reason = quality.get("abstention_reason") or "image-quality not confidently measured"
        return [f"image-quality: abstain ({reason})"]
    if not quality or not quality.get("quality_band"):
        # Dimension not measured for this item — never fabricate a claim.
        return []
    band = quality["quality_band"]
    if band == "sharp":
        return ["image-quality: the photo appears sharp and crisp"]
    if band == "moderate":
        return ["image-quality: the photo appears of moderate quality"]
    return ["image-quality: the photo appears degraded / low quality"]