"""Bounded, noncanonical Stage-B caption-parity execution.

This module intentionally keeps the experiment separate from production corpus
artifacts: it reads only the frozen candidate manifest's selected inputs and
publishes all generated outputs under a new noncanonical research root.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image

from stratum.config import CAPTION_PROMPT, DEFAULT_ASPECT_BUCKETS
from stratum.pipeline.bucket import assign_aspect_bucket, parse_bucket_dims
from stratum.pipeline.caption import ensure_single_paragraph
from stratum2.pipeline.caption2 import CAPTION2_PROMPT_TEMPLATE, build_prompt
from stratum2.pipeline.determinations import derive_determinations

from .dossier import (
    assemble_dossier,
    build_evidence_payload,
    compress_dossier_to_context,
    context4k_text,
)
from .dossier_expand import expand_dossier

from .contracts import ContractError, validate_comparison_parity_plan, validate_program
from .clothing import ClothingError, compute_clothing
from .proportions import ProportionError, compute_proportions
from .hair import HairError, compute_hair
from .skin_color import SkinColorError, compute_skin_tone
from .lighting import LightingError, compute_lighting
from .setting import SettingError, compute_setting
from .texture import TextureError, compute_texture
from .pose_articulation import PoseArticulationError, compute_pose_articulation
from .pointmap_depth import PointmapDepthError, compute_pointmap_depth
from .matting_alpha import MattingAlphaError, compute_matting_alpha
from .face_geometry import FaceGeometryError, compute_face_geometry
from .object_relations import ObjectRelationsError, compute_object_relations
from .scene_category import SceneCategoryError, compute_scene_category
from .gaze_head import GazeHeadError, compute_gaze_head, GAZE_HEAD_MODEL_ASSET
from .camera_viewing_angle import (
    CameraViewingAngleError,
    compute_camera_viewing_angle,
)
from .image_focus import (
    ImageFocusError,
    compute_image_focus,
)


class StageBRunError(RuntimeError):
    """Raised when a Stage-B run cannot preserve its frozen safety contract."""


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_OUTPUT_SEGMENT = re.compile(r"[A-Za-z0-9._-]+")
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
_REQUIRED_REVIEW_FIELDS = (
    "supported_claims",
    "unsupported_claims",
    "omissions",
    "contradictions",
    "abstentions",
)
_CONTEXT_PROMPT_TEMPLATE = """You are an expert descriptive captioner for a text-to-image dataset.
Your task is to write a single, rich, dense paragraph describing the provided image.

DECLARED SPECIALIST EVIDENCE:
{evidence_text}

Use declared specialist evidence only as bounded support; do not turn absent evidence into a claim.
Add what declared evidence omits from the image itself: mood, lighting quality, color palette,
fabric, texture, skin details, expression, background, and any posture/activity that is visually
obvious and not contradicted by the declared evidence.

Write strictly objective prose. No conversational filler, no preambles like "This image shows".
Start the description immediately.
"""


@dataclass(frozen=True)
class StageBGenerationSettings:
    """Pinned local Ollama request settings for every comparison condition."""

    endpoint: str
    model_name: str
    model_digest: str
    temperature: float
    seed: int
    num_predict: int
    top_k: int
    top_p: float
    context_window: int
    timeout_seconds: int

    def __post_init__(self) -> None:
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in _LOCAL_HOSTS:
            raise StageBRunError("Stage-B aggregator endpoint must be a local loopback HTTP endpoint")
        if parsed.path != "/api/generate":
            raise StageBRunError("Stage-B aggregator endpoint must target /api/generate")
        if not self.model_name or self.model_name != self.model_name.strip() or any(
            character.isspace() for character in self.model_name
        ):
            raise StageBRunError("Stage-B model_name must be a canonical non-empty identifier")
        if not _SHA256_RE.fullmatch(self.model_digest):
            raise StageBRunError("Stage-B model_digest must be a lowercase SHA-256 digest")
        if self.temperature != 0.0 or self.top_k != 1 or self.top_p != 1.0:
            raise StageBRunError(
                "Stage-B requires deterministic sampling: temperature=0, top_k=1, top_p=1"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise StageBRunError("Stage-B seed must be a non-negative integer")
        if isinstance(self.num_predict, bool) or not isinstance(self.num_predict, int) or self.num_predict <= 0:
            raise StageBRunError("Stage-B num_predict must be a positive integer")
        if isinstance(self.context_window, bool) or not isinstance(self.context_window, int) or self.context_window < self.num_predict:
            raise StageBRunError("Stage-B context_window must be an integer no smaller than num_predict")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, int) or self.timeout_seconds <= 0:
            raise StageBRunError("Stage-B timeout_seconds must be a positive integer")

    @property
    def fingerprint(self) -> str:
        return _sha256(_canonical_json(asdict(self)).encode("utf-8"))

    def request_options(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "seed": self.seed,
            "num_predict": self.num_predict,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "num_ctx": self.context_window,
        }


Generator = Callable[[Image.Image, str, StageBGenerationSettings], str]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_fingerprint(value: Mapping[str, Any], field: str) -> str:
    payload = {key: item for key, item in value.items() if key != field}
    return _sha256(_canonical_json(payload).encode("utf-8"))


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageBRunError(f"{label} must be a mapping")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise StageBRunError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _safe_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise StageBRunError(f"{label} must be a non-empty normalized POSIX path")
    if "\\" in value or "\x00" in value:
        raise StageBRunError(f"{label} must be a normalized POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise StageBRunError(f"{label} must stay under its declared root")
    if path.as_posix() != value:
        raise StageBRunError(f"{label} must be normalized")
    return value


def _safe_output_segment(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_OUTPUT_SEGMENT.fullmatch(value):
        raise StageBRunError(f"{label} must be a safe output path segment")
    return value


def _resolved_existing_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise StageBRunError(f"{label} must be an existing directory: {path}") from exc
    if not resolved.is_dir():
        raise StageBRunError(f"{label} must be an existing directory: {path}")
    return resolved


def _require_contained(path: Path, root: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise StageBRunError(f"{label} is unavailable: {path}") from exc
    if not resolved.is_relative_to(root):
        raise StageBRunError(f"{label} escapes its declared root")
    return resolved


def _component(component_id: str, descriptor: Mapping[str, Any]) -> dict[str, str]:
    payload = {"id": component_id, **dict(descriptor)}
    return {"id": component_id, "fingerprint": _sha256(_canonical_json(payload).encode("utf-8"))}


def _evidence_fingerprint(evidence: Mapping[str, Any]) -> str:
    return _canonical_fingerprint(evidence, "fingerprint")


def _no_specialist_evidence() -> dict[str, Any]:
    evidence: dict[str, Any] = {"kind": "none", "id": "no-specialist-evidence-v1"}
    evidence["fingerprint"] = _evidence_fingerprint(evidence)
    return evidence


def _body_type_evidence() -> dict[str, Any]:
    """Declared deterministic body-type / proportion specialist (arm #32)."""
    module_path = Path(compute_proportions.__code__.co_filename)
    code_hash = _sha256(module_path.read_bytes())
    evidence: dict[str, Any] = {
        "kind": "specialist_bundle",
        "id": "in-memory-body-type-proportions-v1",
        "specialists": [
            {
                "id": "in-memory-body-type-proportions-v1",
                "scope": "Pose2 Goliath-308 derived continuous anthropometric proportions only (shoulder/hip widths, ratios, torso/limb lengths); never body-type labels or posture/activity semantics.",
                "inputs": "Frozen selected-item pose2.npy only; recomputed in memory during this bounded run with no crawlr/stratum write.",
                "output_semantics": "Provenance-bearing continuous ratio measurements or explicit abstention, not semantic ground truth or caption claims.",
                "provenance": (
                    "research_harness.proportions.compute_proportions "
                    f"SHA-256 {code_hash}; computed in memory during this bounded run with no crawlr/stratum write."
                ),
                "abstention_policy": "Abort the selected item before model generation if required artifacts are missing, unreadable, or detector count is not exactly one; every ratio emits None (never fabricated) when its supporting joints are absent or below confidence threshold; detector disagreement remains a quality anomaly, never prompt content.",
                "known_failure_modes": "Tight crops, low-keypoint-confidence frames, and partially visible limbs make ratios abstain; width ratios are in pixel units and depend on camera frame, so absolute widths are not metric.",
                "qualification_gate": "Candidate evidence only; no effectiveness claim is permitted until the frozen comparison receives completed rubric and adversarial reviews.",
            }
        ],
    }
    evidence["fingerprint"] = _evidence_fingerprint(evidence)
    return evidence


def _serialize_proportions(proportions: Mapping[str, Any]) -> str:
    """Deterministic natural-language rendering of a proportions measurement dict.

    Only **scale-invariant** ratios are verbalized (shoulder:hip, leg:torso,
    limb asymmetry) plus explicit abstention lines. Absolute pixel measurements
    are deliberately NOT verbalized: they are camera-frame-dependent, do not
    survive cross-picture comparison, and a text-to-image model cannot
    interpret them. Raw pixel values still exist in the machine-readable
    `evidence_payload` JSON (dossier / compressor input) as
    between_shoulders/between_hips/torso_length/lengths (px), but they are not
    caption claims.
    """
    lines = [
        "BODY-TYPE PROPORTIONS (deterministic, scale-invariant ratios from Goliath-308 pose2 keypoints):"
    ]
    if not proportions.get("subject_present"):
        lines.append("- no reliable body-keypoint subject present -> abstain from body-type claims")
        return "\n".join(lines)

    ratio_items: tuple[tuple[str, str], ...] = (
        ("shoulder_hip_ratio", "shoulder:hip width ratio"),
        ("leg_torso_ratio", "mean leg:torso length ratio"),
    )
    for key, label in ratio_items:
        value = proportions.get(key)
        if value is None:
            reason = None
            if key == "shoulder_hip_ratio":
                reason = proportions.get("shoulder_hip_ratio_abstention_reason")
            if reason:
                lines.append(f"- {label}: not measurable — {reason}")
            else:
                lines.append(f"- {label}: not measurable (joint absent or low confidence)")
        else:
            # Emit as a human-interpretable ratio description, not a bare number
            lines.append(f"- {label}: {value:.2f}")

    # Limb asymmetry is a scale-invariant relational fact: same-frame ratio.
    llen = proportions.get("left_leg_length")
    rlen = proportions.get("right_leg_length")
    if llen is not None and rlen is not None and rlen > 0:
        asym = llen / rlen
        direction = "left leg longer" if asym > 1.02 else ("right leg longer" if asym < 0.98 else "legs of similar length")
        lines.append(f"- leg length asymmetry (left:right): {direction} (ratio {asym:.2f})")
    else:
        lines.append("- leg length asymmetry: not measurable (one or both legs absent or low confidence)")
    return "\n".join(lines)


def _clothing_evidence() -> dict[str, Any]:
    """Declared deterministic clothing/apparel specialist (arm #29)."""
    module_path = Path(compute_clothing.__code__.co_filename)
    code_hash = _sha256(module_path.read_bytes())
    evidence: dict[str, Any] = {
        "kind": "specialist_bundle",
        "id": "in-memory-clothing-apparel-v1",
        "specialists": [
            {
                "id": "in-memory-clothing-apparel-v1",
                "scope": "seg2 DOME-29 clothing/apparel classes only (Apparel, Upper_Clothing, Lower_Clothing, Socks, Shoes) plus per-class dominant color from source pixels; never body-type, posture, or identity semantics.",
                "inputs": "Frozen selected-item seg2.npy and the source RGB pixels already decoded by this bounded run; recomputed in memory with no crawlr/stratum write.",
                "output_semantics": "Provenance-bearing continuous coverage fractions and deterministic dominant colors per garment class, or explicit abstention, not semantic ground truth or caption claims.",
                "provenance": (
                    "research_harness.clothing.compute_clothing "
                    f"SHA-256 {code_hash}; computed in memory during this bounded run with no crawlr/stratum write."
                ),
                "abstention_policy": "Abort the selected item before model generation if required artifacts are missing, unreadable, or detector count is not exactly one; a garment class is measured only when it clears a raw-pixel floor and a foreground-coverage gate, otherwise it abstains (never fabricated); detector disagreement remains a quality anomaly, never prompt content.",
                "known_failure_modes": "Tight crops, segmentation errors, and heavily occluded garments can make classes abstain; dominant colors depend on lighting and white balance; generic Apparel class may not distinguish garment silhouettes.",
                "qualification_gate": "Candidate evidence only; no effectiveness claim is permitted until the frozen comparison receives completed rubric and adversarial reviews.",
            }
        ],
    }
    evidence["fingerprint"] = _evidence_fingerprint(evidence)
    return evidence


def _serialize_clothing(clothing: Mapping[str, Any]) -> str:
    """Deterministic natural-language rendering of a clothing measurement dict.

    Verbalizes only scale-invariant, caption-relevant facts: which garment
    classes are present, their subject-foreground coverage, and the quantized
    dominant color. Absolute pixel counts and raw RGB are deliberately NOT
    verbalized (camera/size-dependent); they exist in the machine-readable
    `evidence_payload` JSON (dossier / compressor input).
    """
    lines = [
        "CLOTHING/APPAREL (deterministic, seg2 DOME-29 garment classes + source pixel dominant colors):"
    ]
    if not clothing.get("subject_present"):
        lines.append("- no reliable foreground subject present -> abstain from clothing claims")
        return "\n".join(lines)

    garments = clothing.get("garments") or []
    if not garments:
        lines.append(
            "- no garment class cleared the measurement gate -> abstain from clothing/apparel claims "
            "(exposed skin is not inferred as an absence of clothing)"
        )
    for garment in garments:
        coverage = garment.get("coverage")
        color = garment.get("dominant_color_name")
        color_text = f", dominant color {color}" if color else ""
        coverage_text = f"{coverage:.2f} of subject foreground" if coverage is not None else ""
        lines.append(
            f"- {garment['class'].replace('_', ' ')} present{color_text}"
            + (f" ({coverage_text})" if coverage_text else "")
        )
    return "\n".join(lines)


def _hair_evidence() -> dict[str, Any]:
    """Declared deterministic hair specialist (arm #30)."""
    module_path = Path(compute_hair.__code__.co_filename)
    code_hash = _sha256(module_path.read_bytes())
    evidence: dict[str, Any] = {
        "kind": "specialist_bundle",
        "id": "in-memory-hair-v1",
        "specialists": [
            {
                "id": "in-memory-hair-v1",
                "scope": "seg2 DOME-29 Hair(4) region only: coverage of subject foreground, dominant hair color from source pixels, vertical position band, and a hair-to-face vertical-extent length proxy; never facial, identity, or posture semantics.",
                "inputs": "Frozen selected-item seg2.npy and the source RGB pixels already decoded by this bounded run; recomputed in memory with no crawlr/stratum write.",
                "output_semantics": "Provenance-bearing continuous coverage fractions, a deterministic dominant color, and scale-invariant relational ratios or explicit abstention, not semantic ground truth or caption claims.",
                "provenance": (
                    "research_harness.hair.compute_hair "
                    f"SHA-256 {code_hash}; computed in memory during this bounded run with no crawlr/stratum write."
                ),
                "abstention_policy": "Abort the selected item before model generation if required artifacts are missing, unreadable, or detector count is not exactly one; the Hair region is measured only when it clears a raw-pixel floor and a foreground-coverage gate, otherwise it abstains (never fabricated); detector disagreement remains a quality anomaly, never prompt content.",
                "known_failure_modes": "Tight crops and segmentation errors can make the Hair region abstain or under-measure; dominant color depends on lighting and white balance and is a quantized name, not a spectral measurement; a hair-to-face ratio is undefined when the Face_Neck region is absent/degenerate.",
                "qualification_gate": "Candidate evidence only; no effectiveness claim is permitted until the frozen comparison receives completed rubric and adversarial reviews.",
            }
        ],
    }
    evidence["fingerprint"] = _evidence_fingerprint(evidence)
    return evidence


def _serialize_hair(hair: Mapping[str, Any]) -> str:
    """Deterministic natural-language rendering of a hair measurement dict.

    Verbalizes only scale-invariant, caption-relevant facts: hair presence,
    subject-foreground coverage, quantized dominant color, vertical band, and
    the hair-to-face extent ratio (length proxy). Absolute pixel counts and raw
    RGB are deliberately NOT verbalized (camera/size-dependent); they exist in
    the machine-readable `evidence_payload` JSON (dossier / compressor input).
    """
    lines = [
        "HAIR (deterministic, seg2 DOME-29 Hair(4) region + source pixel dominant color):"
    ]
    if not hair.get("subject_present"):
        lines.append("- no reliable foreground subject present -> abstain from hair claims")
        return "\n".join(lines)
    if not hair.get("hair_present"):
        lines.append(
            "- no reliable hair region cleared the measurement gate -> abstain from hair presence/color/detail claims "
            "(an absent hair mask is not evidence the subject is bald)"
        )
        return "\n".join(lines)

    coverage = hair.get("hair_coverage")
    color = hair.get("hair_dominant_color_name")
    position = hair.get("hair_position")
    ratio = hair.get("hair_face_extent_ratio")
    if coverage is not None:
        lines.append(f"- hair present, covering {coverage:.2f} of subject foreground")
    if color:
        lines.append(f"- hair dominant color: {color}")
    if position:
        lines.append(f"- hair occupies the {position} region of the frame")
    if ratio is not None:
        lines.append(f"- hair-to-face vertical extent ratio (length proxy): {ratio:.2f}")
    return "\n".join(lines)


def _skin_color_evidence() -> dict[str, Any]:
    """Declared deterministic skin-tone specialist (arm #31)."""
    module_path = Path(compute_skin_tone.__code__.co_filename)
    code_hash = _sha256(module_path.read_bytes())
    evidence: dict[str, Any] = {
        "kind": "specialist_bundle",
        "id": "in-memory-skin-tone-v1",
        "specialists": [
            {
                "id": "in-memory-skin-tone-v1",
                "scope": "seg2 DOME-29 exposed-skin regions only (Face_Neck, Torso, and limb/hand/foot skin classes) plus dominant skin tone from source pixels; never identity, posture, or facial-expression semantics.",
                "inputs": "Frozen selected-item seg2.npy and the source RGB pixels already decoded by this bounded run; recomputed in memory with no crawlr/stratum write.",
                "output_semantics": "Provenance-bearing continuous exposure coverage fractions and a deterministic quantized dominant skin-tone name/hex, or explicit abstention, not semantic ground truth or caption claims.",
                "provenance": (
                    "research_harness.skin_color.compute_skin_tone "
                    f"SHA-256 {code_hash}; computed in memory during this bounded run with no crawlr/stratum write."
                ),
                "abstention_policy": "Abort the selected item before model generation if required artifacts are missing, unreadable, or detector count is not exactly one; skin tone is measured only when the aggregate exposed-skin region clears a raw-pixel floor and a foreground-coverage gate, otherwise it abstains (never fabricated); detector disagreement remains a quality anomaly, never prompt content.",
                "known_failure_modes": "Skin tone depends on lighting and white balance and is a quantized palette name, not a spectral measurement; tight crops, makeup, or tan lines can shift the mean; regions covered by clothing/garments abstain from contributing; darker tones compress under low key light.",
                "qualification_gate": "Candidate evidence only; no effectiveness claim is permitted until the frozen comparison receives completed rubric and adversarial reviews.",
            }
        ],
    }
    evidence["fingerprint"] = _evidence_fingerprint(evidence)
    return evidence


def _serialize_skin_color(skin: Mapping[str, Any]) -> str:
    """Deterministic natural-language rendering of a skin-tone measurement dict.

    Verbalizes only scale-invariant, caption-relevant facts: whether exposed
    skin is measurable, its subject-foreground coverage, and the quantized
    dominant tone name (overall, plus face vs body where both measured).
    Absolute pixel counts and raw RGB are deliberately NOT verbalized
    (camera/size/white-balance dependent); they exist in the machine-readable
    `evidence_payload` JSON (dossier / compressor input).
    """
    lines = [
        "SKIN TONE (deterministic, seg2 DOME-29 exposed-skin regions + source pixel dominant tone):"
    ]
    if not skin.get("subject_present"):
        lines.append("- no reliable foreground subject present -> abstain from skin-tone claims")
        return "\n".join(lines)
    if not skin.get("exposed_skin_present"):
        lines.append(
            "- no exposed-skin region cleared the measurement gate -> abstain from skin-tone claims "
            "(covered skin is not inferred as an absence of skin tone)"
        )
        return "\n".join(lines)

    coverage = skin.get("skin_coverage")
    tone = skin.get("skin_tone_name")
    face_tone = skin.get("face_tone_name")
    body_tone = skin.get("body_tone_name")
    if coverage is not None:
        lines.append(f"- exposed skin covers {coverage:.2f} of subject foreground")
    if tone:
        lines.append(f"- dominant skin tone: {tone}")
    if face_tone and face_tone != tone:
        lines.append(f"- face/neck tone: {face_tone}")
    if body_tone and body_tone != tone:
        lines.append(f"- body tone: {body_tone}")
    return "\n".join(lines)


def _lighting_evidence() -> dict[str, Any]:
    """Declared deterministic lighting specialist (arm #33)."""
    module_path = Path(compute_lighting.__code__.co_filename)
    code_hash = _sha256(module_path.read_bytes())
    evidence: dict[str, Any] = {
        "kind": "specialist_bundle",
        "id": "in-memory-lighting-v1",
        "specialists": [
            {
                "id": "in-memory-lighting-v1",
                "scope": "Camera-space normal2 direction statistics over the subject plus scale-invariant source luminance/histogram/dynamic-range stats only; never identity, posture, or facial-expression semantics.",
                "inputs": "Frozen selected-item normal2.npy + seg2.npy and the source RGB pixels already decoded by this bounded run; recomputed in memory with no crawlr/stratum write.",
                "output_semantics": "Provenance-bearing continuous lighting statistics with a deterministic quantized luma band, dynamic-range band, shadow fraction, surround ratio, and a coarse key-light direction name, or explicit abstention, not semantic ground truth or caption claims.",
                "provenance": (
                    "research_harness.lighting.compute_lighting "
                    f"SHA-256 {code_hash}; computed in memory during this bounded run with no crawlr/stratum write."
                ),
                "abstention_policy": "Abort the selected item before model generation if required artifacts are missing, unreadable, or detector count is not exactly one; lighting statistics are measured only when the foreground clears a raw-pixel floor and a coverage floor and the direction fit has sufficient valid camera-facing normals, otherwise the respective fact abstains (never fabricated); detector disagreement remains a quality anomaly, never prompt content.",
                "known_failure_modes": "Luma/dynamic-range are camera-response dependent so only relative bands and ratios are verbalized; the Lambertian least-squares direction fit is a heuristic that can alias albedo to lighting on textured/clothed subjects; backlit silhouettes and very flat fill lighting can leave the direction undetermined; extreme over/under-exposure compresses the luma histogram.",
                "qualification_gate": "Candidate evidence only; no effectiveness claim is permitted until the frozen comparison receives completed rubric and adversarial reviews.",
            }
        ],
    }
    evidence["fingerprint"] = _evidence_fingerprint(evidence)
    return evidence


def _context4k_evidence() -> dict[str, Any]:
    """Declared deterministic context4k compact specialist (arm #36 round-trip).

    The evidence is the per-item evidence-linked compact context (the ≤4K
    dossier compression: claim-by-claim prose where every claim carries its
    supporting evidence ids), assembled deterministically in memory from the
    frozen selected inputs — the same five validated dimension specialists
    plus relational determinations that the dossier runner uses. This is the
    round-trip claim-support audit's evidence condition: captions are generated
    FROM the compact context and compared against the matched plain-4K
    summarization baseline.
    """
    module_path = Path(assemble_dossier.__code__.co_filename)
    code_hash = _sha256(module_path.read_bytes())
    evidence: dict[str, Any] = {
        "kind": "specialist_bundle",
        "id": "in-memory-context4k-compact-v1",
        "specialists": [
            {
                "id": "in-memory-context4k-compact-v1",
                "scope": "Per-item evidence-linked compact context (≤4K dossier compression) assembled from the five validated deterministic dimension specialists plus relational determinations; never identity, posture labels, or new facts beyond the evidence supply.",
                "inputs": "Frozen selected-item pose2.npy, seg2.npy, normal2.npy and the source RGB pixels already decoded by this bounded run; dossier assembled and compressed in memory with no crawlr/stratum write.",
                "output_semantics": "Provenance-bearing compact claims where every claim carries its supporting evidence ids (deterministic, evidence-bound), not semantic ground truth or caption claims; absolute pixel measurements stay in the machine-readable evidence payload and are never verbalized.",
                "provenance": (
                    "research_harness.dossier.assemble_dossier + compress_dossier_to_context "
                    f"SHA-256 {code_hash}; computed in memory during this bounded run with no crawlr/stratum write."
                ),
                "abstention_policy": "Abort the selected item before model generation if required artifacts are missing, unreadable, or detector count is not exactly one; an abstained measurement remains an honest abstention line and is never rescued into an invented value; the compact is never padded with fabricated filler to reach a budget; detector disagreement remains a quality anomaly, never prompt content.",
                "known_failure_modes": "The compact is only as rich as the deterministic evidence supply (387-648 dossier tokens/item on the frozen cohort, under the 4001 structural floor — recorded honestly as under_budget); absolute pixel widths and raw luminances are excluded by design; a truncated last claim records its truncation rather than silently dropping the evidence link.",
                "qualification_gate": "Candidate evidence only; no effectiveness claim is permitted until the frozen comparison receives completed rubric and adversarial reviews.",
            }
        ],
    }
    evidence["fingerprint"] = _evidence_fingerprint(evidence)
    return evidence


def _vlm_dense_evidence() -> dict[str, Any]:
    """Declared learned VLM dense-description specialist (arm #47).

    The evidence is the per-item dense multi-view description block emitted by a
    locally-run open-weight VLM (qwen3-vl:32b via Ollama) over the full frame +
    seg2-derived focal crops of the single curated woman source image. The block
    itself is NOT recomputed in-memory: it is pre-generated under a frozen
    prompt/crop policy on owned hardware (Strix, via the scheduler) and its bytes
    are bound in the frozen plan by `vlm_blocks_sha256` per item. The caption
    runner loads, verifies, and renders the block as part of the evidence text —
    this is the dossier-grow (option B) evidence part that extends the
    deterministic dossier compact toward the expanded-dossier structural floor.
    """
    evidence: dict[str, Any] = {
        "kind": "specialist_bundle",
        "id": "pre-generated-in-memory-vlm-dense-description-v1",
        "specialists": [
            {
                "id": "local-gemma3-27b-ollama-v1",
                "scope": "Dense multi-view descriptive prose for a single curated woman source image (full-frame + seg2 focal crops: face, upper body, garment regions); claims tagged observed/inferred/abstained; scale-invariant ratios only in prose.",
                "inputs": "Pre-generated per-item dense-description block (full frame + seg2-derived focal crops) verified byte-for-byte by the frozen plan's vlm_blocks_sha256; loaded from a noncanonical stage root, never from the derived tree.",
                "output_semantics": "Structured markdown dense-description block with six subsections (SUBJECT/GARMENTS/HAIR/SKIN/POSE/SETTING); every factual statement tagged [OBSERVED]/[INFERRED]/[ABSTAIN]; no absolute pixel or raw-coordinate numbers in prose (scale-invariant ratios only); machine-readable measurements stay in evidence_payload.",
                "provenance": (
                    "Open-weight gemma3:27b (Ollama digest a418f5838eaf...) run on owned "
                    "hardware (Strix, scheduler-managed) 2026-08-06 under a frozen prompt + "
                    "deterministic crop policy; no hosted third-party inference of the sensitive "
                    "corpus. MODEL FLIP (2026-08-06): the pre-registered qwen3-vl:32b candidate "
                    "FAILED qualification on the real corpus — silent empty responses on both "
                    "vision and plain-text requests (server-side decode wedge, not restartable "
                    "without root); gemma3:27b is the working already-local vision model, "
                    "verified to emit the full tagged block on corpus items. Block bytes pinned "
                    "by vlm_blocks_sha256 in the frozen comparison plan."
                ),
                "abstention_policy": "Claims are tagged [ABSTAIN] with a reason when a region lacks sufficient visible pixels (motion blur/occlusion/DOF); detector disagreement is a quality anomaly, never content; no absolute pixel claim in prose; a block that fails leakage/abstention checks or decodes empty is reported and abstains at generation, never silently promoted to evidence.",
                "known_failure_modes": "Learned VLM verbosity may over-specify visible attributes or drift from source truth on tight crops; abstention tagging can be sparser than warranted; same-model-family coupling (gemma3:27b also drives caption generation) may make the evidence more decodable by the caption model — recorded as a caveat, mitigated by the gemma4:e4b independent reviewer; a fixed 2x2 collage may underrepresent extreme crops.",
                "qualification_gate": "Candidate evidence only; no effectiveness claim is permitted until the frozen comparison receives completed rubric and adversarial reviews.",
            }
        ],
    }
    evidence["fingerprint"] = _evidence_fingerprint(evidence)
    return evidence


def _serialize_vlm_dense(block_text: str) -> str:
    """Render the frozen VLM dense-description block as caption evidence text."""
    header = (
        "VLM DENSE DESCRIPTION (learned multi-view evidence block, qwen3-vl:32b, "
        "full frame + focal crops; claims tagged [OBSERVED]/[INFERRED]/[ABSTAIN]; "
        "the model abstains rather than invents on low-detail regions):"
    )
    block = block_text.strip() if block_text else ""
    if not block:
        return header + "\n- (no VLM dense block available for this item — abstain from VLM-derived claims)"
    return header + "\n" + block


def _serialize_lighting(lighting: Mapping[str, Any]) -> str:
    """Deterministic natural-language rendering of a lighting measurement dict.

    Verbalizes only scale-invariant, caption-relevant facts: whether lighting
    is measurable, the quantized luma band, dynamic-range band, shadow band,
    surround band, and the coarse key-light direction. Continuous values, the
    fitted light vector, and the fit residual are deliberately NOT verbalized
    (camera/response/scale dependent); they exist in the machine-readable
    `evidence_payload` JSON (dossier / compressor input).
    """
    lines = [
        "LIGHTING (deterministic, normal2 direction statistics + source luminance/histogram/dynamic-range):"
    ]
    if not lighting.get("subject_present"):
        lines.append("- no reliable foreground subject present -> abstain from lighting claims")
        return "\n".join(lines)
    if not lighting.get("lighting_measurable"):
        lines.append("- lighting statistics did not clear the measurement gate -> abstain from lighting claims")
        reason = lighting.get("abstention_reason")
        if reason:
            lines.append(f"  (reason: {reason})")
        return "\n".join(lines)

    band = lighting.get("luma_band")
    if band:
        lines.append(f"- overall exposure: {band}")
    dr = lighting.get("dynamic_range_band")
    if dr:
        lines.append(f"- tonal dynamic range: {dr}")
    shadow = lighting.get("shadow_band")
    if shadow and shadow != "little shadow":
        lines.append(f"- shadows: {shadow}")
    surround = lighting.get("surround_band")
    lines.append(f"- subject relative to surround: {surround or 'undetermined'}")
    direction = lighting.get("light_direction")
    if direction and direction != "undetermined":
        lines.append(f"- key light direction: {direction}")
    elif lighting.get("light_direction") == "undetermined":
        lines.append("- key light direction: undetermined (insufficient directional shading)")
    return "\n".join(lines)


def _setting_evidence() -> dict[str, Any]:
    """Declared deterministic setting/environment specialist (arm #34)."""
    module_path = Path(compute_setting.__code__.co_filename)
    code_hash = _sha256(module_path.read_bytes())
    evidence: dict[str, Any] = {
        "kind": "specialist_bundle",
        "id": "in-memory-setting-v1",
        "specialists": [
            {
                "id": "in-memory-setting-v1",
                "scope": "DOME-29 Background (class 0) coverage and source-pixel color statistics over the non-subject region only; never identity, posture, or in-subject semantics.",
                "inputs": "Frozen selected-item seg2.npy and the source RGB pixels already decoded by this bounded run; recomputed in memory with no crawlr/stratum write.",
                "output_semantics": "Provenance-bearing continuous setting statistics with a deterministic quantized dominant background color name, tone band, vibrancy band, pattern band, and a scale-invariant background-coverage ratio, or explicit abstention, not semantic ground truth or caption claims.",
                "provenance": (
                    "research_harness.setting.compute_setting "
                    f"SHA-256 {code_hash}; computed in memory during this bounded run with no crawlr/stratum write."
                ),
                "abstention_policy": "Abort the selected item before model generation if required artifacts are missing, unreadable, or detector count is not exactly one; setting statistics are measured only when the Background region clears a raw-pixel floor and a frame-coverage floor, otherwise the fact abstains (never fabricated); detector disagreement remains a quality anomaly, never prompt content.",
                "known_failure_modes": "Color names depend on lighting and white balance and are quantized palette names, not spectral measurements; a background split between two equally large regions can average into a non-existent mean color; fully shallow depth-of-field or flat backdrops make the pattern band 'solid' even when the scene is structured; background coverage is a frame share, not an absolute area.",
                "qualification_gate": "Candidate evidence only; no effectiveness claim is permitted until the frozen comparison receives completed rubric and adversarial reviews.",
            }
        ],
    }
    evidence["fingerprint"] = _evidence_fingerprint(evidence)
    return evidence


def _serialize_setting(setting: Mapping[str, Any]) -> str:
    """Deterministic natural-language rendering of a setting measurement dict.

    Verbalizes only scale-invariant, caption-relevant facts: whether the
    background is measurable, its frame coverage ratio, the quantized dominant
    background color name, tone band, vibrancy band, and pattern band.
    Continuous values, the mean RGB hex, and the deviant fraction are
    deliberately NOT verbalized (camera/response/white-balance dependent);
    they exist in the machine-readable `evidence_payload` JSON (dossier /
    compressor input).
    """
    lines = [
        "SETTING (deterministic, seg2 Background region + source pixel palette):"
    ]
    if not setting.get("subject_present"):
        lines.append("- no reliable foreground subject present -> abstain from setting claims")
        return "\n".join(lines)
    if not setting.get("setting_measurable"):
        lines.append("- background region did not clear the measurement gate -> abstain from setting claims")
        reason = setting.get("abstention_reason")
        if reason:
            lines.append(f"  (reason: {reason})")
        return "\n".join(lines)

    coverage = setting.get("background_coverage")
    if coverage is not None:
        lines.append(f"- background surrounds the subject, covering {coverage:.2f} of the frame")
    color = setting.get("dominant_background_color")
    if color:
        lines.append(f"- dominant background color: {color}")
    tone = setting.get("background_tone_band")
    if tone:
        lines.append(f"- background tone: {tone}")
    vibrancy = setting.get("background_vibrancy_band")
    if vibrancy:
        lines.append(f"- background color intensity: {vibrancy}")
    pattern = setting.get("background_pattern_band")
    if pattern:
        lines.append(f"- background surface: {pattern}")
    return "\n".join(lines)


def _texture_evidence() -> dict[str, Any]:
    """Declared deterministic texture/material specialist (arm #35)."""
    module_path = Path(compute_texture.__code__.co_filename)
    code_hash = _sha256(module_path.read_bytes())
    evidence: dict[str, Any] = {
        "kind": "specialist_bundle",
        "id": "in-memory-texture-v1",
        "specialists": [
            {
                "id": "in-memory-texture-v1",
                "scope": "Per-region-class texture/material statistics (dominant measurable fabric class: Apparel/Upper_Clothing/Lower_Clothing; dominant measurable skin class) from seg2 masks + source pixels; never identity, posture, or in-subject semantics beyond surface roughness/pattern of the dominant class.",
                "inputs": "Frozen selected-item seg2.npy and the source RGB pixels already decoded by this bounded run; recomputed in memory with no crawlr/stratum write.",
                "output_semantics": "Provenance-bearing continuous texture statistics with deterministic quantized texture bands (smooth/some texture/textured), fabric pattern bands (solid/some variation/busy), and scale-invariant edge/deviant fractions, or explicit abstention, not semantic ground truth or caption claims.",
                "provenance": (
                    "research_harness.texture.compute_texture "
                    f"SHA-256 {code_hash}; computed in memory during this bounded run with no crawlr/stratum write."
                ),
                "abstention_policy": "Abort the selected item before model generation if required artifacts are missing, unreadable, or detector count is not exactly one; fabric claims are measured only when the dominant garment class clears a raw-pixel floor and a frame-coverage floor, otherwise the fabric fact abstains (never fabricated); skin claims abstain when no skin class clears the floors; detector disagreement remains a quality anomaly, never prompt content.",
                "known_failure_modes": "Gradient statistics conflate illumination boundaries with material edges (a shadow crease can read as texture); very coarse knits and fine patterns can fall in the same edge-fraction band; fabric print detection depends on color distance from the region mean and can miss tone-on-tone prints; skin bands are calibrated lower than fabric and may mislabel stubble/freckles as texture.",
                "qualification_gate": "Candidate evidence only; no effectiveness claim is permitted until the frozen comparison receives completed rubric and adversarial reviews.",
            }
        ],
    }
    evidence["fingerprint"] = _evidence_fingerprint(evidence)
    return evidence


def _serialize_texture(texture: Mapping[str, Any]) -> str:
    """Deterministic natural-language rendering of a texture measurement dict.

    Verbalizes only scale-invariant, caption-relevant facts: whether texture
    is measurable, the dominant fabric class's texture/pattern bands when a
    garment class cleared the gates, and the dominant skin class's surface
    band. Continuous gradient values, deviant fractions, and pixel counts are
    deliberately NOT verbalized (camera/response/white-balance dependent);
    they exist in the machine-readable `evidence_payload` JSON (dossier /
    compressor input).
    """
    lines = [
        "TEXTURE (deterministic, seg2 region-class masks + source pixel gradients):"
    ]
    if not texture.get("subject_present"):
        lines.append("- no reliable foreground subject present -> abstain from texture claims")
        return "\n".join(lines)
    if not texture.get("texture_measurable"):
        lines.append("- no fabric or skin region cleared the measurement gate -> abstain from texture claims")
        reason = texture.get("abstention_reason")
        if reason:
            lines.append(f"  (reason: {reason})")
        return "\n".join(lines)

    fabric_class = texture.get("fabric_class")
    if fabric_class:
        fabric_tex = texture.get("fabric_texture_band")
        fabric_pat = texture.get("fabric_pattern_band")
        if fabric_tex:
            lines.append(f"- dominant fabric ({fabric_class}): surface {fabric_tex}")
        if fabric_pat:
            lines.append(f"- dominant fabric ({fabric_class}): pattern {fabric_pat}")
    else:
        lines.append("- no measurable fabric region -> abstain from fabric/material claims")

    skin_class = texture.get("skin_class")
    if skin_class:
        skin_tex = texture.get("skin_texture_band")
        if skin_tex:
            lines.append(f"- dominant skin surface ({skin_class}): {skin_tex}")
    else:
        lines.append("- no measurable skin region -> abstain from skin-surface claims")
    return "\n".join(lines)


def _pose_articulation_evidence() -> dict[str, Any]:
    """Declared deterministic pose-articulation / kinematic specialist (arm #62)."""
    module_path = Path(compute_pose_articulation.__code__.co_filename)
    code_hash = _sha256(module_path.read_bytes())
    evidence: dict[str, Any] = {
        "kind": "specialist_bundle",
        "id": "in-memory-pose-articulation-v1",
        "specialists": [
            {
                "id": "in-memory-pose-articulation-v1",
                "scope": "Kinematic articulation of the single subject from pose2 GOLIATH-308 keypoints + seg2 limb masks only: per-joint flexion angles (elbow/knee), in-plane torso/pelvis orientation (twist/lean/pelvis tilt), weight-bearing stance + contrapposto class, limb-overlap/crossing structure (arms crossing the spine, legs crossed, arm proximity to torso), symmetry/asymmetry of flexion; never identity, clothing, or semantic pose labels beyond what joint geometry supports.",
                "inputs": "Frozen selected-item pose2.npy (GOLIATH-308) + seg2.npy (DOME-29); recomputed in memory during this bounded run with no crawlr/stratum write.",
                "output_semantics": "Provenance-bearing scale-invariant kinematic measurements (angles, normalized ratios, stance classes) or explicit abstention, not semantic ground truth or caption claims; only scale-invariant facts are verbalized.",
                "provenance": (
                    "research_harness.pose_articulation.compute_pose_articulation "
                    f"SHA-256 {code_hash}; computed in memory during this bounded run with no crawlr/stratum write."
                ),
                "abstention_policy": "Abort the selected item before model generation if required artifacts are missing, unreadable, or detector count is not exactly one; every angle/ratio/stance emits None (never fabricated) when its supporting joints are absent or below confidence threshold; stance and contrapposto abstain when the weight-bearing signal is ambiguous; detector disagreement remains a quality anomaly, never prompt content.",
                "known_failure_modes": "Low keypoint confidence on occluded/cropped limbs makes flexion angles abstain; in-plane torso twist cannot recover out-of-plane rotation; seg2 arm/torso proximity is a spatial proxy for occlusion, not a true depth ordering; contrapposto detection depends on both a clear weight shift and pelvis tilt and may abstain on partially visible legs.",
                "qualification_gate": "Candidate evidence only; no effectiveness claim is permitted until the frozen comparison receives completed rubric and adversarial reviews.",
            }
        ],
    }
    evidence["fingerprint"] = _evidence_fingerprint(evidence)
    return evidence


def _serialize_pose_articulation(articulation: Mapping[str, Any]) -> str:
    """Deterministic natural-language rendering of a pose-articulation dict.

    Verbalizes ONLY scale-invariant kinematic facts: joint-flexion bands
    (extended / bent), stance class (weight on a leg / centered), contrapposto
    stance, torso/pelvis orientation (twist/lean/tilt in degrees), limb-overlap
    structure (arms crossed in front of torso), and flexion asymmetry. Raw
    joint angles, pixel lengths, and seg2 fractions remain in the machine-
    readable `evidence_payload` JSON (dossier / compressor input) and are not
    caption claims — angles are scale-invariant, but a bare degree number is
    not something a text-to-image model should be asked to render; the banded
    "bent-arm"/"weight-on-one-leg" phrasing is.
    """
    lines = [
        "POSE-ARTICULATION (deterministic, scale-invariant kinematics from Goliath-308 pose2 + seg2):"
    ]
    if not articulation.get("subject_present"):
        lines.append("- no reliable subject keypoints -> abstain from pose/stance/limb claims")
        return "\n".join(lines)

    # Joint-flexion bands (scale-invariant: angles).
    for side, key in (("left", "elbow_flexion_left"), ("right", "elbow_flexion_right")):
        value = articulation.get(key)
        if value is None:
            lines.append(f"- {side} elbow flexion: not measurable (joint absent or low confidence)")
        elif value < 135.0:
            lines.append(f"- {side} arm visibly bent at the elbow")
        else:
            lines.append(f"- {side} arm extended at the elbow")
    for side, key in (("left", "knee_flexion_left"), ("right", "knee_flexion_right")):
        value = articulation.get(key)
        if value is None:
            lines.append(f"- {side} knee flexion: not measurable (joint absent or low confidence)")
        elif value < 135.0:
            lines.append(f"- {side} leg visibly bent at the knee")
        else:
            lines.append(f"- {side} leg extended at the knee")

    # Stance / contrapposto (scale-invariant stance classes).
    stance = articulation.get("stance_class")
    if stance and stance != "centered":
        leg = "left" if stance == "weight-left" else "right"
        lines.append(f"- stance: weight clearly carried on the {leg} leg")
    elif stance == "centered":
        lines.append("- stance: weight evenly centered (no single weight-bearing leg)")
    else:
        lines.append("- stance: not measurable (weight-bearing signal ambiguous)")

    contrapposto = articulation.get("contrapposto")
    if contrapposto is True:
        lines.append("- contrapposto stance: hips tilted with weight on one leg (counter-rotated hips/shoulders)")
    elif contrapposto is False:
        lines.append("- contrapposto stance: absent (hips level, weight-bearing signal even)")

    # Torso/pelvis orientation (scale-invariant in-plane angles).
    twist = articulation.get("torso_twist_deg")
    lean = articulation.get("torso_lean_deg")
    tilt = articulation.get("pelvis_tilt_deg")
    if twist is not None:
        lines.append(f"- torso/hips in-plane twist: {twist:.0f} degrees between shoulder and hip axes")
    if lean is not None:
        lines.append(f"- torso lean from vertical: {lean:.0f} degrees")
    if tilt is not None:
        lines.append(f"- pelvis tilt from horizontal: {tilt:.0f} degrees")

    # Limb-overlap structure (scale-invariant crossing counts / fractions).
    crossing = articulation.get("arm_crossing_count")
    if crossing is not None and crossing > 0:
        lines.append(f"- {crossing} arm(s) cross in front of the torso")
    legs_crossed = articulation.get("legs_crossed")
    if legs_crossed is True:
        lines.append("- legs visually crossed")

    # Flexion asymmetry (scale-invariant: angle difference).
    asy = articulation.get("elbow_flexion_asymmetry_deg")
    if asy is not None and asy > 15.0:
        lines.append(f"- asymmetric arm positioning (left/right elbow flexion differ by {asy:.0f} degrees)")
    return "\n".join(lines)


def _pointmap_depth_evidence() -> dict[str, Any]:
    """Declared deterministic point-map / 3D depth-ordering specialist (arm #58)."""
    module_path = Path(compute_pointmap_depth.__code__.co_filename)
    code_hash = _sha256(module_path.read_bytes())
    evidence: dict[str, Any] = {
        "kind": "specialist_bundle",
        "id": "in-memory-pointmap-depth-v1",
        "specialists": [
            {
                "id": "in-memory-pointmap-depth-v1",
                "scope": "Scale-invariant relative-depth ordering of the single subject from pointmap.npy (Sapiens2 per-pixel CAM-frame 3D cloud) + seg2.npy DOME-29 region masks: region depth ranking (nearest/farthest part), left/right hand depth ordering, hand/arm held in front of the torso plane, and a normalized body depth-relief ratio (robust Z spread / median Z). Never identity, clothing, or semantic labels beyond what the point cloud supports; never absolute metric distances in prose.",
                "inputs": "Frozen selected-item pointmap.npy (source-matched, CAM-frame, background zeroed) + seg2.npy (DOME-29); recomputed in memory during this bounded run with no crawlr/stratum write.",
                "output_semantics": "Provenance-bearing scale-invariant depth-order relations and normalized ratios, or explicit abstention, not semantic ground truth or caption claims; only ordering/ratio/band facts are verbalized (absolute metric Z stays in the machine-readable payload).",
                "provenance": (
                    "research_harness.pointmap_depth.compute_pointmap_depth "
                    f"SHA-256 {code_hash}; computed in memory during this bounded run with no crawlr/stratum write."
                ),
                "abstention_policy": "Abort the selected item before model generation if required artifacts are missing, unreadable, or detector count is not exactly one; abstain (emit None with a surfaced reason) when the point-cloud foreground is degenerate/too sparse, no region clears its depth-support floor, or the subject median Z is outside the human-plausible band; detector disagreement remains a quality anomaly, never prompt content.",
                "known_failure_modes": "Monocular pointmap scale ambiguity (no true metric depth); background pixels are zeroed so background-plane depth is unmeasurable; sparse hand/limb regions may abstain; tight crops or subject-fills-frame degeneracy can weaken the relief signal; occlusion ordering is inferred from median-Z ranks, not physical ray casting.",
                "qualification_gate": "Candidate evidence only; no effectiveness claim is permitted until the frozen comparison receives completed rubric and adversarial reviews.",
            }
        ],
    }
    evidence["fingerprint"] = _evidence_fingerprint(evidence)
    return evidence


def _serialize_pointmap_depth(pointmap_depth: Mapping[str, Any]) -> str:
    """Deterministic natural-language rendering of a pointmap-depth dict.

    Verbalizes ONLY scale-invariant depth facts: region nearest/farthest
    ordering, hand depth ordering (one hand clearly nearer), hand/arm in front
    of the torso plane, and the normalized depth-relief band. Raw CAM-frame
    metric Z values, absolute spreads, and per-region median Zs remain in the
    machine-readable `evidence_payload` JSON (dossier / compressor input) and
    are not caption claims — a bare meter number is camera-placement dependent
    and not something a text-to-image model should be asked to render.
    """
    lines = [
        "POINTMAP-DEPTH (deterministic, scale-invariant depth ordering from Sapiens2 pointmap + seg2):"
    ]
    if not pointmap_depth.get("subject_present"):
        lines.append("- no subject depth present -> abstain from depth-ordering claims")
        return "\n".join(lines)
    if pointmap_depth.get("abstained"):
        reason = pointmap_depth.get("abstention_reason") or "depth not measurable"
        lines.append(f"- depth ordering abstained ({reason})")
        return "\n".join(lines)

    relief_band = pointmap_depth.get("relief_band")
    if relief_band == "pronounced":
        lines.append("- body has pronounced depth relief (limbs clearly nearer than the torso plane)")
    elif relief_band == "moderate":
        lines.append("- body has moderate depth relief (some limbs offset from the torso plane)")
    elif relief_band == "compact":
        lines.append("- body largely on one depth plane (compact)")

    nearest = pointmap_depth.get("nearest_region")
    farthest = pointmap_depth.get("farthest_region")
    if nearest and farthest:
        lines.append(
            f"- {nearest.replace('_', ' ')} is the part nearest the camera; {farthest.replace('_', ' ')} the farthest"
        )
    elif nearest:
        lines.append(f"- {nearest.replace('_', ' ')} is the part nearest the camera")

    hand_ordering = pointmap_depth.get("hand_ordering")
    if hand_ordering:
        lines.append(f"- {hand_ordering} hand is clearly held nearer the camera than the other")
    if pointmap_depth.get("left_hand_in_front"):
        lines.append("- left hand/arm is held in front of the torso plane")
    if pointmap_depth.get("right_hand_in_front"):
        lines.append("- right hand/arm is held in front of the torso plane")
    return "\n".join(lines)


def _matting_alpha_evidence() -> dict[str, Any]:
    """Declared deterministic matting / alpha-fidelity specialist (arm #59)."""
    module_path = Path(compute_matting_alpha.__code__.co_filename)
    code_hash = _sha256(module_path.read_bytes())
    evidence: dict[str, Any] = {
        "kind": "specialist_bundle",
        "id": "in-memory-matting-alpha-v1",
        "specialists": [
            {
                "id": "in-memory-matting-alpha-v1",
                "scope": "Scale-invariant matting/alpha-fidelity facts of the single subject from matting.npy (Sapiens2 per-pixel soft alpha) + seg2.npy DOME-29 (Hair) masks: subject alpha-coverage band (sparse/centered/fills-frame), boundary crispness band (soft/moderate/crisp silhouette edge), and soft-edge character (hair-dominant vs clean skin cutout). Never identity, clothing, or semantic labels beyond what the alpha field supports; never absolute pixel widths in prose.",
                "inputs": "Frozen selected-item matting.npy (source-matched, (H,W) float16 alpha) + seg2.npy (DOME-29); recomputed in memory during this bounded run with no crawlr/stratum write.",
                "output_semantics": "Provenance-bearing scale-invariant alpha-fidelity ratios and bands, or explicit abstention, not semantic ground truth or caption claims; only ratio/band facts are verbalized (absolute pixel areas/widths stay in the machine-readable payload).",
                "provenance": (
                    "research_harness.matting_alpha.compute_matting_alpha "
                    f"SHA-256 {code_hash}; computed in memory during this bounded run with no crawlr/stratum write."
                ),
                "abstention_policy": "Abort the selected item before model generation if required artifacts are missing, unreadable, or detector count is not exactly one; abstain (emit None with a surfaced reason) when the matte is absent/ill-formed/degenerate (all-one/all-zero), the alpha subject mask is too small, or alpha values leave [0,1]; detector disagreement remains a quality anomaly, never prompt content.",
                "known_failure_modes": "Matting edge quality inherits the matting model's accuracy (soft-edged hair may be under- or over-soft); silhouette closedness is near-constant on this cohort (no crops) so it carries NO caption band; band thresholds are calibrated to this frozen cohort and may shift on other cohorts; tight crops or subject-fills-frame degeneracy can compress the coverage band.",
                "qualification_gate": "Candidate evidence only; no effectiveness claim is permitted until the frozen comparison receives completed rubric and adversarial reviews.",
            }
        ],
    }
    evidence["fingerprint"] = _evidence_fingerprint(evidence)
    return evidence


def _serialize_matting_alpha(matting_alpha: Mapping[str, Any]) -> str:
    """Deterministic natural-language rendering of a matting-alpha dict.

    Verbalizes ONLY scale-invariant alpha-fidelity facts: subject coverage band,
    boundary crispness band, and soft-edge character. Raw pixel areas, band
    widths, and silhouette structure remain in the machine-readable
    `evidence_payload` JSON (dossier / compressor input) and are not caption
    claims — a bare pixel width is camera-frame-dependent and unrenderable.
    """
    lines = [
        "MATTING-ALPHA (deterministic, scale-invariant alpha fidelity from Sapiens2 matting + seg2):"
    ]
    if not matting_alpha.get("subject_present"):
        lines.append("- no subject alpha present -> abstain from matting claims")
        return "\n".join(lines)
    if matting_alpha.get("abstained"):
        reason = matting_alpha.get("abstention_reason") or "matte not measurable"
        lines.append(f"- matting abstained ({reason})")
        return "\n".join(lines)

    coverage_band = matting_alpha.get("coverage_band")
    if coverage_band == "sparse":
        lines.append("- subject is a small figure occupying a minor part of the frame")
    elif coverage_band == "fills-frame":
        lines.append("- subject largely fills the frame")
    elif coverage_band == "centered":
        lines.append("- subject occupies the center of the frame")

    crisp_band = matting_alpha.get("boundary_crisp_band")
    if crisp_band == "crisp":
        lines.append("- clean sharp silhouette edge (crisp cutout)")
    elif crisp_band == "soft":
        lines.append("- very soft feathered silhouette edge")
    elif crisp_band == "moderate":
        lines.append("- moderately soft silhouette edge")

    edge_band = matting_alpha.get("soft_edge_band")
    if edge_band == "hair-dominant":
        lines.append("- soft detachable hair strands produce a wispy hairline (hair-dominant soft edge)")
    elif edge_band == "mixed":
        lines.append("- soft edge is a mix of hair and skin/background transitions")
    elif edge_band == "skin-clean":
        lines.append("- clean skin/background cutout with minimal hair flyaway")

    if len(lines) == 1:
        lines.append("- (no scale-invariant matting fact cleared a measurement gate)")
    return "\n".join(lines)


def _face_geometry_evidence() -> dict[str, Any]:
    """Declared open-weight facial-landmark / geometry specialist (arm #60)."""
    module_path = Path(compute_face_geometry.__code__.co_filename)
    code_hash = _sha256(module_path.read_bytes())
    model_path = Path(FACE_GEOMETRY_MODEL_ASSET)
    model_sha = _sha256(model_path.read_bytes()) if model_path.exists() else "MISSING"
    evidence: dict[str, Any] = {
        "kind": "specialist_bundle",
        "id": "in-memory-face-geometry-v1",
        "specialists": [
            {
                "id": "in-memory-face-geometry-v1",
                "scope": "Scale-invariant facial-geometry facts of the single subject from a local MediaPipe FaceLandmarker 478-point mesh over the full frame / seg2 Face_Neck crop (union detection policy): eye-spacing band (interpupillary / face width), mouth-width band, jaw-width band, and (plausibility-gated) mid-face vertical share. Never identity, skin-tone, or makeup semantics; only scale-invariant ratios in prose.",
                "inputs": "Frozen selected-item seg2.npy (DOME-29 Face_Neck mask) + the already-decoded source RGB; local open-weight face_landmarker.task model (MediaPipe, CPU, tasks API). Recomputed in memory during this bounded run with no crawlr/stratum write; model run on owned hardware only.",
                "output_semantics": "Provenance-bearing scale-invariant facial ratios and bands with a surfaced abstention on no-face / degenerate landmark detection, not identity ground truth or caption claims; only ratio/band facts are verbalized (landmark coords + pixel bbox stay in the machine-readable payload).",
                "provenance": (
                    "research_harness.face_geometry.compute_face_geometry "
                    f"SHA-256 {code_hash}; model face_landmarker.task sha256 {model_sha} "
                    "(Open-weight MediaPipe FaceLandmarker, Apache-2.0, local CPU, owned hardware); "
                    "computed in memory during this bounded run with no crawlr/stratum write, no "
                    "hosted third-party inference of the sensitive corpus."
                ),
                "abstention_policy": "Abort the selected item before model generation if required artifacts are missing or detector count is not exactly one; abstain (emit None with a surfaced reason) when FaceLandmarker finds no face on the full frame or the seg2 Face_Neck crop, when the detected mesh bbox is degenerate (face smaller than a floor), or when the mid-face vertical share leaves the human-plausibility band; detector disagreement remains a quality anomaly, never prompt content.",
                "known_failure_modes": "FaceLandmarker is resolution-sensitive (the same face can be found full-frame on some items and only on the seg2 crop on others — the union policy mitigates this and the capability probe measured 21/24 detection); turned heads / extreme DOF / occlusion abstain; band thresholds are calibrated to this frozen cohort and may shift on other cohorts; the mid-face vertical share needs a plausibility gate (extreme tilt can produce an implausible share).",
                "qualification_gate": "Candidate evidence only; no effectiveness claim is permitted until the frozen comparison receives completed rubric and adversarial reviews.",
            }
        ],
    }
    evidence["fingerprint"] = _evidence_fingerprint(evidence)
    return evidence


def _serialize_face_geometry(face_geometry: Mapping[str, Any]) -> str:
    """Deterministic natural-language rendering of a face-geometry dict.

    Verbalizes ONLY scale-invariant facial-ratio bands (eye spacing, mouth
    width, jaw width, plausibility-gated mid-face share). Landmark coordinates,
    pixel bbox, and absolute ratios remain in the machine-readable
    `evidence_payload` JSON and are not caption claims.
    """
    lines = [
        "FACE-GEOMETRY (open-weight facial landmarks, scale-invariant):"
    ]
    if face_geometry.get("abstained"):
        reason = face_geometry.get("abstention_reason") or "face not measurable"
        lines.append(f"- face abstained ({reason})")
        return "\n".join(lines)

    def _band_line(name: str, label: str, band: str | None) -> None:
        if band == "close-set":
            lines.append(f"- eyes are set close together (close-set) relative to face width")
        elif band == "wide-set":
            lines.append(f"- eyes are set wide apart (wide-set) relative to face width")
        elif band == "narrow" and name != "eye":
            lines.append(f"- {label} is narrow relative to the face")
        elif band == "wide" and name != "eye":
            lines.append(f"- {label} is wide relative to the face")
        elif band == "short":
            lines.append(f"- mid-face (nose-to-chin) is short relative to the face")
        elif band == "tall":
            lines.append(f"- mid-face (nose-to-chin) is tall relative to the face")

    _band_line("eye", "eye spacing", face_geometry.get("eye_spacing_band"))
    _band_line("mouth", "mouth", face_geometry.get("mouth_band"))
    _band_line("jaw", "jawline", face_geometry.get("jaw_band"))
    _band_line("mid", "mid-face", face_geometry.get("midface_band"))

    if len(lines) == 1:
        lines.append("- (no distinctive facial ratio outside the typical band)")
    return "\n".join(lines)


def _object_relations_evidence() -> dict[str, Any]:
    """Declared open-weight text-grounded object/accessory specialist (arm #61)."""
    module_path = Path(compute_object_relations.__code__.co_filename)
    code_hash = _sha256(module_path.read_bytes())
    model_dir = Path(OBJECT_RELATIONS_MODEL_ASSET)
    model_sha = (
        _sha256((model_dir / "model.safetensors").read_bytes())
        if (model_dir / "model.safetensors").exists() else "MISSING"
    )
    evidence: dict[str, Any] = {
        "kind": "specialist_bundle",
        "id": "in-memory-object-relations-v1",
        "specialists": [
            {
                "id": "in-memory-object-relations-v1",
                "scope": "Object / accessory presence + coarse spatial placement of the single subject from a locally-run open-weight text-grounded detector (Grounding DINO) over the full frame with a frozen closed vocabulary: count band (none/sparse/moderate/dense), placement band (foreground/background/mix from seg2-subject overlap), canonical class list. Never identity, skin-tone, or aesthetic claims; only scale-invariant names/counts/relative placement in prose.",
                "inputs": ("Frozen selected-item seg2.npy (DOME-29 subject mask) + the already-decoded "
                           "source RGB; local open-weight grounding-dino-base (HF Transformers path, "
                           "CPU, model.safetensors sha256 {model_sha}). Recomputed in memory during this "
                           "bounded run with no crawlr/stratum write; model run on owned hardware only.").format(model_sha=model_sha),
                "output_semantics": ("Provenance-bearing scale-invariant object-presence/placement facts "
                                     "with a surfaced abstention on model/input failure, not semantic ground "
                                     "truth or caption claims; only band/name facts are verbalized (normalized "
                                     "boxes, scores, raw phrases stay in the machine-readable payload)."),
                "provenance": (
                    "research_harness.object_relations.compute_object_relations "
                    f"SHA-256 {code_hash}; model grounding-dino-base (IDEA-Research, Apache-2.0, "
                    "HF Transformers path) sha256 {model_sha}, run locally on owned hardware (CPU, "
                    "no VRAM contention with the caption model); computed in memory during this "
                    "bounded run with no crawlr/stratum write, no hosted third-party inference of "
                    "the sensitive corpus."
                ).format(model_sha=model_sha),
                "abstention_policy": ("Abort the selected item before model generation if required "
                                      "artifacts are missing or detector count is not exactly one; abstain "
                                      "(emit thresholds/None with a surfaced reason) on detector invocation "
                                      "failure. Subject-self detections (the detector firing 'body' on the "
                                      "subject herself) are excluded from count and prose — they are the "
                                      "subject, not objects. Detector disagreement remains a quality "
                                      "anomaly, never prompt content."),
                "known_failure_modes": ("Grounding DINO is a text-grounded open-vocab detector: closed-"
                                        "vocabulary recall depends on the frozen class list (a furniture-"
                                        "centric list was degenerate on this scene-dominant cohort — the "
                                        "cohort-derived list is used); scores are not calibrated across "
                                        "domains (threshold 0.25 calibrated on this frozen cohort); subject-"
                                        "self detections ('body') must be excluded; box coords are frame-"
                                        "dependent (normalized only in payload)."),
                "qualification_gate": ("Candidate evidence only; no effectiveness claim is permitted until "
                                       "the frozen comparison receives completed rubric and adversarial reviews."),
            }
        ],
    }
    evidence["fingerprint"] = _evidence_fingerprint(evidence)
    return evidence


def _scene_category_evidence() -> dict[str, Any]:
    """Declared open-weight CLIP ViT-L/14 zero-shot scene classifier (arm #69)."""
    module_path = Path(compute_scene_category.__code__.co_filename)
    code_hash = _sha256(module_path.read_bytes())
    model_dir = Path(SCENE_CATEGORY_MODEL_ASSET)
    model_sha = (
        _sha256((model_dir / "model.safetensors").read_bytes())
        if (model_dir / "model.safetensors").exists() else "MISSING"
    )
    evidence: dict[str, Any] = {
        "kind": "specialist_bundle",
        "id": "in-memory-scene-category-v1",
        "specialists": [
            {
                "id": "in-memory-scene-category-v1",
                "scope": "Semantic scene category (what-kind-of-place) from a locally-run open-weight CLIP ViT-L/14 "
                         "zero-shot classifier over the full-frame source with a frozen closed scene-category set "
                         "(indoor studio / plain wall backdrop / bedroom / living room / outdoor beach / outdoor "
                         "garden / outdoor field / body of water / urban street / poolside) at a calibrated "
                         "confidence floor. Emits the argmax category label (or a surfaced abstention below the "
                         "floor); never identity, skin-tone, or aesthetic claims; only the scale-invariant label.",
                "inputs": ("Frozen selected-item already-decoded full-frame source RGB (SHA-bound via the item's "
                           "source_sha256); local open-weight openai/clip-vit-large-patch14 (MIT, CPU, "
                           "model.safetensors sha256 {model_sha}). Recomputed in memory during this bounded run "
                           "with no crawlr/stratum write; model run on owned hardware only. seg2/pose2 stay "
                           "validation-only reads.").format(model_sha=model_sha),
                "output_semantics": ("Provenance-bearing scale-invariant scene-category label with a surfaced "
                                     "abstention on low confidence or model/input failure, not semantic ground "
                                     "truth or caption claims; only the label is verbalized (similarity "
                                     "logits/probabilities stay in the machine-readable payload)."),
                "provenance": (
                    "research_harness.scene_category.compute_scene_category "
                    f"SHA-256 {code_hash}; model openai/clip-vit-large-patch14 (MIT, "
                    f"HF Transformers path) sha256 {model_sha}, run locally on owned hardware (CPU, "
                    "no VRAM contention with the caption model); computed in memory during this "
                    "bounded run with no crawlr/stratum write, no hosted third-party inference of "
                    "the sensitive corpus."
                ).format(model_sha=model_sha),
                "abstention_policy": ("Abort the selected item before model generation if required "
                                      "artifacts are missing or detector count is not exactly one; abstain "
                                      "(emit None/label with a surfaced reason) when the argmax softmax "
                                      "confidence falls below the calibrated floor. Never overwrite an "
                                      "ambiguous scene with a confident-looking guess. Classifier "
                                      "disagreement remains a quality anomaly, never prompt content."),
                "known_failure_modes": ("CLIP zero-shot is a closed-set classifier: labels outside the frozen "
                                        "scene vocabulary are forced onto the nearest class (addressed by "
                                        "cohort-deriving the closed set from the arm-#47 VLM scene vocabulary); "
                                        "confidence is not calibrated across domains (floor 0.25 calibrated on "
                                        "this frozen cohort: 24/24 classified, max top-1 share 29%); object-level "
                                        "detail that CLIP cannot distinguish from the scene is out of scope."),
                "qualification_gate": ("Candidate evidence only; no effectiveness claim is permitted until "
                                       "the frozen comparison receives completed rubric and adversarial reviews."),
            }
        ],
    }
    evidence["fingerprint"] = _evidence_fingerprint(evidence)
    return evidence


def _serialize_object_relations(objrel: Mapping[str, Any]) -> str:
    """Deterministic natural-language rendering of an object-relations dict.

    Verbalizes ONLY scale-invariant facts: count band, placement band, and the
    canonical class list. Normalized boxes, scores, and raw phrases remain in
    the machine-readable `evidence_payload` JSON and are not caption claims.
    """
    lines = [
        "OBJECT-RELATIONS (open-weight text-grounded detection, scale-invariant):"
    ]
    if not objrel or not objrel.get("count_band"):
        lines.append("- object-relations not measured for this item")
        return "\n".join(lines)
    if objrel.get("abstained"):
        reason = objrel.get("abstention_reason") or "object detection not measurable"
        lines.append(f"- object detection abstained ({reason})")
        return "\n".join(lines)

    count = objrel.get("count", 0)
    band = objrel.get("count_band", "none")
    if band == "none" or count == 0:
        lines.append("- no scene objects detected above the calibrated threshold")
        return "\n".join(lines)

    classes = objrel.get("classes") or []
    cls_txt = ", ".join(classes[:5])
    if band == "sparse":
        lines.append(f"- a single scene object is present: {cls_txt}")
    elif band == "moderate":
        lines.append(f"- several scene objects are present: {cls_txt}")
    else:
        lines.append(f"- the scene contains multiple distinct objects: {cls_txt}")

    placement = objrel.get("placement_band")
    if placement == "foreground":
        lines.append("- objects overlap the subject (held / on-person)")
    elif placement == "background":
        lines.append("- objects sit behind the subject in the background")
    elif placement == "mix":
        lines.append("- objects are a mix of foreground and background")
    return "\n".join(lines)


def _serialize_scene_category(scene: Mapping[str, Any]) -> str:
    """Deterministic natural-language rendering of a scene-category dict.

    Verbalizes ONLY the scale-invariant semantic category label (or a surfaced
    abstention). Similarity logits / probabilities remain in the machine-
    readable `evidence_payload` JSON and are not caption claims.
    """
    lines = [
        "SCENE-CATEGORY (CLIP ViT-L/14 zero-shot, closed set, scale-invariant):"
    ]
    if scene.get("abstained"):
        reason = scene.get("abstention_reason") or "scene classification not confident"
        lines.append(f"- scene-category abstained ({reason})")
        return "\n".join(lines)
    if not scene or not scene.get("category"):
        lines.append("- scene-category not measured for this item")
        return "\n".join(lines)
    lines.append(f"- the setting is a {scene['category']}")
    return "\n".join(lines)


# Frozen open-weight MediaPipe FaceLandmarker asset (arm #68, same mesh as
# arm #60 face-geometry, new evidence part: camera-relative head orientation).
# face_landmarker.task sha256 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff
GAZE_HEAD_MODEL_ASSET = (
    "/mnt/nas-ai-models/research/stratum/models/face-geometry/face_landmarker.task"
)


def _gaze_head_evidence() -> dict[str, Any]:
    """Declared head-orientation specialist (arm #68, reuses arm #60 mesh)."""
    module_path = Path(compute_gaze_head.__code__.co_filename)
    code_hash = _sha256(module_path.read_bytes())
    model_path = Path(GAZE_HEAD_MODEL_ASSET)
    model_sha = _sha256(model_path.read_bytes()) if model_path.exists() else "MISSING"
    evidence: dict[str, Any] = {
        "kind": "specialist_bundle",
        "id": "in-memory-gaze-head-orientation-v1",
        "specialists": [
            {
                "id": "in-memory-gaze-head-orientation-v1",
                "scope": "Scale-invariant camera-relative head orientation (yaw: facing camera / partially turned / profile or turned away; pitch: level / tilted down / tilted up; roll: level / tilted in-plane) from a locally-run open-weight MediaPipe FaceLandmarker 478-point mesh (same mesh asset as arm #60) over the full frame / seg2 Face_Neck crop (union policy). Never identity, skin-tone, facial-shape, or aesthetic claims; only the scale-invariant direction bands in prose.",
                "inputs": ("Frozen selected-item seg2.npy (DOME-29 Face_Neck mask) + the already-decoded "
                           "source RGB; local open-weight face_landmarker.task (owned hardware, local CPU, "
                           "tasks API, model sha256 {model_sha}). Recomputed in memory during this bounded run "
                           "with no crawlr/stratum write; same frozen mesh asset as the validated arm #60.").format(model_sha=model_sha),
                "output_semantics": ("Provenance-bearing scale-invariant head-orientation bands with a surfaced "
                                     "abstention on no-face / degenerate landmark fit, not semantic ground truth "
                                     "or caption claims; only direction bands are verbalized (raw yaw/pitch/roll "
                                     "degrees and landmark coords stay in the machine-readable payload)."),
                "provenance": (
                    "research_harness.gaze_head.compute_gaze_head "
                    f"SHA-256 {code_hash}; model face_landmarker.task sha256 {model_sha} "
                    "(open-weight MediaPipe FaceLandmarker, Apache-2.0, local CPU, owned hardware); "
                    "computed in memory during this bounded run with no crawlr/stratum write, "
                    "no hosted third-party inference of the sensitive corpus."
                ),
                "abstention_policy": ("Abort the selected item before model generation if required artifacts are "
                                      "missing, unreadable, or detector count is not exactly one; abstain (emit "
                                      "None/band with a surfaced reason) when no face is found on the full frame "
                                      "or the seg2 Face_Neck crop (arm #60 union policy), or when the 6-point PnP "
                                      "fit is degenerate (landmark bbox below the pixel floor). Detector "
                                      "disagreement remains a quality anomaly, never prompt content."),
                "known_failure_modes": ("PnP head pose is best for frontal-to-3/4 views; at near-profile the "
                                        "nose/eye/mouth skeleton can project onto a thin footprint and the fit can "
                                        "unstable — the pixel-span floor gates that. Camera-relative direction is "
                                        "scale-invariant but not depth-absolute; roll is in-plane and only "
                                        "roughly camera-relative. Resolution-sensitive (same asset, arm #60 "
                                        "measured UNION is the honest policy)."),
                "qualification_gate": ("Candidate evidence only; no effectiveness claim is permitted until "
                                       "the frozen comparison receives completed rubric and adversarial reviews."),
            }
        ],
    }
    evidence["fingerprint"] = _evidence_fingerprint(evidence)
    return evidence


def _serialize_gaze_head(gaze: Mapping[str, Any]) -> str:
    """Deterministic natural-language rendering of a gaze/head-orientation dict.

    Verbalizes ONLY the scale-invariant camera-relative direction bands (yaw /
    pitch / roll). Raw yaw/pitch/roll degrees and landmark coordinates remain
    in the machine-readable `evidence_payload` JSON and are not caption claims.
    """
    lines = [
        "GAZE/HEAD-ORIENTATION (open-weight facial landmarks, camera-relative, scale-invariant):"
    ]
    if gaze.get("abstained"):
        reason = gaze.get("abstention_reason") or "face not measurable"
        lines.append(f"- head orientation abstained ({reason})")
        return "\n".join(lines)
    if not gaze or not gaze.get("yaw_band"):
        lines.append("- head orientation not measured for this item")
        return "\n".join(lines)
    yb = gaze.get("yaw_band")
    pb = gaze.get("pitch_band")
    rb = gaze.get("roll_band")
    if yb == "facing camera":
        lines.append("- head is facing the camera")
    elif yb == "partially turned":
        lines.append("- head is partially turned from the camera")
    elif yb == "profile or turned away":
        lines.append("- head is turned toward profile / away from the camera")
    if pb == "tilted down":
        lines.append("- head is tilted down (pitch)")
    elif pb == "tilted up":
        lines.append("- head is tilted up (pitch)")
    if rb == "tilted":
        lines.append("- head is tilted to one side (in-plane roll)")
    return "\n".join(lines)


def _camera_viewing_angle_evidence() -> dict[str, Any]:
    """Declared deterministic camera-viewing-angle / framing specialist (arm #74)."""
    module_path = Path(compute_camera_viewing_angle.__code__.co_filename)
    code_hash = _sha256(module_path.read_bytes())
    evidence: dict[str, Any] = {
        "kind": "specialist_bundle",
        "id": "in-memory-camera-viewing-angle-v1",
        "specialists": [
            {
                "id": "in-memory-camera-viewing-angle-v1",
                "scope": "Scale-invariant camera-relative framing with no new model: shot-scale band (close-up / mid-shot / full-body), vertical headroom band (tight / normal / wide), and camera-height band (eye-level / low-angle / high-angle) from the seg2 subject bbox + full-frame geometry. Never identity, skin-tone, facial, or aesthetic claims; only the scale-invariant framing bands in prose.",
                "inputs": ("Frozen selected-item seg2.npy (DOME-29 subject mask, already read for the "
                           "exactly-one-subject invariant) + the full-frame decoded source image dims; computed "
                           "in memory with no crawlr/stratum write; no new model (CPU)."),
                "output_semantics": ("Provenance-bearing scale-invariant framing bands with a surfaced "
                                     "abstention on degenerate/cropped subject bbox, not semantic ground truth "
                                     "or caption claims; raw bbox extents and frame shares stay in the "
                                     "machine-readable payload, never prose."),
                "provenance": (
                    "research_harness.camera_viewing_angle.compute_camera_viewing_angle "
                    f"SHA-256 {code_hash}; computed in memory during this bounded run with no "
                    "crawlr/stratum write, no new model invoked."
                ),
                "abstention_policy": ("Abort the selected item before model generation if required artifacts "
                                      "are missing, unreadable, or detector count is not exactly one; abstain "
                                      "(emit None/band with a surfaced reason) when the seg2 subject mask is "
                                      "empty, the subject bbox is degenerate, or the subject is cropped at a "
                                      "frame edge (framing over a cropped subject would be misleading). "
                                      "Detector disagreement remains a quality anomaly, never prompt content."),
                "known_failure_modes": ("Framing bands are camera-relative and assume a single standing/posed "
                                        "subject; a subject cropped at frame edges abstains honestly. The "
                                        "subject mask uses the seg2 non-background union (corpus invariant: "
                                        "exactly one woman). Shot-scale and headroom bands depend on framing "
                                        "intent, not absolute size."),
                "qualification_gate": ("Candidate evidence only; no effectiveness claim is permitted until "
                                       "the frozen comparison receives completed rubric and adversarial reviews."),
            }
        ],
    }
    evidence["fingerprint"] = _evidence_fingerprint(evidence)
    return evidence


def _serialize_camera_viewing_angle(framing: Mapping[str, Any]) -> str:
    """Deterministic natural-language rendering of a framing dict.

    Verbalizes ONLY the scale-invariant camera-relative framing bands. Raw
    bbox extents / frame shares remain in the machine-readable
    `evidence_payload` JSON and are not caption claims.
    """
    lines = [
        "CAMERA-VIEWING-ANGLE (framing, scale-invariant):"
    ]
    if framing.get("abstained"):
        reason = framing.get("abstention_reason") or "framing not measurable"
        lines.append(f"- camera framing abstained ({reason})")
        return "\n".join(lines)
    if not framing or not framing.get("headroom_band"):
        lines.append("- camera framing not measured for this item")
        return "\n".join(lines)
    hroom = framing.get("headroom_band")
    # shot_scale_band + camera_height_band are payload-only (88% full-body /
    # 100% eye-level on the probe cohort — degenerate uniform axes, never
    # verbalized).
    if hroom == "tight":
        lines.append("- headroom is tight (head near the frame top)")
    elif hroom == "wide":
        lines.append("- headroom is wide (ample space above the head)")
    return "\n".join(lines)


def _image_focus_evidence() -> dict[str, Any]:
    """Declared deterministic image-focus / depth-of-field specialist (arm #75)."""
    module_path = Path(compute_image_focus.__code__.co_filename)
    code_hash = _sha256(module_path.read_bytes())
    evidence: dict[str, Any] = {
        "kind": "specialist_bundle",
        "id": "in-memory-image-focus-v1",
        "specialists": [
            {
                "id": "in-memory-image-focus-v1",
                "scope": "Scale-invariant optical focus / depth-of-field quality with no new model: subject-region vs full-frame interior acutance band (subject softer / comparable / crispest part of the frame) and background-vs-subject interior acutance ratio band (background blurred / soft / sharp — the depth-of-field axis), both measured on the region INTERIOR of the canonical-512 luminance gradient (silhouette halo excluded). Never identity, skin-tone, facial, or aesthetic claims; only the scale-invariant focus/DOF bands in prose.",
                "inputs": ("Frozen selected-item seg2.npy (DOME-29 subject mask, "
                           "already read for the exactly-one-subject invariant) + the already-decoded "
                           "full-frame source RGB (SHA-bound via source_sha256), resampled to a canonical "
                           "512 long side; computed in memory with no crawlr/stratum write; no new model "
                           "(CPU)."),
                "output_semantics": ("Provenance-bearing scale-invariant focus bands with a surfaced "
                                     "abstention on too-small regions or untextured backgrounds, not "
                                     "semantic ground truth or caption claims; raw acutance numbers and "
                                     "canonical dims stay in the machine-readable payload, never prose."),
                "provenance": (
                    "research_harness.image_focus.compute_image_focus "
                    f"SHA-256 {code_hash}; computed in memory during this bounded run with no "
                    "crawlr/stratum write, no new model invoked."
                ),
                "abstention_policy": ("Abort the selected item before model generation if required artifacts "
                                      "are missing, unreadable, or detector count is not exactly one; abstain "
                                      "(emit band None with a surfaced reason) when the eroded subject or "
                                      "background region is too small, or the background is untextured (no "
                                      "detail to resolve as sharp or blurred). Detector disagreement remains a "
                                      "quality anomaly, never prompt content."),
                "known_failure_modes": ("Focus bands are relative-acutance measurements at a canonical "
                                        "resolution; extreme compression artifacts or heavy downsampling can "
                                        "flatten all gradients. A flat/plain background honestly abstains on "
                                        "the DOF axis rather than guessing. The subject-vs-frame band is "
                                        "meaningless on a full-bleed subject with no background and abandons "
                                        "the DOF axis, keeping the subject band."),
                "qualification_gate": ("Candidate evidence only; no effectiveness claim is permitted until "
                                       "the frozen comparison receives completed rubric and adversarial reviews."),
            }
        ],
    }
    evidence["fingerprint"] = _evidence_fingerprint(evidence)
    return evidence


def _serialize_image_focus(focus: Mapping[str, Any]) -> str:
    """Deterministic natural-language rendering of a focus dict.

    Verbalizes ONLY the scale-invariant focus/DOF bands. Raw acutance
    numbers / canonical dims stay in the machine-readable evidence_payload
    JSON and are not caption claims.
    """
    lines = [
        "IMAGE-FOCUS (focus quality / depth of field, scale-invariant):"
    ]
    if focus.get("abstained"):
        reason = focus.get("abstention_reason") or "focus not measurable"
        lines.append(f"- focus abstained ({reason})")
        return "\n".join(lines)
    if not focus or (not focus.get("subject_focus_band") and not focus.get("dof_band")):
        lines.append("- focus not measured for this item")
        return "\n".join(lines)
    sb = focus.get("subject_focus_band")
    if sb == "subject-crisp":
        lines.append("- subject is the crispest in-focus part of the frame")
    elif sb == "subject-softer":
        lines.append("- subject looks softer than the rest of the frame")
    elif sb == "subject-comparable":
        lines.append("- subject and frame are in similar focus")
    db = focus.get("dof_band")
    if db == "background-blurred":
        lines.append("- background is clearly softer than the subject (shallow depth-of-field look)")
    elif db == "background-sharp":
        lines.append("- background is about as sharp as the subject (deep-focus look)")
    elif db == "background-soft":
        lines.append("- background is somewhat softer than the subject")
    elif focus.get("dof_abstained"):
        lines.append(
            f"- depth-of-field not assessed ({focus.get('dof_abstention_reason')})"
        )
    return "\n".join(lines)


def _geometry_evidence() -> dict[str, Any]:
    module_path = Path(derive_determinations.__code__.co_filename)
    code_hash = _sha256(module_path.read_bytes())
    evidence: dict[str, Any] = {
        "kind": "specialist_bundle",
        "id": "in-memory-geometry-determinations-v1",
        "specialists": [
            {
                "id": "in-memory-geometry-determinations-v1",
                "scope": "Pose2 and segmentation-derived continuous geometry and open-set relations only; never posture/activity labels.",
                "inputs": "Frozen selected-item pose2.npy and seg2.npy only; pointmap and all other derived artifacts are out of scope for this first evidence contrast.",
                "output_semantics": "Provenance-bearing measurements and natural-language relations, not semantic ground truth or caption claims.",
                "provenance": (
                    "stratum2.pipeline.determinations.derive_determinations "
                    f"SHA-256 {code_hash}; computed in memory during this bounded run with no crawlr/stratum write."
                ),
                "abstention_policy": "Abort the selected item before model generation if required artifacts are missing, unreadable, or detector count is not exactly one; detector disagreement remains a quality anomaly, never prompt content.",
                "known_failure_modes": "Tight crops, segmentation errors, pose errors, monocular pointmap scale limits, and incomplete body visibility can limit or omit relations.",
                "qualification_gate": "Candidate evidence only; no effectiveness claim is permitted until the frozen comparison receives completed rubric and adversarial reviews.",
            }
        ],
    }
    evidence["fingerprint"] = _evidence_fingerprint(evidence)
    return evidence


def _candidate_items(candidate: Mapping[str, Any], program: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if candidate.get("kind") != "first500-coverage-balanced-candidate-manifest":
        raise StageBRunError("candidate manifest kind is not a first500 coverage-balanced manifest")
    if candidate.get("program_id") != program["program_id"]:
        raise StageBRunError("candidate manifest program_id does not match the Stage-B program")
    if candidate.get("status") != "PENDING_PRE_COMPUTE_NON_EXECUTING":
        raise StageBRunError("candidate manifest is not the frozen pre-compute Stage-B input")
    observed_fingerprint = _require_sha256(candidate.get("manifest_fingerprint"), "candidate manifest fingerprint")
    if _canonical_fingerprint(candidate, "manifest_fingerprint") != observed_fingerprint:
        raise StageBRunError("candidate manifest fingerprint does not bind its current content")

    canonical = _require_mapping(program.get("canonical_source"), "program canonical_source")
    if candidate.get("canonical_source_root") != canonical.get("path"):
        raise StageBRunError("candidate manifest canonical source root does not match the program")
    if candidate.get("derived_artifact_root") != canonical.get("derived_tree"):
        raise StageBRunError("candidate manifest derived root does not match the program")

    raw_items = candidate.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise StageBRunError("candidate manifest must contain at least one frozen item")
    seen: set[str] = set()
    for raw_item in raw_items:
        item = _require_mapping(raw_item, "candidate manifest item")
        image_id = _safe_output_segment(item.get("image_id"), "candidate item image_id")
        if image_id in seen:
            raise StageBRunError("candidate manifest item image_ids must be unique")
        seen.add(image_id)
        _safe_relative_path(item.get("source_relative_path"), "candidate item source_relative_path")
        _require_sha256(item.get("source_sha256"), "candidate item source_sha256")
        quality = _require_mapping(item.get("quality_status"), "candidate item quality_status")
        if quality.get("detector_disagreement") is not False or quality.get("pose2_detection_count") != 1:
            raise StageBRunError("Stage-B candidate items must be one-pose non-anomaly rows")
        availability = _require_mapping(item.get("artifact_availability"), "candidate item artifact_availability")
        for required in ("pose2.npy", "seg2.npy"):
            if availability.get(required) is not True:
                raise StageBRunError(f"candidate item lacks required readable {required}")
    return list(raw_items)


# Frozen open-weight MediaPipe FaceLandmarker asset (arm #60, new model class).
FACE_GEOMETRY_MODEL_ASSET = (
    "/mnt/nas-ai-models/research/stratum/models/face-geometry/face_landmarker.task"
)

# Frozen open-weight Grounding DINO asset directory (arm #61, new model class).
# model.safetensors sha256 5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21
OBJECT_RELATIONS_MODEL_ASSET = (
    "/mnt/nas-ai-models/research/stratum/models/object-relations"
)

# Frozen open-weight CLIP ViT-L/14 asset directory (arm #69, new model class).
# model.safetensors sha256 a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb
SCENE_CATEGORY_MODEL_ASSET = (
    "/mnt/nas-ai-models/research/stratum/models/scene-category"
)

_EVIDENCE_INPUT_NAMES: dict[str, tuple[str, ...]] = {
    "geometry": ("pose2.npy", "seg2.npy"),
    "body-type": ("pose2.npy", "seg2.npy"),
    "clothing": ("pose2.npy", "seg2.npy"),
    "hair": ("pose2.npy", "seg2.npy"),
    "skin-color": ("pose2.npy", "seg2.npy"),
    "lighting": ("seg2.npy", "normal2.npy"),
    "setting": ("seg2.npy",),
    "texture": ("seg2.npy",),
    "context4k": ("pose2.npy", "seg2.npy", "normal2.npy"),
    "vlm-dense": ("pose2.npy", "seg2.npy", "normal2.npy"),
    "pose-articulation": ("pose2.npy", "seg2.npy"),
    "pointmap-depth": ("pointmap.npy", "seg2.npy"),
    "matting-alpha": ("matting.npy", "seg2.npy"),
    "face-geometry": ("seg2.npy",),
    "object-relations": ("seg2.npy",),
    # Arm #69 scene-category: CLIP ViT-L/14 consumes ONLY the already-decoded
    # full-frame source RGB (SHA-bound via the item's source_sha256). No
    # derived artifact is an evidence input; seg2/pose2 stay validation-only
    # reads for the exactly-one-subject invariant.
    "scene-category": (),
    # Arm #68 gaze-head-orientation: MediaPipe FaceLandmarker consumes the
    # already-decoded full-frame source RGB + the seg2 Face_Neck mask (for the
    # union crop fallback, as in face-geometry); seg2 is read-only validation
    # data, never mutated.
    "gaze-head-orientation": ("seg2.npy",),
    # Arm #74 camera-viewing-angle: deterministic framing from the seg2 subject
    # mask + full-frame dims only; seg2 is read-only validation data, never
    # mutated; no new model.
    "camera-viewing-angle": ("seg2.npy",),
    # Arm #75 image-focus: deterministic focus / depth-of-field quality from
    # the already-decoded source RGB (SHA-bound via source_sha256) + the seg2
    # DOME-29 class mask for the subject/background region split. seg2 is the
    # only derived evidence input; no new model.
    "image-focus": ("seg2.npy",),
}


def _evidence_input_names(evidence_kind: str) -> tuple[str, ...]:
    try:
        return _EVIDENCE_INPUT_NAMES[evidence_kind]
    except KeyError as exc:
        raise StageBRunError(f"unsupported evidence_kind for artifact hashing: {evidence_kind}") from exc


def _freeze_evidence_input_artifact_hashes(
    candidate_manifest: Mapping[str, Any], program: Mapping[str, Any], items: list[Mapping[str, Any]],
    *,
    evidence_kind: str = "geometry",
) -> dict[str, dict[str, str]]:
    """Hash the selected evidence inputs for a frozen Stage-B plan.

    The hashed set is the exact artifact set the declared evidence specialist
    consumes (default pose2+seg2; lighting binds seg2+normal2); source-image
    bytes are already bound by each candidate item's source_sha256.
    """
    canonical = _require_mapping(program["canonical_source"], "program canonical_source")
    derived_root = _resolved_existing_directory(Path(canonical["derived_tree"]), "derived artifact root")
    names = _evidence_input_names(evidence_kind)
    hashes: dict[str, dict[str, str]] = {}
    for item in items:
        image_id = _safe_output_segment(item.get("image_id"), "candidate item image_id")
        artifact_dir = _require_contained(
            derived_root / image_id, derived_root, "selected derived artifact directory"
        )
        hashes[image_id] = {}
        for name in names:
            artifact_path = _require_contained(artifact_dir / name, artifact_dir, f"selected {name}")
            try:
                payload = artifact_path.read_bytes()
                np.load(io.BytesIO(payload), allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise StageBRunError(f"selected {name} is unreadable for {image_id}") from exc
            hashes[image_id][name] = _sha256(payload)
    return hashes


def freeze_stage_b_plan(
    program: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    settings: StageBGenerationSettings,
    *,
    evidence_kind: str = "geometry",
    vlm_blocks_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Bind the exact selected geometry-input bytes before any model execution."""
    plan = build_stage_b_plan(program, candidate_manifest, settings, evidence_kind=evidence_kind,
                              vlm_blocks_sha256=vlm_blocks_sha256)
    items = _candidate_items(candidate_manifest, program)
    plan["evidence_input_artifact_sha256"] = _freeze_evidence_input_artifact_hashes(
        candidate_manifest, program, items, evidence_kind=evidence_kind
    )
    plan["comparison_plan_fingerprint"] = _canonical_fingerprint(plan, "comparison_plan_fingerprint")
    try:
        validate_comparison_parity_plan(plan, program)
    except ContractError as exc:
        raise StageBRunError(f"frozen Stage-B plan violates the comparison contract: {exc}") from exc
    return plan


def build_stage_b_plan(
    program: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    settings: StageBGenerationSettings,
    *,
    evidence_kind: str = "geometry",
    vlm_blocks_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a fully pinned four-condition comparison plan without inference.

    `evidence_kind` selects the declared deterministic evidence specialist for
    the evidence-only condition: ``"geometry"`` (arm #4 default, pose2+seg2
    determinations) or ``"body-type"`` (arm #32, pose2 proportions). The default
    is byte-identical to the historical arm-#4 plan so frozen invariants hold.

    ``"vlm-dense"`` (arm #47) builds a FIVE-condition plan: the three neutral
    controls, the deterministic dossier compact condition
    (``context-raw-context4k``) as the matched baseline, and the variant
    (``context-raw-vlm-dense``) that blends the deterministic compact with the
    pre-generated VLM dense-description block. The per-item block bytes are
    pinned by `vlm_blocks_sha256` (required for this kind), so the plan binds
    the exact learned evidence the caption runner renders.
    """
    if evidence_kind not in (
        "geometry", "body-type", "clothing", "hair", "skin-color", "lighting", "setting", "texture", "context4k",
        "vlm-dense", "pose-articulation", "pointmap-depth", "matting-alpha", "face-geometry",
        "object-relations", "scene-category", "gaze-head-orientation", "camera-viewing-angle",
        "image-focus",
    ):
        raise StageBRunError(f"unsupported Stage-B evidence_kind: {evidence_kind}")
    try:
        validate_program(program)
    except ContractError as exc:
        raise StageBRunError(f"invalid Stage-B program: {exc}") from exc
    items = _candidate_items(candidate_manifest, program)
    manifest_id = candidate_manifest.get("manifest_id")
    if not isinstance(manifest_id, str) or not manifest_id.strip() or manifest_id != manifest_id.strip():
        raise StageBRunError("candidate manifest_id must be a canonical non-empty string")

    legacy_view = _component(
        "legacy-bucketed-crop-view-v1",
        {
            "renderer": "stage_b.resize_to_cover_center_crop",
            "buckets": list(DEFAULT_ASPECT_BUCKETS),
            "resample": "Pillow.BICUBIC",
        },
    )
    raw_view = _component(
        "raw-source-view-v1",
        {"renderer": "stage_b.decoded_source_rgb", "crop": "none", "resize": "none"},
    )
    legacy_prompt = _component(
        "legacy-caption-prompt-v1",
        {"renderer": "stratum.config.CAPTION_PROMPT", "text": CAPTION_PROMPT},
    )
    context_prompt = _component(
        "context-grounded-prompt-v1",
        {
            "renderer": "research_harness.stage_b._context_prompt + stratum2.pipeline.caption2.build_prompt",
            "template": _CONTEXT_PROMPT_TEMPLATE,
            "evidence_renderer_template": CAPTION2_PROMPT_TEMPLATE,
        },
    )
    no_evidence = _no_specialist_evidence()
    if evidence_kind == "body-type":
        evidence = _body_type_evidence()
        evidence_condition_id = "context-raw-body-type"
        comparison_plan_id = "stage-b-first500-bodytype-v1"
        hypothesis = (
            "For the frozen coverage-balanced first-500 cohort, declared deterministic anthropometric "
            "proportions (shoulder:hip width ratio, torso length, limb lengths from Goliath-308 pose2) may "
            "improve supported body/limb description claims without increasing unsupported claims when the "
            "source item, view, prompt template, local model, and generation settings are controlled."
        )
        falsified_if = (
            "The body-type evidence condition does not improve the pre-registered claim-support rubric over its "
            "matched no-specialist condition on body/limb claims, or an apparent difference is attributable to an "
            "uncontrolled view, prompt, aggregator, generation, or evaluation change."
        )
        coverage_notes = (
            "All frozen rows have readable existing core artifacts; existing determinations/caption2/t52 files and pointmap "
            "are not used as evidence inputs. Proportions are computed in memory from the frozen selected pose2 only "
            "(min keypoint confidence 0.5, transform-agnostic continuous ratios with explicit abstention)."
        )
    elif evidence_kind == "clothing":
        evidence = _clothing_evidence()
        evidence_condition_id = "context-raw-clothing"
        comparison_plan_id = "stage-b-first500-clothing-v1"
        hypothesis = (
            "For the frozen coverage-balanced first-500 cohort, declared deterministic DOME-29 clothing/apparel "
            "measurements (garment class coverage from seg2 and per-class dominant colors from source pixels) may "
            "improve supported clothing/apparel description claims without increasing unsupported or contradictory "
            "claims when the source item, view, prompt template, local model, and generation settings are controlled."
        )
        falsified_if = (
            "The clothing-evidence condition does not improve the pre-registered claim-support rubric over its "
            "matched no-specialist condition on clothing/apparel claims, or an apparent difference is attributable "
            "to an uncontrolled view, prompt, aggregator, generation, or evaluation change."
        )
        coverage_notes = (
            "All frozen rows have readable existing core artifacts; existing determinations/caption2/t52 files and pointmap "
            "are not used as evidence inputs. Clothing coverage and dominant colors are computed in memory from the frozen "
            "selected seg2 and the already-decoded source pixels only (presence requires a raw-pixel floor and a "
            "foreground-coverage gate; otherwise the class abstains)."
        )
    elif evidence_kind == "hair":
        evidence = _hair_evidence()
        evidence_condition_id = "context-raw-hair"
        comparison_plan_id = "stage-b-first500-hair-v1"
        hypothesis = (
            "For the frozen coverage-balanced first-500 cohort, declared deterministic DOME-29 hair measurements "
            "(Hair-region coverage from seg2, dominant hair color from source pixels, vertical position band, and a "
            "hair-to-face vertical-extent length proxy) may improve supported hair description claims without "
            "increasing unsupported, contradictory, or invented hair-color/coverage claims when the source item, view, "
            "prompt template, local model, and generation settings are controlled."
        )
        falsified_if = (
            "The hair-evidence condition does not improve the pre-registered claim-support rubric over its "
            "matched no-specialist condition on hair claims, or an apparent improvement is attributable to an "
            "uncontrolled view, prompt, aggregator, generation, or evaluation change."
        )
        coverage_notes = (
            "All frozen rows have readable existing core artifacts; existing determinations/caption2/t52 files and pointmap "
            "are not used as evidence inputs. Hair coverage, dominant color, vertical band, and the hair-to-face extent "
            "ratio are computed in memory from the frozen selected seg2 and the already-decoded source pixels only "
            "(presence requires a raw-pixel floor and a foreground-coverage gate; otherwise the region abstains). Only "
            "scale-invariant facts are verbalized."
        )
    elif evidence_kind == "skin-color":
        evidence = _skin_color_evidence()
        evidence_condition_id = "context-raw-skin-color"
        comparison_plan_id = "stage-b-first500-skin-color-v1"
        hypothesis = (
            "For the frozen coverage-balanced first-500 cohort, declared deterministic DOME-29 exposed-skin "
            "measurements (aggregate exposed-skin coverage from seg2 and a quantized dominant skin tone from source "
            "pixels, with face/neck vs body agreement) may improve supported skin-tone/color description claims without "
            "increasing unsupported, contradictory, or invented skin-color claims when the source item, view, prompt "
            "template, local model, and generation settings are controlled."
        )
        falsified_if = (
            "The skin-color evidence condition does not improve the pre-registered claim-support rubric over its "
            "matched no-specialist condition on skin-color claims, or an apparent improvement is attributable to an "
            "uncontrolled view, prompt, aggregator, generation, or evaluation change."
        )
        coverage_notes = (
            "All frozen rows have readable existing core artifacts; existing determinations/caption2/t52 files and pointmap "
            "are not used as evidence inputs. Exposed-skin coverage and dominant skin tone are computed in memory from "
            "the frozen selected seg2 and the already-decoded source pixels only (presence requires a raw-pixel floor "
            "and a foreground-coverage gate; otherwise the region abstains). Only scale-invariant facts are verbalized."
        )
    elif evidence_kind == "lighting":
        evidence = _lighting_evidence()
        evidence_condition_id = "context-raw-lighting"
        comparison_plan_id = "stage-b-first500-lighting-v1"
        hypothesis = (
            "For the frozen coverage-balanced first-500 cohort, declared deterministic lighting measurements "
            "(overall exposure band, tonal dynamic-range band, shadow fraction, subject-vs-surround ratio, and a coarse "
            "key-light direction fit from normal2) may improve supported lighting/shadow/atmosphere description claims "
            "without increasing unsupported, contradictory, or invented lighting claims when the source item, view, prompt "
            "template, local model, and generation settings are controlled."
        )
        falsified_if = (
            "The lighting-evidence condition does not improve the pre-registered claim-support rubric over its "
            "matched no-specialist condition on lighting/shadow claims, or an apparent improvement is attributable to an "
            "uncontrolled view, prompt, aggregator, generation, or evaluation change."
        )
        coverage_notes = (
            "All frozen rows have readable existing core artifacts; existing determinations/caption2/t52 files and pointmap "
            "are not used as evidence inputs. Exposure band, dynamic-range band, shadow fraction, surround ratio, and the "
            "key-light direction fit are computed in memory from the frozen selected normal2 + seg2 and the already-decoded "
            "source pixels only (presence requires a raw-pixel floor and a foreground-coverage gate; the direction fit "
            "additionally requires sufficient valid camera-facing normals, otherwise it abstains). Only scale-invariant "
            "facts are verbalized; absolute luma/normal values stay in evidence_payload."
        )
    elif evidence_kind == "setting":
        evidence = _setting_evidence()
        evidence_condition_id = "context-raw-setting"
        comparison_plan_id = "stage-b-first500-setting-v1"
        hypothesis = (
            "For the frozen coverage-balanced first-500 cohort, declared deterministic setting/environment "
            "measurements (Background-class frame coverage from seg2, quantized dominant background color, "
            "tone/vibrancy/pattern bands from the non-subject source pixels) may improve supported "
            "setting/environment/background description claims without increasing unsupported, contradictory, "
            "or invented background details when the source item, view, prompt template, local model, and "
            "generation settings are controlled."
        )
        falsified_if = (
            "The setting-evidence condition does not improve the pre-registered claim-support rubric over its "
            "matched no-specialist condition on setting/environment claims, or an apparent difference is "
            "attributable to an uncontrolled view, prompt, aggregator, generation, or evaluation change."
        )
        coverage_notes = (
            "All frozen rows have readable existing core artifacts; existing determinations/caption2/t52 files "
            "and pointmap are not used as evidence inputs. Background coverage, dominant color, and the "
            "tone/vibrancy/pattern bands are computed in memory from the frozen selected seg2 (class 0) and the "
            "already-decoded source pixels only (presence requires a raw-pixel floor and a frame-coverage "
            "floor; otherwise the region abstains). Only scale-invariant facts are verbalized; absolute pixel "
            "counts and raw RGB values stay in evidence_payload."
        )
    elif evidence_kind == "texture":
        evidence = _texture_evidence()
        evidence_condition_id = "context-raw-texture"
        comparison_plan_id = "stage-b-first500-texture-v1"
        hypothesis = (
            "For the frozen coverage-balanced first-500 cohort, declared deterministic texture/material "
            "measurements (per-region-class edge/gradient statistics from seg2 masks + source pixels: "
            "dominant fabric class texture/pattern bands, dominant skin class surface band) may improve "
            "supported texture/material/fabric description claims without increasing unsupported, "
            "contradictory, or invented material details when the source item, view, prompt template, "
            "local model, and generation settings are controlled."
        )
        falsified_if = (
            "The texture-evidence condition does not improve the pre-registered claim-support rubric over its "
            "matched no-specialist condition on texture/material claims, or an apparent difference is "
            "attributable to an uncontrolled view, prompt, aggregator, generation, or evaluation change."
        )
        coverage_notes = (
            "All frozen rows have readable existing core artifacts; existing determinations/caption2/t52 files "
            "and pointmap are not used as evidence inputs. Texture statistics are computed in memory from the "
            "frozen selected seg2 (per class mask) and the already-decoded source pixels only (a region class "
            "requires a raw-pixel floor and a frame-coverage floor; otherwise the fact abstains). Statistics "
            "are per-channel-99.9-percentile-normalized so exposure/white balance cannot inflate gradients; "
            "only scale-invariant bands/fractions are verbalized; absolute gradient values and pixel counts "
            "stay in evidence_payload."
        )
    elif evidence_kind == "pose-articulation":
        evidence = _pose_articulation_evidence()
        evidence_condition_id = "context-raw-pose-articulation"
        comparison_plan_id = "stage-b-first500-pose-articulation-v1"
        hypothesis = (
            "For the frozen coverage-balanced first-500 cohort, declared deterministic pose-articulation "
            "measurements (per-joint elbow/knee flexion angles, in-plane torso/pelvis orientation, weight-bearing "
            "stance + contrapposto class, limb-overlap/crossing structure, flexion asymmetry — all scale-invariant "
            "kinematics computed in memory from pose2 GOLIATH-308 keypoints + seg2 limb masks) may reduce "
            "unsupported pose/stance/limb claims in captions versus its matched no-evidence baseline when the "
            "source item, view, prompt template, local model, and generation settings are controlled."
        )
        falsified_if = (
            "The pose-articulation evidence condition does not reduce unsupported pose/stance/limb claims or "
            "increase supported claims versus its matched no-evidence baseline, or the measured kinematics are "
            "not distinguishable from body-type's static ratios (redundant axis), or an apparent difference is "
            "attributable to an uncontrolled view, prompt, aggregator, generation, or evaluation change."
        )
        coverage_notes = (
            "All frozen rows have readable existing core artifacts; existing determinations/caption2/t52 files "
            "and pointmap are not used as evidence inputs. Kinematic articulation is computed in memory from the "
            "frozen selected pose2 (GOLIATH-308 keypoints) + seg2 (DOME-29 limb/region masks) only. Only "
            "scale-invariant facts are verbalized: joint-flexion bands (bent vs extended), stance class "
            "(weight on a leg / centered), contrapposto, torso/pelvis in-plane angles, arm-crossing structure, "
            "and flexion asymmetry. Absolute pixel positions/lengths and raw angles stay in evidence_payload "
            "and are never caption claims."
        )
    elif evidence_kind == "pointmap-depth":
        evidence = _pointmap_depth_evidence()
        evidence_condition_id = "context-raw-pointmap-depth"
        comparison_plan_id = "stage-b-first500-pointmap-depth-v1"
        hypothesis = (
            "For the frozen coverage-balanced first-500 cohort, declared deterministic point-map depth-order "
            "measurements (region nearest/farthest depth ranking, left/right hand depth ordering, hand/arm held "
            "in front of the torso plane, and the normalized body depth-relief ratio — all scale-invariant and "
            "computed in memory from the frozen pointmap.npy CAM-frame cloud + seg2.npy DOME-29 masks) may "
            "reduce unsupported depth/occlusion/foreground-background claims in captions versus its matched "
            "no-evidence baseline when the source item, view, prompt template, local model, and generation "
            "settings are controlled."
        )
        falsified_if = (
            "The pointmap-depth evidence condition does not reduce unsupported depth/occlusion/foreground-"
            "background claims or increase supported claims versus its matched no-evidence baseline, or the "
            "measured depth ordering is not distinguishable from pose-articulation/body-type's 2D proxies "
            "(redundant axis), or an apparent difference is attributable to an uncontrolled view, prompt, "
            "aggregator, generation, or evaluation change."
        )
        coverage_notes = (
            "All frozen rows have readable existing core artifacts; existing determinations/caption2/t52 files "
            "and pose2 are not used as evidence inputs for the depth measurement itself (pose2 stays a "
            "validation-only read for the exactly-one-subject invariant). Relative depth ordering is computed "
            "in memory from the frozen selected pointmap (Sapiens2 per-pixel CAM-frame cloud, background "
            "zeroed) + seg2 (DOME-29 region masks) only. Only scale-invariant facts are verbalized: region "
            "depth ordering (nearest/farthest part), hand depth ordering, hand/arm in front of the torso "
            "plane, and the depth-relief band. Absolute metric Z values, raw spreads, and per-region median Zs "
            "stay in evidence_payload and are never caption claims."
        )
    elif evidence_kind == "matting-alpha":
        evidence = _matting_alpha_evidence()
        evidence_condition_id = "context-raw-matting-alpha"
        comparison_plan_id = "stage-b-first500-matting-alpha-v1"
        hypothesis = (
            "For the frozen coverage-balanced first-500 cohort, declared deterministic matting/alpha-fidelity "
            "measurements (subject alpha-coverage band, boundary crispness band of the silhouette edge, and "
            "soft-edge character — hair-dominant vs clean skin cutout — all scale-invariant and computed in "
            "memory from the frozen matting.npy alpha matte + seg2.npy DOME-29 masks) may reduce unsupported "
            "edge-quality/hair/silhouette claims in captions versus its matched no-evidence baseline when the "
            "source item, view, prompt template, local model, and generation settings are controlled."
        )
        falsified_if = (
            "The matting evidence condition does not reduce unsupported edge-quality/fine-feature claims or "
            "increase supported claims versus its matched no-evidence baseline, or the measured alpha structure "
            "is not distinguishable from seg2 region coverage (redundant axis), or an apparent difference is "
            "attributable to an uncontrolled view, prompt, aggregator, generation, or evaluation change."
        )
        coverage_notes = (
            "All frozen rows have readable existing core artifacts; existing determinations/caption2/t52 files "
            "and pose2 are not used as evidence inputs for the matting measurement itself (pose2 stays a "
            "validation-only read for the exactly-one-subject invariant). Alpha fidelity is computed in memory "
            "from the frozen selected matting.npy (Sapiens2 per-pixel alpha matte) + seg2 (DOME-29 Hair mask) "
            "only. Only scale-invariant facts are verbalized: coverage band, boundary crispness band, and "
            "soft-edge character. Absolute pixel areas/band widths and silhouette structure stay in "
            "evidence_payload and are never caption claims."
        )
    elif evidence_kind == "face-geometry":
        evidence = _face_geometry_evidence()
        evidence_condition_id = "context-raw-face-geometry"
        comparison_plan_id = "stage-b-first500-face-geometry-v1"
        hypothesis = (
            "For the frozen coverage-balanced first-500 cohort, declared facial-geometry measurements "
            "(eye-spacing, mouth-width, jaw-width, and plausibility-gated mid-face bands from a local "
            "open-weight MediaPipe FaceLandmarker 478-point mesh over the full frame / seg2 Face_Neck "
            "crop, union detection policy, all scale-invariant) may reduce unsupported facial-detail "
            "claims in captions versus its matched no-evidence baseline when the source item, view, "
            "prompt template, local model, and generation settings are controlled."
        )
        falsified_if = (
            "The face-geometry evidence condition does not reduce unsupported facial-detail claims or "
            "increase supported claims versus its matched no-evidence baseline, or the landmark model "
            "fails qualification on real corpus items (high abstention / missing faces), or an apparent "
            "difference is attributable to an uncontrolled view, prompt, aggregator, generation, or "
            "evaluation change."
        )
        coverage_notes = (
            "All frozen rows have readable existing core artifacts; existing determinations/caption2/t52 "
            "files and pose2 are not used as evidence inputs for the face-geometry measurement (pose2 "
            "stays a validation-only read for the exactly-one-subject invariant). Facial geometry is computed "
            "in memory from the frozen selected seg2.npy (DOME-29 Face_Neck mask) + the already-decoded "
            "source RGB via the local open-weight MediaPipe FaceLandmarker (Apache-2.0, owned hardware, "
            "CPU, tasks API; model face_landmarker.task bound by sha256). Only scale-invariant facts are "
            "verbalized: eye-spacing / mouth / jaw / mid-face bands. Landmark coordinates and the pixel "
            "bbox stay in evidence_payload and are never caption claims."
        )
    elif evidence_kind == "object-relations":
        evidence = _object_relations_evidence()
        evidence_condition_id = "context-raw-object-relations"
        comparison_plan_id = "stage-b-first500-object-relations-v1"
        hypothesis = (
            "For the frozen coverage-balanced first-500 cohort, declared object/accessory-presence and "
            "spatial-placement measurements (count band none/sparse/moderate/dense, placement band "
            "foreground/background/mix, canonical class list from a locally-run open-weight Grounding DINO "
            "text-grounded detector over a frozen closed vocabulary, all scale-invariant) may reduce "
            "unsupported object claims (hallucinated objects) or increase supported object/background claims "
            "in captions versus its matched no-evidence baseline when the source item, view, prompt template, "
            "local model, and generation settings are controlled."
        )
        falsified_if = (
            "The object-relations evidence condition does not reduce unsupported object claims or increase "
            "supported claims versus its matched no-evidence baseline, or the grounding-dino model fails "
            "qualification on real corpus items (high abstention / subject-self confusion), or an apparent "
            "difference is attributable to an uncontrolled view, prompt, aggregator, generation, or "
            "evaluation change."
        )
        coverage_notes = (
            "All frozen rows have readable existing core artifacts; existing determinations/caption2/t52 "
            "files and pose2 are not used as evidence inputs for the object-relations measurement (pose2 "
            "stays a validation-only read for the exactly-one-subject invariant). Object presence is computed "
            "in memory from the frozen selected seg2.npy (DOME-29 subject mask) + the already-decoded source "
            "RGB via the local open-weight Grounding DINO (IDEA-Research grounding-dino-base, Apache-2.0, "
            "owned hardware, CPU via HF Transformers; model.safetensors bound by sha256) over a frozen "
            "cohort-derived closed vocabulary at the calibrated box threshold (0.25). Only scale-invariant "
            "facts are verbalized: count band, placement band, and the canonical class list. Subject-self "
            "detections ('body') are excluded, and normalized boxes/scores stay in evidence_payload and are "
            "never caption claims."
        )
    elif evidence_kind == "scene-category":
        evidence = _scene_category_evidence()
        evidence_condition_id = "context-raw-scene-category"
        comparison_plan_id = "stage-b-first500-scene-category-v1"
        hypothesis = (
            "For the frozen coverage-balanced first-500 cohort, declared semantic scene-category "
            "determinations (one closed-set label — indoor studio / plain wall backdrop / bedroom / living "
            "room / outdoor beach / outdoor garden / outdoor field / body of water / urban street / poolside — "
            "from a locally-run open-weight CLIP ViT-L/14 zero-shot classifier over the full-frame source at a "
            "calibrated confidence floor, scale-invariant label only) may reduce unsupported "
            "background/scene claims (what-kind-of-place the setting actually is) or increase supported "
            "scene claims in captions versus its matched no-evidence baseline when the source item, view, "
            "prompt template, local model, and generation settings are controlled."
        )
        falsified_if = (
            "The scene-category evidence condition does not reduce unsupported background/scene claims or "
            "increase supported claims versus its matched no-evidence baseline, or the CLIP classifier fails "
            "qualification on real corpus items (high abstention / random labels), or an apparent difference "
            "is attributable to an uncontrolled view, prompt, aggregator, generation, or evaluation change."
        )
        coverage_notes = (
            "All frozen rows have readable existing core artifacts; existing determinations/caption2/t52 "
            "files and pointmap are not used as evidence inputs; seg2/pose2 stay validation-only reads for "
            "the exactly-one-subject invariant. Scene category is computed in memory from the already-decoded "
            "full-frame source RGB via the local open-weight CLIP ViT-L/14 zero-shot classifier "
            "(openai/clip-vit-large-patch14, MIT, owned hardware, CPU; model.safetensors bound by sha256) "
            "over a frozen closed scene-category set (cohort-derived from the arm-#47 VLM dense-description "
            "scene vocabulary) at the calibrated confidence floor (0.25). Only the scale-invariant label is "
            "verbalized; similarity logits/probabilities stay in evidence_payload and are never caption claims."
        )
    elif evidence_kind == "gaze-head-orientation":
        evidence = _gaze_head_evidence()
        evidence_condition_id = "context-raw-gaze-head-orientation"
        comparison_plan_id = "stage-b-first500-gaze-head-orientation-v1"
        hypothesis = (
            "For the frozen coverage-balanced first-500 cohort, declared camera-interaction "
            "head-orientation determinations (scale-invariant yaw band — facing camera / partially turned / "
            "profile or turned away; pitch band — level / tilted down / tilted up; roll band — level / "
            "tilted in-plane — from a locally-run open-weight MediaPipe FaceLandmarker 478-point mesh over "
            "the full frame / seg2 Face_Neck crop via the union policy, camera-relative direction only) may "
            "reduce unsupported camera-interaction claims ('looking at the camera', 'head turned away') or "
            "increase supported relational claims in captions versus its matched no-evidence baseline when "
            "the source item, view, prompt template, local model, and generation settings are controlled."
        )
        falsified_if = (
            "The gaze/head-orientation evidence condition does not reduce unsupported camera-interaction "
            "claims or increase supported claims versus its matched no-evidence baseline, or the face mesh "
            "fails qualification on real corpus items (high abstention / degenerate orientation), or an "
            "apparent difference is attributable to an uncontrolled view, prompt, aggregator, generation, "
            "or evaluation change."
        )
        coverage_notes = (
            "All frozen rows have readable existing core artifacts; existing determinations/caption2/t52 "
            "files and pointmap are not used as evidence inputs for the orientation measurement (pose2 "
            "stays a validation-only read for the exactly-one-subject invariant). Head orientation is "
            "computed in memory from the frozen selected seg2.npy (DOME-29 Face_Neck mask) + the "
            "already-decoded source RGB via the local open-weight MediaPipe FaceLandmarker (Apache-2.0, "
            "owned hardware, CPU, tasks API; model face_landmarker.task bound by sha256 — the SAME "
            "validated asset as arm #60 face-geometry) with a six-point PnP camera-relative fit. Only "
            "scale-invariant direction bands are verbalized (yaw/pitch/roll bands); raw yaw/pitch/roll "
            "degrees and landmark coordinates stay in evidence_payload and are never caption claims."
        )
    elif evidence_kind == "camera-viewing-angle":
        evidence = _camera_viewing_angle_evidence()
        evidence_condition_id = "context-raw-camera-viewing-angle"
        comparison_plan_id = "stage-b-first500-camera-viewing-angle-v1"
        hypothesis = (
            "For the frozen coverage-balanced first-500 cohort, declared deterministic camera-viewing-angle "
            "determinations (shot-scale band close-up / mid-shot / full-body, headroom band, and "
            "camera-height band eye-level / low-angle / high-angle — all scale-invariant framing from the "
            "seg2 subject bbox + full-frame geometry, no new model) may reduce unsupported "
            "camera-perspective/framing claims ('close-up portrait', 'shot from above') or increase "
            "supported framing claims in captions versus its matched no-evidence baseline when the source "
            "item, view, prompt template, local model, and generation settings are controlled."
        )
        falsified_if = (
            "The camera-viewing-angle evidence condition does not reduce unsupported camera/framing claims "
            "or increase supported claims versus its matched no-evidence baseline, or the framing bands "
            "fail on real corpus items (high abstention / degenerate bbox), or an apparent difference is "
            "attributable to an uncontrolled view, prompt, aggregator, generation, or evaluation change."
        )
        coverage_notes = (
            "All frozen rows have readable existing core artifacts; existing determinations/caption2/t52 "
            "files and pointmap are not used as evidence inputs; pose2 stays a validation-only read for "
            "the exactly-one-subject invariant. Framing is computed in memory from the frozen selected "
            "seg2.npy (DOME-29 subject mask) + the full-frame decoded source image dims — no new model, "
            "CPU only. Only scale-invariant framing bands are verbalized (shot-scale / headroom / "
            "camera-height); raw bbox extents and frame shares stay in evidence_payload and are never "
            "caption claims."
        )
    elif evidence_kind == "image-focus":
        evidence = _image_focus_evidence()
        evidence_condition_id = "context-raw-image-focus"
        comparison_plan_id = "stage-b-first500-image-focus-v1"
        hypothesis = (
            "For the frozen coverage-balanced first-500 cohort, declared deterministic image-focus / "
            "depth-of-field determinations (subject-region vs full-frame interior acutance band and "
            "background-vs-subject interior acutance ratio band — blurred / soft / sharp background, "
            "the DOF axis; all scale-invariant, measured on the region INTERIOR of the canonical-512 "
            "luminance gradient, no new model) may reduce unsupported focus/DOF claims ('in sharp focus', "
            "'blurred background') or increase supported focus claims in captions versus its matched "
            "no-evidence baseline when the source item, view, prompt template, local model, and generation "
            "settings are controlled."
        )
        falsified_if = (
            "The image-focus evidence condition does not reduce unsupported focus/DOF claims or increase "
            "supported claims versus its matched no-evidence baseline, or the focus bands fail on real "
            "corpus items (high abstention / degenerate acutance distributions), or an apparent difference "
            "is attributable to an uncontrolled view, prompt, aggregator, generation, or evaluation change."
        )
        coverage_notes = (
            "All frozen rows have readable existing core artifacts; existing determinations/caption2/t52 "
            "files and pointmap are not used as evidence inputs; pose2 stays a validation-only read for "
            "the exactly-one-subject invariant. Focus is computed in memory from the frozen selected "
            "seg2.npy (DOME-29 subject mask) + the already-decoded full-frame source RGB — no new model, "
            "CPU only. Only scale-invariant focus/DOF bands are verbalized; raw acutance numbers and "
            "canonical dims stay in evidence_payload and are never caption claims."
        )
    elif evidence_kind == "context4k":
        evidence = _context4k_evidence()
        evidence_condition_id = "context-raw-context4k"
        comparison_plan_id = "stage-b-roundtrip-context4k-v1"
        hypothesis = (
            "For the frozen coverage-balanced first-500 cohort, captions generated FROM the evidence-linked <=4K compact "
            "context (the deterministic dossier compressed claim-by-claim with every claim carrying its supporting evidence "
            "ids) may improve supported description claims without increasing unsupported claims versus the matched "
            "plain-4K summarization baseline when the source item, view, prompt template, local model, and generation "
            "settings are controlled."
        )
        falsified_if = (
            "The context4k evidence condition does not improve the pre-registered claim-support rubric over its matched "
            "no-specialist condition, or an apparent difference is attributable to an uncontrolled view, prompt, "
            "aggregator, generation, or evaluation change."
        )
        coverage_notes = (
            "All frozen rows have readable existing core artifacts; existing determinations/caption2/t52 files and pointmap "
            "are not used as evidence inputs. The compact context is assembled deterministically in memory from the frozen "
            "selected pose2+seg2+normal2 and the already-decoded source pixels (the five validated dimension specialists "
            "plus relational determinations), then compressed to the <=4K budget claim-by-claim with evidence ids retained; "
            "an under-budget compact is recorded honestly (the deterministic corpus is 387-648 dossier tokens/item) and is "
            "never padded with fabricated filler. Only scale-invariant facts are verbalized; absolute pixel measurements "
            "stay in the machine-readable evidence payload."
        )
    elif evidence_kind == "vlm-dense":
        if vlm_blocks_sha256 is None:
            raise StageBRunError("vlm-dense evidence_kind requires vlm_blocks_sha256")
        evidence = _vlm_dense_evidence()
        evidence_condition_id = "context-raw-vlm-dense"
        comparison_plan_id = "stage-b-vlm-dense-v1"
        hypothesis = (
            "For the frozen coverage-balanced first-500 cohort, adding the pre-generated open-weight "
            "VLM dense multi-view description block (qwen3-vl:32b, full frame + seg2 focal crops, "
            "tagged observed/inferred/abstained, scale-invariant prose) to the deterministic dossier "
            "compact may materially raise supported description claims beyond the deterministic dossier "
            "alone (option-B evidence growth toward the expanded-dossier floor) without increasing "
            "unsupported or contradictory claims when the source item, view, prompt template, local "
            "model, and generation settings are controlled."
        )
        falsified_if = (
            "The VLM dense-description evidence condition does not improve the pre-registered "
            "claim-support rubric over its matched deterministic-dossier compact condition (the "
            "vlm-marginal contrast), or an apparent difference is attributable to an uncontrolled "
            "view, prompt, aggregator, generation, or evaluation change."
        )
        coverage_notes = (
            "All frozen rows have readable existing core artifacts; existing determinations/caption2/t52 "
            "files and pointmap are not used as evidence inputs. The deterministic dossier compact is "
            "assembled in memory from the frozen pose2+seg2+normal2 and source pixels (the validated "
            "deterministic dimensions) exactly as the arm-36 round-trip condition. The variant blends "
            "that compact with the pre-generated VLM dense block (pinned byte-for-byte by "
            "vlm_blocks_sha256; block provenance: qwen3-vl:32b via local Ollama on Strix under the "
            "scheduler, noncanonical stage root). Only scale-invariant facts are verbalized; absolute "
            "pixel measurements stay in evidence_payload; the <=4K compact ceiling governs the compact "
            "artifact itself, while the evidence text tested here is the expanded dossier record."
        )
    else:
        evidence = _geometry_evidence()
        evidence_condition_id = "context-raw-geometry"
        comparison_plan_id = "stage-b-first500-parity-v1"
        hypothesis = (
            "For the frozen coverage-balanced first-500 cohort, declared in-memory deterministic geometry "
            "may improve supported contextual coverage without increasing unsupported or contradictory claims "
            "when the source item, view, prompt template, local model, and generation settings are controlled."
        )
        falsified_if = (
            "The evidence-only contrast does not improve the pre-registered human claim-support rubric over its "
            "matched no-specialist condition, or an apparent difference is attributable to an uncontrolled view, "
            "prompt, aggregator, generation, or evaluation change."
        )
        coverage_notes = (
            "All frozen rows have readable existing core artifacts; existing determinations/caption2/t52 files and pointmap "
            "are not used as evidence inputs. Geometry is recomputed in memory from selected existing pose2/seg2 inputs only."
        )
    aggregator = {
        "model_id": f"local-ollama-{settings.model_name.replace(':', '-').replace('/', '-')}-{settings.model_digest[:12]}",
        "provenance": (
            f"Already-installed local Ollama model {settings.model_name} digest {settings.model_digest}; "
            f"loopback endpoint {settings.endpoint}; fixed request settings fingerprint {settings.fingerprint}."
        ),
        "generation_fingerprint": settings.fingerprint,
        "local_only": True,
    }

    def condition(
        condition_id: str,
        input_view: Mapping[str, Any],
        prompt: Mapping[str, Any],
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "id": condition_id,
            "pilot_manifest_id": manifest_id,
            "input_view": dict(input_view),
            "prompt": dict(prompt),
            "evidence": dict(evidence),
            "aggregator": dict(aggregator),
        }

    plan: dict[str, Any] = {
        "schema_version": 1,
        "kind": "comparison-parity-plan",
        "comparison_plan_id": comparison_plan_id,
        "program_id": program["program_id"],
        "status": "PENDING",
        "parent_issue": candidate_manifest.get("parent_issue", 4),
        "candidate_manifest_fingerprint": candidate_manifest["manifest_fingerprint"],
        "hypothesis": hypothesis,
        "falsified_if": falsified_if,
        "metric_version": "claim-support-rubric-v1",
        "pilot_manifest": {
            "id": manifest_id,
            "source_root": program["canonical_source"]["path"],
            "frozen": True,
            "selection_rationale": (
                "The pre-frozen first-500 coverage-balanced 12 portrait / 6 squareish / 6 landscape cohort "
                "removes missing core geometry availability as a confound without claiming population representativeness."
            ),
            "coverage_notes": coverage_notes,
            "items": [
                {
                    "image_id": item["image_id"],
                    "source_relative_path": item["source_relative_path"],
                    "source_sha256": item["source_sha256"],
                    "artifact_availability": dict(item["artifact_availability"]),
                }
                for item in items
            ],
        },
        "conditions": [
            condition("legacy-bucketed-no-evidence", legacy_view, legacy_prompt, no_evidence),
            condition("legacy-raw-no-evidence", raw_view, legacy_prompt, no_evidence),
            condition("context-raw-no-evidence", raw_view, context_prompt, no_evidence),
            condition("context-raw-context4k", raw_view, context_prompt, _context4k_evidence()),
            condition(evidence_condition_id, raw_view, context_prompt, evidence),
        ] if evidence_kind == "vlm-dense" else [
            condition("legacy-bucketed-no-evidence", legacy_view, legacy_prompt, no_evidence),
            condition("legacy-raw-no-evidence", raw_view, legacy_prompt, no_evidence),
            condition("context-raw-no-evidence", raw_view, context_prompt, no_evidence),
            condition(evidence_condition_id, raw_view, context_prompt, evidence),
        ],
        "contrasts": [
            {
                "id": "input-view-only",
                "baseline_condition": "legacy-bucketed-no-evidence",
                "variant_condition": "legacy-raw-no-evidence",
                "changed_axes": ["input_view"],
            },
            {
                "id": "prompt-only",
                "baseline_condition": "legacy-raw-no-evidence",
                "variant_condition": "context-raw-no-evidence",
                "changed_axes": ["prompt"],
            },
            {
                "id": "evidence-only",
                "baseline_condition": "context-raw-no-evidence",
                "variant_condition": "context-raw-context4k",
                "changed_axes": ["evidence"],
            },
            {
                "id": "vlm-marginal",
                "baseline_condition": "context-raw-context4k",
                "variant_condition": evidence_condition_id,
                "changed_axes": ["evidence"],
            },
        ] if evidence_kind == "vlm-dense" else [
            {
                "id": "input-view-only",
                "baseline_condition": "legacy-bucketed-no-evidence",
                "variant_condition": "legacy-raw-no-evidence",
                "changed_axes": ["input_view"],
            },
            {
                "id": "prompt-only",
                "baseline_condition": "legacy-raw-no-evidence",
                "variant_condition": "context-raw-no-evidence",
                "changed_axes": ["prompt"],
            },
            {
                "id": "evidence-only",
                "baseline_condition": "context-raw-no-evidence",
                "variant_condition": evidence_condition_id,
                "changed_axes": ["evidence"],
            },
        ],
        "review_protocol": {
            "human_review_required": True,
            "sequence": [
                "selected_input_view",
                "provenance_evidence",
                "candidate_output_or_context",
                "decision_rubric",
            ],
            "fields": list(_REQUIRED_REVIEW_FIELDS),
            "detector_disagreement_handling": "quality_anomaly_not_caption_content",
        },
        "metric_self_audit": {
            "before_comparative_inference": True,
            "known_case_item_id": items[0]["image_id"],
            "null_output_id": "empty-caption-null-v1",
            "evaluator_version": "claim-support-rubric-v1",
        },
        "adversarial_review": {
            "planned": True,
            "checks": [
                "metric_definition_stable",
                "fresh_process_or_second_review",
                "edge_case_inspection",
            ],
        },
        "representation_boundary": {
            "legacy_text_encoder_max_tokens": program["representation"]["legacy_text_encoder_max_tokens"],
            "compact_context_routing": "out_of_scope",
            "no_silent_legacy_routing": True,
        },
    }
    if evidence_kind == "vlm-dense":
        item_ids = {_safe_output_segment(item.get("image_id"), "candidate item image_id") for item in items}
        if set(vlm_blocks_sha256 or {}) != item_ids:
            raise StageBRunError("vlm_blocks_sha256 must cover exactly the frozen items")
        for item_id, block_sha in (vlm_blocks_sha256 or {}).items():
            if not _SHA256_RE.fullmatch(block_sha):
                raise StageBRunError(f"vlm_blocks_sha256[{item_id}] must be a lowercase SHA-256 digest")
        plan["vlm_blocks_sha256"] = dict(sorted((vlm_blocks_sha256 or {}).items()))
        plan["vlm_model_digest"] = "a418f5838eaf"
        plan["vlm_prompt_fingerprint"] = "2106b16cf26fd97457b8d399208262e02fc2f9bb4aa5c35bb9b6b9bdf26cbddf"
    plan["comparison_plan_fingerprint"] = _canonical_fingerprint(plan, "comparison_plan_fingerprint")
    try:
        validate_comparison_parity_plan(plan, program)
    except ContractError as exc:
        raise StageBRunError(f"Stage-B plan violates the comparison contract: {exc}") from exc
    return plan


def _validate_frozen_execution_plan(
    expected_plan: Mapping[str, Any],
    program: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    settings: StageBGenerationSettings,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Validate a persisted plan and extract its exact selected evidence hashes."""
    plan = dict(_require_mapping(expected_plan, "expected comparison plan"))
    observed_fingerprint = _require_sha256(
        plan.get("comparison_plan_fingerprint"), "expected comparison plan fingerprint"
    )
    if _canonical_fingerprint(plan, "comparison_plan_fingerprint") != observed_fingerprint:
        raise StageBRunError("expected comparison plan fingerprint does not bind its current content")
    try:
        validate_comparison_parity_plan(plan, program)
    except ContractError as exc:
        raise StageBRunError(f"expected comparison plan violates the comparison contract: {exc}") from exc

    condition_ids = {condition.get("id") for condition in plan.get("conditions", [])}
    if "context-raw-clothing" in condition_ids:
        rebuild_kind = "clothing"
    elif "context-raw-hair" in condition_ids:
        rebuild_kind = "hair"
    elif "context-raw-body-type" in condition_ids:
        rebuild_kind = "body-type"
    elif "context-raw-skin-color" in condition_ids:
        rebuild_kind = "skin-color"
    elif "context-raw-lighting" in condition_ids:
        rebuild_kind = "lighting"
    elif "context-raw-setting" in condition_ids:
        rebuild_kind = "setting"
    elif "context-raw-texture" in condition_ids:
        rebuild_kind = "texture"
    elif "context-raw-pose-articulation" in condition_ids:
        rebuild_kind = "pose-articulation"
    elif "context-raw-pointmap-depth" in condition_ids:
        rebuild_kind = "pointmap-depth"
    elif "context-raw-matting-alpha" in condition_ids:
        rebuild_kind = "matting-alpha"
    elif "context-raw-face-geometry" in condition_ids:
        rebuild_kind = "face-geometry"
    elif "context-raw-object-relations" in condition_ids:
        rebuild_kind = "object-relations"
    elif "context-raw-scene-category" in condition_ids:
        rebuild_kind = "scene-category"
    elif "context-raw-gaze-head-orientation" in condition_ids:
        rebuild_kind = "gaze-head-orientation"
    elif "context-raw-camera-viewing-angle" in condition_ids:
        rebuild_kind = "camera-viewing-angle"
    elif "context-raw-image-focus" in condition_ids:
        rebuild_kind = "image-focus"
    elif "context-raw-vlm-dense" in condition_ids:
        rebuild_kind = "vlm-dense"
    elif "context-raw-context4k" in condition_ids:
        rebuild_kind = "context4k"
    else:
        rebuild_kind = "geometry"
    rebuilt = build_stage_b_plan(program, candidate_manifest, settings, evidence_kind=rebuild_kind,
                                 vlm_blocks_sha256=plan.get("vlm_blocks_sha256"))
    expected_core = {
        key: value
        for key, value in plan.items()
        if key not in {"comparison_plan_fingerprint", "evidence_input_artifact_sha256"}
    }
    rebuilt_core = {
        key: value
        for key, value in rebuilt.items()
        if key != "comparison_plan_fingerprint"
    }
    if _canonical_json(expected_core) != _canonical_json(rebuilt_core):
        raise StageBRunError("expected comparison plan does not exactly match frozen manifest and settings")

    raw_hashes = _require_mapping(
        plan.get("evidence_input_artifact_sha256"), "expected comparison plan evidence_input_artifact_sha256"
    )
    frozen_items = _candidate_items(candidate_manifest, program)
    item_ids = {_safe_output_segment(item.get("image_id"), "candidate item image_id") for item in frozen_items}
    if set(raw_hashes) != item_ids:
        raise StageBRunError("expected comparison plan evidence artifact hashes must cover exactly the frozen items")
    hashes: dict[str, dict[str, str]] = {}
    expected_names = set(_evidence_input_names(rebuild_kind))
    for image_id in sorted(item_ids):
        row = _require_mapping(raw_hashes[image_id], f"frozen evidence hashes for {image_id}")
        if set(row) != expected_names:
            raise StageBRunError(
                f"frozen evidence hashes for {rebuild_kind} must contain exactly "
                f"{sorted(expected_names)}, got {sorted(row)}"
            )
        hashes[image_id] = {
            name: _require_sha256(row[name], f"frozen evidence hash {image_id}/{name}")
            for name in sorted(expected_names)
        }
    return plan, hashes


def _decode_source(payload: bytes, item: Mapping[str, Any]) -> Image.Image:
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            width, height = opened.size
            source_format = opened.format
            image = opened.convert("RGB")
    except (OSError, ValueError, SyntaxError) as exc:
        raise StageBRunError(f"selected source cannot be decoded: {item['source_relative_path']}") from exc
    dimensions = _require_mapping(item.get("source_dimensions"), "candidate item source_dimensions")
    if dimensions.get("width") != width or dimensions.get("height") != height:
        raise StageBRunError(f"source dimensions drifted for {item['source_relative_path']}")
    if item.get("source_format") != source_format:
        raise StageBRunError(f"source format drifted for {item['source_relative_path']}")
    return image


def _load_selected_item(
    item: Mapping[str, Any],
    source_root: Path,
    derived_root: Path,
    expected_evidence_hashes: Mapping[str, str],
    *,
    include_face_geometry: bool = False,
    include_object_relations: bool = False,
    include_scene_category: bool = False,
    include_gaze_head: bool = False,
    include_camera_viewing_angle: bool = False,
    include_image_focus: bool = False,
) -> dict[str, Any]:
    relative_path = _safe_relative_path(item.get("source_relative_path"), "candidate item source_relative_path")
    source_path = _require_contained(source_root / relative_path, source_root, "selected source")
    payload = source_path.read_bytes()
    observed_sha = _sha256(payload)
    if observed_sha != _require_sha256(item.get("source_sha256"), "candidate item source_sha256"):
        raise StageBRunError(f"source SHA-256 drifted for {relative_path}")
    image = _decode_source(payload, item)

    image_id = _safe_output_segment(item.get("image_id"), "candidate item image_id")
    artifact_dir = _require_contained(derived_root / image_id, derived_root, "selected derived artifact directory")
    availability = _require_mapping(item.get("artifact_availability"), "candidate item artifact_availability")

    def artifact(name: str, *, required: bool) -> np.ndarray | None:
        declared = availability.get(name)
        path = artifact_dir / name
        if required and declared is not True:
            raise StageBRunError(f"frozen manifest lacks required {name} for {image_id}")
        if declared is not True:
            return None
        resolved = _require_contained(path, artifact_dir, f"selected {name}")
        # Only evidence-bound artifacts (members of the plan's frozen evidence
        # hash set) are SHA-verified here; validation-only reads (e.g. pose2 for
        # the detector-count invariant under the lighting arm) are not evidence
        # inputs and are not hash-bound.
        if name not in expected_evidence_hashes:
            try:
                return np.load(resolved.open("rb"), allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise StageBRunError(f"selected {name} is unreadable for {image_id}") from exc
        expected_sha = _require_sha256(expected_evidence_hashes[name], f"frozen evidence hash {image_id}/{name}")
        try:
            payload = resolved.read_bytes()
            observed_sha = _sha256(payload)
            if observed_sha != expected_sha:
                raise StageBRunError(f"selected {name} artifact SHA-256 drifted for {image_id}")
            return np.load(io.BytesIO(payload), allow_pickle=False)
        except StageBRunError:
            raise
        except (OSError, ValueError) as exc:
            raise StageBRunError(f"selected {name} is unreadable for {image_id}") from exc

    pose2 = artifact("pose2.npy", required=True)
    seg2 = artifact("seg2.npy", required=True)
    assert pose2 is not None and seg2 is not None
    if pose2.ndim != 3 or pose2.shape[0] != 1 or pose2.shape[1:] != (308, 3):
        raise StageBRunError(f"selected pose2.npy no longer has one Goliath-308 detection for {image_id}")
    if seg2.ndim != 2:
        raise StageBRunError(f"selected seg2.npy must be two-dimensional for {image_id}")
    determinations = derive_determinations(pose2, seg2)
    if determinations["subject"]["n_detections"] != 1:
        raise StageBRunError(f"detector disagreement emerged for frozen selected item {image_id}")
    try:
        proportions = compute_proportions(pose2)
    except ProportionError as exc:
        raise StageBRunError(f"proportions abort for frozen selected item {image_id}: {exc}") from exc
    try:
        clothing = compute_clothing(seg2, np.asarray(image.convert("RGB"), dtype=np.uint8))
    except ClothingError as exc:
        raise StageBRunError(f"clothing abort for frozen selected item {image_id}: {exc}") from exc
    try:
        hair = compute_hair(seg2, np.asarray(image.convert("RGB"), dtype=np.uint8))
    except HairError as exc:
        raise StageBRunError(f"hair abort for frozen selected item {image_id}: {exc}") from exc
    try:
        skin_tone = compute_skin_tone(seg2, np.asarray(image.convert("RGB"), dtype=np.uint8))
    except SkinColorError as exc:
        raise StageBRunError(f"skin-color abort for frozen selected item {image_id}: {exc}") from exc
    try:
        setting = compute_setting(seg2, np.asarray(image.convert("RGB"), dtype=np.uint8))
    except SettingError as exc:
        raise StageBRunError(f"setting abort for frozen selected item {image_id}: {exc}") from exc
    try:
        texture = compute_texture(seg2, np.asarray(image.convert("RGB"), dtype=np.uint8))
    except TextureError as exc:
        raise StageBRunError(f"texture abort for frozen selected item {image_id}: {exc}") from exc
    try:
        articulation = compute_pose_articulation(pose2, seg2)
    except PoseArticulationError as exc:
        raise StageBRunError(f"pose-articulation abort for frozen selected item {image_id}: {exc}") from exc
    derived_reads = ["pose2.npy", "seg2.npy"]
    pointmap_depth = None
    if "pointmap.npy" in expected_evidence_hashes:
        pointmap2 = artifact("pointmap.npy", required=True)
        assert pointmap2 is not None
        if pointmap2.ndim != 3 or pointmap2.shape[2] != 3:
            raise StageBRunError(f"selected pointmap.npy must be (H, W, 3) for {image_id}")
        if pointmap2.shape[0] != seg2.shape[0] or pointmap2.shape[1] != seg2.shape[1]:
            raise StageBRunError(f"selected pointmap.npy not pixel-aligned with seg2 for {image_id}")
        try:
            pointmap_depth = compute_pointmap_depth(pointmap2, seg2)
        except PointmapDepthError as exc:
            raise StageBRunError(f"pointmap-depth abort for frozen selected item {image_id}: {exc}") from exc
        derived_reads.append("pointmap.npy")
    matting_alpha = None
    if "matting.npy" in expected_evidence_hashes:
        matting = artifact("matting.npy", required=True)
        assert matting is not None
        if matting.ndim != 2:
            raise StageBRunError(f"selected matting.npy must be (H, W) for {image_id}")
        if matting.shape[0] != seg2.shape[0] or matting.shape[1] != seg2.shape[1]:
            raise StageBRunError(f"selected matting.npy not pixel-aligned with seg2 for {image_id}")
        try:
            matting_alpha = compute_matting_alpha(matting, seg2)
        except MattingAlphaError as exc:
            raise StageBRunError(f"matting-alpha abort for frozen selected item {image_id}: {exc}") from exc
        derived_reads.append("matting.npy")
    face_geometry = None
    if include_face_geometry:
        rgb = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
        try:
            face_geometry = compute_face_geometry(
                seg2, rgb, model_asset_path=FACE_GEOMETRY_MODEL_ASSET
            )
        except FaceGeometryError as exc:
            raise StageBRunError(f"face-geometry abort for frozen selected item {image_id}: {exc}") from exc
    object_relations = None
    if include_object_relations:
        rgb = np.ascontiguousarray(np.array(image.convert("RGB"), dtype=np.uint8)).copy()
        try:
            object_relations = compute_object_relations(
                seg2, rgb, model_asset_dir=OBJECT_RELATIONS_MODEL_ASSET
            )
        except ObjectRelationsError as exc:
            raise StageBRunError(f"object-relations abort for frozen selected item {image_id}: {exc}") from exc
    scene_category = None
    if include_scene_category:
        rgb = np.ascontiguousarray(np.array(image.convert("RGB"), dtype=np.uint8)).copy()
        try:
            scene_category = compute_scene_category(rgb, model_asset_dir=SCENE_CATEGORY_MODEL_ASSET)
        except SceneCategoryError as exc:
            raise StageBRunError(f"scene-category abort for frozen selected item {image_id}: {exc}") from exc
    gaze_head = None
    if include_gaze_head:
        rgb = np.ascontiguousarray(np.array(image.convert("RGB"), dtype=np.uint8)).copy()
        try:
            gaze_head = compute_gaze_head(
                seg2, rgb, model_asset_path=GAZE_HEAD_MODEL_ASSET
            )
        except GazeHeadError as exc:
            raise StageBRunError(f"gaze-head-orientation abort for frozen selected item {image_id}: {exc}") from exc
    camera_viewing_angle = None
    if include_camera_viewing_angle:
        img_h, img_w = seg2.shape[0], seg2.shape[1]
        try:
            camera_viewing_angle = compute_camera_viewing_angle(seg2, img_h, img_w)
        except CameraViewingAngleError as exc:
            raise StageBRunError(f"camera-viewing-angle abort for frozen selected item {image_id}: {exc}") from exc
    image_focus = None
    if include_image_focus:
        rgb = np.ascontiguousarray(np.array(image.convert("RGB"), dtype=np.uint8))
        try:
            image_focus = compute_image_focus(rgb, seg2)
        except ImageFocusError as exc:
            raise StageBRunError(f"image-focus abort for frozen selected item {image_id}: {exc}") from exc
    lighting = None
    if "normal2.npy" in expected_evidence_hashes:
        normal2 = artifact("normal2.npy", required=True)
        assert normal2 is not None
        if normal2.ndim != 3 or normal2.shape[2] != 3:
            raise StageBRunError(f"selected normal2.npy must be (H, W, 3) for {image_id}")
        try:
            lighting = compute_lighting(
                normal2, seg2, np.asarray(image.convert("RGB"), dtype=np.uint8)
            )
        except LightingError as exc:
            raise StageBRunError(f"lighting abort for frozen selected item {image_id}: {exc}") from exc
        derived_reads.append("normal2.npy")
    return {
        "item": dict(item),
        "image": image,
        "source_sha256": observed_sha,
        "determinations": determinations,
        "proportions": proportions,
        "clothing": clothing,
        "hair": hair,
        "skin_tone": skin_tone,
        "lighting": lighting,
        "setting": setting,
        "texture": texture,
        "articulation": articulation,
        "pointmap_depth": pointmap_depth,
        "matting_alpha": matting_alpha,
        "face_geometry": face_geometry,
        "object_relations": object_relations,
        "scene_category": scene_category,
        "gaze_head": gaze_head,
        "camera_viewing_angle": camera_viewing_angle,
        "image_focus": image_focus,
        "evidence_input_artifact_sha256": dict(expected_evidence_hashes),
        "source_byte_read_count": 1,
        "derived_reads": derived_reads,
    }


def _bucketed_view(image: Image.Image) -> Image.Image:
    bucket = assign_aspect_bucket(*image.size)
    dims = parse_bucket_dims(bucket)
    if dims is None:
        raise StageBRunError(f"unable to parse selected legacy bucket {bucket}")
    bucket_w, bucket_h = dims
    width, height = image.size
    scale = max(bucket_w / width, bucket_h / height)
    new_width = int(np.ceil(width * scale))
    new_height = int(np.ceil(height * scale))
    resampling = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    resized = image.resize((new_width, new_height), resample=resampling)
    left = max(0, (new_width - bucket_w) // 2)
    top = max(0, (new_height - bucket_h) // 2)
    return resized.crop((left, top, left + bucket_w, top + bucket_h))


def _context_prompt(evidence_text: str) -> str:
    """Render the context prompt without smuggling geometry into a baseline."""
    return _CONTEXT_PROMPT_TEMPLATE.format(evidence_text=evidence_text)


def _rendered_context4k(prepared: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Deterministic dossier -> honest expansion -> <=4K compact evidence text.

    Shared by the arm-36 round-trip condition (`context-raw-context4k`) and the
    arm-47 variant (`context-raw-vlm-dense`, which blends the compact with the
    pre-generated VLM block). Returns (evidence_text, evidence_payload_meta).
    """
    item = prepared["item"]
    lighting = prepared["lighting"]
    if lighting is None:
        lighting = {"lighting_measurable": False, "abstention_reason": "normal2.npy unavailable for this item"}
    dossier = assemble_dossier(
        image_id=item["image_id"],
        proportions=prepared["proportions"],
        clothing=prepared["clothing"],
        hair=prepared["hair"],
        skin=prepared["skin_tone"],
        lighting=lighting,
        setting=prepared.get("setting"),
        texture=prepared.get("texture"),
        articulation=prepared.get("articulation"),
        pointmap_depth=prepared.get("pointmap_depth"),
        matting_alpha=prepared.get("matting_alpha"),
        face_geometry=prepared.get("face_geometry"),
        object_relations=prepared.get("object_relations"),
        scene_category=prepared.get("scene_category"),
        gaze_head=prepared.get("gaze_head"),
        camera_viewing_angle=prepared.get("camera_viewing_angle"),
        image_focus=prepared.get("image_focus"),
        determinations=prepared["determinations"],
    )
    payload = build_evidence_payload(
        image_id=item["image_id"],
        proportions=prepared["proportions"],
        clothing=prepared["clothing"],
        hair=prepared["hair"],
        skin=prepared["skin_tone"],
        lighting=lighting,
        setting=prepared.get("setting"),
        texture=prepared.get("texture"),
        articulation=prepared.get("articulation"),
        pointmap_depth=prepared.get("pointmap_depth"),
        matting_alpha=prepared.get("matting_alpha"),
        face_geometry=prepared.get("face_geometry"),
        object_relations=prepared.get("object_relations"),
        scene_category=prepared.get("scene_category"),
        gaze_head=prepared.get("gaze_head"),
        camera_viewing_angle=prepared.get("camera_viewing_angle"),
        image_focus=prepared.get("image_focus"),
        determinations=prepared["determinations"],
        source_sha256=prepared.get("source_sha256"),
    )
    # Honest evidence-bound expansion (provenance lines + payload-bound
    # detail, never fabricated) then deterministic compression to <=4K.
    try:
        expanded = expand_dossier(dossier, payload)
    except Exception as exc:  # noqa: BLE001 - honesty gate failures must fail closed
        raise StageBRunError(f"context4k honest expansion failed for {item['image_id']}: {exc}") from exc
    compact = compress_dossier_to_context(expanded["expanded_dossier"])
    evidence_text = context4k_text(compact["claims"])
    meta = {
        "compact_token_count": compact["token_count"],
        "compact_under_budget": compact["under_budget"],
        "compact_claim_count": len(compact["claims"]),
        "excluded_claim_count": compact["excluded_claim_count"],
        "expanded_dossier_token_count": expanded["token_count"],
        "base_dossier_token_count": expanded["base_token_count"],
        "dossier_evidence_ids": expanded["evidence_ids"],
    }
    return evidence_text, meta


def _render_condition(
    condition: Mapping[str, Any],
    prepared: Mapping[str, Any],
    vlm_blocks: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Image.Image, str, dict[str, Any] | None]:
    condition_id = condition["id"]
    raw = prepared["image"]
    if condition_id == "legacy-bucketed-no-evidence":
        return _bucketed_view(raw), CAPTION_PROMPT, None
    if condition_id == "legacy-raw-no-evidence":
        return raw.copy(), CAPTION_PROMPT, None
    if condition_id == "context-raw-no-evidence":
        return raw.copy(), _context_prompt("- no specialist evidence declared"), None
    if condition_id == "context-raw-geometry":
        determinations = prepared["determinations"]
        rendered = build_prompt(determinations)
        evidence_text = rendered.split("DETERMINATIONS:\n", 1)[-1].strip()
        return raw.copy(), _context_prompt(evidence_text), determinations
    if condition_id == "context-raw-body-type":
        proportions = prepared["proportions"]
        evidence_text = _serialize_proportions(proportions)
        return raw.copy(), _context_prompt(evidence_text), proportions
    if condition_id == "context-raw-clothing":
        clothing = prepared["clothing"]
        evidence_text = _serialize_clothing(clothing)
        return raw.copy(), _context_prompt(evidence_text), clothing
    if condition_id == "context-raw-hair":
        hair = prepared["hair"]
        evidence_text = _serialize_hair(hair)
        return raw.copy(), _context_prompt(evidence_text), hair
    if condition_id == "context-raw-skin-color":
        skin = prepared["skin_tone"]
        evidence_text = _serialize_skin_color(skin)
        return raw.copy(), _context_prompt(evidence_text), skin
    if condition_id == "context-raw-lighting":
        lighting = prepared["lighting"]
        evidence_text = _serialize_lighting(lighting)
        return raw.copy(), _context_prompt(evidence_text), lighting
    if condition_id == "context-raw-setting":
        setting = prepared["setting"]
        evidence_text = _serialize_setting(setting)
        return raw.copy(), _context_prompt(evidence_text), setting
    if condition_id == "context-raw-texture":
        texture = prepared["texture"]
        evidence_text = _serialize_texture(texture)
        return raw.copy(), _context_prompt(evidence_text), texture
    if condition_id == "context-raw-pose-articulation":
        articulation = prepared["articulation"]
        evidence_text = _serialize_pose_articulation(articulation)
        return raw.copy(), _context_prompt(evidence_text), articulation
    if condition_id == "context-raw-pointmap-depth":
        pointmap_depth = prepared["pointmap_depth"]
        evidence_text = _serialize_pointmap_depth(pointmap_depth)
        return raw.copy(), _context_prompt(evidence_text), pointmap_depth
    if condition_id == "context-raw-matting-alpha":
        matting_alpha = prepared["matting_alpha"]
        evidence_text = _serialize_matting_alpha(matting_alpha)
        return raw.copy(), _context_prompt(evidence_text), matting_alpha
    if condition_id == "context-raw-face-geometry":
        face_geometry = prepared["face_geometry"]
        evidence_text = _serialize_face_geometry(face_geometry)
        return raw.copy(), _context_prompt(evidence_text), face_geometry
    if condition_id == "context-raw-object-relations":
        object_relations = prepared["object_relations"]
        evidence_text = _serialize_object_relations(object_relations)
        return raw.copy(), _context_prompt(evidence_text), object_relations
    if condition_id == "context-raw-scene-category":
        scene_category = prepared["scene_category"]
        evidence_text = _serialize_scene_category(scene_category)
        return raw.copy(), _context_prompt(evidence_text), scene_category
    if condition_id == "context-raw-gaze-head-orientation":
        gaze_head = prepared["gaze_head"]
        evidence_text = _serialize_gaze_head(gaze_head)
        return raw.copy(), _context_prompt(evidence_text), gaze_head
    if condition_id == "context-raw-camera-viewing-angle":
        framing = prepared["camera_viewing_angle"]
        evidence_text = _serialize_camera_viewing_angle(framing)
        return raw.copy(), _context_prompt(evidence_text), framing
    if condition_id == "context-raw-image-focus":
        focus = prepared["image_focus"]
        evidence_text = _serialize_image_focus(focus)
        return raw.copy(), _context_prompt(evidence_text), focus
    if condition_id == "context-raw-context4k":
        evidence_text, meta = _rendered_context4k(prepared)
        return raw.copy(), _context_prompt(evidence_text), meta
    if condition_id == "context-raw-vlm-dense":
        evidence_text, meta = _rendered_context4k(prepared)
        image_id = prepared["item"]["image_id"]
        block_entry = (vlm_blocks or {}).get(image_id)
        block_text = (block_entry or {}).get("block_text", "")
        block_sha = (block_entry or {}).get("block_sha256", "")
        blended = evidence_text + "\n\n" + _serialize_vlm_dense(block_text)
        vlm_meta = dict(meta)
        vlm_meta["vlm_block_present"] = bool(block_text)
        vlm_meta["vlm_block_sha256"] = block_sha
        return raw.copy(), _context_prompt(blended), vlm_meta
    raise StageBRunError(f"unsupported Stage-B condition: {condition_id}")


def _verify_installed_ollama_digest(settings: StageBGenerationSettings) -> None:
    """Bind the local tag to the pre-registered digest before model generation."""
    parsed = urlparse(settings.endpoint)
    tags_endpoint = f"{parsed.scheme}://{parsed.netloc}/api/tags"
    try:
        response = requests.get(tags_endpoint, timeout=settings.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise StageBRunError(f"unable to inspect local Ollama tag metadata: {exc}") from exc
    models = payload.get("models") if isinstance(payload, Mapping) else None
    if not isinstance(models, list):
        raise StageBRunError("local Ollama tag metadata did not contain a models list")
    matching = [model for model in models if isinstance(model, Mapping) and model.get("name") == settings.model_name]
    if len(matching) != 1:
        raise StageBRunError(f"local Ollama model tag is unavailable or ambiguous: {settings.model_name}")
    observed = matching[0].get("digest")
    if observed != settings.model_digest:
        raise StageBRunError(
            f"local Ollama model digest drift for {settings.model_name}: expected {settings.model_digest}, observed {observed}"
        )


def _unload_local_ollama_model(settings: StageBGenerationSettings) -> None:
    """Request local model eviction before a scheduler lease is released."""
    parsed = urlparse(settings.endpoint)
    unload_endpoint = f"{parsed.scheme}://{parsed.netloc}/api/generate"
    try:
        response = requests.post(
            unload_endpoint,
            json={"model": settings.model_name, "keep_alive": 0, "stream": False},
            timeout=settings.timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise StageBRunError(f"unable to unload local Ollama model before scheduler release: {exc}") from exc


def _default_generate(image: Image.Image, prompt: str, settings: StageBGenerationSettings) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, subsampling=0)
    payload = {
        "model": settings.model_name,
        "prompt": prompt,
        "images": [base64.b64encode(buffer.getvalue()).decode("ascii")],
        "stream": False,
        "keep_alive": "10m",
        "options": settings.request_options(),
    }
    try:
        response = requests.post(settings.endpoint, json=payload, timeout=settings.timeout_seconds)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise StageBRunError(f"local Ollama generation failed: {exc}") from exc
    caption = ensure_single_paragraph(data.get("response", ""))
    if not caption:
        raise StageBRunError("local Ollama generation returned an empty caption")
    return caption


def _review_template(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record["record_id"],
        "image_id": record["image_id"],
        "condition_id": record["condition_id"],
        "output_relative_path": record["output_relative_path"],
        "review_status": "unreviewed",
        **{field: [] for field in _REQUIRED_REVIEW_FIELDS},
        "verdict": "PENDING",
    }


def _metric_self_audit_template(known_case_item_id: str) -> dict[str, Any]:
    """Pre-register review calibration without fabricating a completed judgment."""
    return {
        "status": "PENDING_HUMAN_SELF_AUDIT",
        "known_case_item_id": known_case_item_id,
        "null_output_id": "empty-caption-null-v1",
        "required_checks": [
            "Score one generated known-case output against its original selected source and declared evidence.",
            "Score the declared empty-caption null output and confirm it is recorded as an abstention rather than a supported claim.",
            "Confirm every claim-support field accepts explicit evidence-linked entries and empty lists where appropriate.",
        ],
        "limitation": "The runner validates record structure only. It does not fabricate a reviewer judgment or a metric PASS.",
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_canonical_json(dict(record)) + "\n")


def _prepare_output_root(program: Mapping[str, Any], output_root: Path) -> tuple[Path, Path]:
    if not output_root.is_absolute():
        raise StageBRunError("Stage-B output_root must be absolute")
    canonical = _require_mapping(program.get("canonical_source"), "program canonical_source")
    source_root = _resolved_existing_directory(Path(canonical["path"]), "canonical source root")
    derived_root = _resolved_existing_directory(Path(canonical["derived_tree"]), "derived artifact root")
    output = output_root.resolve(strict=False)
    if output.exists() or os.path.lexists(output):
        raise StageBRunError("Stage-B output_root already exists; choose a new versioned run root")
    if output.is_relative_to(source_root) or output.is_relative_to(derived_root):
        raise StageBRunError("Stage-B output_root must not be inside a protected corpus root")
    policy = _require_mapping(program.get("artifact_policy"), "program artifact_policy")
    raw_roots = policy.get("approved_output_roots")
    if not isinstance(raw_roots, list) or not raw_roots:
        raise StageBRunError("program must declare at least one approved noncanonical output root")
    approved_roots = tuple(_resolved_existing_directory(Path(root), "approved output root") for root in raw_roots)
    if not any(output.is_relative_to(root) for root in approved_roots):
        raise StageBRunError("Stage-B output_root must be under an approved noncanonical output root")
    parent = _resolved_existing_directory(output.parent, "Stage-B output parent")
    return output, parent


def execute_stage_b(
    program: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    settings: StageBGenerationSettings,
    *,
    output_root: Path,
    expected_plan: Mapping[str, Any] | None = None,
    generate: Generator | None = None,
    vlm_blocks: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute the frozen four-condition run and publish only noncanonical outputs.

    All selected sources/artifacts are preflighted before the first model call;
    a changed input fails closed and leaves no final run root behind.
    """
    try:
        validate_program(program)
    except ContractError as exc:
        raise StageBRunError(f"invalid Stage-B program: {exc}") from exc
    output, parent = _prepare_output_root(program, output_root)
    if expected_plan is None:
        # Direct callers are retained for synthetic fixture compatibility only;
        # the CLI/registered launcher always supplies a persisted frozen plan.
        plan = build_stage_b_plan(program, candidate_manifest, settings)
        evidence_hashes = _freeze_evidence_input_artifact_hashes(
            candidate_manifest, program, _candidate_items(candidate_manifest, program)
        )
        plan["evidence_input_artifact_sha256"] = evidence_hashes
        plan["comparison_plan_fingerprint"] = _canonical_fingerprint(plan, "comparison_plan_fingerprint")
    else:
        plan, evidence_hashes = _validate_frozen_execution_plan(
            expected_plan, program, candidate_manifest, settings
        )
    items = _candidate_items(candidate_manifest, program)
    canonical = _require_mapping(program["canonical_source"], "program canonical_source")
    source_root = _resolved_existing_directory(Path(canonical["path"]), "canonical source root")
    derived_root = _resolved_existing_directory(Path(canonical["derived_tree"]), "derived artifact root")

    # Arm #60: only the face-geometry run invokes the local open-weight
    # FaceLandmarker (new model class, CPU) — gate on the frozen plan's
    # conditions so other arms never pay the model cost.
    include_face_geometry = any(
        str(condition.get("id")) == "context-raw-face-geometry"
        for condition in (plan.get("conditions") or [])
    )
    include_object_relations = any(
        str(condition.get("id")) == "context-raw-object-relations"
        for condition in (plan.get("conditions") or [])
    )
    # Arm #69: only the scene-category run invokes the local open-weight CLIP
    # classifier (new model class, CPU) — gate on the frozen plan's conditions
    # so other arms never pay the CLIP cost.
    include_scene_category = any(
        str(condition.get("id")) == "context-raw-scene-category"
        for condition in (plan.get("conditions") or [])
    )
    # Arm #68: only the gaze-head-orientation run invokes the locally-run
    # MediaPipe FaceLandmarker (reused arm #60 mesh, CPU) — gate on the frozen
    # plan's conditions so other arms never pay the mesh cost.
    include_gaze_head = any(
        str(condition.get("id")) == "context-raw-gaze-head-orientation"
        for condition in (plan.get("conditions") or [])
    )
    # Arm #74: only the camera-viewing-angle run computes deterministic
    # framing (CPU, no new model) — gate on the frozen plan's conditions.
    include_camera_viewing_angle = any(
        str(condition.get("id")) == "context-raw-camera-viewing-angle"
        for condition in (plan.get("conditions") or [])
    )
    # Arm #75: only the image-focus run computes deterministic focus/DOF
    # bands (CPU, no new model) — gate on the frozen plan's conditions.
    include_image_focus = any(
        str(condition.get("id")) == "context-raw-image-focus"
        for condition in (plan.get("conditions") or [])
    )

    # Preflight all frozen inputs before model invocation so an input epoch cannot
    # silently split a paired comparison halfway through the cohort.
    prepared = [
        _load_selected_item(
            item,
            source_root,
            derived_root,
            evidence_hashes[_safe_output_segment(item.get("image_id"), "candidate item image_id")],
            include_face_geometry=include_face_geometry,
            include_object_relations=include_object_relations,
            include_scene_category=include_scene_category,
            include_gaze_head=include_gaze_head,
            include_camera_viewing_angle=include_camera_viewing_angle,
            include_image_focus=include_image_focus,
        )
        for item in items
    ]
    # Arm #47: when the plan binds VLM dense blocks, every pinned hash must be
    # present and byte-verified before any model call (fail closed, never run a
    # partially-verified evidence set through generation).
    pinned_vlm = plan.get("vlm_blocks_sha256")
    if pinned_vlm is not None:
        for image_id in sorted(pinned_vlm):
            entry = (vlm_blocks or {}).get(image_id)
            if not entry or not entry.get("block_text"):
                raise StageBRunError(f"missing VLM dense block for frozen item {image_id}")
            observed = _sha256((entry.get("block_text") or "").encode("utf-8"))
            if observed != pinned_vlm[image_id]:
                raise StageBRunError(
                    f"VLM dense block SHA-256 drifted for {image_id}: pin {pinned_vlm[image_id]}, observed {observed}"
                )
    if generate is None:
        _verify_installed_ollama_digest(settings)
    generator = generate or _default_generate
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-b-", dir=parent))
    try:
        outputs_dir = temporary / "outputs"
        outputs_dir.mkdir()
        records: list[dict[str, Any]] = []
        review_queue: list[dict[str, Any]] = []
        for prepared_item in prepared:
            item = prepared_item["item"]
            image_id = _safe_output_segment(item["image_id"], "candidate item image_id")
            for condition in plan["conditions"]:
                condition_id = _safe_output_segment(condition["id"], "Stage-B condition id")
                view, prompt, evidence_payload = _render_condition(condition, prepared_item, vlm_blocks)
                caption = ensure_single_paragraph(generator(view, prompt, settings))
                if not caption:
                    raise StageBRunError(f"generator returned an empty caption for {image_id}/{condition_id}")
                condition_dir = outputs_dir / condition_id
                condition_dir.mkdir(exist_ok=True)
                output_relative = Path("outputs") / condition_id / f"{image_id}.txt"
                output_path = temporary / output_relative
                output_path.write_text(caption + "\n", encoding="utf-8")
                record = {
                    "schema_version": 1,
                    "record_id": f"{condition_id}:{image_id}",
                    "image_id": image_id,
                    "source_relative_path": item["source_relative_path"],
                    "source_sha256": prepared_item["source_sha256"],
                    "condition_id": condition_id,
                    "input_view": condition["input_view"],
                    "prompt": {
                        **condition["prompt"],
                        "rendered_sha256": _sha256(prompt.encode("utf-8")),
                        "rendered_text": prompt,
                    },
                    "evidence": condition["evidence"],
                    "evidence_payload": evidence_payload,
                    "generation_fingerprint": settings.fingerprint,
                    "output_relative_path": output_relative.as_posix(),
                    "caption_sha256": _sha256(caption.encode("utf-8")),
                    "caption": caption,
                    "caption_character_count": len(caption),
                    "caption_word_count": len(caption.split()),
                    "selected_source_byte_read_count": prepared_item["source_byte_read_count"],
                    "selected_derived_reads": prepared_item["derived_reads"],
                    "selected_evidence_input_artifact_sha256": prepared_item[
                        "evidence_input_artifact_sha256"
                    ],
                }
                records.append(record)
                review_queue.append(_review_template(record))

        self_audit = _metric_self_audit_template(plan["metric_self_audit"]["known_case_item_id"])
        _write_json(temporary / "stage-b-plan.json", plan)
        _write_jsonl(temporary / "records.jsonl", records)
        _write_jsonl(temporary / "review-queue.jsonl", review_queue)
        _write_json(
            temporary / "run-provenance.json",
            {
                "schema_version": 1,
                "status": "PENDING_INDEPENDENT_REVIEW",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "candidate_manifest_id": candidate_manifest["manifest_id"],
                "candidate_manifest_fingerprint": candidate_manifest["manifest_fingerprint"],
                "comparison_plan_id": plan["comparison_plan_id"],
                "comparison_plan_fingerprint": plan["comparison_plan_fingerprint"],
                "generation": {**asdict(settings), "generation_fingerprint": settings.fingerprint},
                "record_count": len(records),
                "selected_source_byte_read_count": sum(item["source_byte_read_count"] for item in prepared),
                "selected_derived_reads_only": True,
                "metric_self_audit": self_audit,
                "semantic_verdict": "PENDING — human claim-support and adversarial reviews are not fabricated by the runner.",
                "non_authorizations": [
                    "no canonical source mutation",
                    "no derived-tree mutation or backfill",
                    "no legacy artifact overwrite",
                    "no context4k or t52 substitution claim",
                    "no empirical PASS or FAIL verdict",
                ],
            },
        )
        (temporary / "review-guide.md").write_text(
            "# Stage-B sequential review guide\n\n"
            "For each `review-queue.jsonl` row, inspect in this order: selected source view, "
            "in-memory determinations/evidence payload (when present), rendered prompt, and candidate caption. "
            "Record supported claims, unsupported claims, omissions, contradictions, and abstentions. "
            "Do not convert detector anomalies into caption semantics. A PASS remains prohibited until an "
            "independent adversarial review is complete.\n",
            encoding="utf-8",
        )
        if output.exists() or os.path.lexists(output):
            raise StageBRunError("Stage-B output_root appeared during execution; refusing to overwrite it")
        os.rename(temporary, output)
        temporary = None
    finally:
        if generate is None:
            # Best-effort local model eviction before the scheduler lease is
            # released; never mask the primary result with an unload error.
            try:
                _unload_local_ollama_model(settings)
            except StageBRunError:
                pass
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)

    return {
        "status": "PENDING_INDEPENDENT_REVIEW",
        "output_root": str(output),
        "record_count": len(records),
        "comparison_plan_fingerprint": plan["comparison_plan_fingerprint"],
        "candidate_manifest_fingerprint": candidate_manifest["manifest_fingerprint"],
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageBRunError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise StageBRunError(f"{label} must contain a JSON object")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="research-stage-b",
        description="Run the frozen first-500 Stage-B caption parity comparison into a noncanonical root",
    )
    parser.add_argument("program", type=Path)
    parser.add_argument("candidate_manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-plan", required=True, type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--num-predict", type=int, default=384)
    parser.add_argument("--context-window", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--vlm-root", type=Path, default=None,
                        help="noncanonical stage root with per-item vlm-dense.json blocks (arm #47)")
    return parser.parse_args(argv)


def _load_vlm_blocks(vlm_root: Path | None) -> dict[str, dict[str, Any]]:
    """Load per-item {image_id: {block_text, block_sha256}} from a stage root."""
    if vlm_root is None:
        return {}
    root = vlm_root.resolve(strict=True)
    if not root.is_dir():
        raise StageBRunError(f"--vlm-root is not a directory: {vlm_root}")
    blocks: dict[str, dict[str, Any]] = {}
    for artifact_dir in root.iterdir():
        record = artifact_dir / "vlm-dense.json"
        if not artifact_dir.is_dir() or not record.is_file():
            continue
        try:
            payload = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StageBRunError(f"unreadable VLM block record {record}: {exc}") from exc
        image_id = payload.get("meta", {}).get("image_id")
        block_text = payload.get("block_text")
        if not isinstance(image_id, str) or not isinstance(block_text, str):
            raise StageBRunError(f"VLM block record {record} lacks image_id/block_text")
        blocks[image_id] = {
            "block_text": block_text,
            "block_sha256": _sha256(block_text.encode("utf-8")),
        }
    return blocks


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = StageBGenerationSettings(
            endpoint=args.endpoint,
            model_name=args.model,
            model_digest=args.model_digest,
            temperature=0.0,
            seed=args.seed,
            num_predict=args.num_predict,
            top_k=1,
            top_p=1.0,
            context_window=args.context_window,
            timeout_seconds=args.timeout_seconds,
        )
        result = execute_stage_b(
            _read_json(args.program, "program JSON"),
            _read_json(args.candidate_manifest, "candidate manifest JSON"),
            settings,
            output_root=args.output,
            expected_plan=_read_json(args.expected_plan, "expected comparison plan JSON"),
            vlm_blocks=_load_vlm_blocks(args.vlm_root),
        )
    except StageBRunError as exc:
        print(f"research-stage-b: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
