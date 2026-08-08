"""Deterministic per-asset expanded-dossier assembly and context4k compression.

Arm #36 (dossier-context4k), the program-level goal. This module is the
deterministic, CPU-only first stage: it reads only the frozen candidate
manifest's selected inputs (pose2/seg2/normal2 + source pixels), runs the five
now-validated deterministic evidence specialists (body-type proportions,
clothing, hair, skin-color, lighting) plus the relational determinations, and
assembles a claim-by-claim evidence-linked dossier. It then compresses that
dossier into a contract-shaped context4k bundle (per `validate_compression_bundle`):
an expanded dossier (evidence IDs + honest token accounting), a compact context of
claim-level prose where every claim carries its supporting evidence IDs, and the
three configured artifacts (context4k.json, context4k.md, compression.json).

Honesty invariants (owner directives, do not regress):
- Only scale-invariant ratios are verbalized as caption claims; absolute pixel
  measurements stay in the machine-readable evidence payload (the dossier input).
- The planner never fabricates evidence: an absent or abstained measurement is
  recorded as an abstention line, never as an invented fact.
- `count_tokens` is deterministic for a fixed program/bundle: it uses the
  legacy-matching T5 tokenizer when available (deterministic whitespace/naive
  fallback otherwise), so the same dossier always maps to the same token count.
- The compressor fills up to a target token budget deterministically: it selects
  claims in a stable priority order and, when the last included claim would
  overshoot, trims it at a natural token boundary recording the truncation in
  compression.json. It NEVER pads with fabricated filler to reach a budget; if
  the evidence corpus is genuinely too small, it records an honest
  `under_budget` note with the true count (reaching the program floor requires
  the aggregator expansion stage, i.e. the scheduler-bound round-trip audit).

This module writes nothing to crawlr/approved or crawlr/stratum and is purely
additive; all outputs belong under an approved noncanonical research root.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .contracts import ContractError, validate_compression_bundle, validate_program

# ---------------------------------------------------------------------------
# Evidence part identifiers (the registry's evidence_parts for arm #36 map to
# these). Each part produces one or more evidence IDs that claims reference.
# ---------------------------------------------------------------------------
EVIDENCE_PARTS: tuple[str, ...] = (
    "verified-dimension-specialists",
    "relational-determinations",
)
# Per-dimension evidence ids, appended inside the "verified-dimension-specialists" part.
DIMENSION_EVIDENCE_IDS: tuple[str, ...] = (
    "body-type-proportions:v1",
    "clothing-apparel:v1",
    "hair:v1",
    "skin-color-tone:v1",
    "lighting:v1",
    "setting-environment:v1",
    "texture-material:v1",
    "pose-articulation:v1",
    "pointmap-depth:v1",
    "matting-alpha:v1",
    "face-geometry:v1",
    "object-relations:v1",
    "scene-category:v1",
    "gaze-head-orientation:v1",
    "camera-viewing-angle:v1",
    "image-focus:v1",
    "apparent-age:v1",
    "affordance-contact:v1",
    "body-configuration:v1",
    "hairstyle:v1",
    "face-visibility:v1",
    "environment-clearance:v1",
    "eye-color:v1",
    "facial-expression:v1",
)
RELATIONAL_EVIDENCE_ID = "relational-determinations:v1"

# Contract-sane budget rails (in tokens). The program's compact_context
# target/min are both 4000 for this program, so an exactly-4000 context is
# achievable by deterministic trim when the corpus is large enough.
_COMPACT_TARGET = 4000

# Abbreviation used by the naive fallback tokenizer (only when the T5 tokenizer
# cannot be loaded). Chosen so the fallback is a conservative upper bound.
_FALLBACK_TOKENS_PER_CHAR = 1.0 / 4.0  # ~4 chars/token

_SAFE_ID = re.compile(r"[A-Za-z0-9._:-]+")
_SENTENCE_END = re.compile(r".*?(\.(?:\s|$)|$)")


class DossierError(RuntimeError):
    """Raised when the dossier cannot be assembled or compressed safely."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Token counting (deterministic).
# ---------------------------------------------------------------------------
_T5_TOKENIZER = None


def _load_t5_tokenizer():
    """Load the legacy-matching T5 tokenizer once (lazy, cached)."""
    global _T5_TOKENIZER
    if _T5_TOKENIZER is None:
        try:
            from transformers import T5TokenizerFast

            # t5-small/fast share the same sentencepiece vocab as the legacy
            # T5 path; pin to it so token accounting matches the 512-token
            # legacy encoder contract even though context4k is a first-class
            # long-context artifact.
            _T5_TOKENIZER = T5TokenizerFast.from_pretrained("t5-small")
        except Exception:  # pragma: no cover - env-dependent fallback
            _T5_TOKENIZER = False  # sentinel: fall back to heuristic below
    return _T5_TOKENIZER


def count_tokens(text: str, *, use_t5: bool = True) -> int:
    """Deterministic token count for a dossier/context text."""
    if use_t5:
        tokenizer = _load_t5_tokenizer()
        if tokenizer:
            return len(tokenizer.encode(text))
    # Naive deterministic fallback (chars/4, conservative upper bound).
    return max(1, int(math.ceil(len(text) * _FALLBACK_TOKENS_PER_CHAR)))


# ---------------------------------------------------------------------------
# Purely additive natural-language renderings of the five validated dimensions.
# These reuse the same measurement vocabulary the caption serde in stage_b uses
# and VERBALIZE ONLY SCALE-INVARIANT FACTS (ratios/bands/names), keeping
# absolute pixel values in the machine-readable payload.
# ---------------------------------------------------------------------------

def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def render_proportions(proportions: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    if not proportions.get("subject_present"):
        return ["body type: abstain (no reliable keypoint subject present)"]
    ratio = proportions.get("shoulder_hip_ratio")
    if ratio is not None:
        lines.append(f"body type: shoulder:hip width ratio {float(ratio):.2f}")
    else:
        reason = proportions.get("shoulder_hip_ratio_abstention_reason")
        lines.append(
            "body type: shoulder:hip width ratio abstained"
            + (f" ({reason})" if reason else " (joint absent or low confidence)")
        )
    leg_torso = proportions.get("leg_torso_ratio")
    if leg_torso is not None:
        lines.append(f"body type: mean leg:torso length ratio {float(leg_torso):.2f}")
    else:
        lines.append("body type: mean leg:torso length ratio abstained (one or both absent)")
    llen, rlen = proportions.get("left_leg_length"), proportions.get("right_leg_length")
    if llen is not None and rlen is not None and rlen > 0:
        asym = float(llen) / float(rlen)
        direction = "left longer" if asym > 1.02 else ("right longer" if asym < 0.98 else "similar")
        lines.append(f"body type: leg-length asymmetry {direction} (ratio {asym:.2f})")
    else:
        lines.append("body type: leg-length asymmetry abstained (absent or low-confidence joints)")
    return lines


def render_clothing(clothing: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    if not clothing.get("subject_present"):
        return ["clothing: abstain (no reliable foreground subject present)"]
    garments = clothing.get("garments") or []
    if not garments:
        lines.append("clothing: no garment class cleared the gate -> abstain from apparel claims (exposed skin is not an inferred absence)")
    for garment in garments:
        name = garment.get("class", "unknown").replace("_", " ")
        color = garment.get("dominant_color_name")
        coverage = garment.get("coverage")
        text = f"clothing: {name} present"
        if color:
            text += f", dominant color {color}"
        if coverage is not None:
            text += f" (covers {float(coverage):.2f} of subject foreground)"
        lines.append(text)
    return lines


def render_hair(hair: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    if not hair.get("subject_present"):
        return ["hair: abstain (no reliable foreground subject present)"]
    if not hair.get("hair_present"):
        return ["hair: no hair region cleared the gate -> abstain from hair claims (absent mask is not baldness)"]
    coverage = hair.get("hair_coverage")
    color = hair.get("hair_dominant_color_name")
    position = hair.get("hair_position")
    ratio = hair.get("hair_face_extent_ratio")
    if coverage is not None:
        lines.append(f"hair: present, covering {float(coverage):.2f} of subject foreground")
    if color:
        lines.append(f"hair: dominant color {color}")
    if position:
        lines.append(f"hair: occupies the {position} region of the frame")
    if ratio is not None:
        lines.append(f"hair: hair-to-face vertical extent ratio (length proxy) {float(ratio):.2f}")
    return lines


def render_skin_color(skin: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    if not skin.get("subject_present"):
        return ["skin tone: abstain (no reliable foreground subject present)"]
    if not skin.get("exposed_skin_present"):
        return ["skin tone: no exposed skin cleared the gate -> abstain from skin-tone claims"]
    tone = skin.get("skin_tone_name")
    face_tone = skin.get("face_tone_name")
    coverage = skin.get("skin_coverage")
    agree = skin.get("face_body_agree")
    if tone:
        lines.append(f"skin tone: dominant exposed-skin tone {tone}")
    if face_tone and agree is False:
        lines.append(f"skin tone: face tone distinct ({face_tone}) — face/body disagree")
    if coverage is not None:
        lines.append(f"skin tone: exposed skin covers {float(coverage):.2f} of subject foreground")
    return lines


def render_lighting(lighting: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    if not lighting.get("lighting_measurable"):
        reason = lighting.get("abstention_reason")
        lines.append("lighting: abstain (not measurable)" + (f" ({reason})" if reason else ""))
        return lines
    for key, label in (
        ("luma_band", "lighting: exposure"),
        ("dynamic_range_band", "lighting: tonal dynamic range"),
        ("shadow_band", "lighting: shadow level"),
        ("surround_band", "lighting: surround/subject contrast"),
    ):
        value = lighting.get(key)
        if value:
            lines.append(f"{label} {value}")
    direction = lighting.get("light_direction")
    if direction:
        lines.append(f"lighting: key light direction {direction}")
    return lines


def render_setting(setting: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    if not setting.get("setting_measurable"):
        reason = setting.get("abstention_reason")
        lines.append("setting: abstain (background not measurable)" + (f" ({reason})" if reason else ""))
        return lines
    coverage = setting.get("background_coverage")
    if isinstance(coverage, (int, float)):
        lines.append(f"setting: background covers {float(coverage):.2f} of the frame")
    color = setting.get("dominant_background_color")
    if color:
        lines.append(f"setting: dominant background color {color}")
    for key, label in (
        ("background_tone_band", "setting: background tone"),
        ("background_vibrancy_band", "setting: background color intensity"),
        ("background_pattern_band", "setting: background surface"),
    ):
        value = setting.get(key)
        if value:
            lines.append(f"{label} {value}")
    return lines


def render_texture(texture: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    if not texture.get("texture_measurable"):
        reason = texture.get("abstention_reason")
        lines.append("texture: abstain (no measurable fabric or skin region)" + (f" ({reason})" if reason else ""))
        return lines
    fabric_class = texture.get("fabric_class")
    if fabric_class:
        tex = texture.get("fabric_texture_band")
        pat = texture.get("fabric_pattern_band")
        if tex:
            lines.append(f"texture: dominant fabric ({fabric_class}) surface {tex}")
        if pat:
            lines.append(f"texture: dominant fabric ({fabric_class}) pattern {pat}")
    else:
        lines.append("texture: abstain (no measurable fabric region for material claims)")
    skin_class = texture.get("skin_class")
    if skin_class:
        tex = texture.get("skin_texture_band")
        if tex:
            lines.append(f"texture: dominant skin surface ({skin_class}) {tex}")
    return lines


def render_pose_articulation(articulation: Mapping[str, Any]) -> list[str]:
    """Scale-invariant kinematic articulation claims (arm #62)."""
    lines: list[str] = []
    if not articulation.get("subject_present"):
        return ["pose articulation: abstain (no reliable subject keypoints present)"]
    for side, key in (("left", "elbow_flexion_left"), ("right", "elbow_flexion_right")):
        value = articulation.get(key)
        if value is None:
            lines.append(f"pose articulation: {side} elbow flexion abstained (joint absent or low confidence)")
        elif float(value) < 135.0:
            lines.append(f"pose articulation: {side} arm visibly bent at the elbow")
        else:
            lines.append(f"pose articulation: {side} arm extended at the elbow")
    for side, key in (("left", "knee_flexion_left"), ("right", "knee_flexion_right")):
        value = articulation.get(key)
        if value is None:
            lines.append(f"pose articulation: {side} knee flexion abstained (joint absent or low confidence)")
        elif float(value) < 135.0:
            lines.append(f"pose articulation: {side} leg visibly bent at the knee")
        else:
            lines.append(f"pose articulation: {side} leg extended at the knee")
    stance = articulation.get("stance_class")
    if stance and stance != "centered":
        leg = "left" if stance == "weight-left" else "right"
        lines.append(f"pose articulation: weight carried on the {leg} leg")
    elif stance == "centered":
        lines.append("pose articulation: stance centered (no single weight-bearing leg)")
    else:
        lines.append("pose articulation: stance abstained (weight-bearing signal ambiguous)")
    contrapposto = articulation.get("contrapposto")
    if contrapposto is True:
        lines.append("pose articulation: contrapposto stance (hips tilted with weight on one leg)")
    elif contrapposto is False:
        lines.append("pose articulation: contrapposto stance absent (hips level)")
    twist = articulation.get("torso_twist_deg")
    lean = articulation.get("torso_lean_deg")
    tilt = articulation.get("pelvis_tilt_deg")
    if twist is not None:
        lines.append(f"pose articulation: torso/hips in-plane twist {float(twist):.0f} degrees")
    if lean is not None:
        lines.append(f"pose articulation: torso lean from vertical {float(lean):.0f} degrees")
    if tilt is not None:
        lines.append(f"pose articulation: pelvis tilt from horizontal {float(tilt):.0f} degrees")
    crossing = articulation.get("arm_crossing_count")
    if crossing is not None and int(crossing) > 0:
        lines.append(f"pose articulation: {int(crossing)} arm(s) cross in front of the torso")
    if articulation.get("legs_crossed") is True:
        lines.append("pose articulation: legs visually crossed")
    return lines


def render_pointmap_depth(pointmap_depth: Mapping[str, Any]) -> list[str]:
    """Scale-invariant point-map depth-ordering claims (arm #58).

    Verbalizes ONLY depth-ORDER relations and normalized ratios (region
    nearest/farthest ordering, hand nearer-than-other hand, hand/arm held in
    front of the body plane, depth-relief band). Raw CAM-frame metric Z values
    and absolute spreads stay in the machine-readable payload and are never
    caption claims (camera-placement dependent, unrenderable as a bare number).
    """
    lines: list[str] = []
    if not pointmap_depth.get("subject_present"):
        return ["depth ordering: abstain (no subject depth present)"]
    if pointmap_depth.get("abstained"):
        reason = pointmap_depth.get("abstention_reason") or "depth not measurable"
        return [f"depth ordering: abstain ({reason})"]

    relief_band = pointmap_depth.get("relief_band")
    if relief_band == "pronounced":
        lines.append("depth ordering: body has pronounced depth relief (limbs clearly nearer than the torso plane)")
    elif relief_band == "moderate":
        lines.append("depth ordering: body has moderate depth relief (some limbs offset from the torso plane)")
    elif relief_band == "compact":
        lines.append("depth ordering: body largely on one depth plane (compact, limbs near the torso plane)")

    nearest = pointmap_depth.get("nearest_region")
    farthest = pointmap_depth.get("farthest_region")
    if nearest and farthest:
        nearest_label = nearest.replace("_", " ")
        farthest_label = farthest.replace("_", " ")
        lines.append(f"depth ordering: {nearest_label} is the part nearest the camera; {farthest_label} the farthest")
    elif nearest:
        lines.append(f"depth ordering: {nearest.replace('_', ' ')} is the part nearest the camera")

    hand_ordering = pointmap_depth.get("hand_ordering")
    if hand_ordering:
        lines.append(f"depth ordering: {hand_ordering} hand is clearly held nearer the camera than the other")
    left_front = pointmap_depth.get("left_hand_in_front")
    right_front = pointmap_depth.get("right_hand_in_front")
    if left_front:
        lines.append("depth ordering: left hand/arm is held in front of the torso plane")
    if right_front:
        lines.append("depth ordering: right hand/arm is held in front of the torso plane")
    return lines


def render_matting_alpha(matting: Mapping[str, Any]) -> list[str]:
    """Scale-invariant matting / alpha-fidelity claims (arm #59).

    Verbalizes ONLY scale-invariant alpha facts: subject coverage band, boundary
    crispness band (sharp/soft silhouette edge), and soft-edge character
    (hair-dominant vs clean skin cutout). Raw pixel areas and band widths stay
    in the machine-readable payload and are never caption claims (pixel widths
    are camera-frame-dependent and unrenderable as bare numbers).
    """
    lines: list[str] = []
    if not matting.get("subject_present"):
        return ["matting: abstain (no subject alpha present)"]
    if matting.get("abstained"):
        reason = matting.get("abstention_reason") or "matte not measurable"
        return [f"matting: abstain ({reason})"]

    coverage_band = matting.get("coverage_band")
    if coverage_band == "sparse":
        lines.append("matting: subject is a small figure occupying a minor part of the frame")
    elif coverage_band == "fills-frame":
        lines.append("matting: subject largely fills the frame")
    elif coverage_band == "centered":
        lines.append("matting: subject occupies the center of the frame")

    crisp_band = matting.get("boundary_crisp_band")
    if crisp_band == "crisp":
        lines.append("matting: clean sharp silhouette edge (crisp cutout)")
    elif crisp_band == "soft":
        lines.append("matting: very soft feathered silhouette edge")
    elif crisp_band == "moderate":
        lines.append("matting: moderately soft silhouette edge")

    edge_band = matting.get("soft_edge_band")
    if edge_band == "hair-dominant":
        lines.append("matting: soft detachable hair strands produce a wispy hairline (hair-dominant soft edge)")
    elif edge_band == "mixed":
        lines.append("matting: soft edge is a mix of hair and skin/background transitions")
    elif edge_band == "skin-clean":
        lines.append("matting: clean skin/background cutout with minimal hair flyaway")
    return lines


def render_face_geometry(face: Mapping[str, Any]) -> list[str]:
    """Scale-invariant facial-geometry claims (arm #60, MediaPipe FaceLandmarker).

    Verbalizes ONLY scale-invariant facial-ratio bands (eye spacing, mouth
    width, jaw width, plausibility-gated mid-face share). Landmark coordinates,
    pixel bbox, and absolute ratios stay in the machine-readable payload and
    are never caption claims (camera-frame-dependent).
    """
    if face.get("abstained"):
        reason = face.get("abstention_reason") or "face not measurable"
        return [f"face-geometry: abstain ({reason})"]

    lines: list[str] = []
    eye = face.get("eye_spacing_band")
    if eye == "close-set":
        lines.append("face-geometry: eyes are set close together relative to the face")
    elif eye == "wide-set":
        lines.append("face-geometry: eyes are set wide apart relative to the face")
    mouth = face.get("mouth_band")
    if mouth == "narrow":
        lines.append("face-geometry: mouth is narrow relative to the face")
    elif mouth == "wide":
        lines.append("face-geometry: mouth is wide relative to the face")
    jaw = face.get("jaw_band")
    if jaw == "narrow":
        lines.append("face-geometry: jawline is narrow (tapered) relative to the face")
    elif jaw == "wide":
        lines.append("face-geometry: jawline is broad relative to the face")
    mid = face.get("midface_band")
    if mid == "short":
        lines.append("face-geometry: mid-face (nose-to-chin) is short relative to the face")
    elif mid == "tall":
        lines.append("face-geometry: mid-face (nose-to-chin) is tall relative to the face")
    return lines


def render_object_relations(objrel: Mapping[str, Any]) -> list[str]:
    """Scale-invariant object-presence / placement claims (arm #61).

    Verbalizes ONLY scale-invariant facts: count band, placement band, and the
    canonical class list. Normalized boxes, scores, and raw phrases stay in the
    machine-readable payload and are never caption claims.
    """
    if objrel.get("abstained"):
        reason = objrel.get("abstention_reason") or "object detection not measurable"
        return [f"object-relations: abstain ({reason})"]
    if not objrel or not objrel.get("count_band"):
        # Dimension not measured for this item (e.g. non-object-relations
        # runs) — emit no claim, never a fabricated "no objects" one.
        return []
    lines: list[str] = []
    band = objrel.get("count_band")
    count = objrel.get("count", 0)
    classes = objrel.get("classes") or []
    cls_txt = ", ".join(classes[:5])
    if band == "none" or count == 0:
        lines.append("object-relations: no scene objects detected above the calibrated threshold")
        return lines
    if band == "sparse":
        lines.append(f"object-relations: a single scene object is present ({cls_txt})")
    elif band == "moderate":
        lines.append(f"object-relations: several scene objects are present ({cls_txt})")
    else:
        lines.append(f"object-relations: the scene contains multiple distinct objects ({cls_txt})")
    placement = objrel.get("placement_band")
    if placement == "foreground":
        lines.append("object-relations: objects overlap the subject (held / on-person)")
    elif placement == "background":
        lines.append("object-relations: objects sit behind the subject in the background")
    elif placement == "mix":
        lines.append("object-relations: objects are a mix of foreground and background")
    return lines


def render_scene_category(scene: Mapping[str, Any]) -> list[str]:
    """Scale-invariant semantic scene-category claim (arm #69).

    Verbalizes ONLY the scale-invariant semantic category label (or a surfaced
    abstention). CLIP similarity logits / probabilities stay in the
    machine-readable payload and are never caption claims.
    """
    if scene.get("abstained"):
        reason = scene.get("abstention_reason") or "scene classification not confident"
        return [f"scene-category: abstain ({reason})"]
    if not scene or not scene.get("category"):
        # Dimension not measured for this item (e.g. non-scene-category
        # runs) — emit no claim, never a fabricated label.
        return []
    return [f"scene-category: the setting is a {scene['category']}"]


def render_apparent_age(age: Mapping[str, Any]) -> list[str]:
    """Scale-invariant apparent-age band claim (arm #73).

    Verbalizes ONLY the coarse scale-invariant age band (or a surfaced
    abstention). The raw floating age estimate and the gender probe stay in
    the machine-readable payload and are never caption claims.
    """
    if age.get("abstained"):
        reason = age.get("abstention_reason") or "apparent age not measurable"
        return [f"apparent-age: abstain ({reason})"]
    if not age or not age.get("age_band"):
        # Dimension not measured for this item (e.g. non-apparent-age runs) —
        # emit no claim, never a fabricated age statement.
        return []
    band = age["age_band"]
    text = {
        "late-teens-to-early-twenties": "looks late teens to early twenties",
        "early-twenties": "looks early twenties",
        "mid-twenties": "looks mid-twenties",
        "late-twenties-to-thirties": "looks late twenties or older",
    }.get(band)
    if not text:
        return []
    return [f"apparent-age: {text} (coarse scale-invariant band, not an exact age)"]


def render_affordance_contact(contact: Mapping[str, Any]) -> list[str]:
    """Scale-invariant subject self-contact / affordance claim (arm #76).

    Verbalizes ONLY scale-invariant self-contact facts: hand-own-body contact
    count, hand-elevation/gesture count, and the grounded binary. Raw
    normalized wrist distances and pixel values stay in the machine-readable
    payload and are never caption claims.
    """
    if contact.get("abstained"):
        reason = contact.get("abstention_reason") or "affordance not measurable"
        return [f"affordance-contact: abstain ({reason})"]
    if not contact or (
        not contact.get("shoulder_width_norm_ok")
        and "grounded" not in contact
    ):
        # Dimension not measured for this item (e.g. non-affordance runs) —
        # emit no claim, never a fabricated self-contact statement.
        return []
    lines: list[str] = []
    n_contact = int(contact.get("hand_contact_count") or 0)
    if n_contact >= 2:
        lines.append("affordance-contact: both hands rest against her own body")
    elif n_contact == 1:
        lines.append("affordance-contact: one hand rests against her own body")
    n_raised = int(contact.get("hand_elevation_count") or 0)
    if n_raised >= 2:
        lines.append("affordance-contact: both hands are raised (gesturing)")
    elif n_raised == 1:
        lines.append("affordance-contact: one hand is raised (gesturing)")
    if contact.get("grounded"):
        lines.append("affordance-contact: subject is grounded (in contact with the lower frame)")
    if not lines:
        lines.append("affordance-contact: measured (no distinctive self-contact band)")
    return lines


def render_body_configuration(config: Mapping[str, Any]) -> list[str]:
    """Scale-invariant whole-body posture-class claim (arm #83).

    Verbalizes ONLY the coarse posture class (standing / seated / reclined).
    Raw normalized pelvis fractions, pixel extents, knee angles, and torso
    lean stay in the machine-readable payload and are never caption claims.
    """
    if config.get("abstained"):
        reason = config.get("abstention_reason") or "body configuration not measurable"
        return [f"body-configuration: abstain ({reason})"]
    if not config:
        # Dimension not measured for this item (e.g. non-body-configuration
        # runs) — emit no claim, never a fabricated posture statement.
        return []
    cls = config.get("posture_class")
    if cls == "standing":
        return ["body-configuration: subject is standing (upright, legs near-extended)"]
    if cls == "seated":
        return ["body-configuration: subject is seated (hips elevated, knees bent)"]
    if cls == "reclined":
        return ["body-configuration: subject is reclining (torso near-horizontal)"]
    return []


def render_hairstyle(config: Mapping[str, Any]) -> list[str]:
    """Scale-invariant hairstyle claim (arm #82).

    Verbalizes ONLY the coarse length + arrangement bands. Raw normalized
    below-shoulder fractions and pixel spans stay in the machine-readable
    payload and are never caption claims.
    """
    if not config:
        # Dimension not measured for this item (e.g. non-hairstyle runs) —
        # emit no claim, never a fabricated hairstyle.
        return []
    if config.get("abstained"):
        reason = config.get("abstention_reason") or "hairstyle not measurable"
        return [f"hairstyle: abstain ({reason})"]
    if not config.get("hair_present"):
        return ["hairstyle: abstain (no hair region present)"]
    lines: list[str] = []
    length = config.get("hair_length_band")
    if length == "short":
        lines.append("hairstyle: hair is short (does not extend below the shoulders)")
    elif length == "shoulder-length":
        lines.append("hairstyle: hair is shoulder-length")
    elif length == "long":
        lines.append("hairstyle: hair is long (extends below the shoulders)")
    arr = config.get("hair_arrangement_band")
    if arr == "down":
        lines.append("hairstyle: hair hangs down below the shoulders")
    elif arr == "kept-up":
        lines.append("hairstyle: hair is kept above the shoulders (short crop, tied back, or up)")
    return lines


def render_face_visibility(config: Mapping[str, Any]) -> list[str]:
    """Scale-invariant face-prominence claim (arm #84).

    Verbalizes ONLY the coarse visibility band. The raw face-share ratio
    stays in the machine-readable payload and is never a caption claim.
    """
    if not config:
        # Dimension not measured for this item (e.g. non-face-visibility
        # runs) — emit no claim, never a fabricated visibility statement.
        return []
    if config.get("abstained"):
        reason = config.get("abstention_reason") or "face visibility not measurable"
        return [f"face-visibility: abstain ({reason})"]
    if not config.get("face_present"):
        return ["face-visibility: abstain (no face region present)"]
    band = config.get("face_visibility_band")
    if band == "clearly-visible":
        return ["face-visibility: face is clearly visible (face dominates the head region)"]
    if band == "partially-framed":
        return ["face-visibility: face is partially framed by surrounding hair"]
    if band == "hair-dominant":
        return ["face-visibility: hair dominates the head region around a relatively small exposed face"]
    return []


def render_environment_clearance(config: Mapping[str, Any]) -> list[str]:
    """Scale-invariant subject-to-environment clearance claim (arm #85).

    Verbalizes ONLY the coarse clearance band. Raw normalized distances stay
    in the machine-readable payload and are never caption claims.
    """
    if not config:
        # Dimension not measured for this item (e.g. non-environment-clearance
        # runs) — emit no claim, never a fabricated spatial-settings statement.
        return []
    if config.get("abstained"):
        reason = config.get("abstention_reason") or "environment clearance not measurable"
        return [f"environment-clearance: abstain ({reason})"]
    if not config.get("subject_present"):
        return ["environment-clearance: abstain (no foreground subject present)"]
    band = config.get("clearance_band")
    if band == "tight":
        return ["environment-clearance: subject is close to the surrounding backdrop/environment (tight negative space)"]
    if band == "moderate":
        return ["environment-clearance: subject has moderate clearance to the surrounding environment"]
    if band == "spacious":
        return ["environment-clearance: subject is in a spacious setting (ample surrounding open space)"]
    return []


def render_eye_color(config: Mapping[str, Any]) -> list[str]:
    """Scale-invariant eye-color claim (arm #80).

    Verbalizes ONLY the coarse closed-set band. Raw RGB/HSV stats stay in the
    machine-readable payload and are never caption claims.
    """
    if not config:
        # Dimension not measured for this item (e.g. non-eye-color runs) —
        # emit no claim, never a fabricated eye-color statement.
        return []
    if config.get("abstained"):
        reason = config.get("abstention_reason") or "eye color not measurable"
        return [f"eye-color: abstain ({reason})"]
    band = config.get("eye_color_band")
    if band in ("brown", "dark", "blue", "green-hazel", "gray"):
        return [f"eye-color: eyes are {band}"]
    return []


def render_facial_expression(config: Mapping[str, Any]) -> list[str]:
    """Scale-invariant facial-expression claim (arm #81).

    Verbalizes ONLY the coarse expression band. Raw normalized ratios stay in
    the machine-readable payload and are never caption claims.
    """
    if not config:
        # Dimension not measured for this item (e.g. non-facial-expression
        # runs) — emit no claim, never a fabricated expression statement.
        return []
    if config.get("abstained"):
        reason = config.get("abstention_reason") or "expression not measurable"
        return [f"facial-expression: abstain ({reason})"]
    band = config.get("expression_band")
    if band == "neutral":
        return ["facial-expression: neutral expression (mouth relaxed, corners level)"]
    if band == "slight-smile":
        return ["facial-expression: slight smile (mouth corners raised and widened)"]
    if band == "open-smile":
        return ["facial-expression: open smile / laughing (mouth open, corners raised)"]
    return []


def render_gaze_head(gaze: Mapping[str, Any]) -> list[str]:
    """Scale-invariant camera-relative head-orientation claim (arm #68).

    Verbalizes ONLY the scale-invariant direction bands (yaw/pitch/roll); raw
    yaw/pitch/roll degrees and landmark coordinates stay in the machine-
    readable payload and are never caption claims.
    """
    if gaze.get("abstained"):
        reason = gaze.get("abstention_reason") or "face not measurable"
        return [f"gaze-head-orientation: abstain ({reason})"]
    if not gaze or not gaze.get("yaw_band"):
        # Dimension not measured for this item (e.g. non-gaze runs) — emit no
        # claim, never a fabricated camera-interaction statement.
        return []
    lines: list[str] = []
    yb = gaze.get("yaw_band")
    pb = gaze.get("pitch_band")
    rb = gaze.get("roll_band")
    if yb == "facing camera":
        lines.append("gaze-head-orientation: head is facing the camera")
    elif yb == "partially turned":
        lines.append("gaze-head-orientation: head is partially turned from the camera")
    elif yb == "profile or turned away":
        lines.append("gaze-head-orientation: head is turned toward profile / away from the camera")
    if pb == "tilted down":
        lines.append("gaze-head-orientation: head is tilted down (pitch)")
    elif pb == "tilted up":
        lines.append("gaze-head-orientation: head is tilted up (pitch)")
    if rb == "tilted":
        lines.append("gaze-head-orientation: head is tilted to one side (in-plane roll)")
    if not lines:
        lines.append("gaze-head-orientation: head is level and facing the camera")
    return lines


def render_camera_viewing_angle(framing: Mapping[str, Any]) -> list[str]:
    """Scale-invariant camera-relative framing claim (arm #74).

    Verbalizes ONLY the scale-invariant framing bands (shot-scale / headroom /
    camera-height); raw bbox extents and frame shares stay in the machine-
    readable payload and are never caption claims.
    """
    if framing.get("abstained"):
        reason = framing.get("abstention_reason") or "framing not measurable"
        return [f"camera-viewing-angle: abstain ({reason})"]
    if not framing or not framing.get("shot_scale_band"):
        # Dimension not measured for this item (e.g. non-camera-viewing-angle
        # runs) — emit no claim, never a fabricated framing statement.
        return []
    lines: list[str] = []
    hroom = framing.get("headroom_band")
    # shot_scale_band + camera_height_band are payload-only (88% full-body /
    # 100% eye-level on the probe cohort — degenerate uniform axes, never
    # verbalized).
    if hroom == "tight":
        lines.append("camera-viewing-angle: headroom is tight (head near the frame top)")
    elif hroom == "wide":
        lines.append("camera-viewing-angle: headroom is wide (ample space above the head)")
    if not lines:
        lines.append("camera-viewing-angle: framing measured (no distinctive band)")
    return lines


def render_image_focus(focus: Mapping[str, Any]) -> list[str]:
    """Scale-invariant focus / depth-of-field claim (arm #75).

    Verbalizes ONLY the scale-invariant focus/DOF bands (subject-vs-frame
    interior acutance band + background-vs-subject DOF band); raw acutance
    numbers and canonical dims stay in the machine-readable payload and are
    never caption claims.
    """
    if focus.get("abstained"):
        reason = focus.get("abstention_reason") or "focus not measurable"
        return [f"image-focus: abstain ({reason})"]
    if not focus or (not focus.get("subject_focus_band") and not focus.get("dof_band")):
        # Dimension not measured for this item (e.g. non-image-focus runs) —
        # emit no claim, never a fabricated focus statement.
        return []
    lines: list[str] = []
    sb = focus.get("subject_focus_band")
    if sb == "subject-crisp":
        lines.append("image-focus: subject is the crispest in-focus part of the frame")
    elif sb == "subject-softer":
        lines.append("image-focus: subject looks softer than the rest of the frame")
    elif sb == "subject-comparable":
        lines.append("image-focus: subject and frame are in similar focus")
    db = focus.get("dof_band")
    if db == "background-blurred":
        lines.append(
            "image-focus: background is clearly softer than the subject (shallow depth-of-field look)"
        )
    elif db == "background-sharp":
        lines.append(
            "image-focus: background is about as sharp as the subject (deep-focus look)"
        )
    elif db == "background-soft":
        lines.append("image-focus: background is somewhat softer than the subject")
    elif focus.get("dof_abstained"):
        lines.append(
            f"image-focus: depth-of-field not assessed ({focus.get('dof_abstention_reason')})"
        )
    if not lines:
        lines.append("image-focus: focus measured (no distinctive band)")
    return lines


def render_relational(det: Mapping[str, Any]) -> list[str]:
    """Relational determinations from derive_determinations (relational part)."""
    lines: list[str] = []
    parts = det.get("body_parts_visible") or []
    visible = [p.get("part") for p in parts if isinstance(p, Mapping) and p.get("part")]
    if visible:
        lines.append(f"relational: visible body regions {', '.join(sorted(visible))}")
    orientation = det.get("orientation") or {}
    upright = orientation.get("upright_deg")
    if isinstance(upright, (int, float)):
        lines.append(f"relational: torso upright angle {float(upright):.1f} degrees")
    relations = det.get("relations") or []
    if isinstance(relations, list):
        for rel in relations:
            lines.append(f"relational: {rel}")
    return lines


# ---------------------------------------------------------------------------
# Deterministic dossier assembly + context4k compression.
# ---------------------------------------------------------------------------

def _setting_payload(setting: Mapping[str, Any] | None) -> dict[str, Any]:
    if not setting:
        return {}
    payload = {
        "background_coverage": setting.get("background_coverage"),
        "dominant_background_color": setting.get("dominant_background_color"),
        "dominant_background_hex": setting.get("dominant_background_hex"),
        "background_tone_band": setting.get("background_tone_band"),
        "background_vibrancy_band": setting.get("background_vibrancy_band"),
        "background_pattern_band": setting.get("background_pattern_band"),
        "background_deviant_fraction": setting.get("background_deviant_fraction"),
    }
    ab = setting.get("abstention_reason")
    if setting.get("abstained"):
        payload["abstention"] = ab or "abstained"
    return payload


def _texture_payload(texture: Mapping[str, Any] | None) -> dict[str, Any]:
    if not texture:
        return {}
    payload = {
        "fabric_class": texture.get("fabric_class"),
        "fabric_coverage": texture.get("fabric_coverage"),
        "fabric_edge_fraction": texture.get("fabric_edge_fraction"),
        "fabric_deviant_fraction": texture.get("fabric_deviant_fraction"),
        "fabric_texture_band": texture.get("fabric_texture_band"),
        "fabric_pattern_band": texture.get("fabric_pattern_band"),
        "skin_class": texture.get("skin_class"),
        "skin_coverage": texture.get("skin_coverage"),
        "skin_edge_fraction": texture.get("skin_edge_fraction"),
        "skin_mean_gradient": texture.get("skin_mean_gradient"),
        "skin_texture_band": texture.get("skin_texture_band"),
    }
    ab = texture.get("abstention_reason")
    if texture.get("abstained"):
        payload["abstention"] = ab or "abstained"
    return payload


def _pose_articulation_payload(articulation: Mapping[str, Any] | None) -> dict[str, Any]:
    if not articulation:
        return {}
    payload = {
        "elbow_flexion_left": articulation.get("elbow_flexion_left"),
        "elbow_flexion_right": articulation.get("elbow_flexion_right"),
        "knee_flexion_left": articulation.get("knee_flexion_left"),
        "knee_flexion_right": articulation.get("knee_flexion_right"),
        "torso_twist_deg": articulation.get("torso_twist_deg"),
        "torso_lean_deg": articulation.get("torso_lean_deg"),
        "pelvis_tilt_deg": articulation.get("pelvis_tilt_deg"),
        "stance_class": articulation.get("stance_class"),
        "contrapposto": articulation.get("contrapposto"),
        "arm_crossing_count": articulation.get("arm_crossing_count"),
        "legs_crossed": articulation.get("legs_crossed"),
        "left_arm_near_torso_fraction": articulation.get("left_arm_near_torso_fraction"),
        "right_arm_near_torso_fraction": articulation.get("right_arm_near_torso_fraction"),
        "elbow_flexion_asymmetry_deg": articulation.get("elbow_flexion_asymmetry_deg"),
        "knee_flexion_asymmetry_deg": articulation.get("knee_flexion_asymmetry_deg"),
    }
    ab = articulation.get("abstention_reason")
    if articulation.get("abstained"):
        payload["abstention"] = ab or "abstained"
    return payload


def _pointmap_depth_payload(pointmap_depth: Mapping[str, Any] | None) -> dict[str, Any]:
    if not pointmap_depth:
        return {}
    payload = {
        "median_z": pointmap_depth.get("median_z"),
        "z_p10": pointmap_depth.get("z_p10"),
        "z_p90": pointmap_depth.get("z_p90"),
        "depth_relief_ratio": pointmap_depth.get("depth_relief_ratio"),
        "relief_band": pointmap_depth.get("relief_band"),
        "region_median_z": pointmap_depth.get("region_median_z"),
        "depth_ordering": pointmap_depth.get("depth_ordering"),
        "nearest_region": pointmap_depth.get("nearest_region"),
        "farthest_region": pointmap_depth.get("farthest_region"),
        "hand_ordering": pointmap_depth.get("hand_ordering"),
        "hand_dz_ratio": pointmap_depth.get("hand_dz_ratio"),
        "left_hand_in_front": pointmap_depth.get("left_hand_in_front"),
        "right_hand_in_front": pointmap_depth.get("right_hand_in_front"),
    }
    ab = pointmap_depth.get("abstention_reason")
    if pointmap_depth.get("abstained"):
        payload["abstention"] = ab or "abstained"
    return payload


def _matting_alpha_payload(matting: Mapping[str, Any] | None) -> dict[str, Any]:
    if not matting:
        return {}
    payload = {
        "coverage_ratio": matting.get("coverage_ratio"),
        "coverage_band": matting.get("coverage_band"),
        "subject_px": matting.get("subject_px"),
        "frame_px": matting.get("frame_px"),
        "subject_height_px": matting.get("subject_height_px"),
        "boundary_crispness": matting.get("boundary_crispness"),
        "boundary_crisp_band": matting.get("boundary_crisp_band"),
        "hair_soft_share": matting.get("hair_soft_share"),
        "soft_edge_band": matting.get("soft_edge_band"),
        "silhouette_closed": matting.get("silhouette_closed"),
        "silhouette_closedness": matting.get("silhouette_closedness"),
        "border_open_fraction": matting.get("border_open_fraction"),
    }
    ab = matting.get("abstention_reason")
    if matting.get("abstained"):
        payload["abstention"] = ab or "abstained"
    return payload


def _face_geometry_payload(face: Mapping[str, Any] | None) -> dict[str, Any]:
    if not face:
        return {}
    payload = {
        "n_landmarks": face.get("n_landmarks"),
        "via": face.get("via"),
        "z_span_rel": face.get("z_span_rel"),
        "face_bbox_px": face.get("face_bbox_px"),
        "eye_spacing_face_width": face.get("eye_spacing_face_width"),
        "interpupillary_face_width": face.get("interpupillary_face_width"),
        "mouth_face_width": face.get("mouth_face_width"),
        "jaw_face_width": face.get("jaw_face_width"),
        "midface_share": face.get("midface_share"),
        "eye_spacing_band": face.get("eye_spacing_band"),
        "mouth_band": face.get("mouth_band"),
        "jaw_band": face.get("jaw_band"),
        "midface_band": face.get("midface_band"),
        "midface_plausibility_abstained": face.get("midface_plausibility_abstained"),
    }
    ab = face.get("abstention_reason")
    if face.get("abstained"):
        payload["abstention"] = ab or "abstained"
    return payload


def _object_relations_payload(objrel: Mapping[str, Any] | None) -> dict[str, Any]:
    if not objrel:
        return {}
    payload = {
        "count": objrel.get("count"),
        "count_band": objrel.get("count_band"),
        "placement_band": objrel.get("placement_band"),
        "classes": objrel.get("classes"),
        "class_counts": objrel.get("class_counts"),
        "n_front": objrel.get("n_front"),
        "n_behind": objrel.get("n_behind"),
        "n_mixed": objrel.get("n_mixed"),
        "box_threshold": objrel.get("box_threshold"),
        "text_threshold": objrel.get("text_threshold"),
        "detections": objrel.get("detections"),
    }
    ab = objrel.get("abstention_reason")
    if objrel.get("abstained"):
        payload["abstention"] = ab or "abstained"
    return payload


def _scene_category_payload(scene: Mapping[str, Any] | None) -> dict[str, Any]:
    if not scene:
        return {}
    payload = {
        "category": scene.get("category"),
        "confidence": scene.get("confidence"),
        "probabilities": scene.get("probabilities"),
        "abstain_confidence": scene.get("abstain_confidence"),
    }
    ab = scene.get("abstention_reason")
    if scene.get("abstained"):
        payload["abstention"] = ab or "abstained"
    return payload


def _gaze_head_payload(gaze: Mapping[str, Any] | None) -> dict[str, Any]:
    if not gaze:
        return {}
    payload = {
        "yaw_deg": gaze.get("yaw"),
        "pitch_deg": gaze.get("pitch"),
        "roll_deg": gaze.get("roll"),
        "yaw_band": gaze.get("yaw_band"),
        "pitch_band": gaze.get("pitch_band"),
        "roll_band": gaze.get("roll_band"),
        "via": gaze.get("via"),
    }
    ab = gaze.get("abstention_reason")
    if gaze.get("abstained"):
        payload["abstention"] = ab or "abstained"
    return payload


def _camera_viewing_angle_payload(framing: Mapping[str, Any] | None) -> dict[str, Any]:
    if not framing:
        return {}
    payload = {
        "shot_scale_band": framing.get("shot_scale_band"),
        "headroom_band": framing.get("headroom_band"),
        "camera_height_band": framing.get("camera_height_band"),
        "subject_bbox_px": framing.get("subject_bbox_px"),
        "subject_frame_height_share": framing.get("subject_frame_height_share"),
        "headroom_frame_share": framing.get("headroom_frame_share"),
        "subject_center_of_mass_roi": framing.get("subject_center_of_mass_roi"),
    }
    ab = framing.get("abstention_reason")
    if framing.get("abstained"):
        payload["abstention"] = ab or "abstained"
    return payload


def _image_focus_payload(focus: Mapping[str, Any] | None) -> dict[str, Any]:
    if not focus:
        return {}
    payload = {
        "subject_focus_band": focus.get("subject_focus_band"),
        "dof_band": focus.get("dof_band"),
        "subject_acutance_median": focus.get("subject_acutance_median"),
        "background_acutance_median": focus.get("background_acutance_median"),
        "background_p99": focus.get("background_p99"),
        "background_std": focus.get("background_std"),
        "global_acutance_median": focus.get("global_acutance_median"),
        "dof_ratio": focus.get("dof_ratio"),
        "subject_vs_frame_ratio": focus.get("subject_vs_frame_ratio"),
        "subject_share": focus.get("subject_share"),
        "canonical_dims": focus.get("canonical_dims"),
    }
    ab = focus.get("abstention_reason")
    if focus.get("abstained"):
        payload["abstention"] = ab or "abstained"
    if focus.get("dof_abstained"):
        payload["dof_abstention"] = focus.get("dof_abstention_reason") or "dof not assessed"
    return payload


def _apparent_age_payload(age: Mapping[str, Any] | None) -> dict[str, Any]:
    if not age:
        return {}
    payload = {
        "age_years": age.get("age_years"),
        "age_band": age.get("age_band"),
        "gender_probe": age.get("gender_probe"),
        "via": age.get("via"),
        "seg2_face_neck_px": age.get("seg2_face_neck_px"),
        "seg2_subject_px": age.get("seg2_subject_px"),
    }
    ab = age.get("abstention_reason")
    if age.get("abstained"):
        payload["abstention"] = ab or "abstained"
    return payload


def _affordance_contact_payload(contact: Mapping[str, Any] | None) -> dict[str, Any]:
    if not contact:
        return {}
    payload = {
        "hand_contact_count": contact.get("hand_contact_count"),
        "hand_elevation_count": contact.get("hand_elevation_count"),
        "left_hand_visible": contact.get("left_hand_visible"),
        "right_hand_visible": contact.get("right_hand_visible"),
        "left_hand_contact": contact.get("left_hand_contact"),
        "right_hand_contact": contact.get("right_hand_contact"),
        "left_hand_raised": contact.get("left_hand_raised"),
        "right_hand_raised": contact.get("right_hand_raised"),
        "grounded": contact.get("grounded"),
        "shoulder_width_px": contact.get("shoulder_width_px"),
        "left_wrist_trunk_dist_norm": contact.get("left_wrist_trunk_dist_norm"),
        "right_wrist_trunk_dist_norm": contact.get("right_wrist_trunk_dist_norm"),
        "left_wrist_hip_offset_norm": contact.get("left_wrist_hip_offset_norm"),
        "right_wrist_hip_offset_norm": contact.get("right_wrist_hip_offset_norm"),
    }
    ab = contact.get("abstention_reason")
    if contact.get("abstained"):
        payload["abstention"] = ab or "abstained"
    return payload


def _body_configuration_payload(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    payload = {
        "posture_class": config.get("posture_class"),
        "pelvis_height_fraction": config.get("pelvis_height_fraction"),
        "torso_leg_extent_ratio": config.get("torso_leg_extent_ratio"),
        "knee_flexion_left_deg": config.get("knee_flexion_left_deg"),
        "knee_flexion_right_deg": config.get("knee_flexion_right_deg"),
        "median_knee_flexion_deg": config.get("median_knee_flexion_deg"),
        "torso_lean_deg": config.get("torso_lean_deg"),
    }
    ab = config.get("abstention_reason")
    if config.get("abstained"):
        payload["abstention"] = ab or "abstained"
    return payload


def _hairstyle_payload(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    payload = {
        "hair_present": config.get("hair_present"),
        "hair_length_band": config.get("hair_length_band"),
        "hair_arrangement_band": config.get("hair_arrangement_band"),
        "hair_below_shoulder_ratio": config.get("hair_below_shoulder_ratio"),
        "hair_below_shoulder_fraction": config.get("hair_below_shoulder_fraction"),
        "hair_span_ratio": config.get("hair_span_ratio"),
        "hair_centroid_row_fraction": config.get("hair_centroid_row_fraction"),
    }
    ab = config.get("abstention_reason")
    if config.get("abstained"):
        payload["abstention"] = ab or "abstained"
    return payload


def _face_visibility_payload(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    payload = {
        "face_present": config.get("face_present"),
        "face_share_of_head": config.get("face_share_of_head"),
        "face_visibility_band": config.get("face_visibility_band"),
        "face_px": config.get("face_px"),
        "face_frame_coverage": config.get("face_frame_coverage"),
    }
    ab = config.get("abstention_reason")
    if config.get("abstained"):
        payload["abstention"] = ab or "abstained"
    return payload


def _environment_clearance_payload(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    payload = {
        "subject_present": config.get("subject_present"),
        "clearance_band": config.get("clearance_band"),
        "clearance_ratio": config.get("clearance_ratio"),
        "clearance_top": config.get("clearance_top"),
        "clearance_bottom": config.get("clearance_bottom"),
        "clearance_left": config.get("clearance_left"),
        "clearance_right": config.get("clearance_right"),
        "subject_frame_coverage": config.get("subject_frame_coverage"),
    }
    ab = config.get("abstention_reason")
    if config.get("abstained"):
        payload["abstention"] = ab or "abstained"
    return payload


def _eye_color_payload(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    payload = {
        "eye_color_band": config.get("eye_color_band"),
        "sample_count": config.get("sample_count"),
        "mean_rgb": config.get("mean_rgb"),
        "hue_deg": config.get("hue_deg"),
        "saturation": config.get("saturation"),
        "value": config.get("value"),
    }
    ab = config.get("abstention_reason")
    if config.get("abstained"):
        payload["abstention"] = ab or "abstained"
    return payload


def _facial_expression_payload(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    payload = {
        "expression_band": config.get("expression_band"),
        "spread_ratio": config.get("spread_ratio"),
        "openness_ratio": config.get("openness_ratio"),
        "corner_elevation_ratio": config.get("corner_elevation_ratio"),
        "reference_fallback": config.get("reference_fallback"),
    }
    ab = config.get("abstention_reason")
    if config.get("abstained"):
        payload["abstention"] = ab or "abstained"
    return payload


def build_evidence_payload(
    *,
    image_id: str,
    proportions: Mapping[str, Any],
    clothing: Mapping[str, Any],
    hair: Mapping[str, Any],
    skin: Mapping[str, Any],
    lighting: Mapping[str, Any],
    setting: Mapping[str, Any] | None = None,
    texture: Mapping[str, Any] | None = None,
    articulation: Mapping[str, Any] | None = None,
    pointmap_depth: Mapping[str, Any] | None = None,
    matting_alpha: Mapping[str, Any] | None = None,
    face_geometry: Mapping[str, Any] | None = None,
    object_relations: Mapping[str, Any] | None = None,
    scene_category: Mapping[str, Any] | None = None,
    gaze_head: Mapping[str, Any] | None = None,
    camera_viewing_angle: Mapping[str, Any] | None = None,
    image_focus: Mapping[str, Any] | None = None,
    apparent_age: Mapping[str, Any] | None = None,
    affordance_contact: Mapping[str, Any] | None = None,
    body_configuration: Mapping[str, Any] | None = None,
    hairstyle: Mapping[str, Any] | None = None,
    face_visibility: Mapping[str, Any] | None = None,
    environment_clearance: Mapping[str, Any] | None = None,
    eye_color: Mapping[str, Any] | None = None,
    facial_expression: Mapping[str, Any] | None = None,
    determinations: Mapping[str, Any],
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Machine-readable evidence payload (dossier / compressor input).

    Absolute pixel measurements live here and are NEVER verbalized as caption
    claims (camera-frame-dependent). Ratios/bands/names are derived for the
    claim-level text by the render functions.
    """
    payload: dict[str, Any] = {
        "schema_version": 1,
        "program_id": "stratum-contextual-specialist-research",
        "image_id": image_id,
        "evidence_payload": {
            "absolute_pixel_measurements": {
                "between_shoulders": proportions.get("between_shoulders"),
                "between_hips": proportions.get("between_hips"),
                "torso_length": proportions.get("torso_length"),
                "left_leg_length": proportions.get("left_leg_length"),
                "right_leg_length": proportions.get("right_leg_length"),
            },
            "ratios": {
                "shoulder_hip_ratio": proportions.get("shoulder_hip_ratio"),
                "leg_torso_ratio": proportions.get("leg_torso_ratio"),
                "hair_face_extent_ratio": hair.get("hair_face_extent_ratio"),
                "surround_ratio": lighting.get("surround_ratio"),
            },
            "proportions_abstention": proportions.get("shoulder_hip_ratio_abstention_reason"),
            "clothing": {
                g.get("class"): {
                    "coverage": g.get("coverage"),
                    "dominant_color_name": g.get("dominant_color_name"),
                    "dominant_hex": g.get("dominant_hex"),
                }
                for g in (clothing.get("garments") or [])
            },
            "hair": {
                "coverage": hair.get("hair_coverage"),
                "dominant_color_name": hair.get("hair_dominant_color_name"),
                "dominant_hex": hair.get("hair_dominant_hex"),
                "frame_coverage": hair.get("hair_frame_coverage"),
                "position": hair.get("hair_position"),
            },
            "skin": {
                "skin_tone_name": skin.get("skin_tone_name"),
                "skin_tone_hex": skin.get("skin_tone_hex"),
                "face_tone_name": skin.get("face_tone_name"),
                "face_tone_hex": skin.get("face_tone_hex"),
                "skin_coverage": skin.get("skin_coverage"),
                "face_body_agree": skin.get("face_body_agree"),
            },
            "lighting": {
                "mean_luma": lighting.get("mean_luma"),
                "median_luma": lighting.get("median_luma"),
                "dynamic_range": lighting.get("dynamic_range"),
                "shadow_fraction": lighting.get("shadow_fraction"),
                "surround_ratio": lighting.get("surround_ratio"),
                "light_vector": lighting.get("light_vector"),
                "light_residual": lighting.get("light_residual"),
            },
            "setting": _setting_payload(setting),
            "texture": _texture_payload(texture),
            "pose_articulation": _pose_articulation_payload(articulation),
            "pointmap_depth": _pointmap_depth_payload(pointmap_depth),
            "matting_alpha": _matting_alpha_payload(matting_alpha),
            "face_geometry": _face_geometry_payload(face_geometry),
            "object_relations": _object_relations_payload(object_relations),
            "scene_category": _scene_category_payload(scene_category),
            "gaze_head": _gaze_head_payload(gaze_head),
            "camera_viewing_angle": _camera_viewing_angle_payload(camera_viewing_angle),
            "image_focus": _image_focus_payload(image_focus),
            "apparent_age": _apparent_age_payload(apparent_age),
            "affordance_contact": _affordance_contact_payload(affordance_contact),
            "body_configuration": _body_configuration_payload(body_configuration),
            "hairstyle": _hairstyle_payload(hairstyle),
            "face_visibility": _face_visibility_payload(face_visibility),
            "environment_clearance": _environment_clearance_payload(environment_clearance),
            "eye_color": _eye_color_payload(eye_color),
            "facial_expression": _facial_expression_payload(facial_expression),
            "relational": {
                "body_parts_visible": [
                    p.get("part") for p in (determinations.get("body_parts_visible") or []) if isinstance(p, Mapping)
                ],
                "orientation_upright_deg": (determinations.get("orientation") or {}).get("upright_deg"),
                "relations": determinations.get("relations") or [],
            },
        },
    }
    if source_sha256:
        payload["source_sha256"] = source_sha256
    payload["evidence_payload_fingerprint"] = _sha256(
        _canonical_json(payload["evidence_payload"]).encode("utf-8")
    )
    return payload


def assemble_dossier(
    *,
    image_id: str,
    proportions: Mapping[str, Any],
    clothing: Mapping[str, Any],
    hair: Mapping[str, Any],
    skin: Mapping[str, Any],
    lighting: Mapping[str, Any],
    setting: Mapping[str, Any] | None = None,
    texture: Mapping[str, Any] | None = None,
    articulation: Mapping[str, Any] | None = None,
    pointmap_depth: Mapping[str, Any] | None = None,
    matting_alpha: Mapping[str, Any] | None = None,
    face_geometry: Mapping[str, Any] | None = None,
    object_relations: Mapping[str, Any] | None = None,
    scene_category: Mapping[str, Any] | None = None,
    gaze_head: Mapping[str, Any] | None = None,
    camera_viewing_angle: Mapping[str, Any] | None = None,
    image_focus: Mapping[str, Any] | None = None,
    apparent_age: Mapping[str, Any] | None = None,
    affordance_contact: Mapping[str, Any] | None = None,
    body_configuration: Mapping[str, Any] | None = None,
    hairstyle: Mapping[str, Any] | None = None,
    face_visibility: Mapping[str, Any] | None = None,
    environment_clearance: Mapping[str, Any] | None = None,
    eye_color: Mapping[str, Any] | None = None,
    facial_expression: Mapping[str, Any] | None = None,
    determinations: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the claim-by-claim expanded dossier for one item.

    Returns a dossier dict containing per-dimension claim lists where every
    claim is a plain-English prose line carrying its supporting evidence part
    and evidence id. This is the deterministic dossier the compressor consumes.
    """
    dimension_factories: tuple[tuple[str, str, Any], ...] = (
        ("body-type-proportions:v1", "body-type", render_proportions(proportions)),
        ("clothing-apparel:v1", "clothing", render_clothing(clothing)),
        ("hair:v1", "hair", render_hair(hair)),
        ("skin-color-tone:v1", "skin-color", render_skin_color(skin)),
        ("lighting:v1", "lighting", render_lighting(lighting)),
        ("setting-environment:v1", "setting", render_setting(setting or {})),
        ("texture-material:v1", "texture", render_texture(texture or {})),
        ("pose-articulation:v1", "pose-articulation", render_pose_articulation(articulation or {})),
        ("pointmap-depth:v1", "pointmap-depth", render_pointmap_depth(pointmap_depth or {})),
        ("matting-alpha:v1", "matting-alpha", render_matting_alpha(matting_alpha or {})),
        ("face-geometry:v1", "face-geometry", render_face_geometry(face_geometry or {})),
        ("object-relations:v1", "object-relations", render_object_relations(object_relations or {})),
        ("scene-category:v1", "scene-category", render_scene_category(scene_category or {})),
        ("gaze-head-orientation:v1", "gaze-head-orientation", render_gaze_head(gaze_head or {})),
        ("camera-viewing-angle:v1", "camera-viewing-angle", render_camera_viewing_angle(camera_viewing_angle or {})),
        ("image-focus:v1", "image-focus", render_image_focus(image_focus or {})),
        ("apparent-age:v1", "apparent-age", render_apparent_age(apparent_age or {})),
        ("affordance-contact:v1", "affordance-contact", render_affordance_contact(affordance_contact or {})),
        ("body-configuration:v1", "body-configuration", render_body_configuration(body_configuration or {})),
        ("hairstyle:v1", "hairstyle", render_hairstyle(hairstyle or {})),
        ("face-visibility:v1", "face-visibility", render_face_visibility(face_visibility or {})),
        ("environment-clearance:v1", "environment-clearance", render_environment_clearance(environment_clearance or {})),
        ("eye-color:v1", "eye-color", render_eye_color(eye_color or {})),
        ("facial-expression:v1", "facial-expression", render_facial_expression(facial_expression or {})),
        (RELATIONAL_EVIDENCE_ID, "relational", render_relational(determinations)),
    )

    sections: dict[str, list[dict[str, Any]]] = {}
    evidence_ids: list[str] = []
    for evidence_id, section, lines in dimension_factories:
        claims = [{"text": line.strip(), "evidence_ids": [evidence_id]} for line in lines if line and line.strip()]
        sections[section] = claims
        if claims and evidence_id not in evidence_ids:
            evidence_ids.append(evidence_id)

    dossier: dict[str, Any] = {
        "schema_version": 1,
        "image_id": image_id,
        "sections": sections,
        "evidence_ids": evidence_ids,
        "token_count": None,  # filled deterministically below
    }
    dossier_text = expanded_dossier_text(dossier)
    dossier["token_count"] = count_tokens(dossier_text)
    return dossier


def expanded_dossier_text(dossier: Mapping[str, Any]) -> str:
    """Deterministic human-readable serialization of the expanded dossier."""
    lines: list[str] = []
    for section, claims in dossier.get("sections", {}).items():
        lines.append(f"## {section.upper()}")
        for claim in claims:
            lines.append(f"[{','.join(claim.get('evidence_ids', []))}] {claim.get('text')}")
    return "\n".join(lines)


def _claim_priority(claim: Mapping[str, Any]) -> tuple[int, str]:
    """Stable, content-based selection priority for compression.

    Lower sort key = included first. Priority ordering is deterministic across
    runs: evidence-id order (declared dimensions first, relational last), then
    the claim text itself for a deterministic tiebreak.
    """
    ids = claim.get("evidence_ids") or []
    order = {eid: i for i, eid in enumerate(DIMENSION_EVIDENCE_IDS)}
    # Relational evidence id sorts after all dimension ids.
    order[RELATIONAL_EVIDENCE_ID] = 1000
    id_rank = min((order.get(eid, 2000) for eid in ids), default=2000)
    return (id_rank, claim.get("text", ""))


def compress_dossier_to_context(
    dossier: Mapping[str, Any], *, target_tokens: int = _COMPACT_TARGET
) -> dict[str, Any]:
    """Deterministically compress the expanded dossier into a compact context.

    Selects claims in stable priority order up to `target_tokens`; trims the
    last included claim at a natural token boundary if it would overshoot, and
    records the truncation. Never invents padding: if the whole corpus is under
    budget, `under_budget` is set True with the honest count (reaching the floor
    requires the aggregator expansion stage, i.e. the scheduler round-trip).
    """
    all_claims: list[dict[str, Any]] = []
    for claims in dossier.get("sections", {}).values():
        all_claims.extend(claims)
    ordered = sorted(all_claims, key=lambda claim: _claim_priority(claim))

    selected: list[dict[str, Any]] = []
    used = 0
    truncated_text: str | None = None
    for claim in ordered:
        text = claim.get("text", "").strip()
        if not text:
            continue
        tokens = count_tokens(text)
        if used + tokens <= target_tokens:
            selected.append(dict(claim))
            used += tokens
            continue
        # Overshoot: trim this one claim at a sentence/token boundary.
        trimmed, trimmed_tokens = _trim_to_budget(text, target_tokens - used)
        if trimmed_tokens > 0:
            over = dict(claim)
            over["text"] = trimmed
            over["truncated"] = True
            selected.append(over)
            used += trimmed_tokens
            truncated_text = text
        break  # after the first trimmed claim we stop adding

    claims_out = [dict(claim) for claim in selected]
    compact_text = context4k_text(claims_out)
    return {
        "claims": claims_out,
        "context_text": compact_text,
        "token_count": used,
        "target_tokens": target_tokens,
        "under_budget": used < target_tokens,
        "truncated_claim_source": truncated_text,
        "excluded_claim_count": max(0, len(ordered) - len(selected)),
    }


def _trim_to_budget(text: str, budget: int) -> tuple[str, int]:
    """Trim text at a natural sentence boundary to fit `budget` tokens."""
    if budget <= 0:
        return "", 0
    match = _SENTENCE_END.match(text.strip())
    if match:
        head = match.group(1).rstrip()
        if head and count_tokens(head) <= budget:
            return head, count_tokens(head)
    # Fallback: progressive word-level trim.
    tokens = text.strip().split(" ")
    acc: list[str] = []
    used = 0
    for word in tokens:
        candidate = " ".join([*acc, word])
        n = count_tokens(candidate)
        if used + n > budget:
            break
        acc.append(word)
        used = n
    return " ".join(acc), used


def context4k_text(claims: Iterable[Mapping[str, Any]]) -> str:
    """Stable human-readable serialization of the compact context."""
    return "\n".join(f"- {claim.get('text')}" for claim in claims)


def build_compression_bundle(
    *,
    image_id: str,
    dossier: Mapping[str, Any],
    context: Mapping[str, Any],
    program: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the contract-shaped compression bundle validated by
    `validate_compression_bundle`. Raises DossierError when it would not pass."""
    validate_program(program)
    expanded_tokens = int(dossier.get("token_count") or count_tokens(expanded_dossier_text(dossier)))
    compact_tokens = int(context.get("token_count") or count_tokens(context.get("context_text", "")))

    bundle: dict[str, Any] = {
        "schema_version": 1,
        "image_id": image_id,
        "expanded_dossier": {
            "token_count": expanded_tokens,
            "evidence_ids": list(dossier.get("evidence_ids", [])),
        },
        "compact_context": {
            "token_count": compact_tokens,
            "claims": [
                {
                    "text": claim.get("text"),
                    "evidence_ids": list(claim.get("evidence_ids", [])),
                }
                for claim in context.get("claims", [])
            ],
        },
        "artifacts": {
            "structured": "context4k.json",
            "human_readable": "context4k.md",
            "provenance": "compression.json",
        },
    }
    if provenance:
        bundle["compression_provenance"] = provenance

    try:
        validate_compression_bundle(bundle, program)
    except ContractError as exc:
        raise DossierError(f"compression bundle fails the contract: {exc}") from exc
    return bundle


def build_item_context4k_artifacts(bundle: Mapping[str, Any], target_dir: Path) -> dict[str, Any]:
    """Persist the three context4k artifacts (context4k.json/md/compression.json).

    All writes are additive and live under the caller's (approved, noncanonical)
    target directory. Returns a dict reporting what was written.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    structured = json.loads(_canonical_json(bundle))
    structured_path = target_dir / "context4k.json"
    structured_path.write_text(json.dumps(structured, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    claims = bundle.get("compact_context", {}).get("claims", [])
    md_lines = [
        f"# context4k — {bundle.get('image_id')}",
        "",
        "Each claim carries its supporting evidence IDs (deterministic specialists + relational determinations).",
        "",
    ]
    for claim in claims:
        md_lines.append(f"- [{','.join(claim.get('evidence_ids', []))}] {claim.get('text')}")
    md_path = target_dir / "context4k.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    compression = {
        "schema_version": 1,
        "image_id": bundle.get("image_id"),
        "expanded_dossier_token_count": bundle.get("expanded_dossier", {}).get("token_count"),
        "compact_context_token_count": bundle.get("compact_context", {}).get("token_count"),
        "evidence_ids": bundle.get("expanded_dossier", {}).get("evidence_ids", []),
        "claim_count": len(claims),
        "compression_provenance": bundle.get("compression_provenance"),
    }
    compression_path = target_dir / "compression.json"
    compression_path.write_text(json.dumps(compression, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "context4k.json": str(structured_path),
        "context4k.md": str(md_path),
        "compression.json": str(compression_path),
        "claim_count": len(claims),
    }
