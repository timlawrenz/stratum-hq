"""Deterministic object / accessory presence + spatial-relations measurements.

Arm #61. NEW-MODEL-CLASS specialist: runs the open-weight Grounding DINO
text-grounded open-vocabulary detector (`IDEA-Research/grounding-dino-base`,
Apache-2.0, HF Transformers path, CPU on owned hardware) over the full-frame
decoded source and derives scale-invariant facts:

- object-presence count band (none / sparse / moderate / dense) from the number
  of detections above the calibrated box threshold;
- placement band (foreground / background / mix) from the overlap of each
  detected box with the seg2 subject mask — objects overlapping the subject
  (held / on-person) vs objects essentially in the background;
- canonical object-class list (the detected classes, mapped from the raw
  phrase to the frozen closed vocabulary), scale-invariant names only;
- for each detection, the machine-readable payload keeps the normalized box,
  score, canonical class, and subject-overlap — never prose.

Only scale-invariant facts are verbalized (counts, names, relative placement).
Absolute box coordinates stay in ``evidence_payload`` JSON and are never caption
claims (camera-frame-dependent; the measurement-semantics directive).

Honesty guards (each measured on the frozen cohort 2026-08-07):

- THE old furniture-centric vocabulary was DEGENERATE on this cohort (9/24
  detected; the cohort is scene-dominant: water/field/concrete/mirror/window,
  not chairs/desks). The frozen closed vocabulary below is COHORT-DERIVED from
  the already-computed VLM dense-description blocks (option-B arm #47) and the
  calibrated sweep: at box_threshold 0.25, 21/24 items detect >= 1 object.
- SUBJECT-SELF-CONFUSION GUARD: Grounding DINO fires ``body``/``person`` on
  the subject herself (measured: a 0.28-0.45 ``body`` box per several items).
  Raw detections whose canonical class is a subject-confusable standalone word
  are EXCLUDED from both the count and the prose (they are the subject, not an
  object). ``body of water`` is a legitimate scene object and is kept — the
  exclusion is exact-standalone-word, not substring.
- Band calibration (rule arm #34/#35/#58/#59): thresholds below were calibrated
  from the measured cohort distribution (no band >= 75%): count none=8,
  sparse(1)=7, moderate(2-4)=5, dense(5+)=4; placement foreground=4,
  background=4, mix=8, none=8 (mix is the natural majority when the subject
  fills the frame — reported honestly, not re-bucketed).

Capability probe (qualification gate step 2, non-sensitive synthetic + the
frozen cohort on owned hardware): transformers 5.8.1 API — ``post_process
_grounded_object_detection`` takes ``threshold`` (NOT ``box_threshold``) and
``text_labels`` in the result dict; load ~8s, CPU inference ~5s/image — 24
items fit a bounded CPU measurement with no GPU claim and no VRAM contention
with the caption model.

Provenance: open-weight model (grounding-dino-base, safetensors sha256
5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21,
Apache-2.0) run on owned hardware only; no hosted third-party inference of the
sensitive corpus; model_asset_dir dependency-injected so unit tests can point
at a fixture directory and the runner at the frozen model asset.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# ---------------------------------------------------------------------------
# Frozen closed vocabulary (cohort-derived 2026-08-07 — see module docstring).
# Order does not matter; the canonicalizer prefers the longest match.
# NOTE: subject words (body/person/woman/...) are deliberately NOT in the
# vocabulary — they are guarded separately because the detector still emits
# them as raw phrases and they must never become object claims.
# ---------------------------------------------------------------------------
CLOSED_VOCAB: tuple[str, ...] = (
    "potted plant", "plant", "tree", "grass", "flowers", "mirror",
    "window", "door", "wall", "fence", "railing", "swimming pool", "pool",
    "body of water", "ocean", "beach", "sand", "boat", "boat deck",
    "shower", "curb", "skateboard", "graffiti", "pillow", "bed", "bench",
    "chair", "cushion", "towel", "blanket", "vase", "lamp", "candle",
    "earrings", "necklace", "bracelet", "ring", "watch", "hat", "scarf",
    "glasses", "belt", "bag", "purse", "shoes", "sneakers", "heels",
    "sandals",
)

# Exact-standalone subject-confusable words the detector can emit for the
# subject herself — never object claims (exclusion is exact-token, so
# ``body of water`` survives).
_SUBJECT_WORDS: frozenset[str] = frozenset({
    "body", "person", "woman", "man", "people", "human", "subject",
})

# Calibrated thresholds (2026-08-07, frozen cohort).
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20
SPARSE = 1          # count <= 1 -> sparse
MODERATE = 2        # count >= 2 -> moderate
DENSE = 5           # count >= 5 -> dense
FRONT_OVERLAP = 0.50   # box overlap with subject mask > this -> in-front
BEHIND_OVERLAP = 0.15  # box overlap with subject mask < this -> behind

# Model asset (bind the sha256 in the declaration; dir injected by caller).
MODEL_SHA256 = "5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21"


class ObjectRelationsError(RuntimeError):
    pass


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise ObjectRelationsError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise ObjectRelationsError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise ObjectRelationsError(f"seg2 must be integer class labels, got dtype {seg2.dtype}")


def validate_rgb_array(rgb: np.ndarray) -> None:
    if not isinstance(rgb, np.ndarray):
        raise ObjectRelationsError("rgb must be a numpy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ObjectRelationsError(f"rgb must be (H, W, 3), got shape {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise ObjectRelationsError(f"rgb must be uint8, got dtype {rgb.dtype}")


class _GroundingDinoRuntime:
    """Lazy, process-wide Grounding DINO processor + model (CPU)."""

    _processor = None
    _model = None

    @classmethod
    def get(cls, model_asset_dir: str):
        if cls._model is None:
            from transformers import (
                GroundingDinoForObjectDetection,
                GroundingDinoProcessor,
            )
            cls._processor = GroundingDinoProcessor.from_pretrained(model_asset_dir)
            cls._model = GroundingDinoForObjectDetection.from_pretrained(model_asset_dir)
            cls._model.eval()
        return cls._processor, cls._model

    @classmethod
    def reset(cls) -> None:
        cls._processor = None
        cls._model = None


def canonical_class(raw_label: str) -> str:
    """Map a raw detector phrase to the longest closed-vocabulary class.

    Compounds (e.g. ``window window frame door``) collapse to the longest
    vocabulary phrase they contain; an unmatched phrase returns itself (and is
    treated as unclassified at the caller).
    """
    norm = raw_label.strip().lower()
    if not norm:
        return ""
    for candidate in sorted(CLOSED_VOCAB, key=len, reverse=True):
        # Exact-phrase or token-boundary containment.
        if candidate in norm:
            return candidate
    # Token-level fallback: an exact vocab token inside the phrase.
    tokens = set(norm.replace("-", " ").split())
    for candidate in sorted(CLOSED_VOCAB, key=len, reverse=True):
        if any(candidate == tok or candidate in tok for tok in tokens):
            return candidate
    return norm


def _is_subject_self(cls: str) -> bool:
    """True when the canonical class is an exact standalone subject word."""
    return cls in _SUBJECT_WORDS


def _count_band(count: int) -> str:
    if count <= 0:
        return "none"
    if count <= SPARSE:
        return "sparse"
    if count <= MODERATE:
        return "moderate"
    if count >= DENSE:
        return "dense"
    return "moderate"


def _placement_band(n_front: int, n_behind: int, n_mixed: int, total: int) -> str:
    if total == 0:
        return "none"
    if n_front > n_behind and n_front > n_mixed:
        return "foreground"
    if n_behind > n_front and n_behind > n_mixed:
        return "background"
    return "mix"


def compute_object_relations(
    seg2: np.ndarray,
    rgb: np.ndarray,
    *,
    model_asset_dir: str,
    subject_classes: Mapping[int, Any] | None = None,
) -> dict[str, Any]:
    """Compute scale-invariant object/accessory presence + placement facts.

    Detection policy: full-frame Grounding DINO over the frozen closed
    vocabulary at the calibrated thresholds. Boxes are clamped to the frame;
    degenerate boxes are dropped. Subject-confusable standalone detections
    (``body``/``person``/...) are excluded from count and prose — they are the
    subject, not objects. Scale-invariant bands only in prose; normalized
    boxes / scores stay in the machine-readable payload.

    Args:
        seg2: (H, W) integer DOME-29 class labels aligned with rgb.
        rgb: (H, W, 3) uint8 decoded source pixels aligned with seg2.
        model_asset_dir: absolute path to the frozen grounding-dino directory.
        subject_classes: optional iterable/mapping of subject (non-Background)
            class indices (default: all non-Background DOME-29).

    Returns a dict with ``abstained``, count/placement bands, canonical class
    list, and the machine-readable detection payload.
    """
    validate_seg2_array(seg2)
    validate_rgb_array(rgb)
    if seg2.shape[0] != rgb.shape[0] or seg2.shape[1] != rgb.shape[1]:
        raise ObjectRelationsError(
            f"seg2 {seg2.shape} must be pixel-aligned with rgb {rgb.shape}"
        )

    if subject_classes is None:
        subject_mask = seg2 != 0  # all non-Background DOME-29
    else:
        subject_mask = np.isin(seg2, list(subject_classes))
    h, w = rgb.shape[:2]

    processor, model = _GroundingDinoRuntime.get(model_asset_dir)
    vocab_text = " . ".join(CLOSED_VOCAB)

    try:
        import torch

        with torch.no_grad():
            inputs = processor(images=rgb, text=vocab_text, return_tensors="pt")
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[(h, w)],
        )[0]
    except Exception as exc:  # noqa: BLE001
        raise ObjectRelationsError(f"grounding-dino invocation failed: {exc!r}") from exc

    detections: list[dict[str, Any]] = []
    n_front = n_behind = n_mixed = 0
    for i in range(len(results["scores"])):
        score = float(results["scores"][i])
        raw_label = str(results["text_labels"][i]) if i < len(results["text_labels"]) else ""
        cls = canonical_class(raw_label)
        if _is_subject_self(cls):
            # The detector fired on the subject herself — never an object.
            continue
        x1, y1, x2, y2 = [float(v) for v in results["boxes"][i].tolist()]
        # Clamp to frame; drop degenerate (out-of-frame / zero-area) boxes.
        x1i = int(max(0.0, min(x1, w)))
        y1i = int(max(0.0, min(y1, h)))
        x2i = int(max(0.0, min(x2, w)))
        y2i = int(max(0.0, min(y2, h)))
        if x2i <= x1i or y2i <= y1i:
            continue
        area = (x2i - x1i) * (y2i - y1i)
        overlap = int(subject_mask[y1i:y2i, x1i:x2i].sum())
        ov_frac = overlap / area if area > 0 else 0.0
        if ov_frac > FRONT_OVERLAP:
            placement = "in-front"
            n_front += 1
        elif ov_frac < BEHIND_OVERLAP:
            placement = "behind"
            n_behind += 1
        else:
            placement = "mixed"
            n_mixed += 1
        detections.append({
            "raw_label": raw_label,
            "class": cls,
            "score": round(score, 3),
            "box_normalized": [
                round(x1 / w, 4), round(y1 / h, 4),
                round(x2 / w, 4), round(y2 / h, 4),
            ],
            "subject_overlap": round(ov_frac, 3),
            "placement": placement,
        })

    count = len(detections)
    classes: list[str] = [d["class"] for d in detections if d["class"]]
    # Deduplicate class names for the prose list (counts stay per-detection).
    classes_seen: list[str] = []
    for c in classes:
        if c not in classes_seen:
            classes_seen.append(c)

    from collections import Counter
    class_counts = Counter(classes)

    return {
        "abstained": False,
        "detection": "DETECTED",
        "count": count,
        "count_band": _count_band(count),
        "placement_band": _placement_band(n_front, n_behind, n_mixed, count),
        "n_front": n_front,
        "n_behind": n_behind,
        "n_mixed": n_mixed,
        "classes": classes_seen,
        "class_counts": dict(class_counts),
        "box_threshold": BOX_THRESHOLD,
        "text_threshold": TEXT_THRESHOLD,
        "detections": detections,
    }


def render_object_relations(objrel: Mapping[str, Any]) -> list[str]:
    """Scale-invariant object-presence / placement claims (arm #61)."""
    if objrel.get("abstained"):
        reason = objrel.get("abstention_reason") or "object detection not measurable"
        return [f"object-relations: abstain ({reason})"]
    if not objrel or not objrel.get("count_band"):
        # Dimension not measured for this item — never fabricate a claim.
        return []
    count = objrel.get("count", 0)
    band = objrel.get("count_band", "none")
    lines: list[str] = []
    if band == "none":
        lines.append("object-relations: no scene objects detected above the calibrated threshold")
        return lines
    classes = objrel.get("classes") or []
    cls_txt = ", ".join(classes[:5])
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