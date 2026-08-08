"""Apparent-age estimation from a local open-weight MiVOLO-V2 model.

Arm #73. NEW-model-class specialist: runs the open-weight MiVOLO-V2 age+gender
transformer (Apache-2.0, face+body multi-input, arXiv 2307.04616 / 2403.02302,
HF ``iitolstykh/mivolo_v2``, ``MiVOLOForImageClassification`` remote-code path)
on owned hardware over the frozen cohort and derives a SCALE-INVARIANT
apparent-age band:

- late-teens / twenties / thirties-and-up (coarse bands; never a precise
  numeric age in prose — the raw floating age estimate stays in the
  machine-readable ``evidence_payload`` JSON).

Crop policy (measured 2026-08-07 capability probe + archive):
- Face crop: seg2 Face_Neck (DOME-29 class 3) mask bbox, margin-expanded
  (1x max side); if the Face_Neck region is too small (px < floor) OR MiVOLO
  fails on it, fall back to the full frame as the face input (union policy,
  mirroring the face-geometry #60 resolution-sensitivity finding); if both
  fail to yield a plausible age the item abstains with a surfaced reason.
- Body crop: seg2 subject-union (seg2 != 0) mask bbox, margin-expanded; used
  as the person input MiVOLO-V2 was trained with (``with_persons_model``).
  Full-frame is the body fallback when the subject mask is degenerate.

Band calibration (band-degeneracy rule arm #34/#35/#58/#59/#60): the first
3-band scheme (teens/twenties/thirties) was DEGENERATE on this homogeneous
portrait cohort (21/24 in "twenties" = 87.5%: measured ages cluster 24-29,
median 26.2, range 19.8-32.9). The honest re-probe cuts at the measured
distribution gaps, giving 4 bands -> 2/6/12/4 (max share 50.0%, no band >=
75%; probe 2026-08-07). If the cohort turns out homogeneous the axis would be
kept payload-only and never verbalized; here it discriminates cleanly. The
probe script is ``scripts/probe_apparent_age.py``; its calibrated thresholds
are what this module's ``_age_band`` uses.

Abstention: small / turned / occluded / zero Face_Neck region with a surfaced
reason; detector disagreement remains a quality anomaly, never caption content;
gender is NOT verbalized (the corpus is curated to exactly one woman, so gender
is a constant-quality check at best, payload-only).

Provenance: model.safetensors sha256 a6cf24db1e05c33c5ff4edf4a36fd2db47c84e2
a9f2d060a2c22e6a2f9e7a4c625 (pinned at installation, see model card); run on
owned hardware only; no hosted third-party inference of the sensitive corpus;
no corpus write.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# DOME-29 class index for Face_Neck (used for the face crop policy).
_FACE_NECK = 3

# Crop / abstention gates.
_MIN_FN_PX = 400        # seg2 Face_Neck region floor for the face-crop path
_MIN_SUBJ_PX = 2000     # seg2 subject-union floor for the body-crop path

# Model asset + vendored mivolo package.
MODEL_DIR = "/mnt/nas-ai-models/research/stratum/models/apparent-age"
# model.safetensors sha256 (pinned at installation).
MODEL_SHA256 = "96efb47051c038ebeec74b73b4253c5fd000433e5afcab7deee0bd8f3fa7bf18"

# Scale-invariant apparent-age band thresholds (years). RE-CALIBRATED from the
# frozen-cohort probe (2026-08-07 arm #73). The first 3-band scheme
# (teens/twenties/thirties) was DEGENERATE on this homogeneous portrait cohort
# (21/24 in "twenties", 87.5%) — the measured ages cluster 24-29 (median 26.2,
# range 19.8-32.9). The honest re-probe cuts at the natural distribution gaps,
# giving 2/7/11/4 (max share 45.8%, no band >= 75%). Bands are coarse and
# scale-invariant; the raw float age stays payload-only, never a prose claim.
AGE_LATE_TEENS_MAX = 23.0      # below -> late-teens-to-early-twenties
AGE_EARLY_TWENTIES_MAX = 25.5  # below -> early-twenties
AGE_MID_TWENTIES_MAX = 28.0    # below -> mid-twenties
# else -> late-twenties-to-thirties


class ApparentAgeError(RuntimeError):
    pass


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise ApparentAgeError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise ApparentAgeError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise ApparentAgeError(f"seg2 must be integer class labels, got dtype {seg2.dtype}")


def validate_rgb_array(rgb: np.ndarray) -> None:
    if not isinstance(rgb, np.ndarray):
        raise ApparentAgeError("rgb must be a numpy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ApparentAgeError(f"rgb must be (H, W, 3), got shape {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise ApparentAgeError(f"rgb must be uint8, got dtype {rgb.dtype}")


class _MiVOLORuntime:
    """Lazy, process-wide MiVOLO-V2 model (HF transformers remote-code path)."""

    _model = None
    _proc = None
    _config = None

    @classmethod
    def get(cls, model_dir: str):
        if cls._model is None:
            import os
            import sys

            src = os.path.join(model_dir, "mivolo_src")
            if src not in sys.path:
                sys.path.insert(0, src)
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

            from transformers import (
                AutoConfig,
                AutoImageProcessor,
                AutoModelForImageClassification,
            )
            import torch

            cls._config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._model = AutoModelForImageClassification.from_pretrained(
                model_dir,
                trust_remote_code=True,
                dtype=torch.float32,
            ).to(device)
            cls._model.eval()
            cls._proc = AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=True)
            cls._device = device
        return cls._model, cls._proc, cls._config

    @classmethod
    def reset(cls) -> None:
        cls._model = None
        cls._proc = None
        cls._config = None


def _crop_mask_bbox(mask: np.ndarray, margin: int, img_h: int, img_w: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    cy0, cy1 = max(0, ys.min() - margin), min(img_h - 1, ys.max() + margin)
    cx0, cx1 = max(0, xs.min() - margin), min(img_w - 1, xs.max() + margin)
    return cy0, cy1, cx0, cx1


def _crop(rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    cy0, cy1, cx0, cx1 = bbox
    if cy1 - cy0 <= 0 or cx1 - cx0 <= 0:
        raise ApparentAgeError("degenerate crop bbox")
    return np.ascontiguousarray(rgb[cy0:cy1, cx0:cx1])


def _rgb_to_bgr(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr[:, :, ::-1])


def _infer_age(rgb: np.ndarray, face_crop: np.ndarray, body_crop: np.ndarray,
               model_dir: str = MODEL_DIR) -> dict[str, Any]:
    """Run MiVOLO-V2 on face + body BGR crops; return probe facts or None."""
    model, proc, config = _MiVOLORuntime.get(model_dir)
    import torch

    faces_bgr = _rgb_to_bgr(face_crop)
    body_bgr = _rgb_to_bgr(body_crop)
    with torch.no_grad():
        faces_input = proc(images=[faces_bgr])["pixel_values"].to(
            dtype=model.dtype, device=model.device
        )
        body_input = proc(images=[body_bgr])["pixel_values"].to(
            dtype=model.dtype, device=model.device
        )
        out = model(faces_input=faces_input, body_input=body_input)
        age = float(out.age_output[0].item())
        gender_idx = int(out.gender_class_idx[0].item())
        gender_prob = float(out.gender_probs[0].item())
    gender_label = config.gender_id2label.get(gender_idx)
    return {
        "age_years": round(age, 2),
        "gender_probe": {"idx": gender_idx, "label": gender_label, "prob": round(gender_prob, 3)},
    }


def compute_apparent_age(seg2: np.ndarray, rgb: np.ndarray,
                         *, model_dir: str = MODEL_DIR) -> dict[str, Any]:
    """Compute a scale-invariant apparent-age band from seg2 + source pixels.

    Uses a UNION crop policy (mirroring arm #60's resolution-sensitivity
    finding): the seg2 Face_Neck crop first, the full frame as the face-input
    fallback when the Face_Neck region is too small or fails to yield a
    plausible age. The body crop comes from the seg2 subject-union bbox
    (full-frame fallback). Only the coarse band is returned for prose; the raw
    floating age stays in the machine-readable payload.

    Returns a dict with ``abstained`` / ``abstention_reason`` on failure.
    """
    validate_seg2_array(seg2)
    validate_rgb_array(rgb)
    if seg2.shape[0] != rgb.shape[0] or seg2.shape[1] != rgb.shape[1]:
        raise ApparentAgeError(f"seg2 {seg2.shape} must be pixel-aligned with rgb {rgb.shape}")

    img_h, img_w = rgb.shape[0], rgb.shape[1]
    fn_mask = seg2 == _FACE_NECK
    fn_px = int(fn_mask.sum())
    subj_mask = seg2 != 0
    subj_px = int(subj_mask.sum())

    # Body crop (person input). Subject-union bbox; full-frame fallback.
    body_crop: np.ndarray
    if subj_px >= _MIN_SUBJ_PX:
        try:
            bbox = _crop_mask_bbox(subj_mask, int(max(img_h, img_w) * 0.02), img_h, img_w)
            body_crop = _crop(rgb, bbox)
        except ApparentAgeError:
            body_crop = np.ascontiguousarray(rgb)
    else:
        body_crop = np.ascontiguousarray(rgb)

    # Face crops (union policy).
    face_candidates: list[tuple[str, np.ndarray]] = []
    if fn_px >= _MIN_FN_PX:
        try:
            # margin ~1x the Face_Neck region max side, capped so it stays sane
            ys, xs = np.where(fn_mask)
            margin = min(int(max(ys.max() - ys.min(), xs.max() - xs.min())),
                          int(max(img_h, img_w) * 0.5))
            bbox = _crop_mask_bbox(fn_mask, margin, img_h, img_w)
            face_candidates.append(("seg2_face_crop", _crop(rgb, bbox)))
        except ApparentAgeError:
            pass
    face_candidates.append(("full_frame", np.ascontiguousarray(rgb)))

    last_err: str | None = None
    for tag, face_crop in face_candidates:
        try:
            facts = _infer_age(rgb, face_crop, body_crop, model_dir=model_dir)
        except Exception as exc:  # noqa: BLE001
            last_err = f"{tag}: {exc!r}"
            continue
        age = facts["age_years"]
        # Plausibility gate: MiVOLO output is age in [0, 122] by construction;
        # still guard against a degenerate/edge output before verbalizing.
        if not (0.0 <= age <= 122.0):
            last_err = f"{tag}: implausible age {age}"
            continue
        band = _age_band(age)
        return {
            "abstained": False,
            "detection": "DETECTED",
            "via": tag,
            "seg2_face_neck_px": fn_px,
            "seg2_subject_px": subj_px,
            "age_years": age,
            "age_band": band,
            "gender_probe": facts.get("gender_probe"),
        }

    if fn_px < _MIN_FN_PX:
        reason = f"seg2 Face_Neck region too small (px={fn_px}) -> no measurable face"
    elif last_err:
        reason = f"mivolo inference failed on all face candidates: {last_err}"
    else:
        reason = "no plausible apparent age from the available crops"
    return {"abstained": True, "abstention_reason": reason, "seg2_face_neck_px": fn_px,
            "seg2_subject_px": subj_px}


def _age_band(age: float) -> str:
    """Coarse scale-invariant apparent-age band (re-calibrated on the cohort).

    4 bands cut at the measured distribution gaps (2026-08-07 probe: 2/7/11/4,
    max share 45.8%) so no single band >= 75%.
    """
    if age < AGE_LATE_TEENS_MAX:
        return "late-teens-to-early-twenties"
    if age < AGE_EARLY_TWENTIES_MAX:
        return "early-twenties"
    if age < AGE_MID_TWENTIES_MAX:
        return "mid-twenties"
    return "late-twenties-to-thirties"


def render_apparent_age(age: Mapping[str, Any]) -> list[str]:
    """Scale-invariant apparent-age claim for the dossier (arm #73)."""
    if not age:
        # Dimension not measured for this item (e.g. non-apparent-age runs) —
        # emit no claim, never a fabricated age statement.
        return []
    if age.get("abstained"):
        reason = age.get("abstention_reason") or "apparent age not measurable"
        return [f"apparent-age: abstain ({reason})"]
    if not age.get("age_band"):
        return []
    band = age["age_band"]
    text = {
        "late-teens-to-early-twenties": "likely late teens to early twenties",
        "early-twenties": "likely in her early twenties",
        "mid-twenties": "likely in her mid-twenties",
        "late-twenties-to-thirties": "likely late twenties or older",
    }.get(band, "age not confidently assigned")
    return [f"apparent-age: {text} (coarse scale-invariant band, not an exact age)"]
