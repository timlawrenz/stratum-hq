"""Scale-invariant image-focus / depth-of-field measurements (arm #75).

Deterministic focus-quality evidence computed in memory from the source RGB
(SHA-bound via each item's source_sha256) and the frozen seg2 DOME-29 class
mask (region split). This is a genuinely-NEW evidence part: no validated arm
measures optical focus / depth-of-field quality (global or region-relative
sharpness / acutance).

Measurement design (calibrated on the frozen 24-item cohort, 2026-08-07):
- Decode source RGB, convert to luminance, and RESAMPLE to a canonical long
  side of 512 px (LANCZOS; seg2 with NEAREST) so acutance values are
  comparable ACROSS pictures regardless of native resolution (the review
  context also downsamples toward ~512, so the reviewer reads the same
  scale). Per-pixel gradient magnitude via central differences.
- Region split: subject = seg2 != 0 (DOME-29 class 0 is Background),
  background = seg2 == 0. Both masks are ERODED (subject 2 px, background
  3 px, 4-neighbourhood) so acutance is measured on the region INTERIOR,
  never the silhouette boundary halo (gradient-specialist pitfall,
  arm #35).
- DOF ratio        = background interior median acutance / subject interior
                     median acutance. Cohort distribution (probe2): min 0.00,
                     p25 0.38, med 0.62, p75 0.82, max 1.68; band cuts 0.45 /
                     0.80 give 9 / 8 / 7 measured — no degeneracy. Verbalized
                     band: blurred / soft / sharp background.
- Subject-vs-frame = subject interior median / full-frame interior median
                     acutance. Cohort distribution: min 0.62, p25 1.13,
                     med 1.31, p75 1.65, max 3.62; cuts 0.9 / 1.6 give
                     3 / 12 / 9 — no degeneracy. Verbalized band: subject
                     softer / comparable / crispest-part.
- Rejected discriminator: subject median vs frame P99 (all 24/24 <= 0.16 —
                     degenerate; the top-1% edge energy lives in background
                     detail, not a usable focus axis).
- Flat-background guard: when the background interior has essentially NO
  texture (bg_p99 < 4.0), the DOF axis abandons — there is nothing to
  resolve as sharp or blurred (0/24 fired on the cohort, but the honest
  abstention stays).

Only the scale-invariant ratio bands are verbalized; raw acutance numbers /
canonical dims / region shares stay in the machine-readable evidence_payload
and are never caption claims (the reviewer consumes the same rendered
evidence, so px is never rewarded or penalized).

Abstention: when the eroded subject region is too small, or the eroded
background region is too small (e.g. a full-bleed portrait with no visible
background), or the background is untextured, the relevant band abstains with
a surfaced reason. Detector disagreement remains a quality anomaly, never
caption content. No new model is required — deterministic CPU in-memory, no
corpus write.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

# DOME-29 class-0 is Background; subject = everything else (the curated one
# woman, exactly-one-subject corpus invariant).
_SUBJECT_CLASSES = frozenset(range(1, 29))

# Canonical length so acutance is comparable across native resolutions.
CANONICAL_SIDE = 512
# Minimum eroded interior pixels for a region to be measurable.
_MIN_REGION_PX = 100
# DOF band cuts (probe2: 9/8/7 at 0.45/0.80, max share 37.5%).
DOF_BLURRED_MAX = 0.45
DOF_SHARP_MIN = 0.80
# Subject-vs-frame band cuts (probe2: 3/12/9 at 0.9/1.6, max share 50%).
SUBJECT_SOFTER_MAX = 0.9
SUBJECT_CRISP_MIN = 1.6
# Flat-background guard: below this background P99 acutance there is no
# texture to resolve; the DOF axis abstains (0/24 on the cohort).
FLAT_BACKGROUND_P99 = 4.0


class ImageFocusError(RuntimeError):
    pass


def validate_image_focus_inputs(rgb: np.ndarray, seg2: np.ndarray) -> None:
    if not isinstance(rgb, np.ndarray) or not isinstance(seg2, np.ndarray):
        raise ImageFocusError("rgb and seg2 must be numpy arrays")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ImageFocusError(f"rgb must be (H, W, 3), got {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise ImageFocusError(f"rgb must be uint8, got {rgb.dtype}")
    if seg2.ndim != 2:
        raise ImageFocusError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise ImageFocusError(f"seg2 must be integer class labels, got dtype {seg2.dtype}")
    if seg2.shape != rgb.shape[:2]:
        raise ImageFocusError(
            f"seg2 {seg2.shape} not pixel-aligned with rgb {rgb.shape[:2]}"
        )


def _resample(rgb: np.ndarray, seg2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resample both to a canonical long side, preserving aspect."""
    height, width = rgb.shape[:2]
    scale = CANONICAL_SIDE / max(width, height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    img = Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_rgb = np.asarray(img, dtype=np.uint8)
    seg_small = np.asarray(
        Image.fromarray(seg2).resize((new_w, new_h), Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    return resized_rgb, seg_small


def _luminance_and_gradient(rgb: np.ndarray) -> np.ndarray:
    lum = (
        0.299 * rgb[:, :, 0].astype(np.float64)
        + 0.587 * rgb[:, :, 1].astype(np.float64)
        + 0.114 * rgb[:, :, 2].astype(np.float64)
    )
    gy, gx = np.gradient(lum)
    return np.sqrt(gx * gx + gy * gy)


def _median_acutance(grad: np.ndarray, mask: np.ndarray) -> float | None:
    interior = grad[mask]
    if interior.size < _MIN_REGION_PX:
        return None
    return float(np.median(interior))


def compute_image_focus(rgb: np.ndarray, seg2: np.ndarray) -> dict[str, Any]:
    """Compute scale-invariant focus / depth-of-field bands.

    Args:
        rgb: (H, W, 3) uint8 decoded source RGB.
        seg2: (H, W) integer DOME-29 class labels aligned with rgb.

    Returns a dict with ``abstained`` (whole-item), per-band abstention
    reasons where only one axis is unmeasurable, and the band values.
    """
    validate_image_focus_inputs(rgb, seg2)
    resized_rgb, seg_small = _resample(rgb, seg2)
    grad = _luminance_and_gradient(resized_rgb)

    subject = seg_small != 0
    background = seg_small == 0
    # Region INTERIOR only (silhouette-halo-free; subject 2 px, bg 3 px).
    subject_i = binary_erosion(subject, iterations=2).astype(bool)
    background_i = binary_erosion(background, iterations=3).astype(bool)

    subj_med = _median_acutance(grad, subject_i)
    if subj_med is None:
        return {
            "abstained": True,
            "abstention_reason": f"eroded subject region too small ({int(subject_i.sum())} px < {_MIN_REGION_PX})",
        }

    bg_med = _median_acutance(grad, background_i)
    bg_p99 = (
        float(np.percentile(grad[background_i], 99))
        if background_i.any()
        else None
    )
    global_med = float(np.median(grad)) if grad.size else None

    out: dict[str, Any] = {
        "abstained": False,
        "detection": "MEASURED",
        "canonical_dims": [resized_rgb.shape[1], resized_rgb.shape[0]],
        "subject_share": round(float(subject.mean()), 4),
        # machine-readable payload only (never prose)
        "subject_acutance_median": subj_med,
        "background_acutance_median": bg_med,
        "background_p99": bg_p99,
        "background_std": (
            float(grad[background_i].std()) if background_i.any() else None
        ),
        "global_acutance_median": global_med,
    }

    # --- Background / depth-of-field band (bg vs subject interior median) ---
    dof_band = None
    dof_reason = None
    if bg_med is None:
        dof_reason = (
            f"eroded background region too small ({int(background_i.sum())} px "
            f"< {_MIN_REGION_PX}) — no visible background to compare"
        )
    elif bg_p99 is not None and bg_p99 < FLAT_BACKGROUND_P99:
        dof_reason = "background is untextured — nothing to resolve as sharp or blurred"
    elif subj_med <= 0:
        dof_reason = (
            "subject region has no measurable interior texture — focus ratio undefined"
        )
    else:
        ratio = bg_med / subj_med
        out["dof_ratio"] = round(ratio, 4)
        if ratio <= DOF_BLURRED_MAX:
            dof_band = "background-blurred"
        elif ratio >= DOF_SHARP_MIN:
            dof_band = "background-sharp"
        else:
            dof_band = "background-soft"
    out["dof_band"] = dof_band
    if dof_reason:
        out["dof_abstained"] = True
        out["dof_abstention_reason"] = dof_reason

    # --- Subject focus band (subject vs full-frame interior median) ---
    subj_band = None
    if global_med is not None and global_med > 0:
        ratio = subj_med / global_med
        out["subject_vs_frame_ratio"] = round(ratio, 4)
        if ratio <= SUBJECT_SOFTER_MAX:
            subj_band = "subject-softer"
        elif ratio >= SUBJECT_CRISP_MIN:
            subj_band = "subject-crisp"
        else:
            subj_band = "subject-comparable"
    out["subject_focus_band"] = subj_band
    return out


def render_image_focus(focus: Mapping[str, Any] | None) -> list[str]:
    """Scale-invariant focus / DOF claims for the dossier (arm #75)."""
    if not focus:
        # Dimension not measured for this item (e.g. non-image-focus runs) —
        # emit no claim, never a fabricated focus statement.
        return []
    if focus.get("abstained"):
        reason = focus.get("abstention_reason") or "focus not measurable"
        return [f"image-focus: abstain ({reason})"]
    if not focus.get("subject_focus_band") and not focus.get("dof_band"):
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
        lines.append(
            "image-focus: background is somewhat softer than the subject"
        )
    elif focus.get("dof_abstained"):
        lines.append(
            f"image-focus: depth-of-field not assessed ({focus.get('dof_abstention_reason')})"
        )

    if not lines:
        lines.append("image-focus: focus measured (no distinctive band)")
    return lines