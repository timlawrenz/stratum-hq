"""Deterministic matting / alpha-fidelity measurements from `matting.npy` + `seg2`.

Arm #59. Reads the existing `matting.npy` (Sapiens2 per-pixel soft alpha /
matte, source-matched ``(H, W)`` float16 in ``[0, 1]``, 1.0 == fully opaque
subject, graduated soft edges carry hair / fine-feature alpha) and `seg2.npy`
(DOME-29 class labels, class 4 == Hair) and emits scale-invariant alpha-fidelity
facts:

- subject alpha-integral coverage ratio (fraction of the frame the opaque
  subject occupies), banded sparse / centered / fills-frame;
- boundary crispness: median alpha-gradient magnitude over the 1-px silhouette
  ring (normalized by the alpha range [0,1], so units are alpha-change per px
  — a local derivative, not an absolute pixel measure) — a scale-invariant
  edge-sharpness descriptor, banded soft / moderate / crisp;
- soft-edge character: the share of the semi-transparent band that lies inside
  the seg2 Hair class (hair flyaway) vs skin/background anti-aliasing, banded
  skin-clean / mixed / hair-dominant — this is the "soft detachable hair
  strands" axis;
- silhouette structure (machine-readable payload only): single-connected-
  component share and frame-border-open fraction. On the frozen cohort this
  measurements is near-constant (24/24 closed, no crops), so it carries NO
  caption band — documented honestly as a non-discriminator on this cohort.

Only scale-invariant facts are verbalized: ratios, normalized gradient/derivative
descriptors, and names. Absolute pixel values (coverage px, band px, region
areas) stay in the machine-readable ``evidence_payload`` (dossier / compressor
input) and are never caption claims — a pixel width is camera-frame-dependent
and not something a text-to-image model should be asked to render.

Every measurement honors the exactly-one-subject invariant and abstains (emits
None / abstention reason) when the matte is absent or ill-formed or degenerate
(all-one/all-zero), coverage is below a floor (no opaque subject present), or
the pose2 detector count is not exactly one (the caller enforces the latter).

Calibration (2026-08-07, band-calibration rule arm #34/#35/#58): on-paper
thresholds for `soft_edge_band` (0.015/0.030) and `detail_band` (0.05/0.15)
were DEGENERATE on this cohort (21/24 "sharp", 23/24 "fine-detail",
silhouette 24/24 "closed"). Recalibrated from the measured 24-item probe:
boundary crispness p50 spread 11 crisp / 9 moderate / 4 soft (max share 46%);
hair-soft share 6 skin-clean / 14 mixed / 4 hair-dominant (max 58%); coverage
5 sparse / 2 fills / 17 centered (max 71%). Silhouette closedness is honestly
non-discriminating (24/24) and kept payload-only.

Provenance: deterministic CPU measurement from existing core artifacts; no model
invocation, no corpus write.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from stratum2.config import DOME_29

_DOME_INDEX = {name: i for i, name in enumerate(DOME_29)}

HAIR = _DOME_INDEX["Hair"]

# Alpha thresholds (constant, not calibrated).
_OPAQUE = 0.9   # "fully opaque core"
_SEMI = 0.05    # below this is transparent outside
_SUBJECT = 0.5  # subject mask threshold

# Measurement gates / floors.
MIN_SUBJECT_PX = 500      # alpha>=0.5 must clear this many px to measure
MIN_RING_PX = 50          # silhouette ring support floor for crispness
MIN_SEMI_PX = 50          # soft-band support floor for edge-character

# Scale-invariant band thresholds — CALIBRATED from the frozen-cohort probe
# (2026-08-07, arm #59), see band-calibration note in the docstring.
COVERAGE_SPARSE = 0.20    # below this: small figure in frame
COVERAGE_FILL = 0.55      # above this: subject fills most of the frame
BOUNDARY_CRISP = 0.24     # median ring gradient >= this: crisp silhouette edge
BOUNDARY_SOFT = 0.16      # below this: feathered / very soft edge
HAIR_DOMINANT = 0.50      # soft-band hair share >= this: hair flyaway dominates
SKIN_CLEAN = 0.20         # soft-band hair share < this: clean skin cutout


class MattingAlphaError(RuntimeError):
    pass


def validate_matting_array(alpha: np.ndarray) -> None:
    if not isinstance(alpha, np.ndarray):
        raise MattingAlphaError("matting must be a numpy array")
    if alpha.ndim != 2:
        raise MattingAlphaError(f"matting must be (H, W), got shape {alpha.shape}")
    if alpha.dtype != np.float16 and alpha.dtype != np.float32:
        raise MattingAlphaError(f"matting must be float16/float32, got dtype {alpha.dtype}")


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise MattingAlphaError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise MattingAlphaError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise MattingAlphaError(f"seg2 must be integer class labels, got dtype {seg2.dtype}")


def _subject_mask(alpha: np.ndarray) -> np.ndarray:
    return alpha >= _SUBJECT


def _opaque_mask(alpha: np.ndarray) -> np.ndarray:
    return alpha >= _OPAQUE


def _subject_height_px(mask: np.ndarray) -> int:
    rows = np.flatnonzero(np.any(mask, axis=1))
    if rows.size == 0:
        return 0
    return int(rows[-1] - rows[0] + 1)


def _boundary_crispness(alpha: np.ndarray, subject: np.ndarray) -> float | None:
    """Median |grad alpha| over the 1-px silhouette ring (alpha-change per px).

    A sharp cutout localizes the 0->1 transition to a 1-px ring (high median
    gradient); a feathered / soft edge spreads it over several px (low median
    gradient). Scale-invariant descriptor normalized by the alpha range [0,1].
    """
    interior = ndimage.binary_erosion(subject, structure=np.ones((3, 3)))
    ring = subject & ~interior
    if int(ring.sum()) < MIN_RING_PX:
        return None
    ay, ax = np.gradient(alpha.astype(np.float64))
    mag = np.sqrt(ax * ax + ay * ay)
    return float(np.percentile(mag[ring], 50))


def _hair_soft_share(alpha: np.ndarray, seg2: np.ndarray) -> float | None:
    """Share of the semi-transparent band that lies inside the Hair class.

    High => the soft transition is hair flyaway / wispy hairline (the
    "soft detachable hair strands" signal); low => the soft band is mostly
    skin/background anti-aliasing (a clean cutout).
    """
    semi = (alpha >= _SEMI) & (alpha < _OPAQUE)
    if int(semi.sum()) < MIN_SEMI_PX:
        return None
    hair = seg2 == HAIR
    hs = int((semi & hair).sum())
    ss = int((semi & ~hair).sum())
    if (hs + ss) == 0:
        return None
    return float(hs) / float(hs + ss)


def _silhouette_structure(subject: np.ndarray) -> dict[str, Any]:
    labels, n = ndimage.label(subject)
    if n == 0:
        return {"closed": False, "largest_component_share": 0.0, "border_open_fraction": 1.0}
    sizes = ndimage.sum(np.ones_like(subject), labels, index=np.arange(1, n + 1))
    largest_share = float(float(np.max(sizes)) / float(subject.sum()))
    border = subject[0, :].sum() + subject[-1, :].sum() + subject[:, 0].sum() + subject[:, -1].sum()
    border_open = 0.0
    if subject.sum() > 0:
        border_open = float(border) / float(subject.sum())
    return {
        "closed": largest_share >= 0.97 and border_open < 0.12,
        "largest_component_share": round(largest_share, 4),
        "border_open_fraction": round(border_open, 4),
    }


def compute_matting_alpha(
    alpha: np.ndarray,
    seg2: np.ndarray,
    *,
    min_subject_px: int = MIN_SUBJECT_PX,
) -> dict[str, Any]:
    """Compute deterministic alpha-fidelity measurements with explicit abstention.

    Args:
        alpha: (H, W) float matte aligned with seg2, in [0, 1].
        seg2: (H, W) integer DOME-29 class labels aligned with alpha.

    Returns a dict with scale-invariant alpha-fidelity facts only. Raw pixel
    values (coverage px, band px, areas) live in the machine-readable payload,
    never as caption claims.
    """
    validate_matting_array(alpha)
    validate_seg2_array(seg2)
    if alpha.shape[0] != seg2.shape[0] or alpha.shape[1] != seg2.shape[1]:
        raise MattingAlphaError(
            f"matting {alpha.shape} must be pixel-aligned with seg2 {seg2.shape}"
        )

    a = alpha.astype(np.float64)
    if float(np.nanmin(a)) < 0.0 or float(np.nanmax(a)) > 1.0:
        return _abstain(
            "matting values outside the [0, 1] alpha band -> degenerate matte",
            alpha_min=float(np.nanmin(a)),
            alpha_max=float(np.nanmax(a)),
        )
    subject = _subject_mask(a)
    subject_px = int(subject.sum())
    if subject_px < min_subject_px:
        return _abstain(
            "alpha subject mask too small for stable alpha-fidelity measurement",
            subject_px=subject_px,
            subject_present=subject_px > 0,
        )

    opaque = _opaque_mask(a)
    frame_px = int(a.size)
    coverage_ratio = float(opaque.sum()) / float(frame_px)

    crispness = _boundary_crispness(a, subject)
    hair_share = _hair_soft_share(a, seg2)
    silhouette = _silhouette_structure(subject)

    # ---- scale-invariant bands (calibrated from the frozen-cohort probe) ----
    if coverage_ratio < COVERAGE_SPARSE:
        coverage_band = "sparse"
    elif coverage_ratio < COVERAGE_FILL:
        coverage_band = "centered"
    else:
        coverage_band = "fills-frame"

    if crispness is None:
        crisp_band = None
    elif crispness >= BOUNDARY_CRISP:
        crisp_band = "crisp"
    elif crispness >= BOUNDARY_SOFT:
        crisp_band = "moderate"
    else:
        crisp_band = "soft"

    if hair_share is None:
        edge_band = None
    elif hair_share >= HAIR_DOMINANT:
        edge_band = "hair-dominant"
    elif hair_share >= SKIN_CLEAN:
        edge_band = "mixed"
    else:
        edge_band = "skin-clean"

    return {
        "subject_present": True,
        "abstained": False,
        "abstention_reason": None,
        "matting_measurable": True,
        "subject_px": subject_px,            # payload only
        "frame_px": frame_px,                # payload only
        "coverage_ratio": round(coverage_ratio, 4),  # scale-invariant ratio
        "coverage_band": coverage_band,
        "subject_height_px": _subject_height_px(subject),  # payload only
        "boundary_crispness": None if crispness is None else round(crispness, 4),
        "boundary_crisp_band": crisp_band,
        "hair_soft_share": None if hair_share is None else round(hair_share, 4),
        "soft_edge_band": edge_band,
        "silhouette_closed": silhouette["closed"],       # payload-only (non-discriminating)
        "silhouette_closedness": silhouette["largest_component_share"],
        "border_open_fraction": silhouette["border_open_fraction"],
    }


def _abstain(reason: str, **counts: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "subject_present": True,
        "abstained": True,
        "abstention_reason": reason,
        "matting_measurable": False,
        "subject_px": None,
        "frame_px": None,
        "coverage_ratio": None,
        "coverage_band": None,
        "subject_height_px": None,
        "boundary_crispness": None,
        "boundary_crisp_band": None,
        "hair_soft_share": None,
        "soft_edge_band": None,
        "silhouette_closed": None,
        "silhouette_closedness": None,
        "border_open_fraction": None,
    }
    result["subject_present"] = counts.pop("subject_present", True)
    result.update(counts)
    result.setdefault("subject_px", 0)
    return result
