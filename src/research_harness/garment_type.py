"""Deterministic garment-type / silhouette-category measurement from `seg2`.

Arm #97. NEW deterministic evidence part (no new model). Reads the existing
`seg2.npy` (DOME-29 semantic labels, uint8, at full source resolution) and emits
a coarse SCALE-INVARIANT garment-type / silhouette band:

- upper garment present / absent: whether the subject's upper body is covered by
  an upper-body garment (Apparel or Upper_Clothing beyond a raw-pixel + share
  floor) vs exposed torso/limb skin.
- lower garment present / absent: whether the lower body is covered by
  Lower_Clothing (or a full-length Apparel reaching the lower region).
- garment_type_band: `upper-lower-covered` / `upper-only` / `lower-only` /
  `skin-dominant`, or an honest abstention.

This is a genuinely-NEW claim axis beyond clothing #29's class coverage +
dominant colors: the CATEGORY split ('she wears a top', 'in leggings', 'dressed')
that #29 validated but did not verbalize as a garment-type/silhouette category.
The band closes the 'wearing a top / dressed / leggings' garment-category claim
space with deterministic support.

ONLY scale-invariant facts are verbalized (the coarse band). Raw class-coverage
ratios and pixel counts stay in the machine-readable `evidence_payload` and are
never caption claims (measurement-semantics directive).

Abstention: abstains when clothing/apparel classes are absent or degenerate
(e.g. a fully unclothed subject with no garment classes), never fabricating a
garment type. Detector disagreement is a quality anomaly, never caption content.
CPU-only, in-memory, no corpus write, no new model.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from stratum2.config import DOME_29

# DOME-29 class indices (authoritative in stratum2.config.DOME_29). Cross-checked
# in tests so this module can never silently drift from the real seg2 layout.
APPAREL = DOME_29.index("Apparel")            # dresses / one-pieces / loose garments
UPPER_CLOTHING = DOME_29.index("Upper_Clothing")
LOWER_CLOTHING = DOME_29.index("Lower_Clothing")
TORSO = DOME_29.index("Torso")                # exposed torso skin
FACE_NECK = DOME_29.index("Face_Neck")

# Skin classes measured for exposed-skin / skin-dominance (never a semantic claim).
SKIN_LIMBS = (
    DOME_29.index("Left_Upper_Arm"),
    DOME_29.index("Right_Upper_Arm"),
    DOME_29.index("Left_Lower_Arm"),
    DOME_29.index("Right_Lower_Arm"),
    DOME_29.index("Left_Upper_Leg"),
    DOME_29.index("Right_Upper_Leg"),
    DOME_29.index("Left_Lower_Leg"),
    DOME_29.index("Right_Lower_Leg"),
    DOME_29.index("Left_Hand"),
    DOME_29.index("Right_Hand"),
    DOME_29.index("Left_Foot"),
    DOME_29.index("Right_Foot"),
)

# Presence floor: a garment class must clear a raw pixel count AND a
# foreground-share before it is treated as present (mirror clothing.py).
MIN_CLASS_PX = 200
MIN_COVERAGE = 0.01

# Upper-region skin floor: above this exposed-torso+arm share of the upper
# region, the subject reads skin-dominant (upper bare) when no garment cleared.
UPPER_SKIN_FLOOR = 0.02

UPPER_GARMENT_CLASSES = (APPAREL, UPPER_CLOTHING)
LOWER_GARMENT_CLASSES = (LOWER_CLOTHING, APPAREL)


class GarmentTypeError(RuntimeError):
    pass


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise GarmentTypeError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise GarmentTypeError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise GarmentTypeError(f"seg2 must be integer class labels, got dtype {seg2.dtype}")


def compute_garment_type(
    seg2: np.ndarray,
    *,
    min_px: int = MIN_CLASS_PX,
    min_coverage: float = MIN_COVERAGE,
    upper_skin_floor: float = UPPER_SKIN_FLOOR,
) -> dict[str, Any]:
    """Compute the deterministic garment-type band with honest abstention.

    Args:
        seg2: (H, W) DOME-29 class labels at full source resolution.
        min_px / min_coverage: presence floors for a class/region to be measured.
        upper_skin_floor: upper-region exposed-skin share above which a bare
            upper body reads skin-dominant.

    Returns a dict with scale-invariant facts only:
    - subject_present / abstained / abstention_reason
    - upper_garment_present / lower_garment_present (bool)
    - skin_dominant (bool)
    - garment_type_band (upper-lower-covered / upper-only / lower-only /
      skin-dominant) or None
    - raw coverage ratios for the machine-readable payload.
    """
    validate_seg2_array(seg2)

    out: dict[str, Any] = {
        "subject_present": True,
        "abstained": False,
        "abstention_reason": None,
        "upper_garment_present": False,
        "lower_garment_present": False,
        "skin_dominant": False,
        "garment_type_band": None,
        "upper_garment_coverage": None,
        "lower_garment_coverage": None,
        "apparel_share": None,
        "torso_skin_coverage": None,
        "upper_skin_coverage": None,
        "lower_skin_coverage": None,
    }

    fg_pixels = int((seg2 > 0).sum())
    if fg_pixels <= 0:
        out.update({
            "subject_present": False,
            "abstained": True,
            "abstention_reason": "no foreground subject present",
        })
        return out
    denom = max(fg_pixels, 1)

    up_g = int(np.isin(seg2, UPPER_GARMENT_CLASSES).sum())
    up_share = up_g / denom
    out["upper_garment_coverage"] = round(up_share, 4)
    out["apparel_share"] = round(int((seg2 == APPAREL).sum()) / denom, 4)

    # Exposed-skin measures (payload + skin-dominance classification).
    torso_share = int((seg2 == TORSO).sum()) / denom
    upper_limbs = SKIN_LIMBS[:8]
    upper_skin_share = int(np.isin(seg2, [TORSO, *upper_limbs]).sum()) / denom
    lower_limbs = SKIN_LIMBS[4:]
    lower_skin_share = int(np.isin(seg2, lower_limbs).sum()) / denom
    out["torso_skin_coverage"] = round(torso_share, 4)
    out["upper_skin_coverage"] = round(upper_skin_share, 4)
    out["lower_skin_coverage"] = round(lower_skin_share, 4)

    # ---- Lower-region garment coverage ----
    # Lower_Clothing share is referenced against the lower region (lower limbs +
    # Lower_Clothing + lower-half Apparel). Apparel is ambiguous (dress vs top):
    # its lower-half pixels (below the subject centroid) count toward the lower
    # garment, approximating a dress reaching the lower body.
    subject_rows = np.nonzero(seg2 > 0)[0]
    apparel_mask = seg2 == APPAREL
    arows = np.nonzero(apparel_mask)[0]
    apparel_lower_half = 0
    if arows.size and subject_rows.size:
        subject_cy = float(subject_rows.mean())
        apparel_lower_half = int((arows > subject_cy).sum())
    lower_region_mask = np.isin(seg2, [*lower_limbs, LOWER_CLOTHING])
    lower_region_px = int((lower_region_mask | (seg2 == APPAREL)).sum())
    lower_denom = max(lower_region_px, 1)
    lower_g_px = int((seg2 == LOWER_CLOTHING).sum()) + (
        apparel_lower_half if int(apparel_mask.sum()) >= min_px else 0
    )
    lo_share = lower_g_px / lower_denom
    out["lower_garment_coverage"] = round(lo_share, 4)

    # ---- Presence classification (calibrated from frozen-cohort probe) ----
    upper_present = int(np.isin(seg2, (APPAREL, UPPER_CLOTHING)).sum()) >= min_px and up_share >= min_coverage
    lower_present = int((seg2 == LOWER_CLOTHING).sum()) >= min_px and lo_share >= min_coverage

    if upper_present and lower_present:
        out["upper_garment_present"] = True
        out["lower_garment_present"] = True
        out["garment_type_band"] = "upper-lower-covered"
    elif upper_present:
        out["upper_garment_present"] = True
        out["garment_type_band"] = "upper-only"
    elif lower_present:
        out["lower_garment_present"] = True
        out["garment_type_band"] = "lower-only"
    elif upper_skin_share >= upper_skin_floor or torso_share >= upper_skin_floor:
        # No garment region cleared; substantial exposed skin -> skin-dominant.
        out["skin_dominant"] = True
        out["garment_type_band"] = "skin-dominant"
    else:
        out.update({
            "abstained": True,
            "abstention_reason": "garment regions absent or degenerate (cannot classify garment type)",
        })
    return out


def render_garment_type(config: Mapping[str, Any]) -> list[str]:
    """Scale-invariant garment-type claims for the dossier (arm #97).

    Verbalizes ONLY the coarse garment-type band. Raw coverage ratios and pixel
    counts stay in the machine-readable payload.
    """
    if not config:
        # Dimension not measured for this item (e.g. non-garment-type runs) —
        # emit no claim, never a fabricated garment type.
        return []
    if config.get("abstained"):
        reason = config.get("abstention_reason") or "garment type not measurable"
        return [f"garment-type: abstain ({reason})"]
    if not config.get("subject_present"):
        return ["garment-type: abstain (no foreground subject)"]
    band = config.get("garment_type_band")
    if band == "upper-lower-covered":
        return ["garment-type: subject is dressed (upper and lower body covered)"]
    if band == "upper-only":
        return ["garment-type: upper body clothed, lower body exposed (e.g. wearing a top, legs uncovered)"]
    if band == "lower-only":
        return ["garment-type: lower body covered, upper body exposed"]
    if band == "skin-dominant":
        return ["garment-type: skin-dominant (no garment region cleared; exposed skin dominates)"]
    return []
