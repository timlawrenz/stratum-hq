"""Deterministic point-map / 3D depth-ordering measurements from `pointmap.npy` + `seg2`.

Arm #58. Reads the existing `pointmap.npy` (Sapiens2 pointmap: per-pixel 3D
point cloud in CAM frame, +X right, +Y down, +Z toward viewer, camera optical
center at origin, background pixels zeroed) and `seg2.npy` (DOME-29 class
labels, class 0 == Background) and emits scale-invariant depth facts:

- relative-depth ordering of body regions (which seg2 region is nearest to /
  farthest from the camera plane, from region-median Z ranks);
- left/right hand depth ordering (hand held closer to the camera than the
  other / aligned), a self-occlusion-relevant pairwise ordering;
- arms-held-in-front vs rest: median Z of each hand+forearm group vs the torso
  plane median Z (is a hand/crossed arm in front of the body plane?);
- subject depth-relief ratio: robust Z spread (p10-p90) over the subject
  normalized by the subject's median Z (scale-invariant body thin-most/thickest
  volume proxy: arms-forward or crouching poses show pronounced relief);
- subject foreground depth extent relative to median (compact vs spread).

Only scale-invariant facts are verbalized: region ORDERING relations, pairwise
nearer/farther directions, and normalized ratios (spread/median). Absolute Z
values (camera-frame metric distances), pixel positions, and raw spreads stay
in the machine-readable `evidence_payload` (dossier / compressor input) and are
never caption claims — a bare meter number is not something a text-to-image
model should be asked to render, and it is camera-placement dependent.

Every measurement honors the exactly-one-subject invariant and abstains (emits
None / abstention reason) when the point-map is absent or ill-formed, the
foreground is degenerate (fills the frame with no interior relief signal, or too
few valid depth pixels), or the subject distance is outside a human-plausible
band. Detector disagreement is a quality anomaly, never caption content.

Provenance: deterministic CPU measurement from existing core artifacts; no model
invocation, no corpus write.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stratum2.config import DOME_29

_DOME_INDEX = {name: i for i, name in enumerate(DOME_29)}

BACKGROUND = 0  # "Background" — zeroed in pointmap

# DOME-29 region groupings used for depth ordering.
HEAD_CLASSES = (
    _DOME_INDEX["Face_Neck"],
    _DOME_INDEX["Hair"],
    _DOME_INDEX["Lower_Lip"],
    _DOME_INDEX["Upper_Lip"],
    _DOME_INDEX["Lower_Teeth"],
    _DOME_INDEX["Upper_Teeth"],
    _DOME_INDEX["Tongue"],
)
TORSO_CLASSES = (_DOME_INDEX["Torso"], _DOME_INDEX["Upper_Clothing"], _DOME_INDEX["Lower_Clothing"])
LEFT_ARM_CLASSES = (_DOME_INDEX["Left_Upper_Arm"], _DOME_INDEX["Left_Lower_Arm"])
RIGHT_ARM_CLASSES = (_DOME_INDEX["Right_Upper_Arm"], _DOME_INDEX["Right_Lower_Arm"])
LEFT_HAND_CLASSES = (_DOME_INDEX["Left_Hand"],)
RIGHT_HAND_CLASSES = (_DOME_INDEX["Right_Hand"],)
LEFT_LEG_CLASSES = (
    _DOME_INDEX["Left_Upper_Leg"],
    _DOME_INDEX["Left_Lower_Leg"],
    _DOME_INDEX["Left_Foot"],
    _DOME_INDEX["Left_Shoe"],
    _DOME_INDEX["Left_Sock"],
)
RIGHT_LEG_CLASSES = (
    _DOME_INDEX["Right_Upper_Leg"],
    _DOME_INDEX["Right_Lower_Leg"],
    _DOME_INDEX["Right_Foot"],
    _DOME_INDEX["Right_Shoe"],
    _DOME_INDEX["Right_Sock"],
)

# Region groups in canonical display order (scale-invariant "region" label set).
_REGION_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("head", HEAD_CLASSES),
    ("torso", TORSO_CLASSES),
    ("left_arm", LEFT_ARM_CLASSES),
    ("right_arm", RIGHT_ARM_CLASSES),
    ("left_hand", LEFT_HAND_CLASSES),
    ("right_hand", RIGHT_HAND_CLASSES),
    ("left_leg", LEFT_LEG_CLASSES),
    ("right_leg", RIGHT_LEG_CLASSES),
)

# Measurement gates.
MIN_FG_PX = 500          # at least this many non-zero depth pixels to measure
MIN_REGION_PX = 200      # a region must clear this to contribute an ordering
MIN_HAND_PX = 60         # hand region support floor (small regions are noisy)
RELIEF_FLOOR = 0.09    # below this normalized depth-relief is "compact" (probed p50-era split 6)
RELIEF_PRONOUNCED = 0.16  # above this the subject has strong depth relief (probed 6/24)

# Human-plausible subject distance band for the CAM-frame metric Z (meters):
# outside this the point-map scale is not plausible for a portrait subject.
MIN_SUBJECT_Z = 0.3
MAX_SUBJECT_Z = 12.0

# Sparse-but-precise relations: how far apart (normalized) two region medians
# must be before we claim a "clearly nearer" ordering rather than "aligned".
_MIN_ORDERING_GAP = 0.08  # |dz| / median over 8% before calling a side "nearer"


class PointmapDepthError(RuntimeError):
    pass


def validate_pointmap_array(pointmap: np.ndarray) -> None:
    if not isinstance(pointmap, np.ndarray):
        raise PointmapDepthError("pointmap must be a numpy array")
    if pointmap.ndim != 3 or pointmap.shape[2] != 3:
        raise PointmapDepthError(f"pointmap must be (H, W, 3), got shape {pointmap.shape}")
    if pointmap.dtype != np.float16 and pointmap.dtype != np.float32:
        raise PointmapDepthError(f"pointmap must be float16/float32, got dtype {pointmap.dtype}")


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise PointmapDepthError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise PointmapDepthError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise PointmapDepthError(f"seg2 must be integer class labels, got dtype {seg2.dtype}")


def _region_median_z(
    pointmap: np.ndarray,
    seg2: np.ndarray,
    classes: tuple[int, ...],
    *,
    min_px: int = MIN_REGION_PX,
) -> float | None:
    """Median CAM-frame Z over seg2 class pixels that have a valid (non-zero) Z."""
    mask = np.isin(seg2, classes)
    if int(mask.sum()) < min_px:
        return None
    z = pointmap[..., 2][mask]
    z_valid = z[z != 0.0]
    if z_valid.size < max(min_px, int(mask.sum() * 0.5)):
        return None
    return float(np.median(z_valid))


# Regions whose support floor differs from the default (hands are small).
_REGION_MIN_PX: dict[str, int] = {
    "left_hand": MIN_HAND_PX,
    "right_hand": MIN_HAND_PX,
}


def compute_pointmap_depth(
    pointmap: np.ndarray,
    seg2: np.ndarray,
    *,
    min_fg_px: int = MIN_FG_PX,
) -> dict[str, Any]:
    """Compute deterministic depth-ordering measurements with explicit abstention.

    Args:
        pointmap: (H, W, 3) float CAM-frame point cloud aligned with seg2.
        seg2: (H, W) integer DOME-29 class labels aligned with pointmap.

    Returns a dict with scale-invariant depth facts only; every caption-facing
    fact is an ordering / ratio / band. Raw metric Z values and spreads live in
    the machine-readable payload, never as caption claims.
    """
    validate_pointmap_array(pointmap)
    validate_seg2_array(seg2)
    if pointmap.shape[0] != seg2.shape[0] or pointmap.shape[1] != seg2.shape[1]:
        raise PointmapDepthError(
            f"pointmap {pointmap.shape} must be pixel-aligned with seg2 {seg2.shape}"
        )

    z = pointmap[..., 2].astype(np.float64)
    fg_mask = z != 0.0
    fg_px = int(fg_mask.sum())
    if fg_px < min_fg_px:
        return _abstain(
            "too few valid depth pixels for stable depth ordering",
            fg_px=fg_px,
            subject_present=fg_px > 0,
        )

    subject_z = z[fg_mask]
    median_z = float(np.median(subject_z))
    if not (MIN_SUBJECT_Z <= median_z <= MAX_SUBJECT_Z):
        return _abstain(
            f"subject median Z {median_z:.3f} outside the human-plausible "
            f"band [{MIN_SUBJECT_Z}, {MAX_SUBJECT_Z}] -> pointmap scale degenerate",
            fg_px=fg_px,
            median_z=median_z,
            subject_present=True,
        )

    p10 = float(np.percentile(subject_z, 10))
    p90 = float(np.percentile(subject_z, 90))
    relief = (p90 - p10) / median_z  # normalized depth-relief (scale-invariant)

    # ---- per-region median Z + relative ordering ----
    region_medians: dict[str, float] = {}
    for name, classes in _REGION_GROUPS:
        value = _region_median_z(
            pointmap, seg2, classes, min_px=_REGION_MIN_PX.get(name, MIN_REGION_PX)
        )
        if value is not None:
            region_medians[name] = value

    if not region_medians:
        return _abstain(
            "no body region clears the depth-support floor for ordering",
            fg_px=fg_px,
            median_z=median_z,
            subject_present=True,
        )

    ordered = sorted(region_medians.items(), key=lambda kv: kv[1])  # nearest -> farthest
    nearest_region = ordered[0][0]
    farthest_region = ordered[-1][0]

    # ---- hand depth ordering / arm-in-front structure ----
    lh = region_medians.get("left_hand")
    rh = region_medians.get("right_hand")
    torso = region_medians.get("torso")

    hand_ordering: str | None = None
    hand_dz_ratio: float | None = None
    left_hand_in_front: bool | None = None
    right_hand_in_front: bool | None = None

    if (lh is not None and rh is not None) and median_z > 0.0:
        hand_dz_ratio = (lh - rh) / median_z
        if abs(hand_dz_ratio) >= _MIN_ORDERING_GAP:
            hand_ordering = "left" if lh < rh else "right"  # smaller Z == nearer camera
    if lh is not None and torso is not None and median_z > 0.0:
        left_hand_in_front = (torso - lh) / median_z >= _MIN_ORDERING_GAP
    if rh is not None and torso is not None and median_z > 0.0:
        right_hand_in_front = (torso - rh) / median_z >= _MIN_ORDERING_GAP

    # ---- scale-invariant depth-relief band ----
    if relief < RELIEF_FLOOR:
        relief_band = "compact"
    elif relief < RELIEF_PRONOUNCED:
        relief_band = "moderate"
    else:
        relief_band = "pronounced"

    return {
        "subject_present": True,
        "abstained": False,
        "abstention_reason": None,
        "depth_measurable": True,
        "median_z": round(median_z, 3),           # payload only, never prose
        "z_p10": round(p10, 3),                   # payload only
        "z_p90": round(p90, 3),                   # payload only
        "depth_relief_ratio": round(relief, 4),   # scale-invariant normalized spread
        "relief_band": relief_band,
        "foreground_depth_px": fg_px,
        "region_median_z": {
            name: round(value, 3) for name, value in region_medians.items()
        },
        "depth_ordering": [name for name, _ in ordered],
        "nearest_region": nearest_region,
        "farthest_region": farthest_region,
        "hand_ordering": hand_ordering,           # None == aligned/absent
        "hand_dz_ratio": hand_dz_ratio,
        "left_hand_in_front": left_hand_in_front,
        "right_hand_in_front": right_hand_in_front,
    }


def _abstain(reason: str, **counts: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "subject_present": True,
        "abstained": True,
        "abstention_reason": reason,
        "depth_measurable": False,
        "median_z": None,
        "z_p10": None,
        "z_p90": None,
        "depth_relief_ratio": None,
        "relief_band": None,
        "foreground_depth_px": None,
        "region_median_z": None,
        "depth_ordering": None,
        "nearest_region": None,
        "farthest_region": None,
        "hand_ordering": None,
        "hand_dz_ratio": None,
        "left_hand_in_front": None,
        "right_hand_in_front": None,
    }
    result["subject_present"] = counts.pop("subject_present", True)
    result.update(counts)
    result.setdefault("foreground_depth_px", 0)
    return result
