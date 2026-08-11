"""Deterministic hairstyle / hair-length + hair-arrangement measurement.

Arm #82. NEW deterministic evidence part (no new model). Shows hair #30
(color + region coverage) does NOT cover shape: 'long/short hair', 'hair tied
back / up' are caption claims with no shape support. This specialist reads the
existing `seg2.npy` (DOME-29 Hair mask, class 4) plus the `pose2.npy`
(GOLIATH-308 shoulder + neck keypoints, [x, y, conf]) and emits:

- hair_length_band:  short / shoulder-length / long  (scale-invariant, relative
  to the shoulder LINE — how far the hair's lowest extent hangs below the
  shoulder midpoint, normalized by shoulder width);
- hair_arrangement_band: down / kept-up  (scale-invariant shape signal — whether
  the hair's vertical mass hangs below the shoulder line. A 'kept-up' band
  covers hairstyles that keep hair above the shoulders — short crops, buns,
  tied-backs — which a Hair silhouette cannot be geometrically separated into
  individually on this cohort; trying to call a short crop 'tied-back' would
  fabrication. Documented as an honest collapse of the on-paper
  up/tied-back/down scheme (band-degeneracy recovery): only down vs kept-up is
  genuinely discriminating.)

Only scale-invariant facts are verbalized (the coarse bands); raw pixel
positions, pixel spans, and normalized fractions stay in the machine-readable
`evidence_payload` (camera-frame-dependent absolutes are never caption claims).

Abstention: abstains when the Hair region is absent/tiny, the shoulder/neck
keypoints are unreliable, or the shape geometry is ambiguous (head cropped,
shoulder mid unreliable, degenerate span). Never fabricate a hairstyle;
detector disagreement is a quality anomaly, never caption content. CPU-only,
in-memory, no corpus write, no new model.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from stratum2.config import DOME_29, GOLIATH_308

# DOME-29 class indices (authoritative in stratum2.config.DOME_29).
HAIR = DOME_29.index("Hair")
FACE_NECK = DOME_29.index("Face_Neck")

# GOLIATH-308 keypoint names.
_GOLIATH_INDEX = {name: i for i, name in enumerate(GOLIATH_308)}

CORE_MIN_CONF = 0.5

# Presence floor: a hair region must clear a raw pixel count before it is
# treated as measured (mirror hair.py).
MIN_CLASS_PX = 200


class HairstyleError(RuntimeError):
    pass


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise HairstyleError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise HairstyleError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise HairstyleError(f"seg2 must be integer class labels, got dtype {seg2.dtype}")


def validate_pose2_array(pose: np.ndarray) -> None:
    if not isinstance(pose, np.ndarray):
        raise HairstyleError("pose2 must be a numpy array")
    if pose.shape == (1, 308, 3):
        pose = pose[0]
    if pose.shape != (308, 3):
        raise HairstyleError(
            f"pose2 must be shape (308,3) or (1,308,3), got {pose.shape}"
        )


def _normalize_pose(pose: np.ndarray) -> np.ndarray:
    validate_pose2_array(pose)
    if pose.shape == (1, 308, 3):
        return pose[0]
    return pose


def _pt(pose: np.ndarray, name: str) -> tuple[float, float] | None:
    idx = _GOLIATH_INDEX[name]
    x, y, conf = float(pose[idx, 0]), float(pose[idx, 1]), float(pose[idx, 2])
    if x < 0 or y < 0 or conf < CORE_MIN_CONF:
        return None
    return (x, y)


def _shoulder_line(pose: np.ndarray) -> tuple[float, float] | None:
    """Return (shoulder_mid_y, shoulder_width_px) or None when unreliable."""
    ls = _pt(pose, "left_shoulder") or _pt(pose, "left_acromion")
    rs = _pt(pose, "right_shoulder") or _pt(pose, "right_acromion")
    if not (ls and rs):
        return None
    width = float(np.hypot(rs[0] - ls[0], rs[1] - ls[1]))
    if width <= 0:
        return None
    return ((ls[1] + rs[1]) / 2.0, width)


def _neck_y(pose: np.ndarray) -> float | None:
    neck = _pt(pose, "neck")
    if neck is not None:
        return neck[1]
    # Fallback: shoulder-mid y (the neck sits just above the shoulder line).
    sl = _shoulder_line(pose)
    if sl is not None:
        return sl[0]
    return None


def compute_hairstyle(
    seg2: np.ndarray,
    pose2: np.ndarray,
    *,
    min_px: int = MIN_CLASS_PX,
) -> dict[str, Any]:
    """Compute deterministic hairstyle bands with honest abstention.

    Args:
        seg2: (H, W) uint8 DOME-29 class labels (Hair = class 4).
        pose2: (308,3) or (1,308,3) GOLIATH-308 keypoints.
        min_px: raw-pixel floor for the Hair region.

    Returns a dict with scale-invariant hair facts only:
    - subject_present / hair_present / abstained
    - hair_length_band (short / shoulder-length / long)
    - hair_arrangement_band (up / down / tied-back)
    - raw scale-invariant geometry (below-shoulder ratio + fraction, span
      ratio, centroid row fraction) for the machine-readable payload.
    """
    validate_seg2_array(seg2)
    pose = _normalize_pose(pose2)

    out: dict[str, Any] = {
        "subject_present": True,
        "abstained": False,
        "abstention_reason": None,
        "hair_present": False,
        "hair_length_band": None,
        "hair_arrangement_band": None,
        "hair_below_shoulder_ratio": None,
        "hair_below_shoulder_fraction": None,
        "hair_span_ratio": None,
        "hair_centroid_row_fraction": None,
        "hair_extent_below_neck_px": None,
        "shoulder_width_px": None,
    }

    hair_mask = seg2 == HAIR
    hair_px = int(hair_mask.sum())
    if hair_px < min_px:
        out.update({
            "abstained": True,
            "abstention_reason": "Hair region absent or below the raw-pixel floor",
        })
        return out
    out["hair_present"] = True

    sl = _shoulder_line(pose)
    neck_y = _neck_y(pose)

    rows, cols = np.nonzero(hair_mask)
    if rows.size == 0:
        out.update({
            "abstained": True,
            "abstention_reason": "Hair region degenerate (no pixels after floor check)",
        })
        return out

    hair_top = float(rows.min())
    hair_bot = float(rows.max())
    hair_span = hair_bot - hair_top
    out["hair_centroid_row_fraction"] = round(float(rows.mean()) / seg2.shape[0], 4)

    # ---- Length band needs the shoulder line (scale denominator) ----
    if sl is None:
        out.update({
            "abstained": True,
            "abstention_reason": "shoulder/neck keypoints unreliable -> length and arrangement bands abstain",
            "hair_present": True,
        })
        return out
    shoulder_y, shoulder_w = sl
    out["shoulder_width_px"] = round(shoulder_w, 1)

    below_shoulder_px = int((rows > shoulder_y).sum())
    out["hair_below_shoulder_fraction"] = round(below_shoulder_px / hair_px, 4)
    below_shoulder_extent = max(0.0, hair_bot - shoulder_y)
    out["hair_below_shoulder_ratio"] = round(below_shoulder_extent / shoulder_w, 4)
    out["hair_span_ratio"] = round(hair_span / shoulder_w, 4)
    if neck_y is not None:
        out["hair_extent_below_neck_px"] = round(
            max(0.0, hair_bot - neck_y), 1
        )

    # ---- Length band (hair_below_shoulder_ratio, scale-invariant) ----
    # Calibrated 2026-08-08 from the frozen-cohort probe (see reference):
    # bands cut on how far the hair's lowest extent hangs below the shoulder
    # line in shoulder-width units.
    bsr = out["hair_below_shoulder_ratio"]
    if bsr is not None:
        if bsr < 0.15:
            out["hair_length_band"] = "short"
        elif bsr < 0.60:
            out["hair_length_band"] = "shoulder-length"
        else:
            out["hair_length_band"] = "long"

    # ---- Arrangement band (extent, scale-invariant) ----
    # Re-cut 2026-08-08 from the frozen-cohort probe (band-degeneracy
    # recovery): the on-paper up/tied-back/down scheme was DEGENERATE — 'up'
    # never fired on this cohort (7/7 non-down items were short crops with
    # span>=0.55, mislabeled 'tied-back'; calling a short crop tied-back would
    # be fabrication). The geometry genuinely discriminates ONLY whether hair
    # hangs below the shoulder line: 'down' (material below-shoulder mass)
    # vs 'kept-up' (hair above/at the shoulders — short crops, buns, ties).
    bsf = out["hair_below_shoulder_fraction"]
    if bsf is not None:
        if bsf >= 0.10 and (bsr is not None and bsr >= 0.15):
            out["hair_arrangement_band"] = "down"
        else:
            out["hair_arrangement_band"] = "kept-up"

    return out


def render_hairstyle(config: Mapping[str, Any]) -> list[str]:
    """Scale-invariant hairstyle claims for the dossier (arm #82).

    Verbalizes ONLY the coarse length + arrangement bands. Raw normalized
    fractions / pixel spans stay in the machine-readable payload.
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
