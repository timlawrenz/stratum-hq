"""Deterministic smile / facial-expression measurement from `pose2`.

Arm #81. NEW deterministic evidence part (no new model). Reads the existing
`pose2.npy` (GOLIATH-308 mouth-corner + eye-center keypoints, [x, y, conf])
and emits a coarse scale-invariant expression band:

- expression_band: neutral / slight-smile / open-smile (or abstain).
- empty/open mouth signature (openness_ratio) and
- mouth-corner spread (spread_ratio) + corner elevation (smile curvature).

Scale denom: the inter-eye-center distance (inter-pupil/inter-iris distance)
when reliable — a stable face-width proxy that survives cross-picture
comparison and a text-to-image model can interpret. When the eyes are not both
reliable, we fall back to the mouth's own height as a local scale (still
scale-invariant) but flagged `reference_fallback`.

ONLY scale-invariant facts are verbalized (the coarse band); raw pixel
distances and normalized ratios stay in the machine-readable
`evidence_payload` (camera-frame-dependent absolutes are never caption claims).

Expression geometry:
- spread_ratio = mouth-corner width / face reference. A smile pulls the mouth
  corners outward AND raises them (curvature up), so spread + corner elevation
  together separate neutral from smiling more cleanly than spread alone.
- openness_ratio = vertical mouth opening / face reference. An open laugh has
  a large mouth opening; a closed smile has near-zero opening.

Abstention: abstains when the mouth is occluded / mouth-corner keypoints are
low-confidence or the face reference is unusable. Never fabricate a smile;
detector disagreement is a quality anomaly, never caption content.
CPU-only, in-memory, no corpus write, no new model.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from stratum2.config import GOLIATH_308

_GOLIATH_INDEX = {name: i for i, name in enumerate(GOLIATH_308)}

CORE_MIN_CONF = 0.5


class FacialExpressionError(RuntimeError):
    pass


def validate_pose2_array(pose: np.ndarray) -> None:
    if not isinstance(pose, np.ndarray):
        raise FacialExpressionError("pose2 must be a numpy array")
    if pose.shape == (1, 308, 3):
        pose = pose[0]
    if pose.shape != (308, 3):
        raise FacialExpressionError(
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


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def compute_facial_expression(pose2: np.ndarray) -> dict[str, Any]:
    """Compute the deterministic expression band with honest abstention.

    Args:
        pose2: (308,3) or (1,308,3) GOLIATH-308 keypoints.

    Returns a dict with scale-invariant facts only:
    - abstained / abstention_reason
    - expression_band (neutral / slight-smile / open-smile)
    - spread_ratio / openness_ratio / corner_elevation ratio (payload)
    """
    pose = _normalize_pose(pose2)

    out: dict[str, Any] = {
        "abstained": False,
        "abstention_reason": None,
        "expression_band": None,
        "spread_ratio": None,
        "openness_ratio": None,
        "corner_elevation_ratio": None,
        "mouth_width_px": None,
        "mouth_open_px": None,
        "face_reference_px": None,
        "reference_fallback": False,
    }

    lmc = _pt(pose, "l_outer_corner_of_mouth")
    rmc = _pt(pose, "r_outer_corner_of_mouth")
    lms = _pt(pose, "l_inner_corner_of_mouth")
    rms = _pt(pose, "r_inner_corner_of_mouth")
    up_lip = _pt(pose, "midpoint_3_of_upper_outer_lip") or _pt(pose, "midpoint_3_of_upper_inner_lip")
    lo_lip = _pt(pose, "midpoint_3_of_lower_outer_lip") or _pt(pose, "midpoint_3_of_lower_inner_lip")

    if not (lmc and rmc and lms and rms and up_lip and lo_lip):
        out.update({
            "abstained": True,
            "abstention_reason": "mouth keypoints unreliable (mouth occluded / low-confidence)",
        })
        return out

    # Face-width reference: inter-eye-center distance (stable), fallback to the
    # mouth's own height as a local scale.
    le = _pt(pose, "l_center_of_iris") or _pt(pose, "l_center_of_pupil")
    re = _pt(pose, "r_center_of_iris") or _pt(pose, "r_center_of_pupil")
    if le and re:
        reference = _dist(le, re)
        fallback = False
    else:
        # No usable eye reference: fall back to the mouth-corner spread as a
        # local scale (still scale-invariant). Using mouth HEIGHT here would
        # make openness ratio trivially 1.0 (degenerate), so width is used.
        mouth_w = _dist(lmc, rmc)
        reference = mouth_w if mouth_w > 0 else _dist(up_lip, lo_lip)
        fallback = True
    out["face_reference_px"] = round(reference, 1)
    out["reference_fallback"] = bool(fallback)
    if reference <= 0:
        out.update({
            "abstained": True,
            "abstention_reason": "face reference degenerate (cannot normalize expression)",
        })
        return out

    mouth_width = _dist(lmc, rmc)
    inner_width = _dist(lms, rms)
    mouth_open = _dist(up_lip, lo_lip)
    out["mouth_width_px"] = round(mouth_width, 1)
    out["mouth_open_px"] = round(mouth_open, 1)

    spread = mouth_width / reference
    openness = mouth_open / reference
    out["spread_ratio"] = round(spread, 4)
    out["openness_ratio"] = round(openness, 4)

    # Corner elevation: how high the outer mouth corners sit relative to the
    # mouth-center line, normalized by reference. A smile curves the corners
    # UP (smaller y in image coords than the mouth centerline between the
    # upper-lip mid and the corners). Sign (+elevated / -dropped).
    mouth_mid_y = (up_lip[1] + lo_lip[1]) / 2.0
    corner_y = (lmc[1] + rmc[1]) / 2.0
    corner_elev = (mouth_mid_y - corner_y) / reference
    out["corner_elevation_ratio"] = round(corner_elev, 4)

    # ---- Classify (re-cut 2026-08-08 from the frozen-cohort probe) ----
    # Band-degeneracy recovery: the first openness-only cut (3 bands) was
    # DEGENERATE — 17/19 items collapsed into 'slight-smile' (max_share 0.89)
    # because this portrait cohort's mouth SPREAD is near-constant and open
    # laughs are rare. The genuinely-discriminating axes are:
    #   - OPENNESS (mouth opening): two clear repo outliers (openness >= 0.28)
    #     are open laughs / wide smiles;
    #   - CORNER ELEVATION (smile curvature): items with the mouth corners
    #     raised >= 0.05 reference above the lip midline read as smiles; the
    #     rest (level or dropped corners, elev < 0.05) read neutral.
    # Re-cut -> open-smile (openness) > slight-smile (elevation) > neutral.
    if openness >= 0.28:
        out["expression_band"] = "open-smile"
    elif corner_elev >= 0.05:
        out["expression_band"] = "slight-smile"
    else:
        out["expression_band"] = "neutral"
    return out


def render_facial_expression(config: Mapping[str, Any]) -> list[str]:
    """Scale-invariant expression claim for the dossier (arm #81).

    Verbalizes ONLY the coarse expression band. Raw normalized ratios stay in
    the machine-readable payload.
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
