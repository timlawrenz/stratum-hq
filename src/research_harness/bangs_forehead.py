"""Deterministic bangs / forehead-hair-coverage measurement from seg2 + pose2.

Arm #110 (registered 2026-08-10 via the gated propose-dimensions channel,
brainstorm-new-data). NEW evidence part `bangs-forehead` (no new model, CPU,
deterministic from the frozen derived artifacts).

Measurement (robust to bangs-wearers, unlike a Face_Neck-top-anchored strip):
- The forehead band is anchored to the RELIABLE pose2 GOLIATH-308 eye line
  (mean of left/right eye centers): rows `[eye_y - 0.20*face_height, eye_y]`,
  spanning the forehead width from the seg2 Face_Neck bounding box (10% inset).
  This is the region right above the brows where bangs drape and where a
  swept-back hairline leaves smooth skin; scalp hair sits well above it.
- face_height = seg2 Face_Neck bbox height below the eye line (chin proxy).
- bangs coverage = seg2 DOME-29 Hair pixels inside the forehead band /
  band area. Scale-invariant by construction (a ratio over the same band).
- Verbalized band: bangs (coverage >= 0.35) / partial-fringe (0.12..0.35) /
  swept-back (<= 0.12), or abstain when the forehead band is unusable.

This axis is distinct from the validated hairstyle #82 (length + down/kept-up
arrangement) and hair #30 (color/coverage): it measures the hair draping over
the forehead specifically ('she has bangs / a fringe', 'hair swept back
revealing the forehead').

ONLY the scale-invariant band is verbalized; the raw normalized coverage ratio
stays in the machine-readable `evidence_payload`. Abstention: absent/occluded
face, unreliable eye line, absent/tiny hair or face region, or a degenerate
band; never fabricate a bangs state. CPU-only, in-memory, no corpus write.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from stratum2.config import DOME_29, GOLIATH_308

_FACE_NECK = DOME_29.index("Face_Neck")
_HAIR = DOME_29.index("Hair")

_GOLIATH_INDEX = {name: i for i, name in enumerate(GOLIATH_308)}
_EYE_R = GOLIATH_308.index("right_eye")
_EYE_L = GOLIATH_308.index("left_eye")
CORE_MIN_CONF = 0.5

# Band floors — CALIBRATED from the frozen-cohort probe (2026-08-10).
BANGS_FLOOR = 0.35      # coverage >= this -> bangs
SWEPT_BACK_MAX = 0.12   # coverage <= this -> swept-back; else partial-fringe

# Forehead band geometry.
BAND_HEIGHT_FRACTION = 0.20  # band height above the eye line, in face heights
MIN_FACE_PX = 400            # seg2 Face_Neck region floor
MIN_HAIR_PX = 200            # seg2 Hair region floor
MIN_BAND_PX = 80             # minimum band area to measure a ratio


class BangsForeheadError(RuntimeError):
    pass


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise BangsForeheadError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise BangsForeheadError(f"seg2 must be two-dimensional, got shape {seg2.shape}")


def validate_pose2_array(pose: np.ndarray) -> None:
    if not isinstance(pose, np.ndarray):
        raise BangsForeheadError("pose2 must be a numpy array")
    if pose.shape == (1, 308, 3):
        pose = pose[0]
    if pose.shape != (308, 3):
        raise BangsForeheadError(f"pose2 must be shape (308,3) or (1,308,3), got {pose.shape}")


def _eye_line_y(pose: np.ndarray) -> float | None:
    ys = []
    for idx in (_EYE_R, _EYE_L):
        y, conf = float(pose[idx, 1]), float(pose[idx, 2])
        x = float(pose[idx, 0])
        if x >= 0 and y >= 0 and conf >= CORE_MIN_CONF:
            ys.append(y)
    if len(ys) < 2:
        return None
    return float(np.mean(ys))


def compute_bangs_forehead(seg2: np.ndarray, pose2: np.ndarray,
                           *, bangs_floor: float = BANGS_FLOOR,
                           swept_back_max: float = SWEPT_BACK_MAX) -> dict[str, Any]:
    """Compute the scale-invariant bangs / swept-back / partial-fringe band.

    Returns a dict with ``abstained`` / ``abstention_reason`` on failure, or
    the band + raw coverage ratio (payload).
    """
    validate_seg2_array(seg2)
    validate_pose2_array(pose2)
    if pose2.shape == (1, 308, 3):
        pose2 = pose2[0]

    out: dict[str, Any] = {
        "subject_present": True,
        "abstained": False,
        "abstention_reason": None,
        "bangs_band": None,
        "forehead_hair_coverage": None,
        "forehead_band_px": 0,
    }

    face_mask = seg2 == _FACE_NECK
    face_px = int(face_mask.sum())
    hair_mask = seg2 == _HAIR
    hair_px = int(hair_mask.sum())
    if face_px < MIN_FACE_PX:
        out.update({"abstained": True, "abstention_reason": (
            f"seg2 Face_Neck region too small (px={face_px} < {MIN_FACE_PX})")})
        return out
    if hair_px < MIN_HAIR_PX:
        out.update({"abstained": True, "abstention_reason": (
            f"seg2 Hair region too small (px={hair_px} < {MIN_HAIR_PX})")})
        return out

    rows, cols = np.nonzero(face_mask)
    top_f = int(cols.min())
    right_f = int(cols.max()) + 1
    bottom_f = int(rows.max()) + 1

    eye_y = _eye_line_y(pose2)
    if eye_y is None:
        out.update({"abstained": True,
                    "abstention_reason": "eye-line landmarks unreliable"})
        return out

    face_h = float(max(1, bottom_f - eye_y))
    band_top = int(eye_y - BAND_HEIGHT_FRACTION * face_h)
    band_bottom = int(eye_y)
    inset = int(0.10 * (right_f - top_f))
    band_left = top_f + inset
    band_right = max(band_left + 1, right_f - inset)
    if band_top < 0:
        band_top = 0
    band_h = band_bottom - band_top
    band_w = band_right - band_left
    if band_h < 1 or band_w < 1 or band_h * band_w < MIN_BAND_PX:
        out.update({"abstained": True, "abstention_reason": (
            f"forehead band too small ({band_h}x{band_w}px < {MIN_BAND_PX})")})
        return out

    band = np.zeros_like(face_mask, dtype=bool)
    band[band_top:band_bottom, band_left:band_right] = True
    band_px = int(band.sum())
    hair_in_band = int(np.logical_and(band, hair_mask).sum())
    coverage = hair_in_band / band_px if band_px else 0.0
    out["forehead_band_px"] = band_px
    out["forehead_hair_coverage"] = round(float(coverage), 4)
    out["eye_y"] = round(eye_y, 2)
    out["face_height_px"] = round(face_h, 2)

    if coverage >= bangs_floor:
        out["bangs_band"] = "bangs"
    elif coverage <= swept_back_max:
        out["bangs_band"] = "swept-back"
    else:
        out["bangs_band"] = "partial-fringe"
    return out


def render_bangs_forehead(config: Mapping[str, Any]) -> list[str]:
    """Scale-invariant bangs claim for the dossier (arm #110)."""
    if not config:
        return []
    if config.get("abstained"):
        reason = config.get("abstention_reason") or "bangs not measurable"
        return [f"bangs-forehead: abstain ({reason})"]
    band = config.get("bangs_band")
    if band == "bangs":
        return ["bangs-forehead: hair falls across the forehead as bangs/fringe"]
    if band == "partial-fringe":
        return ["bangs-forehead: a partial fringe frames the forehead"]
    if band == "swept-back":
        return ["bangs-forehead: hair is swept back revealing the forehead"]
    return []
