"""Deterministic face-prominence / face-to-hair visibility from `seg2`.

Arm #84. NEW deterministic evidence part (no new model). Reads the existing
`seg2.npy` (DOME-29 class labels, uint8, full source resolution) and measures
how much of the local HEAD region is visible face surface vs hair
(scale-invariant):

- face_present: does the Face_Neck region clear a raw-pixel floor?
- face_share_of_head: Face_Neck px divided by (Face_Neck + Hair) px inside the
  Face_Neck bbox dilated to the local head window. A face generously framed by
  hair has a SMALL share; a face dominating the head region has a LARGE share.
- face_visibility_band: clearly-visible / partially-framed / hair-dominant.

Only the coarse scale-invariant band is verbalized; raw ratios stay in the
machine-readable `evidence_payload`.

**Band-degeneracy recovery (measured 2026-08-08):** the on-paper
occlusion-overlap measure (Face_Neck & Hand/Arm/Hair overlap) is DEGENERATE
on hard-label seg2 — one class per pixel means Face_Neck can NEVER overlap an
occluding class at a pixel, so occlusion_fraction was 0.000 for 23/23 (max
share 1.00). Re-probed the discriminator and re-cut to the face-to-hair
prominence ratio, which is well-distributed (0.371–0.888, p25 0.50, median
0.55, p75 0.70) and grounds the caption claims this arm targets ('her face is
framed by her hair' / 'face clearly visible'). The arm reports prominence of
the visible face, NOT a pixel-overlap occlusion figure (which is structurally
impossible here); 'hair-dominant' means hair makes up the large majority of
the head region around a relatively small exposed face, stated as prominence,
never as a fabricated occlusion percentage.

Abstention: abstains when there is no reliable Face_Neck region (no face in
frame). Never fabricate a visibility class; detector disagreement is a quality
anomaly, never caption content. CPU-only, in-memory, no corpus write, no new
model.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from stratum2.config import DOME_29

# DOME-29 class indices (authoritative in stratum2.config.DOME_29).
FACE_NECK = DOME_29.index("Face_Neck")
HAIR = DOME_29.index("Hair")

# A face region must clear a raw-pixel floor (mirror hair.py / clothing.py).
MIN_CLASS_PX = 200

# Local head window margin: max(floor, proportional to face extent) so the
# face-to-hair ratio genuinely scales with the frame (fixed-pixel margins
# break the scale-invariance invariant).
LOCAL_WINDOW_MARGIN_MIN = 20
LOCAL_WINDOW_MARGIN_FRACTION = 0.20  # of the face bbox long side


def _local_window(face: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, cols = np.nonzero(face)
    if rows.size == 0:
        return None
    h, w = face.shape
    extent = max(int(rows.max()) - int(rows.min()),
                 int(cols.max()) - int(cols.min()))
    margin = max(LOCAL_WINDOW_MARGIN_MIN, int(round(extent * LOCAL_WINDOW_MARGIN_FRACTION)))
    return (
        max(0, int(rows.min()) - margin),
        min(h, int(rows.max()) + margin),
        max(0, int(cols.min()) - margin),
        min(w, int(cols.max()) + margin),
    )
# Band cuts calibrated on the frozen 24-item cohort (2026-08-08 probe):
# face_share_of_head distribution min 0.371 / p25 0.498 / median 0.545 /
# p75 0.697 / max 0.888 (measured 23/24).
CLEARLY_VISIBLE_MIN = 0.65   # face dominates the head region
FRAMED_MIN = 0.45            # between: face partially framed by hair


class FaceVisibilityError(RuntimeError):
    pass


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise FaceVisibilityError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise FaceVisibilityError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise FaceVisibilityError(f"seg2 must be integer class labels, got dtype {seg2.dtype}")


def compute_face_visibility(seg2: np.ndarray, *, min_px: int = MIN_CLASS_PX) -> dict[str, Any]:
    """Compute the deterministic face-prominence band with honest abstention.

    Args:
        seg2: (H, W) uint8 DOME-29 class labels.
        min_px: raw-pixel floor for the Face_Neck region to be measurable.

    Returns a dict with scale-invariant facts only:
    - face_present / abstained
    - face_share_of_head (payload ratio)
    - face_visibility_band (clearly-visible / partially-framed / hair-dominant)
    """
    validate_seg2_array(seg2)

    out: dict[str, Any] = {
        "face_present": False,
        "abstained": False,
        "abstention_reason": None,
        "face_share_of_head": None,
        "face_visibility_band": None,
        "face_px": None,
        "face_frame_coverage": None,
    }
    face = seg2 == FACE_NECK
    face_px = int(face.sum())
    out["face_px"] = face_px
    out["face_frame_coverage"] = round(face_px / max(seg2.size, 1), 6)
    if face_px < min_px:
        out.update({
            "abstained": True,
            "abstention_reason": "no reliable Face_Neck region in frame (face absent / below floor)",
        })
        return out
    out["face_present"] = True

    win = _local_window(face)
    if win is None:
        out.update({
            "abstained": True,
            "abstention_reason": "Face_Neck region degenerate (no pixels after floor check)",
        })
        return out
    r0, r1, c0, c1 = win
    local = seg2[r0:r1, c0:c1]
    hair_px = int((local == HAIR).sum())
    face_local = int((local == FACE_NECK).sum())
    denom = face_local + hair_px
    if denom <= 0:
        out.update({
            "abstained": True,
            "abstention_reason": "head region empty after floor check",
        })
        return out
    share = face_local / denom
    out["face_share_of_head"] = round(share, 4)

    if share >= CLEARLY_VISIBLE_MIN:
        out["face_visibility_band"] = "clearly-visible"
    elif share >= FRAMED_MIN:
        out["face_visibility_band"] = "partially-framed"
    else:
        out["face_visibility_band"] = "hair-dominant"
    return out


def render_face_visibility(config: Mapping[str, Any]) -> list[str]:
    """Scale-invariant face-prominence claim for the dossier (arm #84).

    Verbalizes ONLY the coarse visibility band. The raw face-share ratio
    stays in the machine-readable payload.
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

