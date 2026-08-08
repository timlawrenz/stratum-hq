"""Deterministic subject-to-environment clearance / negative-space measurement.

Arm #85. NEW deterministic evidence part (no new model). Reads the existing
`seg2.npy` (DOME-29 class labels, uint8, full source resolution) and measures
the subject's local clearance to the Background class, scale-invariant:

- subject_present / subject bbox
- normalized directional clearances (top / bottom / left / right): the
  Background gap from the subject bbox edge to the frame edge in that
  direction, divided by the subject bbox extent on the SAME axis (a pure
  ratio — survives cross-picture comparison and a text-to-image model).
- clearance_ratio: median of the LEFT and RIGHT normalized clearances (the
  horizontal negative space around the subject — the axis that separates
  'close to a backdrop/wall' from 'in an open space' on a portrait cohort).
- clearance_band: tight / moderate / spacious.

Only the coarse scale-invariant band is verbalized; raw normalized distances
stay in the machine-readable `evidence_payload`.

This is distinct from setting #34 (whole-background coverage/color/tone) and
camera-viewing-angle #74 (framing/headroom): this arm measures the LOCAL
negative space around the subject silhouette. 'Leaning against a wall' /
'in an open space' are caption claims this arm grounds or abstains on.

**Band-degeneracy recovery (measured 2026-08-08):** the on-paper median of all
FOUR directional clearances was DEGENERATE on this portrait cohort — the
tall full-body subject bbox makes the vertical (top/bottom) gaps near-zero for
most items, so 19/22 items collapsed into 'tight' (max_share 0.86). Re-probed
the discriminator: the LEFT/RIGHT horizontal negative space separates cleanly
(10 tight / 9 moderate / 3 spacious, max_share 0.45) and matches the
spatial-settings claims this arm targets ('close to a wall/backdrop' is a
horizontal proximity on a portrait).

Abstention: abstains when the subject fills the frame edge-to-edge (no
Background clearance measurable in any direction) or the subject mask is
degenerate. Never fabricate a clearance class; detector disagreement is a
quality anomaly, never caption content. CPU-only, in-memory, no corpus write,
no new model.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class EnvironmentClearanceError(RuntimeError):
    pass


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise EnvironmentClearanceError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise EnvironmentClearanceError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise EnvironmentClearanceError(f"seg2 must be integer class labels, got dtype {seg2.dtype}")


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(values))


def compute_environment_clearance(seg2: np.ndarray) -> dict[str, Any]:
    """Compute the deterministic clearance band with honest abstention.

    Args:
        seg2: (H, W) uint8 DOME-29 class labels (Background = class 0).

    Returns a dict with scale-invariant facts only:
    - subject_present / abstained
    - normalized directional clearances (top/bottom/left/right)
    - clearance_ratio (median normalized clearance)
    - clearance_band (tight / moderate / spacious)
    """
    validate_seg2_array(seg2)

    out: dict[str, Any] = {
        "subject_present": False,
        "abstained": False,
        "abstention_reason": None,
        "clearance_top": None,
        "clearance_bottom": None,
        "clearance_left": None,
        "clearance_right": None,
        "clearance_ratio": None,
        "clearance_band": None,
        "subject_bbox_h": None,
        "subject_bbox_w": None,
        "subject_frame_coverage": None,
    }
    subject = seg2 > 0
    subject_px = int(subject.sum())
    if subject_px <= 0:
        out.update({
            "abstained": True,
            "abstention_reason": "no foreground subject present",
        })
        return out
    out["subject_present"] = True

    rows, cols = np.nonzero(subject)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    h, w = seg2.shape
    bbox_h = max(r1 - r0, 1)
    bbox_w = max(c1 - c0, 1)
    out["subject_bbox_h"] = bbox_h
    out["subject_bbox_w"] = bbox_w
    out["subject_frame_coverage"] = round(subject_px / max(seg2.size, 1), 6)

    # Normalized directional clearances (Background gap / subject bbox extent
    # on the same axis). A subject filling the frame edge-to-edge in a
    # direction yields 0 in that direction; all-zero -> abstain (frame-fill).
    ct = r0 / bbox_h
    cb = (h - 1 - r1) / bbox_h
    cl = c0 / bbox_w
    cr = (w - 1 - c1) / bbox_w
    out["clearance_top"] = round(min(ct, 9.999), 4)
    out["clearance_bottom"] = round(min(cb, 9.999), 4)
    out["clearance_left"] = round(min(cl, 9.999), 4)
    out["clearance_right"] = round(min(cr, 9.999), 4)

    ratios = [ct, cb, cl, cr]
    if max(ratios) <= 1e-6:
        out.update({
            "abstained": True,
            "abstention_reason": "subject fills the frame edge-to-edge (no Background clearance measurable)",
        })
        return out
    # Horizontal negative space = median of left/right clearances (the
    # discriminating axis on a portrait cohort; vertical gaps near-zero for
    # tall full-body subjects). Band cuts calibrated on the frozen 24-item
    # cohort (2026-08-08 probe): left/right median min 0.0 / p25 0.03 /
    # median 0.23 / p75 0.42 / max 1.40 -> cuts 0.15 / 0.60.
    med = _median([cl, cr])
    out["clearance_ratio"] = round(med, 4)

    if med < 0.15:
        out["clearance_band"] = "tight"
    elif med < 0.60:
        out["clearance_band"] = "moderate"
    else:
        out["clearance_band"] = "spacious"
    return out


def render_environment_clearance(config: Mapping[str, Any]) -> list[str]:
    """Scale-invariant clearance claim for the dossier (arm #85).

    Verbalizes ONLY the coarse clearance band. Raw normalized distances stay
    in the machine-readable payload.
    """
    if not config:
        # Dimension not measured for this item (e.g. non-environment-clearance
        # runs) — emit no claim, never a fabricated spatial-settings statement.
        return []
    if config.get("abstained"):
        reason = config.get("abstention_reason") or "environment clearance not measurable"
        return [f"environment-clearance: abstain ({reason})"]
    if not config.get("subject_present"):
        return ["environment-clearance: abstain (no foreground subject present)"]
    band = config.get("clearance_band")
    if band == "tight":
        return ["environment-clearance: subject is close to the surrounding backdrop/environment (tight negative space)"]
    if band == "moderate":
        return ["environment-clearance: subject has moderate clearance to the surrounding environment"]
    if band == "spacious":
        return ["environment-clearance: subject is in a spacious setting (ample surrounding open space)"]
    return []
