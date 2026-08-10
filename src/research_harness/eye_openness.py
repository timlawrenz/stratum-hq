"""Deterministic eye-openness / eyelid-state measurement from `pose2`.

Arm #113 (registered 2026-08-10 via the gated propose-dimensions channel,
selected by the epsilon-greedy EXPLORE slot, selection_progress 32). NEW
evidence part `eye-openness` (no new model, CPU, deterministic from the frozen
pose2 GOLIATH-308 artifact).

Measurement (scale-invariant, per eye):
- The eyelid line is traced by 16 GOLIATH-308 keypoints per eye (8 upper-lid
  points and 8 lower-lid points: outer end, midpoints 1-6, centerpoint).
  NOTE: GOLIATH-308 eyelid lines cover only the OUTER half of the eye (outer
  canthus -> mid-eye); there is no inner-canthus keypoint, and pairs near the
  canthi have ~zero gap even for open eyes, so a MEDIAN pair gap is biased
  low. We use the MAX vertical pair gap (the widest lid aperture, EAR-style):
  for each upper-lid point, the vertical distance to the lower-lid point with
  the nearest x; the MAXIMUM over the 8 pairs is the aperture.
- openness_ratio = max lid gap / IPD  (scale-invariant; ~0 for a closed eye,
  larger for an open eye). IPD (left_eye/right_eye keypoints) is the same
  inter-eye reference facial-expression #81 uses.
- Frontal-plausibility gate: measure only when IPD is present and plausible
  (>= MIN_IPD_PX) and the eyelid line is fully keypointed; else that eye
  abstains with a surfaced reason.
- iris state: full (iris center + >=2 iris-border keypoints at conf>=0.5),
  partial (center only), none. A closed eye drops the iris keypoints AND the
  eyelid-line keypoints; when the eyelid line is unusable AND iris state is
  none AND the face-side eye center is also absent, the eye reads
  closed-signature. When face keypoints ARE present but the eye region is
  absent, this is the closed-eye signature (GOLIATH drops the whole eye-region
  keypoint set for closed/occluded eyes).

Verbalized band (item level, coarse):
- eyes-open / lidded / eyes-closed; band floors CALIBRATED from the
  frozen-cohort probe (2026-08-10) — see module constants.
- never identity, eye-color, or gaze claims; raw ratios stay in the
  machine-readable `evidence_payload`.

Abstention: neither eye measurable (turned-away or fully occluded face),
degenerately small face (IPD below floor), or model/array failure; never
fabricate an eyelid state. CPU-only, in-memory, no corpus write.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from stratum2.config import GOLIATH_308

_G = {name: i for i, name in enumerate(GOLIATH_308)}

CORE_MIN_CONF = 0.5

# Eyelid line keypoint sets per side (order within each set is irrelevant:
# pairs are formed by nearest-x matching).
_UPPER = {
    side: [f"{side}_outer_end_of_upper_eyelid_line"] +
          [f"{side}_midpoint_{i}_of_upper_eyelid_line" for i in range(1, 7)] +
          [f"{side}_centerpoint_of_upper_eyelid_line"]
    for side in ("l", "r")
}
_LOWER = {
    side: [f"{side}_outer_end_of_lower_eyelid_line"] +
          [f"{side}_midpoint_{i}_of_lower_eyelid_line" for i in range(1, 7)] +
          [f"{side}_centerpoint_of_lower_eyelid_line"]
    for side in ("l", "r")
}
_IRIS_CENTER = {side: f"{side}_center_of_iris" for side in ("l", "r")}
_IRIS_BORDER = {
    side: [f"{side}_border_of_iris_{p}" for p in (3, 6, 9, 12)]
    for side in ("l", "r")
}
_EYE_CENTER = {"l": "left_eye", "r": "right_eye"}

# Minimum plausible interpupillary distance (px) to measure a stable
# scale-invariant ratio; below this the face is too small/degraded.
MIN_IPD_PX = 12.0

# ---------------------------------------------------------------------------
# Band floors — CALIBRATED from the frozen-cohort probe (2026-08-10).
# Distribution (max gap / IPD, n=46 eyes): min 0.0365 / p25 0.0598 /
# median 0.1295 / p75 0.1809 / max 0.5024 (the three eye-color-abstained
# items — 0yo0gx..., 0mel7e..., 08v25q... — are exactly the three lowest
# apertures, corroborating the axis). CLOSED_MAX is set below the lowest
# measured ratio: the closed band fires on the keypoint-dropped
# closed-signature, not on threshold-fitting; LIDDED_MAX=0.10 separates the
# low-aperture cluster (lidded) from the normal cluster (open) with no band
# >= 75% (measured lidded 7 / open 15 / closed-signature 1).
CLOSED_MAX = 0.025    # ratio <= this -> closed (reserved; below all measured)
LIDDED_MAX = 0.10     # ratio in (CLOSED_MAX, LIDDED_MAX] -> lidded
# ratio > LIDDED_MAX -> open
# ---------------------------------------------------------------------------


class EyeOpennessError(RuntimeError):
    pass


def validate_pose2_array(pose: np.ndarray) -> None:
    if not isinstance(pose, np.ndarray):
        raise EyeOpennessError("pose2 must be a numpy array")
    if pose.shape == (1, 308, 3):
        pose = pose[0]
    if pose.shape != (308, 3):
        raise EyeOpennessError(f"pose2 must be shape (308,3) or (1,308,3), got {pose.shape}")


def _normalize_pose(pose: np.ndarray) -> np.ndarray:
    validate_pose2_array(pose)
    if pose.shape == (1, 308, 3):
        return pose[0]
    return pose


def _pt(pose: np.ndarray, name: str) -> tuple[float, float] | None:
    idx = _G[name]
    x, y, conf = float(pose[idx, 0]), float(pose[idx, 1]), float(pose[idx, 2])
    if x < 0 or y < 0 or conf < CORE_MIN_CONF:
        return None
    return (x, y)


def _iris_state(pose: np.ndarray, side: str) -> str:
    """full (center + >=2 borders) / partial (center only) / none."""
    center = _pt(pose, _IRIS_CENTER[side])
    if center is None:
        return "none"
    borders = [p for p in (_pt(pose, n) for n in _IRIS_BORDER[side]) if p is not None]
    return "full" if len(borders) >= 2 else "partial"


def _interpupillary(pose: np.ndarray) -> float | None:
    lc = _pt(pose, _EYE_CENTER["l"])
    rc = _pt(pose, _EYE_CENTER["r"])
    if lc is None or rc is None:
        return None
    return float(np.hypot(rc[0] - lc[0], rc[1] - lc[1]))


def _eye_openness(pose: np.ndarray, side: str, ipd: float | None) -> dict[str, Any]:
    """Per-eye eyelid geometry. Returns dict with 'eye_state' in
    {measured, closed-signature, unmeasurable} + surfaced reason."""
    upper_raw = [_pt(pose, n) for n in _UPPER[side]]
    lower_raw = [_pt(pose, n) for n in _LOWER[side]]
    if any(p is None for p in upper_raw + lower_raw):
        # Eyelid line dropped by the detector (closed eyes drop the whole
        # eye-region keypoint set). Distinguish closed-signature from
        # turned-away face via iris + face-side eye-center presence.
        iris_state = _iris_state(pose, side)
        if iris_state == "none" and _pt(pose, _EYE_CENTER[side]) is None:
            return {"side": side, "eye_state": "unmeasurable",
                    "reason": "eye-region keypoints dropped (closed or face side absent)"}
        return {
            "side": side,
            "eye_state": "closed-signature",
            "reason": "eyelid line dropped and no full iris keypoints (closed-eye signature)",
            "iris_state": iris_state,
        }

    if ipd is None or ipd < MIN_IPD_PX:
        return {"side": side, "eye_state": "unmeasurable",
                "reason": f"interpupillary distance unavailable/too small (ipd={ipd})"}
    upper: list[tuple[float, float]] = [p for p in upper_raw if p is not None]
    lower: list[tuple[float, float]] = [p for p in lower_raw if p is not None]
    assert len(upper) == 8 and len(lower) == 8
    xs_u = [p[0] for p in upper]
    ys_u = [p[1] for p in upper]
    xs_l = [p[0] for p in lower]
    ys_l = [p[1] for p in lower]
    # Pair each upper-lid point with the nearest-x lower-lid point; the
    # MAX pair gap is the widest lid aperture (EAR-style; median is biased
    # low by ~zero-gap canthus pairs).
    gaps: list[float] = []
    for ux, uy in zip(xs_u, ys_u):
        best = min(range(len(xs_l)), key=lambda i: abs(xs_l[i] - ux))
        gaps.append(abs(uy - ys_l[best]))
    gap = float(max(gaps)) if gaps else 0.0
    ratio = gap / ipd if ipd > 0 else 0.0
    return {
        "side": side,
        "eye_state": "measured",
        "openness_ratio": round(float(ratio), 4),
        "max_lid_gap_px": round(gap, 3),
        "ipd_px": round(ipd, 3),
        "iris_state": _iris_state(pose, side),
    }


def _band_for_ratio(ratio: float) -> str:
    if ratio <= CLOSED_MAX:
        return "closed"
    if ratio <= LIDDED_MAX:
        return "lidded"
    return "open"


def compute_eye_openness(pose2: np.ndarray) -> dict[str, Any]:
    """Compute the deterministic eye-openness band with honest abstention.

    Returns a dict with:
    - abstained / abstention_reason
    - eye_openness_band (open / lidded / closed)
    - per_eye: per-side dicts (payload)
    - the item-level mean openness ratio (payload, never prose)
    """
    pose = _normalize_pose(pose2)

    out: dict[str, Any] = {
        "abstained": False,
        "abstention_reason": None,
        "eye_openness_band": None,
        "openness_ratio": None,
        "per_eye": {},
    }

    ipd = _interpupillary(pose)
    eyes = {}
    for side in ("l", "r"):
        eyes[side] = _eye_openness(pose, side, ipd)
    out["per_eye"] = eyes

    measured = [e for e in eyes.values() if e.get("eye_state") == "measured"]
    closed_sig = [e for e in eyes.values() if e.get("eye_state") == "closed-signature"]

    if not measured:
        # No usable geometry on either side.
        if len(closed_sig) == 2:
            out.update({
                "eye_openness_band": "closed",
                "abstained": False,
                "openness_ratio": 0.0,
            })
            return out
        out.update({
            "abstained": True,
            "abstention_reason": (
                "neither eye has usable eyelid geometry (turned-away, occluded, "
                "or too-small face)"
            ),
        })
        return out

    ratios = [e["openness_ratio"] for e in measured]
    item_ratio = float(np.mean(ratios))
    out["openness_ratio"] = round(item_ratio, 4)

    if closed_sig and len(measured) == 1 and measured[0]["openness_ratio"] <= LIDDED_MAX:
        # One eye measured but low, the other closed-by-signature -> lidded.
        out["eye_openness_band"] = "lidded"
        return out

    out["eye_openness_band"] = _band_for_ratio(item_ratio)
    return out


def render_eye_openness(config: Mapping[str, Any]) -> list[str]:
    """Scale-invariant eyelid-state claim for the dossier (arm #113)."""
    if not config:
        return []
    if config.get("abstained"):
        reason = config.get("abstention_reason") or "eye openness not measurable"
        return [f"eye-openness: abstain ({reason})"]
    band = config.get("eye_openness_band")
    if band == "open":
        return ["eye-openness: eyes are open"]
    if band == "lidded":
        return ["eye-openness: eyelids are partially lowered (lidded)"]
    if band == "closed":
        return ["eye-openness: eyes are closed"]
    return []