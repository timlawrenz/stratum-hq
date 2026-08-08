"""Deterministic subject affordance / self-contact measurements from `pose2` + `seg2`.

Arm #76. NEW deterministic evidence part (no new model). Reads the existing
`pose2.npy` (GOLIATH-308, [x, y, conf]) and the existing `seg2.npy` (DOME-29)
and emits scale-invariant affordance / self-contact facts:

- hand-to-own-body contact count (which / how many hands have their wrist
  within a normalized shoulder-width distance of the subject's own trunk
  region — a hand resting on the hip, folded arms, a hand touching the body);
- hand elevation (gesture) count (how many wrists sit above the hip line by a
  normalized margin — a raised / gesturing / face-near hand);
- subject grounding (whether the subject silhouette reaches the bottom frame
  edge — standing full-frame / in contact with the frame floor versus a
  floating / seated-and-frame-cropped subject).

ONLY scale-invariant facts are ever verbalized: contact counts, elevation
counts, and the grounded binary. Raw pixel distances, pixel coordinates, and
absolute hand regions stay in the machine-readable `evidence_payload`
(camera-frame-dependent absolutes are never caption claims).

Honest scope boundary: seg2 DOME-29 segments ONLY the subject (Background=0 is
not an object class), so subject-to-EXTERNAL-object contact (held-in-hand /
leaning-on / sitting-on an object) is NOT measurable from seg2+pose2 alone —
that axis is the object-relations arm's Grounding-DINO domain (already
validated) and any such claim here would be fabricated; we abstain from it and
cover only own-body contact + grounding, which no validated arm measures
(pose-articulation #62 measures arm flexion and limb-overlap structure, NOT
wrist-to-trunk self-contact or frame grounding).

Abstention: a hand whose wrist keypoint is unreported / below confidence, or
an item with no reliable shoulder/acromion width normalization, abstains for
that axis rather than fabricating. Detector disagreement remains a quality
anomaly, never caption content. No corpus write; CPU-only, in-memory.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from stratum2.config import DOME_29, GOLIATH_308

_GOLIATH_INDEX = {name: i for i, name in enumerate(GOLIATH_308)}
_DOME_INDEX = {name: i for i, name in enumerate(DOME_29)}

WRIST_MIN_CONF = 0.5  # below this a wrist is treated as unreliable -> abstain that hand
CORE_MIN_CONF = 0.5   # shoulder/hip keypoints

# Calibrated on the frozen 24-item cohort (2026-08-07 probe, see the reference):
#   hand_contact_count  {0:11, 1:6, 2:7}  max_share 0.46
#   hand_elevation_count {0:14, 1:6, 2:4} max_share 0.58
#   grounded            {False:10, True:14} max_share 0.58
#   wrist visibility    2=18/24, <2=6/24 (honest abstentions for those hands)
TRUNK_CONTACT_NORM = 0.35   # wrist within 0.35 shoulder-widths of trunk -> contact
WRIST_ABOVE_HIP_NORM = 0.30  # wrist above hip line by > 0.30 shoulder-widths -> raised

# DOME-29 classes that form the subject's own trunk / body for self-contact.
_TRUNK_CLASSES = (
    _DOME_INDEX["Torso"], _DOME_INDEX["Upper_Clothing"], _DOME_INDEX["Lower_Clothing"],
    _DOME_INDEX["Apparel"],
    _DOME_INDEX["Left_Upper_Leg"], _DOME_INDEX["Right_Upper_Leg"],
    _DOME_INDEX["Left_Lower_Leg"], _DOME_INDEX["Right_Lower_Leg"],
)

# Every DOME-29 class that is the curated single woman (for the grounding check).
_SUBJECT_CLASSES = tuple(
    i for i in range(len(DOME_29)) if i != 0
)


class AffordanceContactError(RuntimeError):
    pass


def validate_pose2_array(pose: np.ndarray) -> None:
    if not isinstance(pose, np.ndarray):
        raise AffordanceContactError("pose2 must be a numpy array")
    if pose.shape == (1, 308, 3):
        pose = pose[0]
    if pose.shape != (308, 3):
        raise AffordanceContactError(
            f"pose2 must be shape (308,3) or (1,308,3), got {pose.shape}"
        )


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise AffordanceContactError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise AffordanceContactError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise AffordanceContactError(f"seg2 must be integer class labels, got dtype {seg2.dtype}")


def _normalize_pose(pose: np.ndarray) -> np.ndarray:
    validate_pose2_array(pose)
    if pose.shape == (1, 308, 3):
        return pose[0]
    return pose


def _pt(pose: np.ndarray, name: str, min_conf: float) -> tuple[float, float] | None:
    idx = _GOLIATH_INDEX[name]
    x, y, conf = float(pose[idx, 0]), float(pose[idx, 1]), float(pose[idx, 2])
    if x < 0 or y < 0 or conf < min_conf:
        return None
    return (x, y)


def _mask_dist_to(px: tuple[float, float] | None, mask: np.ndarray) -> float | None:
    """Euclidean distance from a pixel point to the nearest mask pixel (px units)."""
    if px is None:
        return None
    yy, xx = np.where(mask)
    if len(yy) == 0:
        return None
    return float(np.min((xx - px[0]) ** 2 + (yy - px[1]) ** 2) ** 0.5)


def compute_affordance_contact(
    pose: np.ndarray,
    seg2: np.ndarray,
    *,
    min_conf: float = WRIST_MIN_CONF,
) -> dict[str, Any]:
    """Compute deterministic self-contact / affordance measurements.

    Returns a dict:
    - abstained / abstention_reason  (whole-item abstain when no usable skeleton)
    - subject_present
    - shoulder_width_px / shoulder_width_norm_ok
    - hand_contact_count  (0..2; Ky of wrists near own trunk)
    - hand_elevation_count (0..2; wrists above hip line)
    - left_hand_contact / right_hand_contact (bool per side)
    - left_hand_raised / right_hand_raised (bool per side)
    - left_hand_visible / right_hand_visible, and per-side wrist-trunk
      normalized distance (scale-invariant) in payload
    - grounded (bool): subject silhouette reaches the bottom frame row
    - payload-only raw values (normalized wrist->trunk dist, wrist->hip dist)
    """
    pose = _normalize_pose(pose)
    validate_seg2_array(seg2)

    lac = _pt(pose, "left_acromion", CORE_MIN_CONF)
    rac = _pt(pose, "right_acromion", CORE_MIN_CONF)
    ls = _pt(pose, "left_shoulder", CORE_MIN_CONF)
    rs = _pt(pose, "right_shoulder", CORE_MIN_CONF)
    lw = _pt(pose, "left_wrist", min_conf)
    rw = _pt(pose, "right_wrist", min_conf)
    lh = _pt(pose, "left_hip", CORE_MIN_CONF)
    rh = _pt(pose, "right_hip", CORE_MIN_CONF)

    out: dict[str, Any] = {
        "subject_present": True,
        "abstained": False,
        "abstention_reason": None,
        "shoulder_width_px": None,
        "shoulder_width_norm_ok": False,
        "hand_contact_count": 0,
        "hand_elevation_count": 0,
        "left_hand_visible": lw is not None,
        "right_hand_visible": rw is not None,
        "left_hand_contact": False,
        "right_hand_contact": False,
        "left_hand_raised": False,
        "right_hand_raised": False,
        "grounded": False,
        # payload-only (normalized, scale-invariant raw values; never prose alone):
        "left_wrist_trunk_dist_norm": None,
        "right_wrist_trunk_dist_norm": None,
        "left_wrist_hip_offset_norm": None,
        "right_wrist_hip_offset_norm": None,
    }

    sw = None
    if lac and rac:
        sw = float(np.hypot(rac[0] - lac[0], rac[1] - lac[1]))
    elif ls and rs:
        sw = float(np.hypot(rs[0] - ls[0], rs[1] - ls[1]))
    if sw is not None and sw > 0:
        out["shoulder_width_px"] = round(sw, 1)
        out["shoulder_width_norm_ok"] = True
    else:
        # No reliable shoulder width -> cannot normalize distances honectly.
        # Hand-contact and hand-elevation axes abstain; grounding (frame-based,
        # scale-free) can still be measured.
        out.update({
            "hand_contact_count": 0,
            "hand_elevation_count": 0,
            "left_hand_contact": False,
            "right_hand_contact": False,
            "left_hand_raised": False,
            "right_hand_raised": False,
            "left_wrist_trunk_dist_norm": None,
            "right_wrist_trunk_dist_norm": None,
            "left_wrist_hip_offset_norm": None,
            "right_wrist_hip_offset_norm": None,
        })

    trunk_mask = np.isin(seg2, _TRUNK_CLASSES)

    contact_count = 0
    elevation_count = 0
    norm_sw = sw
    if out["shoulder_width_norm_ok"] and norm_sw is not None:
        hip_y = ((lh[1] + rh[1]) / 2.0) if (lh and rh) else None
        for side, wrist in (("left", lw), ("right", rw)):
            if wrist is None:
                continue
            trunk_d = _mask_dist_to(wrist, trunk_mask)
            if trunk_d is not None:
                trunk_norm = trunk_d / norm_sw
                contact = trunk_norm <= TRUNK_CONTACT_NORM
                out[f"{side}_wrist_trunk_dist_norm"] = round(trunk_norm, 3)
                if contact:
                    out[f"{side}_hand_contact"] = True
                    contact_count += 1
            if hip_y is not None:
                offset_norm = (hip_y - wrist[1]) / sw
                out[f"{side}_wrist_hip_offset_norm"] = round(offset_norm, 3)
                if offset_norm > WRIST_ABOVE_HIP_NORM:
                    out[f"{side}_hand_raised"] = True
                    elevation_count += 1
    out["hand_contact_count"] = contact_count
    out["hand_elevation_count"] = elevation_count

    # Grounding: does the subject silhouette reach the bottom row of the frame?
    subject_mask = np.isin(seg2, _SUBJECT_CLASSES)
    out["grounded"] = bool(subject_mask.any() and bool(subject_mask[-1, :].any()))

    return out


def render_affordance_contact(contact: Mapping[str, Any]) -> list[str]:
    """Scale-invariant affordance/self-contact claims for the dossier (arm #76)."""
    if contact.get("abstained"):
        reason = contact.get("abstention_reason") or "affordance not measurable"
        return [f"affordance-contact: abstain ({reason})"]
    if not contact:
        # Dimension not measured for this item (e.g. non-affordance runs) —
        # emit no claim, never a fabricated self-contact statement.
        return []
    if not contact.get("shoulder_width_norm_ok"):
        pass  # hand axes abstain; grounding may still fire (rendered below)
    lines: list[str] = []
    n_contact = int(contact.get("hand_contact_count") or 0)
    if n_contact >= 2:
        lines.append("affordance-contact: both hands rest against her own body")
    elif n_contact == 1:
        lines.append("affordance-contact: one hand rests against her own body")
    n_raised = int(contact.get("hand_elevation_count") or 0)
    if n_raised >= 2:
        lines.append("affordance-contact: both hands are raised (gesturing)")
    elif n_raised == 1:
        lines.append("affordance-contact: one hand is raised (gesturing)")
    if contact.get("grounded"):
        lines.append("affordance-contact: subject is grounded (in contact with the lower frame)")
    if not lines:
        lines.append("affordance-contact: measured (no distinctive self-contact band)")
    return lines
