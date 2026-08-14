"""Deterministic hand-face-ratio (relational hand-size) measurement.

Arm #124 (registered 2026-08-11 via the gated propose-dimensions channel,
currently PROPOSAL — un-elected while hold #132 gates every Stage-B round
trip). NEW relational evidence part `hand-face-ratio` that FUSES two
already-qualified open-weight MediaPipe models on owned hardware — the
21-point HandLandmarker (arm #109, `hand_landmarker.task` sha256
fbc2a300...) and the 478-point FaceLandmarker (arm #60, `face_landmarker.task`
sha256 64184e229b...) — to emit a scale-invariant "hand size relative to the
face" band (small / typical / large). No new model class is introduced;

both underlying models are already qualified specialists, so this arm is
strictly a NEW relational EVIDENCE PART + detector fusion.

Non-redundance (registry selection rationale): hand-gesture #109 covers the
gesture CLASS (finger-extension pattern) and hand-raised framing; face-geometry
#60 covers face-INTERNAL proportions only. RELATIVE HAND SIZE vs the face
('small delicate hands', 'large hands') is an unbound relational axis that
neither can ground — it requires a concurrent hands + face pairing.

Measurement (scale-invariant, from the 2D meshes):
- Face width: straight-line distance between the widest cheek landmarks
  (CHEEK_L 234 / CHEEK_R 454) of the 478-point mesh, mapped to FULL-FRAME
  pixel coordinates (union detection: full frame, then the seg2 Face_Neck
  crop with its crop-origin offset — every landmark is projected back into
  full-frame px so hand and face distances share one scale).
- Hand palm length: distance from the wrist (0) to the middle-finger MCP (9)
  of the 21-point mesh — a GESTURE-ROBUST hand-size proxy (the wrist and the
  MCP row stay put under finger curl, unlike the fingertip 'span').
- `hand_face_ratio` = palm length / face width (both in full-frame px →
  camera-distance invariant, survives cross-picture comparison; never an
  absolute-pixel claim).
- Corroborating payload (never the primary band): max pairwise landmark
  extent / face width ('span' — GESTURE-DEPENDENT, collapses under a fist,
  disclosed), palm width (index MCP 5 → pinky MCP 17) / face width, per-hand
  ratio, sides, and detection-grade ('via') notes.
- Plausibility gate: measure only when the hand bbox clears MIN_HAND_PX AND
  the face is detected AND the ratio is inside the human-plausible band
  (palm_length / face_width ~[0.35, 1.30] for adults; outside that range the
  hand is held at a very different depth than the face and the read is
  confounded) — otherwise abstain with a surfaced reason.

Verbalized band (item level, coarse): small / typical / large, read from the
hand_face_ratio at COHORT-CALIBRATED tercile cuts (SMALL_MAX / LARGE_MIN
below, set 2026-08-11 from the frozen-cohort calibration probe artifact
/mnt/nas-ai-models/research/stratum/hand-face-ratio-calibration-probe.json;
10/24 measured — hands out of frame/occluded on the rest, honest abstention —
split 3/3/4, max_share 0.40), exactly as nose-geometry #121 / lip-fullness
#122 / eye-shape #123 did.

Abstention: no face detected on full frame or the seg2 Face_Neck crop
(mapped to full-frame px), no hand detected on the full frame or the 2x
LANCZOS upscale (arm-#109 policy), degenerate hand bbox (< MIN_HAND_PX),
or the ratio outside the human-plausible band (hand held at a very different
depth than the face / severe pose / misdetection); never fabricate a
hand-size read. CPU-only in memory, no corpus write, models on owned
hardware (no hosted third-party inference of the sensitive corpus).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .face_geometry import (
    CHEEK_L,
    CHEEK_R,
    FACE_NECK,
    _MIN_FN_PX,
    _detect_mesh_on,
    validate_rgb_array as _validate_rgb_array,
    validate_seg2_array as _validate_seg2_array,
)
from .hand_gesture import _detect_hands
from .hand_gesture import validate_rgb_array as _validate_rgb_array_hand

# ---------------------------------------------------------------------------
# Canonical mesh landmark indices.
# ---------------------------------------------------------------------------
# FaceMesh 478-point: widest cheek points (face width).
FACE_W_L, FACE_W_R = CHEEK_L, CHEEK_R
# HandMesh 21-point: wrist / middle-MCP / index-MCP / pinky-MCP.
HAND_WRIST = 0
HAND_MIDDLE_MCP = 9
HAND_INDEX_MCP = 5
HAND_PINKY_MCP = 17

# Detection gates / floors.
_MIN_HAND_PX = 24            # hand landmark bbox side must clear this (px)
_MIN_FACE_W_PX = 15          # face width must clear this (px)
_NUM_HANDS = 2               # bound the hand detector output

# Human-plausibility band for palm-length / face-width (scale-invariant).
# Typical adult (females) palm length ~10-11cm vs bizygomatic face width
# ~12.5-13.5cm -> ratio ~0.75-0.88. Outside [0.35, 1.30] the hand sits at a
# very different depth than the face (foregrounded hand / misleading
# perspective) and the read is confounded -> abstain.
HAND_FACE_PLAUSIBLE = (0.35, 1.30)

# Model asset sha256s (bind them in the declaration; paths injected).
FACE_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
HAND_MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


class HandFaceRatioError(RuntimeError):
    pass


def validate_rgb_array(rgb: np.ndarray) -> None:
    """Validate a uint8 (H, W, 3) RGB array; raise HandFaceRatioError."""
    try:
        _validate_rgb_array(rgb)
    except Exception as exc:  # noqa: BLE001 - re-raise with the local type
        raise HandFaceRatioError(str(exc)) from exc


def validate_seg2_array(seg2: np.ndarray) -> None:
    """Validate a 2D integer seg2 label array; raise HandFaceRatioError."""
    try:
        _validate_seg2_array(seg2)
    except Exception as exc:  # noqa: BLE001 - re-raise with the local type
        raise HandFaceRatioError(str(exc)) from exc


def _hand_pts(hand, img_w: float, img_h: float) -> np.ndarray:
    """Hand landmarks -> (21, 2) array in full-frame pixel coordinates.

    MediaPipe returns NORMALIZED coords relative to the frame it ran on. The
    hand detector runs on the full frame or a 2x LANCZOS upscale; a pure
    homothetic upscale maps normalized coords 1:1 to the full frame, so
    img_w/img_h are always the FULL-FRAME dims.
    """
    pts = np.array([(p.x * img_w, p.y * img_h) for p in hand], dtype=np.float64)
    return np.ascontiguousarray(pts)


def _face_pts(mesh, img_w: float, img_h: float) -> np.ndarray:
    """Face landmarks -> (478, 2) array in FULL-FRAME pixel coordinates."""
    return np.array(
        [(p.x * img_w, p.y * img_h) for p in mesh], dtype=np.float64
    )


def _face_on_full_frame(rgb: np.ndarray, face_model_asset_path: str):
    """Face mesh detected on the full frame (returns pts or None)."""
    mesh = _detect_mesh_on(np.ascontiguousarray(rgb), face_model_asset_path)
    if mesh is None:
        return None
    h, w = rgb.shape[0], rgb.shape[1]
    pts = _face_pts(mesh, w, h)
    face_w_px = float(np.linalg.norm(pts[FACE_W_R] - pts[FACE_W_L]))
    if face_w_px < _MIN_FACE_W_PX:
        return None
    return {"pts": pts, "face_width_px": face_w_px, "via": "full_frame"}


def _face_on_seg_crop(seg2, rgb, face_model_asset_path: str):
    """Face mesh detected on the seg2 Face_Neck crop, mapped to full-frame px.

    The crop is NOT a homothetic scaling of the full frame, so crop-normalized
    coordinates must be projected back into full-frame pixel space with the
    crop origin + size before any distance is computed (keeps hand and face
    geometry on one shared scale).
    """
    mask = seg2 == FACE_NECK
    fn_px = int(mask.sum())
    if fn_px < _MIN_FN_PX:
        return None
    ys, xs = np.where(mask)
    h, w = ys.max() - ys.min(), xs.max() - xs.min()
    margin = int(max(h, w))
    cy0, cy1 = max(0, ys.min() - margin), min(seg2.shape[0] - 1, ys.max() + margin)
    cx0, cx1 = max(0, xs.min() - margin), min(seg2.shape[1] - 1, xs.max() + margin)
    crop = np.ascontiguousarray(rgb[cy0:cy1, cx0:cx1])
    mesh = _detect_mesh_on(crop, face_model_asset_path)
    if mesh is None:
        return None
    cw, ch = crop.shape[1], crop.shape[0]
    pts = np.array(
        [
            (cx0 + p.x * cw, cy0 + p.y * ch)
            for p in mesh
        ],
        dtype=np.float64,
    )
    face_w_px = float(np.linalg.norm(pts[FACE_W_R] - pts[FACE_W_L]))
    if face_w_px < _MIN_FACE_W_PX:
        return None
    return {"pts": pts, "face_width_px": face_w_px, "via": "seg2_face_crop"}


def _measure(
    hand_pts: np.ndarray,
    face_width_px: float,
) -> dict[str, Any] | None:
    """Per-hand scale-invariant ratio vs face width, or None (degenerate).

    ``hand_pts`` is (21, 2) in full-frame px; ``face_width_px`` is the
    full-frame face width. The primary ratio (palm length / face width) is
    gesture-robust (wrist and MCP row are stable under finger curl); the
    max-pairwise-extent 'span' is reported payload-only and disclosed as
    gesture-dependent.
    """
    palm_len_px = float(np.linalg.norm(hand_pts[HAND_MIDDLE_MCP] - hand_pts[HAND_WRIST]))
    palm_w_px = float(np.linalg.norm(hand_pts[HAND_PINKY_MCP] - hand_pts[HAND_INDEX_MCP]))
    span_px = float(
        np.max(np.linalg.norm(hand_pts[:, None, :] - hand_pts[None, :, :], axis=-1))
    )
    if face_width_px <= 1e-6 or palm_len_px <= 1e-6:
        return None
    ratio = palm_len_px / face_width_px
    if not (HAND_FACE_PLAUSIBLE[0] <= ratio <= HAND_FACE_PLAUSIBLE[1]):
        return {
            "measureable": False,
            "reason": (
                f"palm-length/face-width ratio {round(ratio, 3)} outside the "
                f"human-plausible band {HAND_FACE_PLAUSIBLE} — the hand is "
                "held at a very different depth than the face (misleading "
                "perspective) or the detection is spurious"
            ),
        }
    return {
        "measureable": True,
        "hand_face_ratio": round(ratio, 4),
        "palm_length_px": round(palm_len_px, 3),
        "palm_span_px_payload": round(span_px, 3),
        "span_face_width_payload": round(span_px / face_width_px, 4),
        "palm_width_face_width_payload": round(palm_w_px / face_width_px, 4),
    }


def compute_hand_face_ratio(
    rgb: np.ndarray,
    seg2: np.ndarray,
    *,
    face_model_asset_path: str,
    hand_model_asset_path: str,
) -> dict[str, Any]:
    """Compute the scale-invariant hand-face-ratio bands with honest abstention.

    Detection policies (measured on the frozen cohort by arms #109 and #60):
    hands = full frame then 2x LANCZOS upscale (the pass with more hands wins,
    bounded at num_hands=2); face = full frame then seg2 Face_Neck crop
    (UNION), every landmark projected to full-frame px. Only scale-invariant
    ratios and the coarse band are returned for prose; raw landmark-derived
    values stay in the machine-readable payload. If at least one hand is
    measurable and the face is detected, the item band is the mean of the
    per-hand ratios (single-hand fallback disclosed).
    """
    validate_seg2_array(seg2)
    validate_rgb_array(rgb)
    if seg2.shape[0] != rgb.shape[0] or seg2.shape[1] != rgb.shape[1]:
        raise HandFaceRatioError(
            f"seg2 {seg2.shape} must be pixel-aligned with rgb {rgb.shape}"
        )

    # Face first (the denominator) — union policy, mapped to full-frame px.
    face = _face_on_full_frame(rgb, face_model_asset_path)
    if face is None:
        face = _face_on_seg_crop(seg2, rgb, face_model_asset_path)
    if face is None:
        return {
            "abstained": True,
            "abstention_reason": (
                "no face detected on the full frame or the seg2 Face_Neck "
                "crop (face too small / turned away / occluded)"
            ),
            "seg2_face_neck_px": int((seg2 == FACE_NECK).sum()),
        }
    face_width_px = face["face_width_px"]

    # Hands — arm #109 policy.
    img_h, img_w = rgb.shape[0], rgb.shape[1]
    arr = np.ascontiguousarray(rgb)
    hands = _detect_hands(arr, hand_model_asset_path)
    via = "full_frame"
    if not hands:
        from PIL import Image
        up = np.ascontiguousarray(np.asarray(
            Image.fromarray(rgb).resize((img_w * 2, img_h * 2), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        ))
        hands_up = _detect_hands(up, hand_model_asset_path)
        if len(hands_up) > len(hands):
            hands = hands_up
            via = "upscaled_2x"
    if not hands:
        return {
            "abstained": True,
            "abstention_reason": (
                "no hand detected on the full frame or the 2x upscale "
                "(hands turned away, occluded, or out of frame)"
            ),
            "face_width_px": round(face_width_px, 3),
            "via": via,
        }

    per_hand: list[dict[str, Any]] = []
    for i, hand in enumerate(hands):
        hp = _hand_pts(hand, img_w, img_h)
        xs, ys = hp[:, 0], hp[:, 1]
        if xs.max() - xs.min() < _MIN_HAND_PX or ys.max() - ys.min() < _MIN_HAND_PX:
            continue
        m = _measure(hp, face_width_px)
        if m is None or not m.get("measureable"):
            per_hand.append({
                "index": i,
                "measureable": False,
                "reason": (m.get("reason") if m else
                           "degenerate hand geometry (bbox below the size floor)"),
            })
            continue
        per_hand.append({
            "index": i,
            "measureable": True,
            "hand_face_ratio": m["hand_face_ratio"],
            "palm_length_px": m["palm_length_px"],
            "span_face_width_payload": m["span_face_width_payload"],
            "palm_width_face_width_payload": m["palm_width_face_width_payload"],
            "hand_bbox_px": [round(float(xs.min())), round(float(ys.min())),
                             round(float(xs.max())), round(float(ys.max()))],
        })

    measured = [p for p in per_hand if p.get("measureable")]
    if not measured:
        return {
            "abstained": True,
            "abstention_reason": (
                "hands detected but none measurable (degenerate bbox or ratio "
                "outside the human-plausible depth band)"
            ),
            "hands_detected": len(per_hand),
            "face_width_px": round(face_width_px, 3),
            "via": via,
        }

    ratio_mean = float(np.mean([p["hand_face_ratio"] for p in measured]))
    fact = {
        "abstained": False,
        "face_width_px": round(face_width_px, 3),
        "face_detection_via": face["via"],
        "hand_detection_via": via,
        "hands_detected": len(per_hand),
        "hands_measured": len(measured),
        "hand_face_ratio": round(ratio_mean, 4),
        "per_hand": per_hand,
    }
    if len(measured) == 1:
        fact["single_hand_disclosed"] = True
    return _apply_bands(fact)


# ---------------------------------------------------------------------------
# Band floors — COHORT-CALIBRATED tercile cuts (2026-08-11, frozen-cohort
# calibration probe, artifact
# /mnt/nas-ai-models/research/stratum/hand-face-ratio-calibration-probe.json:
# 10/24 measured, 14 honest abstains (no hand / no face detected — hands
# turned away, occluded, or out of frame on this portrait/crop-heavy cohort;
# consistent with the hand-gesture #109 probe), measured ratio range
# 0.4146-0.8387). The probe's p33/p66 cuts split the measured cohort
# 3 small / 3 typical / 4 large (max_share 0.40); the provisional canon cuts
# (0.65/1.00) were non-degenerate (0.50) but no measured item clears 1.00,
# so the "large" band could never fire. Authoritative cuts below are the
# measured cohort terciles, following the nose-geometry #121 pattern
# (measured 7/7/7 at p33/p66 cuts). Bands are corpus-relative within the
# frozen cohort; the qualification gate (no band >= 75%) passes at max_share
# 0.40. Coverage disclosure: only 10/24 (41.7%) items are measurable; the
# axis abstains honestly elsewhere.
# ---------------------------------------------------------------------------
SMALL_MAX = 0.571   # palm-length/face-width below this -> small / delicate hands (cohort p33)
LARGE_MIN = 0.690   # above this -> large hands (cohort p66)


def set_band_floors(small_max: float, large_min: float) -> None:
    """Override the band cuts (probe convenience; the module's authoritative
    constants are the module-level SMALL_MAX / LARGE_MIN above)."""
    global SMALL_MAX, LARGE_MIN  # noqa: PLW0603
    SMALL_MAX = small_max
    LARGE_MIN = large_min


def _apply_bands(fact: dict[str, Any]) -> dict[str, Any]:
    """Attach the hand-face-ratio band (small / typical / large)."""
    r = fact.get("hand_face_ratio")
    if r is None:
        fact["hand_size_band"] = None
        fact["banding_unavailable"] = True
    elif r < SMALL_MAX:
        fact["hand_size_band"] = "small"
    elif r > LARGE_MIN:
        fact["hand_size_band"] = "large"
    else:
        fact["hand_size_band"] = "typical"
    return fact


def render_hand_face_ratio(cfg: Mapping[str, Any] | None) -> list[str]:
    """Scale-invariant hand-face-ratio claims for the dossier (arm #124)."""
    if not cfg:
        return []
    if cfg.get("abstained"):
        reason = cfg.get("abstention_reason") or "hand size relative to face not measurable"
        return [f"hand-face-ratio: abstain ({reason})"]
    if cfg.get("banding_unavailable"):
        return ["hand-face-ratio: measured but not banded (payload)"]
    lines: list[str] = []
    band = cfg.get("hand_size_band")
    if band == "small":
        lines.append("hand-face-ratio: the hands read small relative to the face (delicate hands)")
    elif band == "large":
        lines.append("hand-face-ratio: the hands read large relative to the face")
    elif band == "typical":
        lines.append("hand-face-ratio: the hands read typical in size relative to the face")
    return lines
