"""Deterministic hand-gesture measurement from a local MediaPipe HandLandmarker.

Arm #109. NEW-model-class specialist: runs the open-weight MediaPipe
HandLandmarker (21-point-per-hand mesh, `hand_landmarker.task`, Apache-2.0,
local CPU via the tasks API, XNNPACK) on owned hardware and derives
scale-invariant hand facts:

- per-visible-hand gesture class (open-palm / fist / pointing /
  relaxed-curl) from the deterministic within-hand finger-extension
  geometry (tip-to-PIP vs PIP-to-MCP segment ratios — scale-invariant);
- one-vs-two-hands (the number of detected hands);
- hand-raised flag per hand, gated on the pose2 GOLIATH-308 wrist/shoulder
  reference (wrist above the shoulder line by >= 0.5 shoulder-widths —
  scale-invariant body-relative framing); abstains honestly when the wrist
  or shoulders are unreported.

Only scale-invariant categorical facts are verbalized (the
measurement-semantics directive: absolute pixel positions are
camera-frame-dependent and a text-to-image model cannot render them). Raw
landmark coordinates, per-finger extension vector, confidence, and the
handedness output stay in the machine-readable ``evidence_payload`` JSON
and are never caption claims.

Detection policy (measured 2026-08-10 capability probe): MediaPipe Hands is
resolution-sensitive on this portrait cohort; the deterministic policy is
full-frame first, then a 2x LANCZOS upscale when zero hands are found; the
pass with more hands wins (bounded at num_hands=2). Crop slices are
``np.ascontiguousarray`` (MediaPipe silently drops non-contiguous views).

Band calibration (2026-08-10, band-calibration rule arm #34/#35/#59): the
frozen-cohort probe must show no gesture band >= 75% of measured hands and
>= 14/24 items with at least one detected hand (coverage floor pre-registered
on issue #109); otherwise the axis is re-cut or kept payload-only.

Provenance: local open-weight model (hand_landmarker.task, sha256 fbc2a300…
run on owned hardware only; no hosted third-party inference of the sensitive
corpus; no corpus write. model_asset_path is dependency-injected so unit
tests can point at a fixture and the runner at the frozen model asset.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# ---------------------------------------------------------------------------
# Landmark index constants (canonical 21-point MediaPipe Hands mesh).
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# (MCP, PIP, DIP, TIP) per finger, canonical order.
_FINGERS = [
    ("index", INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    ("middle", MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    ("ring", RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    ("pinky", PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
]
# Thumb chain uses its MCP as the base joint (MCP-IP-TIP; no DIP).
THUMB_MCP_I, THUMB_IP_I, THUMB_TIP_I = THUMB_MCP, THUMB_IP, THUMB_TIP

# Detection gates / floors.
_MIN_HAND_PX = 24            # landmark bbox side must clear this (px)
_NUM_HANDS = 2               # bound the detector output

# Finger-extension thresholds — within-hand segment ratios (scale-invariant).
# An extended finger has tip-to-PIP spanning ~2 segments vs PIP-to-MCP ~1
# (ratio ~2.0); a folded finger collapses tip near the palm (ratio ~0.5-1.0).
_EXTEND_RATIO = 1.25         # regular fingers
_THUMB_EXTEND_RATIO = 1.10   # shorter thumb chain

# Hand-raised gate (pose2 GOLIATH-308 body reference, scale-invariant).
WRIST_MIN_CONF = 0.5
CORE_MIN_CONF = 0.5
HAND_RAISED_NORM = 0.50      # wrist above shoulder line by >= 0.5 shoulder-widths

# Gesture classes (closed set).
GESTURE_CLASSES = ("open-palm", "fist", "pointing", "relaxed-curl")

# Model asset sha256 (float16 task, staged 2026-08-10).
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"

# Model asset (bind the sha256 in the declaration; path injected by caller).
HAND_GESTURE_MODEL_ASSET = "/mnt/nas-ai-models/research/stratum/models/hand-gesture/hand_landmarker.task"


class HandGestureError(RuntimeError):
    pass


def validate_rgb_array(rgb: np.ndarray) -> None:
    if not isinstance(rgb, np.ndarray):
        raise HandGestureError("rgb must be a numpy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise HandGestureError(f"rgb must be (H, W, 3), got shape {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise HandGestureError(f"rgb must be uint8, got dtype {rgb.dtype}")


def validate_pose2_array(pose: np.ndarray) -> None:
    if not isinstance(pose, np.ndarray):
        raise HandGestureError("pose2 must be a numpy array")
    if pose.shape == (1, 308, 3):
        pose = pose[0]
    if pose.shape != (308, 3):
        raise HandGestureError(f"pose2 must be shape (308,3) or (1,308,3), got {pose.shape}")


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


class _HandLandmarkerRuntime:
    """Lazy, process-wide MediaPipe HandLandmarker (CPU, tasks API)."""

    _landmarker = None

    @classmethod
    def get(cls, model_asset_path: str):
        if cls._landmarker is None:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                HandLandmarker,
                HandLandmarkerOptions,
                RunningMode,
            )
            cls._landmarker = HandLandmarker.create_from_options(
                HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_asset_path),
                    running_mode=RunningMode.IMAGE,
                    num_hands=_NUM_HANDS,
                    min_hand_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )
        return cls._landmarker

    @classmethod
    def reset(cls) -> None:
        cls._landmarker = None


def _to_mediapipe_image(arr: np.ndarray):
    from mediapipe import Image as MpImage
    from mediapipe.tasks.python.vision.core.image import ImageFormat
    return MpImage(image_format=ImageFormat.SRGB, data=arr)


def _detect_hands(arr: np.ndarray, model_asset_path: str) -> list[Any]:
    """Return the list of detected hand landmark sets (each 21 points)."""
    try:
        res = _HandLandmarkerRuntime.get(model_asset_path).detect(
            _to_mediapipe_image(arr)
        )
    except Exception as exc:  # noqa: BLE001
        raise HandGestureError(f"hand landmarker invocation failed: {exc!r}") from exc
    if not res or not getattr(res, "hand_landmarks", None):
        return []
    return list(res.hand_landmarks)


def _finger_extended(hand: np.ndarray, mcp: int, pip: int, dip: int, tip: int,
                     ratio: float) -> bool:
    """Scale-invariant extension test on the within-hand segment chain.

    dist(tip, PIP) vs dist(PIP, MCP): extended fingers keep the tip far
    beyond the PIP joint (ratio ~2 segment lengths); folded fingers collapse
    tip toward the palm (ratio ~0.5-1.0). The threshold is a within-hand
    ratio, so it is invariant to hand size / camera distance.
    """
    base = _euclid(hand[pip], hand[mcp])
    if base <= 1e-9:
        return False
    return _euclid(hand[tip], hand[pip]) > ratio * base


def _thumb_extended(hand: np.ndarray, mcp: int, ip: int, tip: int,
                    ratio: float) -> bool:
    """Scale-invariant thumb extension test (MCP-IP-TIP chain, no DIP)."""
    base = _euclid(hand[ip], hand[mcp])
    if base <= 1e-9:
        return False
    return _euclid(hand[tip], hand[ip]) > ratio * base


def _gesture_class(hand: np.ndarray) -> tuple[str, dict[str, bool | int]]:
    """Deterministic gesture class from the finger-extension vector."""
    ext: dict[str, bool | int] = {}
    for name, mcp, pip, dip, tip in _FINGERS:
        ext[name] = _finger_extended(hand, mcp, pip, dip, tip, _EXTEND_RATIO)
    ext["thumb"] = _thumb_extended(hand, THUMB_MCP_I, THUMB_IP_I, THUMB_TIP_I,
                                   _THUMB_EXTEND_RATIO)

    fingers = [ext[n] for n in ("index", "middle", "ring", "pinky")]
    n_ext = sum(fingers)
    if all(fingers) and ext["thumb"]:
        cls = "open-palm"
    elif n_ext == 0 and not ext["thumb"]:
        cls = "fist"
    elif n_ext == 0 and ext["thumb"]:
        cls = "relaxed-curl"          # thumb-up variant
    elif n_ext == 1 and ext["index"]:
        cls = "pointing"
    else:
        cls = "relaxed-curl"          # peace sign / partial curl / other
    pay: dict[str, bool | int] = {name: bool(v) for name, v in ext.items()}
    pay["fingers_extended_count"] = int(n_ext)
    return cls, pay


def _pose2_pt(pose: np.ndarray, name: str) -> tuple[float, float] | None:
    from stratum2.config import GOLIATH_308
    idx = {n: i for i, n in enumerate(GOLIATH_308)}[name]
    x, y, conf = float(pose[idx, 0]), float(pose[idx, 1]), float(pose[idx, 2])
    if x < 0 or y < 0 or conf < CORE_MIN_CONF:
        return None
    return (x, y)


def _hand_raised(pose: np.ndarray, hand_center: tuple[float, float], side: str) -> dict[str, Any]:
    """Body-relative hand-raised flag from pose2 GOLIATH-308 (scale-invariant).

    hand_center is in full-frame pixel coordinates. The wrist of the matched
    side, when reported with confidence, anchors the hand; the shoulder line
    and shoulder width provide the scale-invariant body reference. Y grows
    downward in image space, so "above" is a smaller y.
    """
    ls = _pose2_pt(pose, "left_shoulder")
    rs = _pose2_pt(pose, "right_shoulder")
    if ls is None or rs is None:
        return {"measured": False, "reason": "shoulders unreported (conf < 0.5)"}
    shoulder_w = float(np.hypot(rs[0] - ls[0], rs[1] - ls[1]))
    if shoulder_w <= 1e-6:
        return {"measured": False, "reason": "degenerate shoulder width"}
    wrist = _pose2_pt(pose, f"{side}_wrist")
    if wrist is None:
        return {"measured": False, "reason": f"{side} wrist unreported (conf < 0.5)"}
    shoulder_y = (ls[1] + rs[1]) / 2.0
    above = (shoulder_y - hand_center[1]) / shoulder_w
    raised = above >= HAND_RAISED_NORM
    return {
        "measured": True,
        "hand_raised": bool(raised),
        "wrist_above_shoulder_norm": round(float(above), 4),
        "side": side,
    }


def _match_sides(pose: np.ndarray, centers: list[tuple[float, float]]) -> list[str | None]:
    """Match detected hands to pose2 wrist sides (greedy, distinct).

    Each detected hand is assigned the nearest-unclaimed reported wrist. If
    more hands than wrists remain, the leftover hand takes the opposite side
    of the single assigned one (deterministic; body-frame x-order is not
    reliable across facing directions). No wrists reported -> all None.
    """
    wrists: dict[str, tuple[float, float]] = {}
    for side in ("left", "right"):
        w = _pose2_pt(pose, f"{side}_wrist")
        if w is not None:
            wrists[side] = w
    n = len(centers)
    sides: list[str | None] = [None] * n
    if not wrists:
        return sides
    remaining = list(wrists.items())
    indices = list(range(n))
    while indices and remaining:
        best: tuple[float, int, int, str] | None = None
        for i in indices:
            for j, (side, w) in enumerate(remaining):
                d = float(np.hypot(w[0] - centers[i][0], w[1] - centers[i][1]))
                if best is None or d < best[0]:
                    best = (d, i, j, side)
        assert best is not None
        _, i, j, side = best
        sides[i] = side
        indices.remove(i)
        remaining.pop(j)
    assigned = [s for s in sides if s is not None]
    if len(assigned) == 1 and any(s is None for s in sides):
        opposite = {"left": "right", "right": "left"}[assigned[0]]
        for i in range(n):
            if sides[i] is None:
                sides[i] = opposite
    return sides


def compute_hand_gesture(
    rgb: np.ndarray,
    pose2: np.ndarray,
    *,
    model_asset_path: str,
) -> dict[str, Any]:
    """Compute scale-invariant hand-gesture facts from the source frame + pose2.

    Args:
        rgb: (H, W, 3) uint8 decoded source pixels (full frame).
        pose2: (308, 3) or (1, 308, 3) GOLIATH-308 keypoints aligned with rgb.
        model_asset_path: absolute path to the frozen hand_landmarker.task.

    Returns a dict with ``abstained`` / ``abstention_reason``, the item
    gesture claim, per-hand payloads, the one-vs-two-hands count, and
    per-hand raised flags. Absolute landmark coordinates stay payload-only.
    """
    validate_rgb_array(rgb)
    validate_pose2_array(pose2)
    if pose2.shape == (1, 308, 3):
        pose2 = pose2[0]

    # Resolution policy: full frame first; on zero hands retry a 2x LANCZOS
    # upscale (MediaPipe Hands is resolution-sensitive on portrait crops).
    arr = np.ascontiguousarray(rgb)
    hands = _detect_hands(arr, model_asset_path)
    via = "full_frame"
    if not hands:
        from PIL import Image
        h, w = rgb.shape[0], rgb.shape[1]
        up = np.ascontiguousarray(np.asarray(
            Image.fromarray(rgb).resize((w * 2, h * 2), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        ))
        hands_up = _detect_hands(up, model_asset_path)
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
            "hands_detected": 0,
            "via": via,
        }

    img_h, img_w = rgb.shape[0], rgb.shape[1]
    candidates: list[tuple[Any, tuple[float, float]]] = []
    for i, hand in enumerate(hands):
        pts = np.array([(p.x, p.y, p.z) for p in hand])
        xs = pts[:, 0] * img_w
        ys = pts[:, 1] * img_h
        bbox_w, bbox_h = xs.max() - xs.min(), ys.max() - ys.min()
        if bbox_w < _MIN_HAND_PX or bbox_h < _MIN_HAND_PX:
            continue  # degenerate mesh: ignore this hand
        candidates.append((hand, (float(xs.mean()), float(ys.mean()))))

    if not candidates:
        return {
            "abstained": True,
            "abstention_reason": "detected hands are degenerate (bbox below the size floor)",
            "hands_detected": len(hands),
            "via": via,
        }

    sides = _match_sides(pose2, [c[1] for c in candidates])
    per_hand: list[dict[str, Any]] = []
    for i, (hand, center) in enumerate(candidates):
        pts = np.array([(p.x, p.y, p.z) for p in hand])
        xs = pts[:, 0] * img_w
        ys = pts[:, 1] * img_h
        cls, ext_pay = _gesture_class(pts[:, :2])
        side = sides[i]
        raised = _hand_raised(pose2, center, side) if side else {
            "measured": False, "reason": "no pose2 wrist match"
        }
        per_hand.append({
            "index": i,
            "side": side,
            "gesture_class": cls,
            "hand_raised": raised,
            "hand_bbox_px": [round(float(xs.min())), round(float(ys.min())),
                             round(float(xs.max())), round(float(ys.max()))],
            "palm_center_px": [round(center[0]), round(center[1])],
            "extension": ext_pay,
        })

    out: dict[str, Any] = {
        "abstained": False,
        "hands_detected": len(per_hand),
        # One-vs-two-hands is measured but SILENCED to payload-only: the
        # 2026-08-10 frozen-cohort probe measured one-hand 13 / two-hands 1
        # (max_share 0.929 >= 0.75 degeneracy gate), so the count axis does
        # not discriminate on this cohort and never enters prose.
        "via": via,
        "per_hand": per_hand,
    }

    # Item gesture claim: verbalize both hands when both are measured and
    # differ; the highest-confidence side otherwise (side from the pose2
    # wrist match; a matched-side order keeps the claim deterministic).
    def _key(h: dict[str, Any]) -> str:
        return h["side"] or f"hand-{h['index']}"

    ordered = sorted(per_hand, key=_key)
    if len(ordered) == 2:
        out["gesture_claim"] = (
            f"{ordered[0]['side'] or 'one'} hand {ordered[0]['gesture_class']}, "
            f"{ordered[1]['side'] or 'other'} hand {ordered[1]['gesture_class']}"
        )
    else:
        h0 = ordered[0]
        side_txt = f"{h0['side']} " if h0["side"] else ""
        out["gesture_claim"] = f"{side_txt}hand {h0['gesture_class']}"
    out["gesture_band"] = ordered[0]["gesture_class"]
    out["raised_flags"] = [
        {"side": h.get("side"), "hand_raised": h["hand_raised"].get("hand_raised"),
         "measured": h["hand_raised"].get("measured")}
        for h in ordered
    ]
    return out


def render_hand_gesture(config: Mapping[str, Any] | None) -> list[str]:
    """Scale-invariant hand-gesture claims for the dossier (arm #109)."""
    if not config:
        return []
    if config.get("abstained"):
        reason = config.get("abstention_reason") or "hand gesture not measurable"
        return [f"hand-gesture: abstain ({reason})"]
    lines: list[str] = []
    claim = config.get("gesture_claim")
    if claim:
        lines.append(f"hand-gesture: {claim}")
    for f in config.get("raised_flags") or []:
        if not f.get("measured"):
            continue
        side = f.get("side") or "a"
        if f.get("hand_raised"):
            lines.append(f"hand-gesture: the {side} hand is raised above the shoulder line")
    if not lines:
        lines.append("hand-gesture: hands visible but no distinctive gesture claim")
    return lines