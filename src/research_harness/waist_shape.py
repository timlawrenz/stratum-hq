"""Deterministic waist-shape / hip-waist ratio measurement from seg2 + pose2.

Arm #106 (registered 2026-08-10, capability-blocked by the v1 probe). This is
the **v2 limb-excluded torso-profile estimator** selected by the recorded
decision (a) on issue #106 ("develop a better waist estimator — limb-excluded
torso profile") under the open-world sourcing directive. GOLIATH-308 has no
waist keypoint, so the waist row is located deterministically from the
DOME-29 torso/clothing strip between the pose2 hip line and shoulder line.

v2 measurement (all pre-registered before the v2 probe, 2026-08-10):
- strip   = seg2 pixels of {Torso, Upper_Clothing, Lower_Clothing}
- band    = rows [y_shoulder_mid .. y_hip_mid] (pose2, conf >= 0.5)
- width(y)= x-hull of the strip at row y (max-min+1); run(y) = longest
  contiguous strip run in the row.
- occlusion guards (the v1 failure modes, measured on real pixels):
  (a) arm-dominated rows: arm/hand-class px >= 0.5 * (strip+arm px) -> reject
      (arms covering the waist shrink the torso strip to a sliver, e.g.
      1528uw min row 57px with 130 arm px);
  (b) fragmented strips: run(y) < 0.75 * width(y) -> reject (edge occlusion
      splits the strip, e.g. 0yo0gx rows with run 7-40 vs hull 80-145);
  (c) sliver guard: in the waist sub-band only rows with
      width(y) >= 0.55 * hip_w survive (a genuine waist is never less than
      ~55% of hip width; catches the 0.1-0.35-ratio neck/arm slivers).
- hip row   = argmax width(y) over clean rows in [y_hip_mid - 0.30*T, y_hip_mid]
- waist row = argmin width(y) over clean rows in [y_shoulder_mid + 0.40*T,
              y_hip_mid - 0.12*T]  (safely below the bust line / above the
              hip flare; excludes the neck-row minima near the band top that
              corrupted v1, e.g. 03r1r 109px neck rows vs 163px true waist)
- hip_waist_ratio = hip_w / waist_w (scale-invariant; both in px in payload)
- plane gate on the shoulder/hip segments (<=45 deg from horizontal) and the
  human-plausible band [0.7, 2.4] follow research_harness.proportions
  semantics exactly (owner directive — never weakened). NOTE: the sliver
  guard (c) implies measured ratios are always within ~[1.0, 1.82]
  (waist >= 0.55*hip, and the hip sub-band holds the widest clean row), so
  the [0.7, 2.4] plausibility branch is belt-and-suspenders by construction
  (pinned by test_measured_ratio_always_within_sliver_implied_bounds).
- verbalized bands (pre-registered cuts):
    hip:waist < 1.10                      -> straight
    1.10 <= hip:waist < 1.30              -> moderate
    hip:waist >= 1.30 and should:hip<0.90 -> wider-hips
    hip:waist >= 1.30 and should:hip>=0.90 -> hourglass
  (shoulder:hip via pose2; disambiguates hourglass vs pear, both of which
  share a large hip:waist; body-type #32 measures shoulder:hip alone and
  cannot separate them).

ONLY scale-invariant ratios/bands are verbalized; raw px widths stay in the
machine-readable evidence_payload. Abstention is per-item with a recorded
reason; never fabricate a waist class. CPU-only, in-memory, no corpus write.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from stratum2.config import DOME_29, GOLIATH_308

# --- DOME-29 classes (frozen index table) ---
_TORSO = DOME_29.index("Torso")                    # 22
_UPPER_CLOTHING = DOME_29.index("Upper_Clothing")  # 23
_LOWER_CLOTHING = DOME_29.index("Lower_Clothing")  # 13
_ARM_HAND_CLASSES = frozenset(
    DOME_29.index(n) for n in (
        "Left_Lower_Arm", "Right_Lower_Arm",
        "Left_Upper_Arm", "Right_Upper_Arm",
        "Left_Hand", "Right_Hand",
    )
)
TORSO_CLASSES = frozenset({_TORSO, _UPPER_CLOTHING, _LOWER_CLOTHING})

# --- GOLIATH-308 joints ---
_GOLIATH_INDEX = {name: i for i, name in enumerate(GOLIATH_308)}
_JOINTS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")

# --- Pre-registered v2 protocol constants ---
MIN_CONF = 0.5            # joint reliability floor (proportions semantics)
PLANE_MAX_ANGLE = 45.0    # degrees from horizontal (proportions semantics)
MIN_BAND_PX = 60          # torso band height floor (px)
HIP_BAND_FRACTION = 0.30  # hip sub-band: bottom 30% of the torso band
WAIST_TOP_FRACTION = 0.40  # waist sub-band starts 40% below the shoulder line
WAIST_BOTTOM_FRACTION = 0.12  # waist sub-band ends 12% above the hip line
MIN_CLEAN_FRACTION = 0.20  # >=20% of a sub-band's rows must be clean
FRAG_RUN_FRACTION = 0.75  # run/hull coherence floor (occlusion guard b)
ARM_DOMINANCE = 0.50      # arm px >= 50% of (strip+arm) px -> reject (guard a)
SLIVER_FRACTION = 0.55    # waist row width must be >= 55% of hip_w (guard c)
RATIO_PLAUSIBLE_MIN = 0.7
RATIO_PLAUSIBLE_MAX = 2.4
BAND_STRAIGHT_MAX = 1.10        # hip:waist < 1.10 -> straight
BAND_MODERATE_MAX = 1.30        # [1.10, 1.30) -> moderate
WIDER_HIPS_SHOULDER_HIP = 0.90  # hip:waist >= 1.30 and shoulder:hip < 0.90


class WaistShapeError(RuntimeError):
    pass


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise WaistShapeError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise WaistShapeError(f"seg2 must be two-dimensional, got shape {seg2.shape}")


def validate_pose2_array(pose: np.ndarray) -> None:
    if not isinstance(pose, np.ndarray):
        raise WaistShapeError("pose2 must be a numpy array")
    if pose.shape == (1, 308, 3):
        pose = pose[0]
    if pose.shape != (308, 3):
        raise WaistShapeError(f"pose2 must be shape (308,3) or (1,308,3), got {pose.shape}")


def _joint(pose: np.ndarray, name: str) -> tuple[float, float, float] | None:
    idx = _GOLIATH_INDEX[name]
    x, y, conf = (float(pose[idx, 0]), float(pose[idx, 1]), float(pose[idx, 2]))
    if x < 0 or y < 0 or conf < MIN_CONF:
        return None
    return (x, y, conf)


def _seg_angle_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.degrees(np.arctan2(abs(a[1] - b[1]), abs(a[0] - b[0]))))


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _row_profile(seg2: np.ndarray, y: int) -> tuple[int, int, int, int]:
    """Return (hull, run, strip_px, arm_px) for one row of the strip mask."""
    m = seg2[y, :]
    strip = np.isin(m, list(TORSO_CLASSES))
    strip_px = int(strip.sum())
    if strip_px == 0:
        return (0, 0, 0, int(np.isin(m, list(_ARM_HAND_CLASSES)).sum()))
    xs = np.where(strip)[0]
    hull = int(xs[-1] - xs[0] + 1)
    d = np.diff(xs)
    breaks = np.where(d > 1)[0]
    run = int(np.diff(np.concatenate(([-1], breaks, [len(xs) - 1]))).max())
    arm_px = int(np.isin(m, list(_ARM_HAND_CLASSES)).sum())
    return (hull, run, strip_px, arm_px)


def compute_waist_shape(seg2: np.ndarray, pose2: np.ndarray) -> dict[str, Any]:
    """Compute the v2 limb-excluded hip:waist ratio band (or abstain)."""
    validate_seg2_array(seg2)
    validate_pose2_array(pose2)
    if pose2.shape == (1, 308, 3):
        pose2 = pose2[0]
    H = int(seg2.shape[0])

    out: dict[str, Any] = {
        "subject_present": True,
        "abstained": True,
        "abstention_reason": None,
        "waist_shape_band": None,
        "hip_waist_ratio": None,
        "hip_w_px": None,
        "waist_w_px": None,
        "hip_row_y": None,
        "waist_row_y": None,
        "shoulder_hip_ratio": None,
        "seg_angles_deg": None,
    }

    pts = {}
    for name in _JOINTS:
        p = _joint(pose2, name)
        if p is None:
            out["abstention_reason"] = "hip/shoulder low conf"
            return out
        pts[name] = p
    ls, rs = pts["left_shoulder"], pts["right_shoulder"]
    lh, rh = pts["left_hip"], pts["right_hip"]

    a_s = _seg_angle_deg(ls[:2], rs[:2])
    a_h = _seg_angle_deg(lh[:2], rh[:2])
    out["seg_angles_deg"] = [round(a_s, 1), round(a_h, 1)]
    if a_s > PLANE_MAX_ANGLE or a_h > PLANE_MAX_ANGLE:
        out["abstention_reason"] = (
            f"plane-mix s{int(a_s)} h{int(a_h)} — width ratio not a body measure")
        return out

    y_s = (ls[1] + rs[1]) / 2.0
    y_h = (lh[1] + rh[1]) / 2.0
    band = y_h - y_s
    if band < MIN_BAND_PX:
        out["abstention_reason"] = f"torso region too short ({band:.0f}px)"
        return out
    if y_s < 0 or y_h >= H:
        out["abstention_reason"] = "torso band out of frame"
        return out

    shoulder_hip = (_dist(ls[:2], rs[:2]) + 1.0) / (_dist(lh[:2], rh[:2]) + 1.0)
    out["shoulder_hip_ratio"] = round(shoulder_hip, 4)

    hip_top = int(y_h - HIP_BAND_FRACTION * band)
    hip_bot = int(y_h)
    waist_top = int(y_s + WAIST_TOP_FRACTION * band)
    waist_bot = int(y_h - WAIST_BOTTOM_FRACTION * band)

    def clean_rows(y0: int, y1: int) -> list[tuple[int, int]]:
        rows = []
        for y in range(y0, y1 + 1):
            hull, run, strip_px, arm_px = _row_profile(seg2, y)
            if hull <= 0:
                continue
            if arm_px >= ARM_DOMINANCE * (strip_px + arm_px):
                continue  # guard (a): arm-dominated row
            if run < FRAG_RUN_FRACTION * hull:
                continue  # guard (b): fragmented strip
            rows.append((y, hull))
        return rows

    # --- hip width from the lower sub-band ---
    hip_rows = clean_rows(hip_top, hip_bot)
    if len(hip_rows) < MIN_CLEAN_FRACTION * (hip_bot - hip_top + 1):
        out["abstention_reason"] = "hip region occluded or unusable"
        return out
    hip_y, hip_w = max(hip_rows, key=lambda t: t[1])
    out["hip_w_px"] = hip_w
    out["hip_row_y"] = hip_y

    # --- waist width from the mid sub-band with the sliver guard ---
    waist_rows = []
    for y, hull in clean_rows(waist_top, waist_bot):
        if hull >= SLIVER_FRACTION * hip_w:  # guard (c)
            waist_rows.append((y, hull))
    if len(waist_rows) < MIN_CLEAN_FRACTION * (waist_bot - waist_top + 1):
        out["abstention_reason"] = "waist region occluded or unusable"
        return out
    waist_y, waist_w = min(waist_rows, key=lambda t: t[1])
    out["waist_w_px"] = waist_w
    out["waist_row_y"] = waist_y

    ratio = hip_w / waist_w
    if not (RATIO_PLAUSIBLE_MIN <= ratio <= RATIO_PLAUSIBLE_MAX):
        out["abstention_reason"] = (
            f"implausible ratio {ratio:.2f} outside human band "
            f"[{RATIO_PLAUSIBLE_MIN}, {RATIO_PLAUSIBLE_MAX}] — projection artifact")
        return out
    out["hip_waist_ratio"] = round(ratio, 4)

    if ratio < BAND_STRAIGHT_MAX:
        out["waist_shape_band"] = "straight"
    elif ratio < BAND_MODERATE_MAX:
        out["waist_shape_band"] = "moderate"
    elif shoulder_hip < WIDER_HIPS_SHOULDER_HIP:
        out["waist_shape_band"] = "wider-hips"
    else:
        out["waist_shape_band"] = "hourglass"

    out["abstained"] = False
    out["abstention_reason"] = None
    return out


def render_waist_shape(config: Mapping[str, Any]) -> list[str]:
    """Scale-invariant waist-shape claim for the dossier (arm #106 v2)."""
    if not config:
        return []
    if config.get("abstained"):
        reason = config.get("abstention_reason") or "waist not measurable"
        return [f"waist-shape: abstain ({reason})"]
    band = config.get("waist_shape_band")
    if band == "straight":
        return ["waist-shape: waist is not markedly narrower than the hips (straight silhouette)"]
    if band == "moderate":
        return ["waist-shape: waist is moderately narrower than the hips"]
    if band == "hourglass":
        return ["waist-shape: waist is markedly narrower than the hips while shoulders and hips are comparable (hourglass silhouette)"]
    if band == "wider-hips":
        return ["waist-shape: hips are markedly wider than both the waist and the shoulders (pear / wider-hips silhouette)"]
    return []