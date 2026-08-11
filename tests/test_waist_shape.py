"""TDD coverage for the waist-shape v2 evidence specialist (arm #106, decision (a)).

Deterministic hip:waist ratio band from the limb-excluded torso profile
(seg2 DOME-29 torso/clothing strip + pose2 GOLIATH-308 gating), scale-
invariant. v2 fixes the v1 failure mode (row-width minima corrupted by
arm/hand pixels) with three pre-registered occlusion guards. Only the
scale-invariant ratio/band is verbalized; raw px widths stay in the payload.
Pure and tested without any model; no GPU needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.waist_shape import (
    WaistShapeError,
    compute_waist_shape,
    render_waist_shape,
    validate_pose2_array,
    validate_seg2_array,
)
from stratum2.config import DOME_29, GOLIATH_308

_TORSO = DOME_29.index("Torso")
_UPPER = DOME_29.index("Upper_Clothing")
_ARM = DOME_29.index("Left_Upper_Arm")
_G = {name: i for i, name in enumerate(GOLIATH_308)}

LS, RS = _G["left_shoulder"], _G["right_shoulder"]
LH, RH = _G["left_hip"], _G["right_hip"]


def _pose(*, y_s: float = 300.0, y_h: float = 760.0, w_s: float = 220.0,
          w_h: float = 170.0, conf: float = 0.95, cx: float = 400.0) -> np.ndarray:
    pose = np.zeros((308, 3), dtype=float)
    pose[:, 2] = conf
    for i, y, w in ((LS, y_s, w_s), (RS, y_s, w_s), (LH, y_h, w_h), (RH, y_h, w_h)):
        pose[i] = (cx - w / 2 if i in (LS, LH) else cx + w / 2, y, conf)
    return pose


def _torso(seg: np.ndarray, *, y_s: int, y_h: int, cx: int,
           shoulder_hw: float, waist_hw: float, hip_hw: float,
           waist_frac: float = 0.55) -> np.ndarray:
    """Paint a Torso strip whose half-width is a linear shoulder->waist->hip profile."""
    band = y_h - y_s
    yw = int(y_s + waist_frac * band)
    for y in range(y_s, y_h + 1):
        if y <= yw:
            t = (y - y_s) / max(1, yw - y_s)
            hw = shoulder_hw + t * (waist_hw - shoulder_hw)
        else:
            t = (y - yw) / max(1, y_h - yw)
            hw = waist_hw + t * (hip_hw - waist_hw)
        hw = max(0, int(hw))
        seg[y, int(cx - hw):int(cx + hw) + 1] = _TORSO
    return seg


def _arm(seg: np.ndarray, *, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    seg[y0:y1 + 1, x0:x1 + 1] = _ARM
    return seg


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

def test_validate_seg2() -> None:
    with pytest.raises(WaistShapeError):
        validate_seg2_array(np.zeros(4))
    with pytest.raises(WaistShapeError):
        validate_seg2_array(np.zeros((3, 3, 3)))


def test_validate_pose2() -> None:
    with pytest.raises(WaistShapeError):
        validate_pose2_array(np.zeros((308, 4)))
    with pytest.raises(WaistShapeError):
        validate_pose2_array(np.zeros((5, 3)))


# ---------------------------------------------------------------------------
# Abstention gates (G1-G3) — same semantics as research_harness.proportions
# ---------------------------------------------------------------------------

def test_abstains_low_conf_joints() -> None:
    seg = _torso(np.zeros((1000, 800), dtype=np.uint8), y_s=300, y_h=760,
                 cx=400, shoulder_hw=110, waist_hw=70, hip_hw=90)
    pose = _pose()
    pose[LH, 2] = 0.2
    r = compute_waist_shape(seg, pose)
    assert r["abstained"] is True
    assert "low conf" in r["abstention_reason"]


def test_abstains_plane_mix() -> None:
    seg = _torso(np.zeros((1000, 800), dtype=np.uint8), y_s=300, y_h=760,
                 cx=400, shoulder_hw=110, waist_hw=70, hip_hw=90)
    pose = _pose()
    # right hip raised far above the left -> hip segment ~55 deg from horizontal
    pose[RH, 1] = pose[LH, 1] - 260.0
    r = compute_waist_shape(seg, pose)
    assert r["abstained"] is True
    assert "plane-mix" in r["abstention_reason"]


def test_abstains_torso_too_short() -> None:
    seg = _torso(np.zeros((1000, 800), dtype=np.uint8), y_s=300, y_h=340,
                 cx=400, shoulder_hw=110, waist_hw=70, hip_hw=90)
    r = compute_waist_shape(seg, _pose(y_s=300.0, y_h=340.0))
    assert r["abstained"] is True
    assert "too short" in r["abstention_reason"]


def test_abstains_band_out_of_frame() -> None:
    seg = np.zeros((1000, 800), dtype=np.uint8)
    _torso(seg, y_s=900, y_h=999, cx=400, shoulder_hw=110, waist_hw=70, hip_hw=90)
    # valid joints, but the hip line lies beyond the seg frame
    r = compute_waist_shape(seg, _pose(y_s=900.0, y_h=1100.0))
    assert r["abstained"] is True
    assert "out of frame" in r["abstention_reason"]


# ---------------------------------------------------------------------------
# v2 occlusion guards (the v1 corruption modes)
# ---------------------------------------------------------------------------

def test_occlusion_guard_arm_dominated_rows_do_not_become_the_waist() -> None:
    """An arm plank over the waist rows must not shrink the measured waist."""
    seg = np.zeros((1000, 800), dtype=np.uint8)
    _torso(seg, y_s=300, y_h=760, cx=400, shoulder_hw=110, waist_hw=70, hip_hw=100)
    # arm covers the RIGHT half of the strip at the waist rows (~520..580):
    # rows keep a torso remnant (~76px) but are arm-dominated (guard a).
    _arm(seg, y0=520, y1=580, x0=405, x1=560)
    r = compute_waist_shape(seg, _pose())
    assert r["abstained"] is False, r["abstention_reason"]
    # v1 would have read the ~76px remnant as the waist; v2 must use the
    # clean rows near the true waist (~140-150px)
    assert 120 <= r["waist_w_px"] <= 170
    assert r["hip_w_px"] >= 160


def test_occlusion_guard_fragmented_strip_rejected() -> None:
    """A forearm over the strip middle splits the run -> row rejected."""
    seg = np.zeros((1000, 800), dtype=np.uint8)
    _torso(seg, y_s=300, y_h=760, cx=400, shoulder_hw=110, waist_hw=70, hip_hw=100)
    # arm covers the MIDDLE of the strip at the waist rows -> run splits
    _arm(seg, y0=540, y1=560, x0=390, x1=450)
    r = compute_waist_shape(seg, _pose())
    assert r["abstained"] is False, r["abstention_reason"]
    assert 120 <= r["waist_w_px"] <= 170
    assert r["hip_w_px"] >= 160


def test_occlusion_guard_sliver_rejected() -> None:
    """A tiny coherent sliver (v1's 61px corruption) must not become the waist."""
    seg = np.zeros((1000, 800), dtype=np.uint8)
    _torso(seg, y_s=300, y_h=760, cx=400, shoulder_hw=110, waist_hw=70, hip_hw=100)
    # replace the true waist row with a narrow coherent sliver
    seg[553, 330:471] = 0
    seg[553, 395:406] = _TORSO
    r = compute_waist_shape(seg, _pose())
    assert r["abstained"] is False, r["abstention_reason"]
    assert r["waist_w_px"] >= 110  # sliver (11px) rejected by the 0.55*hip guard
    # and the hip is not the sliver row either
    assert r["hip_w_px"] >= 160


def test_abstains_waist_region_fully_occluded() -> None:
    """Whole mid-band covered by arms -> honest abstention, no fabricated ratio."""
    seg = np.zeros((1000, 800), dtype=np.uint8)
    _torso(seg, y_s=300, y_h=760, cx=400, shoulder_hw=110, waist_hw=70, hip_hw=90)
    _arm(seg, y0=500, y1=700, x0=280, x1=520)
    r = compute_waist_shape(seg, _pose())
    assert r["abstained"] is True
    assert "waist region" in r["abstention_reason"]


# ---------------------------------------------------------------------------
# Band semantics (pre-registered cuts)
# ---------------------------------------------------------------------------

def test_band_straight() -> None:
    seg = np.zeros((1000, 800), dtype=np.uint8)
    _torso(seg, y_s=300, y_h=760, cx=400, shoulder_hw=110, waist_hw=96, hip_hw=100)
    r = compute_waist_shape(seg, _pose())
    assert r["abstained"] is False, r["abstention_reason"]
    assert r["waist_shape_band"] == "straight"


def test_band_moderate() -> None:
    seg = np.zeros((1000, 800), dtype=np.uint8)
    _torso(seg, y_s=300, y_h=760, cx=400, shoulder_hw=120, waist_hw=85, hip_hw=100)
    r = compute_waist_shape(seg, _pose())
    assert r["abstained"] is False, r["abstention_reason"]
    assert r["waist_shape_band"] == "moderate"


def test_band_hourglass_with_comparable_shoulders() -> None:
    seg = np.zeros((1000, 800), dtype=np.uint8)
    _torso(seg, y_s=300, y_h=760, cx=400, shoulder_hw=110, waist_hw=65, hip_hw=100)
    r = compute_waist_shape(seg, _pose(w_s=220.0, w_h=170.0))  # shoulder:hip ~1.29
    assert r["abstained"] is False, r["abstention_reason"]
    assert r["waist_shape_band"] == "hourglass"


def test_band_wider_hips_with_narrow_shoulders() -> None:
    seg = np.zeros((1000, 800), dtype=np.uint8)
    _torso(seg, y_s=300, y_h=760, cx=400, shoulder_hw=110, waist_hw=65, hip_hw=100)
    # narrow shoulders -> pear silhouette
    r = compute_waist_shape(seg, _pose(w_s=150.0, w_h=170.0))
    assert r["abstained"] is False, r["abstention_reason"]
    assert r["waist_shape_band"] == "wider-hips"


@pytest.mark.parametrize("waist_hw,hip_hw", [
    (100, 100), (95, 100), (85, 100), (70, 100), (65, 105), (60, 95),
])
def test_measured_ratio_always_within_sliver_implied_bounds(waist_hw: float, hip_hw: float) -> None:
    """The occlusion guards imply ratio in ~[1.0, 1.82]: the sliver guard
    (waist >= 0.55*hip) caps hip:waist at 1/0.55 and the hip sub-band
    (bottom 30% of the torso band) always holds the widest clean row, so a
    measured ratio must be >= 1.0 and <= 1.82. The owner's [0.7, 2.4]
    plausibility band is therefore enforced by construction (documented in
    the module); this pins the implied bound so future edits cannot silently
    widen or narrow it."""
    seg = np.zeros((1000, 800), dtype=np.uint8)
    _torso(seg, y_s=300, y_h=760, cx=400, shoulder_hw=120,
           waist_hw=waist_hw, hip_hw=hip_hw)
    r = compute_waist_shape(seg, _pose())
    if r["abstained"]:
        return  # abstentions are honest; only measured ratios are bounded
    assert 1.0 <= r["hip_waist_ratio"] <= 1.82


# ---------------------------------------------------------------------------
# Scale invariance + payload discipline
# ---------------------------------------------------------------------------

def test_ratio_scale_invariant_under_resolution_change() -> None:
    def build(scale: int) -> dict:
        seg = np.zeros((1000 * scale, 800 * scale), dtype=np.uint8)
        _torso(seg, y_s=300 * scale, y_h=760 * scale, cx=400 * scale,
               shoulder_hw=110 * scale, waist_hw=65 * scale, hip_hw=100 * scale)
        return compute_waist_shape(seg, _pose(
            y_s=300.0 * scale, y_h=760.0 * scale,
            w_s=220.0 * scale, w_h=170.0 * scale, cx=400.0 * scale))

    r1, r2 = build(1), build(2)
    assert r1["abstained"] is False and r2["abstained"] is False
    # the verbalized band is scale-invariant; the exact ratio digit carries
    # only int-pixel hull quantization (~1px resolution)
    assert r1["waist_shape_band"] == r2["waist_shape_band"]
    assert abs(r1["hip_waist_ratio"] - r2["hip_waist_ratio"]) < 0.02
    # px widths scale with resolution (within rounding) but stay payload-only
    assert abs(r2["waist_w_px"] - 2 * r1["waist_w_px"]) <= 2
    assert abs(r2["hip_w_px"] - 2 * r1["hip_w_px"]) <= 2


def test_px_stays_payload_only_in_render() -> None:
    seg = np.zeros((1000, 800), dtype=np.uint8)
    _torso(seg, y_s=300, y_h=760, cx=400, shoulder_hw=110, waist_hw=65, hip_hw=100)
    r = compute_waist_shape(seg, _pose())
    assert r["abstained"] is False
    claims = render_waist_shape(r)
    assert claims and len(claims) == 1
    assert "px" not in claims[0].lower()
    assert "ratio" not in claims[0]  # only the coarse band is verbalized


def test_render_abstain_and_empty() -> None:
    assert render_waist_shape({}) == []
    seg = np.zeros((1000, 800), dtype=np.uint8)
    r = compute_waist_shape(seg, _pose(y_s=300.0, y_h=340.0))
    assert r["abstained"] is True
    claims = render_waist_shape(r)
    assert claims and "abstain" in claims[0]