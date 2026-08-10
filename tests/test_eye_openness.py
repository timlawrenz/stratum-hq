"""TDD coverage for the eye-openness evidence specialist (arm #113).

Deterministic eyelid-state band (open / lidded / closed) from pose2
GOLIATH-308 eyelid-line keypoints normalized by interpupillary distance,
plus the closed-eye keypoint-dropped signature. Scale-invariant; only the
coarse band is verbalized; raw ratios stay in the payload. Pure and tested
without any model; no GPU needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.eye_openness import (
    EyeOpennessError,
    compute_eye_openness,
    render_eye_openness,
    validate_pose2_array,
)

from stratum2.config import GOLIATH_308

_G = {name: i for i, name in enumerate(GOLIATH_308)}


def _pose(ipd_px: float = 60.0, eye_y: float = 300.0) -> np.ndarray:
    """Default pose: frontal face, both eyes at y=eye_y, confident.

    Eyelid geometry: upper lid points at y=eye_y-10 (open), lower lid points
    at y=eye_y+10 (open) -> max gap 20px; IPD 60px -> ratio ~0.33 (open).
    """
    sf = ipd_px / 60.0  # uniform scale factor derived from the IPD
    ey = eye_y
    pose = np.zeros((308, 3), dtype=float)
    pose[:, 2] = 1.0
    pose[_G["left_eye"]] = (120.0 * sf, ey, 0.95)
    pose[_G["right_eye"]] = (180.0 * sf, ey, 0.95)
    for side, cx in (("l", 135.0), ("r", 165.0)):
        for name in ("upper", "lower"):
            pts = [f"{side}_{end}_of_{name}_eyelid_line" for end in ("outer_end",)] + \
                  [f"{side}_midpoint_{i}_of_{name}_eyelid_line" for i in range(1, 7)] + \
                  [f"{side}_centerpoint_of_{name}_eyelid_line"]
            dy = -10.0 if name == "upper" else 10.0
            for j, n in enumerate(pts):
                pose[_G[n]] = ((cx - 20.0 + j * 5.0) * sf, ey + dy * sf, 0.95)
        pose[_G[f"{side}_center_of_iris"]] = (cx * sf, ey, 0.95)
        for p in (3, 6, 9, 12):
            pose[_G[f"{side}_border_of_iris_{p}"]] = (cx * sf, ey, 0.95)
    return pose


def _pose_closed(ipd_px: float = 60.0, eye_y: float = 300.0) -> np.ndarray:
    """Closed-eye signature: face visible (eye centers + nose present) but
    the eyelid lines and iris keypoints are all dropped by the detector."""
    pose = _pose(ipd_px=ipd_px, eye_y=eye_y)
    # Drop the eyelid-line and iris keypoints (closed eyes collapse the lid
    # lines; GOLIATH drops the whole eye-region keypoint set).
    for name in list(_G):
        if "eyelid" in name or "iris" in name or "pupil" in name:
            pose[_G[name]] = (0.0, 0.0, 0.0)
    return pose


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

def test_validate_pose2() -> None:
    with pytest.raises(EyeOpennessError):
        validate_pose2_array(np.zeros((308, 2)))
    with pytest.raises(EyeOpennessError):
        validate_pose2_array(np.zeros((200, 3), dtype=float))


# ---------------------------------------------------------------------------
# Band classification
# ---------------------------------------------------------------------------

def test_open_eyes_band_open() -> None:
    r = compute_eye_openness(_pose())
    assert r["abstained"] is False
    assert r["eye_openness_band"] == "open"
    assert r["openness_ratio"] > 0.10


def test_lidded_eyes_band_lidded() -> None:
    pose = _pose()
    # Open template: upper lid y=290, lower y=310 (gap 20/60 = 0.33).
    # Squint: lower the upper lid so the gap shrinks to ~5px -> ratio ~0.083,
    # inside (CLOSED_MAX=0.025, LIDDED_MAX=0.10].
    for side in ("l", "r"):
        pts = [f"{side}_{end}_of_upper_eyelid_line" for end in ("outer_end",)] + \
              [f"{side}_midpoint_{i}_of_upper_eyelid_line" for i in range(1, 7)] + \
              [f"{side}_centerpoint_of_upper_eyelid_line"]
        for n in pts:
            pose[_G[n]][1] += 15.0  # upper lid -> y=305, gap = 5px -> 0.083
    r = compute_eye_openness(pose)
    assert r["abstained"] is False
    assert r["eye_openness_band"] == "lidded"


def test_closed_signature_band_closed() -> None:
    r = compute_eye_openness(_pose_closed())
    assert r["abstained"] is False
    assert r["eye_openness_band"] == "closed"
    assert r["openness_ratio"] == 0.0


def test_all_zero_pose_abstains() -> None:
    r = compute_eye_openness(np.zeros((308, 3), dtype=float))
    assert r["abstained"] is True
    assert r["eye_openness_band"] is None


def test_scale_invariance() -> None:
    """Scaling the whole face must keep the same band."""
    def build(sf: float):
        return compute_eye_openness(_pose(ipd_px=60.0 * sf, eye_y=300.0 * sf))

    r_small = build(1.0)
    r_big = build(2.0)
    assert r_small["eye_openness_band"] == r_big["eye_openness_band"]
    assert r_small["openness_ratio"] == pytest.approx(r_big["openness_ratio"], abs=0.02)


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------

def test_single_eye_dropped_uses_other() -> None:
    pose = _pose()
    for n in [f"r_{end}_of_upper_eyelid_line" for end in ("outer_end",)] + \
             [f"r_midpoint_{i}_of_upper_eyelid_line" for i in range(1, 7)] + \
             [f"r_centerpoint_of_upper_eyelid_line"]:
        pose[_G[n]] = (0, 0, 0.0)
    r = compute_eye_openness(pose)
    assert r["abstained"] is False
    assert r["eye_openness_band"] == "open"


def test_both_eyes_dropped_with_face_context_closed() -> None:
    r = compute_eye_openness(_pose_closed())
    assert r["eye_openness_band"] == "closed"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_open() -> None:
    lines = render_eye_openness(compute_eye_openness(_pose()))
    assert any("open" in ln for ln in lines)


def test_render_not_measured_empty() -> None:
    assert render_eye_openness({}) == []


def test_render_abstain() -> None:
    lines = render_eye_openness({"abstained": True, "abstention_reason": "no face"})
    assert any("abstain" in ln for ln in lines)


def test_payload_honest_no_px_in_prose() -> None:
    joined = " ".join(render_eye_openness(compute_eye_openness(_pose())))
    assert "120" not in joined and "300" not in joined  # no raw px in prose