"""TDD coverage for the eye-color evidence specialist (arm #80).

Deterministic iris-hue band from pose2 GOLIATH-308 iris/pupil keypoints +
source RGB, annulus sampling between pupil and iris borders. Only the coarse
closed-set band is verbalized; raw HSV stats stay payload-only. Pure and
tested without any model; no GPU needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.eye_color import (
    EyeColorError,
    compute_eye_color,
    render_eye_color,
    validate_pose2_array,
)

from stratum2.config import GOLIATH_308

_G = {name: i for i, name in enumerate(GOLIATH_308)}


def _pose_eye(side: str, center=(200.0, 200.0, 0.9), radius=8.0,
              pupil_radius=3.0) -> np.ndarray:
    pose = np.zeros((308, 3), dtype=float)
    pose[:, 2] = 1.0
    for n, ang in ((f"{side}_border_of_iris_3", 0.0),
                   (f"{side}_border_of_iris_6", 90.0),
                   (f"{side}_border_of_iris_9", 180.0),
                   (f"{side}_border_of_iris_12", 270.0)):
        pose[_G[n]] = (center[0] + radius * np.cos(np.radians(ang)),
                       center[1] + radius * np.sin(np.radians(ang)), 0.9)
    for n, ang in ((f"{side}_border_of_pupil_3", 45.0),
                   (f"{side}_border_of_pupil_6", 135.0),
                   (f"{side}_border_of_pupil_9", 225.0),
                   (f"{side}_border_of_pupil_12", 315.0)):
        pose[_G[n]] = (center[0] + pupil_radius * np.cos(np.radians(ang)),
                       center[1] + pupil_radius * np.sin(np.radians(ang)), 0.9)
    pose[_G[f"{side}_center_of_iris"]] = (*center[:2], 0.9)
    pose[_G[f"{side}_center_of_pupil"]] = (*center[:2], 0.9)
    return pose


def _two_eyes() -> np.ndarray:
    """A full pose with both eyes' iris/pupil keypoints defined."""
    pose = np.zeros((308, 3), dtype=float)
    pose[:, 2] = 1.0
    for side, cx in (("l", 200.0), ("r", 240.0)):
        p = _pose_eye(side, center=(cx, 200.0, 0.9))
        for n in GOLIATH_308:
            if n.startswith(f"{side}_") and ("border" in n or "center" in n):
                pose[_G[n]] = p[_G[n]]
    return pose


def _rgb_with_iris(color: tuple[int, int, int]) -> np.ndarray:
    """400x400 RGB: white-ish skin with a brown/dark iris ring at the eyes."""
    rgb = np.full((400, 400, 3), 220, dtype=np.uint8)
    for cy, cx in ((200, 200), (200, 240)):  # (row, col) — right eye at row 200, col 240
        for dy in range(-10, 11):
            for dx in range(-10, 11):
                d = (dx * dx + dy * dy) ** 0.5
                if 3.5 <= d <= 8.5:  # iris annulus band (cover sampler radius range)
                    rgb[cy + dy, cx + dx] = color
    return rgb


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

def test_validate_pose2() -> None:
    with pytest.raises(EyeColorError):
        validate_pose2_array(np.zeros((308, 2)))
    with pytest.raises(EyeColorError):
        validate_pose2_array(np.zeros((200, 3), dtype=float))


# ---------------------------------------------------------------------------
# Measurement + bands
# ---------------------------------------------------------------------------

def test_brown_iris() -> None:
    rgb = _rgb_with_iris((100, 60, 30))  # warm brown
    pose = _two_eyes()
    r = compute_eye_color(pose, rgb)
    assert r["abstained"] is False
    assert r["eye_color_band"] == "brown"
    assert r["sample_count"] >= 8


def test_dark_iris() -> None:
    rgb = _rgb_with_iris((30, 24, 22))
    pose = _two_eyes()
    r = compute_eye_color(pose, rgb)
    assert r["abstained"] is False
    assert r["eye_color_band"] == "dark"


def test_blue_iris() -> None:
    rgb = _rgb_with_iris((70, 110, 180))
    pose = _two_eyes()
    r = compute_eye_color(pose, rgb)
    assert r["abstained"] is False
    assert r["eye_color_band"] == "blue"


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------

def test_no_iris_abstains() -> None:
    pose = np.zeros((308, 3), dtype=float)  # all conf 0, no keypoints
    rgb = _rgb_with_iris((100, 60, 30))
    r = compute_eye_color(pose, rgb)
    assert r["abstained"] is True
    assert r["eye_color_band"] is None
    assert r["abstention_reason"]


def test_closed_eye_cropped_abstains() -> None:
    # Only unreliable (low-conf) iris keypoints -> abstain.
    pose = _two_eyes()
    pose[:, 2] = 0.0  # zero confidence everywhere
    pose[_G["l_center_of_iris"]] = (200.0, 200.0, 0.9)
    pose[_G["r_center_of_iris"]] = (240.0, 200.0, 0.9)
    # no borders have conf -> annulus cannot form
    r = compute_eye_color(pose, _rgb_with_iris((100, 60, 30)))
    assert r["abstained"] is True


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_brown() -> None:
    rgb = _rgb_with_iris((100, 60, 30))
    r = compute_eye_color(_two_eyes(), rgb)
    lines = render_eye_color(r)
    assert any("brown" in ln for ln in lines)


def test_render_not_measured_empty() -> None:
    assert render_eye_color({}) == []


def test_render_abstain() -> None:
    r = {"abstained": True, "abstention_reason": "no iris"}
    lines = render_eye_color(r)
    assert any("abstain" in ln for ln in lines)


def test_render_no_rgb_in_prose() -> None:
    rgb = _rgb_with_iris((100, 60, 30))
    r = compute_eye_color(_two_eyes(), rgb)
    joined = " ".join(render_eye_color(r))
    assert "RGB" not in joined and "HSV" not in joined
    assert "(" not in joined
