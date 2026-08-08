"""TDD coverage for the environment-clearance evidence specialist (arm #85).

Deterministic subject-to-environment clearance band (tight / moderate /
spacious) from seg2 Background split, scale-invariant (normalized by subject
bbox extent on the same axis). Only the coarse band is verbalized; raw
normalized distances stay in the payload. Pure and tested without any model;
no GPU needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from research_harness.environment_clearance import (
    EnvironmentClearanceError,
    compute_environment_clearance,
    render_environment_clearance,
    validate_seg2_array,
)


def _seg(h: int = 600, w: int = 800) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _mk(r0: int, r1: int, c0: int, c1: int, h: int = 600, w: int = 800) -> np.ndarray:
    seg = np.zeros((h, w), dtype=np.uint8)
    seg[r0:r1, c0:c1] = 4
    return seg


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

def test_validate_seg2() -> None:
    with pytest.raises(EnvironmentClearanceError):
        validate_seg2_array(np.zeros((5, 5, 1), dtype=np.uint8))
    with pytest.raises(EnvironmentClearanceError):
        validate_seg2_array(np.zeros((5, 5), dtype=np.float32))
    with pytest.raises(EnvironmentClearanceError):
        validate_seg2_array("nope")


# ---------------------------------------------------------------------------
# Clearance bands
# ---------------------------------------------------------------------------

def test_tight_horizontal_clearance() -> None:
    """Subject bbox spans the full frame width -> tight (left/right ~0)."""
    seg = _mk(r0=50, r1=550, c0=0, c1=800)  # full width: col 0..799 of 800
    r = compute_environment_clearance(seg)
    assert r["abstained"] is False
    assert r["clearance_band"] == "tight"
    assert r["clearance_ratio"] == pytest.approx(0.0, abs=1e-6)
    assert r["clearance_top"] > 0 and r["clearance_bottom"] > 0  # vertical still nonzero


def test_spacious_horizontal_clearance() -> None:
    """Small subject centered with wide side gaps -> spacious."""
    seg = _mk(r0=100, r1=220, c0=120, c1=240)  # bbox 120x120 centered-ish
    r = compute_environment_clearance(seg)
    assert r["clearance_band"] == "spacious"
    assert r["clearance_ratio"] is not None
    assert r["clearance_ratio"] >= 0.60
    # left/right gaps: (120)/(120)=1.0, (800-1-240)/(120)=4.66 -> median 1.0


def test_moderate_clearance() -> None:
    """Mid ground: subject with recognizable but not huge side gaps."""
    seg = _mk(r0=50, r1=550, c0=200, c1=600)  # bbox 400x400, frame 800 wide
    r = compute_environment_clearance(seg)
    assert r["clearance_band"] in ("moderate", "spacious")
    # left/right gaps: 200/400=0.5, (800-1-600)/400=0.4975 -> median ~0.5


def test_scale_invariant() -> None:
    """Doubling frame + subject keeps the same band and ratio (pure ratio)."""
    small = _mk(r0=50, r1=550, c0=200, c1=600, h=600, w=800)
    r_small = compute_environment_clearance(small)
    big = _mk(r0=100, r1=1100, c0=400, c1=1200, h=1200, w=1600)
    r_big = compute_environment_clearance(big)
    assert r_small["clearance_band"] == r_big["clearance_band"]
    assert r_big["clearance_ratio"] == pytest.approx(
        r_small["clearance_ratio"], abs=1e-3
    )


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------

def test_no_subject_abstains() -> None:
    r = compute_environment_clearance(np.zeros((600, 800), dtype=np.uint8))
    assert r["abstained"] is True
    assert "no foreground subject" in (r["abstention_reason"] or "")


def test_frame_fill_abstains() -> None:
    """Subject fills the whole frame edge-to-edge -> no clearance -> abstain."""
    seg = np.ones((600, 800), dtype=np.uint8) * 4
    r = compute_environment_clearance(seg)
    assert r["abstained"] is True
    assert "edge-to-edge" in (r["abstention_reason"] or "")
    assert r["clearance_band"] is None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_tight() -> None:
    lines = render_environment_clearance(compute_environment_clearance(
        _mk(r0=50, r1=550, c0=10, c1=790)))
    assert any("close" in ln for ln in lines)


def test_render_spacious() -> None:
    lines = render_environment_clearance(compute_environment_clearance(
        _mk(r0=100, r1=220, c0=120, c1=240)))
    assert any("spacious" in ln for ln in lines)


def test_render_not_measured_empty() -> None:
    assert render_environment_clearance({}) == []


def test_render_abstain() -> None:
    r = {"abstained": True, "abstention_reason": "no background clearance"}
    lines = render_environment_clearance(r)
    assert any("abstain" in ln for ln in lines)


def test_render_no_ratio_in_prose() -> None:
    r = compute_environment_clearance(_mk(r0=100, r1=220, c0=120, c1=240))
    joined = " ".join(render_environment_clearance(r))
    assert "0." not in joined  # normalized ratios stay payload-only
