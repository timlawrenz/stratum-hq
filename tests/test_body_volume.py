"""TDD coverage for the body-volume measurement module (arm #96, Anny re-scope).

Tests the deterministic, model-independent parts of the module: input
validation, the scale-invariant mesh metric, the band floors, and honest
abstention on the Multi-HMR empty-return path (stubbed — no 1.5GB checkpoint
is loaded in CI). The GPU/model qualification was the capability probe
(2026-08-10: coverage below the pre-registered floor -> arm marked blocked).
"""

from __future__ import annotations

import numpy as np
import pytest

import research_harness.body_volume as bv
from research_harness.body_volume import (
    BodyVolumeError,
    _mesh_metrics,
    compute_body_volume,
    render_body_volume,
    validate_rgb_array,
    validate_seg2_array,
)


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

def test_validate_seg2() -> None:
    with pytest.raises(BodyVolumeError):
        validate_seg2_array(np.zeros((5, 5, 1), dtype=np.uint8))
    with pytest.raises(BodyVolumeError):
        validate_seg2_array(np.zeros((5, 5), dtype=np.float32))


def test_validate_rgb() -> None:
    with pytest.raises(BodyVolumeError):
        validate_rgb_array(np.zeros((5, 5), dtype=np.uint8))
    with pytest.raises(BodyVolumeError):
        validate_rgb_array(np.zeros((5, 5, 3), dtype=np.float32))


def test_misaligned_seg2_rgb_raises() -> None:
    with pytest.raises(BodyVolumeError):
        compute_body_volume(np.zeros((10, 10), dtype=np.uint8),
                            np.zeros((8, 8, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Mesh metric (scale-invariant normalized volume)
# ---------------------------------------------------------------------------

def _trimesh():
    import trimesh  # noqa: F401
    return trimesh


def test_mesh_metrics_box() -> None:
    """A size-1 icosphere: finite mesurable height and positive norm."""
    trimesh = _trimesh()
    b = trimesh.creation.icosphere(subdivisions=3, radius=1.0)  # 642 vertices
    vertices = np.asarray(b.vertices, dtype=np.float64)
    vertices = vertices - vertices.min(axis=0)
    norm, height = _mesh_metrics(vertices, np.asarray(b.faces, dtype=np.int64))
    assert 1.5 < height < 2.5  # diameter ~2, percentile-shrunk
    assert norm is not None and norm > 0


def test_mesh_metrics_scale_invariant() -> None:
    """Scaling a mesh leaves volume/height^3 unchanged (ratio-based)."""
    trimesh = _trimesh()

    def box(s: float):
        b = trimesh.creation.icosphere(subdivisions=3, radius=s)
        vertices = np.asarray(b.vertices, dtype=np.float64)
        vertices = vertices - vertices.min(axis=0)
        return _mesh_metrics(vertices, np.asarray(b.faces, dtype=np.int64))

    n1, h1 = box(1.0)
    n2, h2 = box(3.0)
    assert h2 / h1 == pytest.approx(3.0, abs=1e-2)  # 3x radius -> 3x height
    assert n1 == pytest.approx(n2, abs=1e-3)


def test_mesh_metrics_tiny_mesh_returns_none() -> None:
    v3d = np.zeros((3, 3), dtype=np.float64)
    assert _mesh_metrics(v3d, np.zeros((1, 3), dtype=np.int64)) == (None, None)


# ---------------------------------------------------------------------------
# Empty-return path (Multi-HMR returns the ("{}", []) tuple on zero detections)
# ---------------------------------------------------------------------------

class _DummyEmptyModel:
    """Model returning exactly what Multi-HMR returns for zero detections."""
    def __call__(self, x, **kwargs):
        return ({}, [])

    body_model = type("BM", (), {"faces": np.zeros((0, 4)), "phenotype_labels": []})


def test_empty_return_abstains_honestly(monkeypatch) -> None:
    """({}, []) from the model -> honest abstain, never a fabricated volume."""
    seg2 = np.zeros((200, 200), dtype=np.uint8)
    seg2[50:150, 50:150] = 1
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    monkeypatch.setattr(bv._RunTime, "_model", _DummyEmptyModel())
    monkeypatch.setattr(bv._RunTime, "_device", "cpu")
    monkeypatch.setattr(bv, "MIN_SUBJECT_PX", 1)  # subject present
    r = compute_body_volume(seg2, rgb, checkpoint="does-not-matter")
    assert r["abstained"] is True
    assert "no detector person" in (r["abstention_reason"] or "")
    monkeypatch.setattr(bv._RunTime, "_model", None)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_empty() -> None:
    assert render_body_volume({}) == []


def test_render_abstain() -> None:
    lines = render_body_volume({"abstained": True, "abstention_reason": "no mesh"})
    assert any("abstain" in ln for ln in lines)


def test_render_bands() -> None:
    for band, needle in (("slim", "slender"), ("average", "average"),
                         ("fuller", "fuller")):
        lines = render_body_volume({"body_volume_band": band})
        assert any(needle in ln for ln in lines)


def test_render_no_band_returns_empty() -> None:
    assert render_body_volume({"body_volume_band": None}) == []


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------

def test_band_floors_ordered() -> None:
    assert bv.VOLUME_BANDS is not None
    slim_hi, avg_hi = bv.VOLUME_BANDS
    assert 0 < slim_hi < avg_hi
    assert bv.MIN_MESH_HEIGHT < bv.MAX_MESH_HEIGHT
    assert bv.MIN_NORM_VOLUME < bv.MAX_NORM_VOLUME


def test_detection_threshold_calibrated() -> None:
    # Calibrated 2026-08-10: the checkpoint default (0.3) only recovers 14/24
    # frozen items; 0.1 recovers 21/24 (documented in module + probe).
    assert bv.DET_THRESH == 0.1
