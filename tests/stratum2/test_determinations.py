from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stratum2.config import GOLIATH_308, DOME_29
# We will import the actual processor once it's written
# from stratum2.pipeline.determinations import process


def make_synthetic_pose2(
    num_persons: int = 1, pose_type: str = "standing"
) -> np.ndarray:
    """Create a synthetic pose2.npy array (N, 308, 3) for testing."""
    pose = np.zeros((num_persons, 308, 3), dtype=np.float32)
    if num_persons == 0:
        return pose

    for i in range(num_persons):
        # Base setup: centered, full-confidence
        pose[i, :, 2] = 0.9  # 90% confidence default

        # Place key points (image coords: +x right, +y down)
        # Center of frame is roughly 500, 500

        # Default: standing upright
        neck_idx = GOLIATH_308.index("neck")
        lhip_idx = GOLIATH_308.index("left_hip")
        rhip_idx = GOLIATH_308.index("right_hip")
        lsho_idx = GOLIATH_308.index("left_shoulder")
        rsho_idx = GOLIATH_308.index("right_shoulder")
        lwri_idx = GOLIATH_308.index("left_wrist")
        rwri_idx = GOLIATH_308.index("right_wrist")

        if pose_type == "standing":
            pose[i, neck_idx, :2] = [500, 200]
            pose[i, lsho_idx, :2] = [400, 250]
            pose[i, rsho_idx, :2] = [600, 250]
            pose[i, lhip_idx, :2] = [450, 600]
            pose[i, rhip_idx, :2] = [550, 600]
            # Arms down
            pose[i, lwri_idx, :2] = [380, 550]
            pose[i, rwri_idx, :2] = [620, 550]

        elif pose_type == "inverted":
            pose[i, neck_idx, :2] = [500, 800]
            pose[i, lhip_idx, :2] = [450, 400]
            pose[i, rhip_idx, :2] = [550, 400]

        elif pose_type == "horizontal":
            pose[i, neck_idx, :2] = [200, 500]
            pose[i, lhip_idx, :2] = [600, 450]
            pose[i, rhip_idx, :2] = [600, 550]

        elif pose_type == "arms_raised":
            pose[i, neck_idx, :2] = [500, 400]
            pose[i, lsho_idx, :2] = [400, 450]
            pose[i, lhip_idx, :2] = [450, 800]
            pose[i, rhip_idx, :2] = [550, 800]
            # Wrist above shoulder
            pose[i, lwri_idx, :2] = [400, 200]
            pose[i, rwri_idx, :2] = [600, 200]

    return pose


def make_synthetic_seg2(mask_type: str = "full_body", shape=(1000, 1000)) -> np.ndarray:
    """Create a synthetic seg2.npy array (H, W) uint8."""
    seg = np.zeros(shape, dtype=np.uint8)

    torso_idx = DOME_29.index("Torso")
    face_idx = DOME_29.index("Face_Neck")

    if mask_type == "full_body":
        seg[100:900, 300:700] = torso_idx
        seg[50:150, 400:600] = face_idx
    elif mask_type == "face_only":
        seg[0:1000, 0:1000] = face_idx

    return seg


def setup_fixture_dir(
    tmp_path: Path, num_persons=1, pose_type="standing", mask_type="full_body"
):
    """Writes synthetic arrays to tmp_path and returns the path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    pose2 = make_synthetic_pose2(num_persons, pose_type)
    seg2 = make_synthetic_seg2(mask_type)

    np.save(str(tmp_path / "pose2.npy"), pose2)
    np.save(str(tmp_path / "seg2.npy"), seg2)

    return tmp_path


# ==========================================
# TESTS
# ==========================================


def test_determinations_single_subject_anomaly(tmp_path):
    from stratum2.pipeline.determinations import process

    # Test N=0
    d0 = setup_fixture_dir(tmp_path / "d0", num_persons=0)
    process(image_path=Path("dummy.jpg"), output_dir=d0)
    res0 = json.loads((d0 / "determinations.json").read_text())
    assert res0["subject"]["n_detections"] == 0
    assert res0["subject"]["detector_anomaly"] == "no_detection"

    # Test N=2
    d2 = setup_fixture_dir(tmp_path / "d2", num_persons=2)
    process(image_path=Path("dummy.jpg"), output_dir=d2)
    res2 = json.loads((d2 / "determinations.json").read_text())
    assert res2["subject"]["n_detections"] == 2
    assert res2["subject"]["detector_anomaly"] == "extra_detections(2)"


def test_determinations_geometry_upright_deg(tmp_path):
    from stratum2.pipeline.determinations import process

    # Standing
    d_stand = setup_fixture_dir(tmp_path / "stand", pose_type="standing")
    process(image_path=Path("dummy.jpg"), output_dir=d_stand)
    r_stand = json.loads((d_stand / "determinations.json").read_text())
    assert 0 <= r_stand["orientation"]["upright_deg"] <= 10  # roughly 0

    # Inverted
    d_inv = setup_fixture_dir(tmp_path / "inv", pose_type="inverted")
    process(image_path=Path("dummy.jpg"), output_dir=d_inv)
    r_inv = json.loads((d_inv / "determinations.json").read_text())
    assert 170 <= r_inv["orientation"]["upright_deg"] <= 190  # roughly 180

    # Horizontal
    d_horiz = setup_fixture_dir(tmp_path / "horiz", pose_type="horizontal")
    process(image_path=Path("dummy.jpg"), output_dir=d_horiz)
    r_horiz = json.loads((d_horiz / "determinations.json").read_text())
    assert 80 <= r_horiz["orientation"]["upright_deg"] <= 100  # roughly 90


def test_determinations_crop_silence(tmp_path):
    from stratum2.pipeline.determinations import process

    # Face only
    d_face = setup_fixture_dir(tmp_path / "face", mask_type="face_only")
    process(image_path=Path("dummy.jpg"), output_dir=d_face)
    r_face = json.loads((d_face / "determinations.json").read_text())

    # Assert silence on relations (no confidently wrong limb relations)
    assert len(r_face.get("relations", [])) == 0

    # But still has extent and visible parts
    parts = [p["part"] for p in r_face["body_parts_visible"]]
    assert "face" in parts
    assert "torso" not in parts


def test_determinations_no_enum_leakage(tmp_path):
    from stratum2.pipeline.determinations import process

    d_stand = setup_fixture_dir(tmp_path / "stand2", pose_type="standing")
    process(image_path=Path("dummy.jpg"), output_dir=d_stand)
    res = json.loads((d_stand / "determinations.json").read_text())

    rel_text = " ".join(res.get("relations", []))
    assert "_" not in rel_text, "Found snake_case token in relations"
    assert "standing" not in rel_text.lower(), "Found banned posture enum in relations"
    assert "seated" not in rel_text.lower(), "Found banned posture enum in relations"
