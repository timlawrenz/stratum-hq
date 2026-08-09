"""Regression tests for per-region seg<->pose corroboration.

These pin the design approved by Tim: a region's keypoints are trusted for
relations anchored on that region iff seg2 shows that region's pixels above a
minimal fraction. No cross-part inference (hand survives without torso).
"""

import json
from pathlib import Path

import numpy as np

from stratum2.config import DOME_29, GOLIATH_308
from tests.stratum2.test_determinations import (
    make_synthetic_pose2,
    make_synthetic_pointmap,
)


def _seg2_with_regions(regions: dict[str, tuple[slice, slice]], shape=(1000, 1000)):
    """Build a seg2 array where each named region's classes fill a rect.

    regions: {region_name: (y_slice, x_slice)} using the approved class mapping.
    """
    REGION_CLASSES = {
        "face": [3, 4, 24, 25, 26, 27, 28],
        "torso": [22, 23, 13],
        "left_arm": [11, 7],
        "right_arm": [20, 16],
        "left_hand": [6],
        "right_hand": [15],
        "left_leg": [12, 8, 9, 10],
        "right_leg": [21, 17, 18, 19],
    }
    seg = np.zeros(shape, dtype=np.uint8)
    for region, (ys, xs) in regions.items():
        cls = REGION_CLASSES[region][0]
        seg[ys, xs] = cls
    return seg


def _write_dir(tmp_path: Path, pose2, seg2):
    tmp_path.mkdir(parents=True, exist_ok=True)
    np.save(str(tmp_path / "pose2.npy"), pose2)
    np.save(str(tmp_path / "seg2.npy"), seg2)
    np.save(
        str(tmp_path / "pointmap.npy"),
        make_synthetic_pointmap(shape=seg2.shape, mask_type="full_body"),
    )
    return tmp_path


def test_clothed_shoulder_keeps_torso_and_arm_relation(tmp_path):
    """00000 onesie case: torso present ONLY via Upper_Clothing.

    Torso must be corroborated (shoulders kept), arm relation must fire.
    """
    from stratum2.pipeline.determinations import process

    pose2 = make_synthetic_pose2(1, "standing")
    # Torso present via Upper_Clothing (23); arms present via arm classes;
    # legs present so leg relations also corroborate.
    seg2 = np.zeros((1000, 1000), dtype=np.uint8)
    seg2[50:150, 400:600] = 3  # face
    seg2[150:700, 350:650] = 23  # Upper_Clothing covering torso region
    seg2[250:560, 360:400] = 11  # Left_Upper_Arm
    seg2[560:700, 360:400] = 7  # Left_Lower_Arm
    seg2[250:560, 600:640] = 20  # Right_Upper_Arm
    seg2[560:700, 600:640] = 16  # Right_Lower_Arm
    seg2[700:900, 430:470] = 12  # Left_Upper_Leg
    seg2[700:900, 530:570] = 21  # Right_Upper_Leg
    d = _write_dir(tmp_path / "clothed", pose2, seg2)

    process(image_path=Path("dummy.jpg"), output_dir=d)
    doc = json.loads((d / "determinations.json").read_text())

    parts = {p["part"] for p in doc["body_parts_visible"]}
    assert "torso" in parts, "torso must be reported via Upper_Clothing"
    rels = " ".join(doc["relations"])
    assert "left arm extended downward" in rels
    assert "upright_deg" in doc["orientation"], (
        "upright_deg must fire with torso corroborated"
    )


def test_hand_in_hair_portrait_hand_survives_torso_silent(tmp_path):
    """Hand in hair close-up: Left_Hand pixels present, NO torso/arm pixels.

    Hand-level relations must survive; torso-anchored measurements
    (upright_deg, camera height) and arm relations must go silent.
    """
    from stratum2.pipeline.determinations import process

    pose2 = make_synthetic_pose2(1, "hands_together")
    seg2 = np.zeros((1000, 1000), dtype=np.uint8)
    seg2[0:1000, 0:1000] = 3  # face fills frame
    # Both hand classes present near the wrists; NO torso/arm classes.
    # Each patch is 130x110 = 14,300 px (> 1% of 1,000,000 = 10,000 threshold).
    seg2[450:580, 420:530] = 6  # Left_Hand
    seg2[450:560, 500:610] = 15  # Right_Hand
    d = _write_dir(tmp_path / "handhair", pose2, seg2)

    process(image_path=Path("dummy.jpg"), output_dir=d)
    doc = json.loads((d / "determinations.json").read_text())

    parts = {p["part"] for p in doc["body_parts_visible"]}
    assert "left_hand" in parts, "hand region must be reported from seg2"
    assert "torso" not in parts

    rels = " ".join(doc["relations"])
    # hands together is anchored on hands only -> survives
    assert "hands together" in rels
    # arm relations anchored on arms -> silent
    assert "arm extended" not in rels
    # upright_deg anchored on torso -> silent
    assert "upright_deg" not in doc["orientation"]
    # camera height anchored on torso/shoulder -> silent
    assert "height_rel_shoulder_m" not in doc.get("camera", {})


def test_true_face_only_all_nonface_silent(tmp_path):
    """True tight face crop: only face pixels. Everything non-face silent."""
    from stratum2.pipeline.determinations import process

    pose2 = make_synthetic_pose2(1, "standing")
    seg2 = np.zeros((1000, 1000), dtype=np.uint8)
    seg2[0:1000, 0:1000] = 3  # face only
    d = _write_dir(tmp_path / "faceonly", pose2, seg2)

    process(image_path=Path("dummy.jpg"), output_dir=d)
    doc = json.loads((d / "determinations.json").read_text())

    parts = {p["part"] for p in doc["body_parts_visible"]}
    assert parts == {"face"}
    # Only the face relation may fire on a face-only crop; all limb/orientation
    # and camera-height determinations stay silent.
    assert doc["relations"] == ["face turned toward camera"]
    assert "upright_deg" not in doc["orientation"]
    assert "height_rel_shoulder_m" not in doc.get("camera", {})


def test_pose2_not_mutated(tmp_path):
    """pose2 array must be read-only inside the pass (no confidence wiping)."""
    from stratum2.pipeline.determinations import process

    pose2 = make_synthetic_pose2(1, "standing")
    original = pose2.copy()
    seg2 = np.zeros((1000, 1000), dtype=np.uint8)
    seg2[0:1000, 0:1000] = 3  # face only -> would previously trigger a wipe
    d = _write_dir(tmp_path / "readonly", pose2, seg2)

    # process loads pose2 from disk; to test non-mutation we re-read after.
    process(image_path=Path("dummy.jpg"), output_dir=d)
    on_disk = np.load(d / "pose2.npy")
    np.testing.assert_array_equal(on_disk, original)
