import json
from pathlib import Path

from tests.stratum2.test_determinations import setup_fixture_dir


def test_determinations_relations(tmp_path):
    from stratum2.pipeline.determinations import process

    # 1. Arm orientations: left arm downward, right arm downward
    # (By default our 'standing' fixture has wrists below shoulders, and legs below hips)
    d_stand = setup_fixture_dir(tmp_path / "arms_down", pose_type="standing")
    process(image_path=Path("dummy.jpg"), output_dir=d_stand)
    r_stand = json.loads((d_stand / "determinations.json").read_text())
    rels_stand = " ".join(r_stand["relations"])
    assert "left arm extended downward" in rels_stand
    assert "right arm extended downward" in rels_stand
    assert "left leg extended downward" in rels_stand
    assert "right leg extended downward" in rels_stand

    # 2. Arm orientations: left arm upward, right arm upward
    d_raised = setup_fixture_dir(tmp_path / "arms_up", pose_type="arms_raised")
    process(image_path=Path("dummy.jpg"), output_dir=d_raised)
    r_raised = json.loads((d_raised / "determinations.json").read_text())
    rels_raised = " ".join(r_raised["relations"])
    assert "left arm extended upward" in rels_raised
    assert "right arm extended upward" in rels_raised

    # 3. Hands together vs hands apart
    d_together = setup_fixture_dir(tmp_path / "together", pose_type="hands_together")
    process(image_path=Path("dummy.jpg"), output_dir=d_together)
    r_together = json.loads((d_together / "determinations.json").read_text())
    rels_together = " ".join(r_together["relations"])
    assert "hands together" in rels_together
    assert "hands together" not in rels_stand  # Default standing has hands apart

    # 4. Facing
    d_profile = setup_fixture_dir(tmp_path / "profile", pose_type="profile_left")
    process(image_path=Path("dummy.jpg"), output_dir=d_profile)
    r_profile = json.loads((d_profile / "determinations.json").read_text())
    rels_profile = " ".join(r_profile["relations"])
    assert "face in profile" in rels_profile
    assert "face turned toward camera" in rels_stand


def test_determinations_held_object(tmp_path):
    from stratum2.pipeline.determinations import process

    d = setup_fixture_dir(
        tmp_path / "held", pose_type="hands_together", mask_type="held_object"
    )
    process(image_path=Path("dummy.jpg"), output_dir=d)
    res = json.loads((d / "determinations.json").read_text())
    rels = " ".join(res["relations"])
    assert "hands gripping an object at" in rels
