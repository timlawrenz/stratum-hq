import json
from pathlib import Path

from tests.stratum2.test_determinations import setup_fixture_dir


def test_determinations_camera(tmp_path):
    from stratum2.pipeline.determinations import process

    # Standing has a flat pointmap at Z=2.5
    # Shoulders are at y=250 in the fixture. Y grid maps y=100..900 -> Y=-0.5..+1.0
    # Therefore shoulder Y should be approx -0.21.
    # Camera origin (Y=0) is roughly y=366 in image space.
    # So height_rel_shoulder_m should be roughly -0.21.

    d_stand = setup_fixture_dir(tmp_path / "camera", pose_type="standing")
    process(image_path=Path("dummy.jpg"), output_dir=d_stand)
    r_stand = json.loads((d_stand / "determinations.json").read_text())

    assert "camera" in r_stand
    cam = r_stand["camera"]
    assert abs(cam["distance_m"] - 2.5) < 0.1
    # Renamed field: shoulder_height_rel_camera_m is the pointmap Y at the
    # shoulder (camera frame, +Y down, camera at origin). Shoulder at Y=-0.21
    # means the shoulder sits 0.21m above the camera.
    assert -0.3 < cam["shoulder_height_rel_camera_m"] < -0.1
