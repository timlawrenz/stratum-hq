from pathlib import Path
import json

from tests.stratum2.test_determinations import setup_fixture_dir


def test_determinations_idempotent(tmp_path):
    from stratum2.pipeline.determinations import process

    d = setup_fixture_dir(tmp_path / "idem")

    # First run
    res1 = process(image_path=Path("dummy.jpg"), output_dir=d)
    assert res1 is True
    out1 = (d / "determinations.json").read_bytes()

    # Second run
    res2 = process(image_path=Path("dummy.jpg"), output_dir=d)
    assert res2 is True
    out2 = (d / "determinations.json").read_bytes()

    assert out1 == out2


def test_determinations_schema(tmp_path):
    from stratum2.pipeline.determinations import process

    d = setup_fixture_dir(tmp_path / "schema")
    process(image_path=Path("dummy.jpg"), output_dir=d)

    doc = json.loads((d / "determinations.json").read_text())

    # Assert top-level keys
    assert "schema_version" in doc
    assert "subject" in doc
    assert "subject_extent" in doc
    assert "body_parts_visible" in doc
    assert "orientation" in doc
    assert "relations" in doc

    # Assert body_parts structure
    parts = doc["body_parts_visible"]
    for p in parts:
        assert "part" in p
        assert "pixel_frac" in p
        assert "kp_conf" in p
