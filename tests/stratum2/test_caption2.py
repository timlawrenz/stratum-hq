import json
from pathlib import Path
from unittest.mock import patch

from stratum2.pipeline.caption2 import process


def test_caption2_dependency_gate(tmp_path):
    # Setup dir with image but NO determinations.json
    (tmp_path / "dummy.jpg").write_bytes(b"fake_image_data")

    # Should return False (skip/fail) rather than hallucinating
    res = process(image_path=tmp_path / "dummy.jpg", output_dir=tmp_path)
    assert res is False
    assert not (tmp_path / "caption2.txt").exists()


@patch("stratum.pipeline.caption.OllamaCaptionBackend.generate")
def test_caption2_passthrough_and_isolation(mock_generate, tmp_path):
    # Setup dir WITH determinations and legacy stratum1 files
    # Needs a real valid JPEG so PIL can open it
    from PIL import Image
    import numpy as np

    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(tmp_path / "dummy.jpg")

    det = {"schema_version": 2, "subject": {"n_detections": 1}}
    (tmp_path / "determinations.json").write_text(json.dumps(det))

    # Legacy files
    (tmp_path / "caption.txt").write_text("Old caption")
    (tmp_path / "t5_hidden.npy").write_bytes(b"old_t5")

    # Mock the LLM call
    mock_generate.return_value = "This is the grounded caption2 output."

    res = process(image_path=tmp_path / "dummy.jpg", output_dir=tmp_path)
    assert res is True

    # 1. Pass-through gate
    out_text = (tmp_path / "caption2.txt").read_text()
    assert out_text == "This is the grounded caption2 output."

    # Prompt check
    called_prompt = mock_generate.call_args[1]["prompt"]
    assert "DETERMINATIONS:" in called_prompt
    assert "exactly one primary subject detected" in called_prompt

    # 2. Isolation gate
    assert (tmp_path / "caption.txt").read_text() == "Old caption"
    assert (tmp_path / "t5_hidden.npy").read_bytes() == b"old_t5"


@patch("stratum.pipeline.caption.OllamaCaptionBackend.generate")
def test_caption2_idempotency(mock_generate, tmp_path):
    from PIL import Image
    import numpy as np

    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(tmp_path / "dummy.jpg")

    (tmp_path / "determinations.json").write_text("{}")

    # Pre-existing output
    (tmp_path / "caption2.txt").write_text("Existing caption")

    res = process(image_path=tmp_path / "dummy.jpg", output_dir=tmp_path)
    assert res is True

    # Ollama was NOT called
    mock_generate.assert_not_called()
    assert (tmp_path / "caption2.txt").read_text() == "Existing caption"
