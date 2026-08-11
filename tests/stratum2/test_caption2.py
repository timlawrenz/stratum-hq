import json
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


@patch("stratum.pipeline.caption.OllamaCaptionBackend.generate")
def test_caption2_forwards_nondefault_max_tokens_to_backend(mock_generate, tmp_path):
    """A direct caption2 call passes a non-default budget to the backend."""
    from PIL import Image
    import numpy as np

    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(tmp_path / "dummy.jpg")
    (tmp_path / "determinations.json").write_text(json.dumps({"schema_version": 2}))
    mock_generate.return_value = "Budget-controlled caption."

    assert process(
        image_path=tmp_path / "dummy.jpg",
        output_dir=tmp_path,
        max_tokens=731,
    ) is True

    assert mock_generate.call_args.kwargs["max_tokens"] == 731


@patch("stratum.pipeline.caption.OllamaCaptionBackend.generate")
def test_caption2_cli_forwards_budget_and_omits_detector_anomaly(mock_generate, tmp_path):
    """The full CLI path preserves budget control and strips anomaly prompt content."""
    from PIL import Image
    import numpy as np

    from stratum.discovery import output_dir_for_image
    from stratum2.cli import cmd_process, parse_args

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    image_path = input_dir / "dummy.jpg"
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(image_path)

    artifact_dir = output_dir_for_image(image_path, input_dir, output_dir)
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "determinations.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "subject": {
                    "n_detections": 2,
                    "detector_anomaly": "extra_detections(2)",
                },
            }
        )
    )
    mock_generate.return_value = "CLI-controlled caption."

    args = parse_args(
        [
            "process",
            str(input_dir),
            "--output",
            str(output_dir),
            "--passes",
            "caption2",
            "--caption-max-tokens",
            "731",
            "--device",
            "cpu",
            "--progress-every",
            "0",
        ]
    )

    assert cmd_process(args) == 0
    assert mock_generate.call_args.kwargs["max_tokens"] == 731
    prompt = mock_generate.call_args.kwargs["prompt"]
    assert "detector anomaly" not in prompt.lower()
    assert "extra_detections" not in prompt
    assert (artifact_dir / "caption2.txt").read_text() == "CLI-controlled caption."
