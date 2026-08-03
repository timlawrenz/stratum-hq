from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from stratum2.pipeline.t52 import process


def test_t52_dependency_gate(tmp_path):
    res = process(
        image_path=Path("dummy.jpg"), output_dir=tmp_path, tokenizer=None, encoder=None
    )
    assert res is False
    assert not (tmp_path / "t52_hidden.npy").exists()


def test_t52_shape_dtype_and_isolation(tmp_path):
    (tmp_path / "caption2.txt").write_text("A rich descriptive caption2.")

    # Legacy files
    (tmp_path / "t5_hidden.npy").write_bytes(b"old_t5")

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": [np.zeros((512,), dtype=np.int64)],
        "attention_mask": [np.ones((512,), dtype=np.uint8)],
    }

    # We don't actually need to run the real PyTorch model for the shape gate,
    # just mock the helper function so we can assert it produces exactly what
    # the contract says.
    with patch("stratum.pipeline.t5.compute_t5_hidden_states") as mock_compute:
        mock_compute.return_value = np.zeros((512, 1024), dtype=np.float16)

        res = process(
            image_path=Path("dummy.jpg"),
            output_dir=tmp_path,
            tokenizer=mock_tokenizer,
            encoder=MagicMock(),
        )
        assert res is True

        # Shape/dtype gate
        hidden = np.load(tmp_path / "t52_hidden.npy")
        assert hidden.shape == (512, 1024)
        assert hidden.dtype == np.float16

        mask = np.load(tmp_path / "t52_mask.npy")
        assert mask.shape == (512,)
        assert mask.dtype == np.uint8

        # Isolation gate
        assert (tmp_path / "t5_hidden.npy").read_bytes() == b"old_t5"


def test_t52_idempotency(tmp_path):
    (tmp_path / "caption2.txt").write_text("Text")
    (tmp_path / "t52_hidden.npy").write_bytes(b"existing_hidden")
    (tmp_path / "t52_mask.npy").write_bytes(b"existing_mask")

    with patch("stratum.pipeline.t5.compute_t5_hidden_states") as mock_compute:
        res = process(
            image_path=Path("dummy.jpg"),
            output_dir=tmp_path,
            tokenizer=None,
            encoder=None,
        )
        assert res is True
        mock_compute.assert_not_called()
        assert (tmp_path / "t52_hidden.npy").read_bytes() == b"existing_hidden"
