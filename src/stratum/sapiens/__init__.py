"""Sapiens TorchScript model loading and shared preprocessing.

Downloads Sapiens-1B TorchScript checkpoints from HuggingFace on first use
and provides common preprocessing for all Sapiens tasks (segmentation, depth,
surface normals).

Models expect 1024×768 (H×W) input images normalised with
mean=[123.5, 116.5, 103.5], std=[58.5, 57.0, 57.5].
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from stratum.config import SAPIENS_INPUT_HEIGHT, SAPIENS_INPUT_WIDTH

SAPIENS_CACHE_DIR = Path(
    os.environ.get("SAPIENS_CACHE_DIR", Path.home() / ".cache" / "sapiens")
)

SAPIENS_MEAN = np.array([123.5, 116.5, 103.5], dtype=np.float32)
SAPIENS_STD = np.array([58.5, 57.0, 57.5], dtype=np.float32)


def _download_checkpoint(repo_id: str, filename: str) -> Path:
    """Download a TorchScript checkpoint from HuggingFace if not cached."""
    cache_dir = SAPIENS_CACHE_DIR / repo_id.replace("/", "--")
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / filename
    if model_path.exists():
        return model_path
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(cache_dir),
    )
    return model_path


def load_sapiens_model(repo_id: str, filename: str, device: str = "cpu"):
    """Load a Sapiens TorchScript model onto the specified device."""
    import torch

    path = _download_checkpoint(repo_id, filename)
    model = torch.jit.load(str(path), map_location=device)
    model.eval()
    return model


def preprocess(image: np.ndarray):
    """Prepare an image for Sapiens inference.

    Args:
        image: RGB image as H×W×3 uint8 numpy array.

    Returns:
        Tensor of shape ``(1, 3, 1024, 768)`` in float32.
    """
    import cv2
    import torch

    img = cv2.resize(
        image,
        (SAPIENS_INPUT_WIDTH, SAPIENS_INPUT_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    img = img.astype(np.float32)
    img = (img - SAPIENS_MEAN) / SAPIENS_STD
    # HWC → CHW → NCHW
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def postprocess_resize(result, target_h: int, target_w: int):
    """Bilinear-interpolate model output to target dimensions.

    Args:
        result: Model output tensor, shape ``(1, C, H_model, W_model)``.
        target_h: Target height.
        target_w: Target width.

    Returns:
        Resized tensor of shape ``(1, C, target_h, target_w)``.
    """
    import torch.nn.functional as F

    return F.interpolate(result, size=(target_h, target_w), mode="bilinear", align_corners=False)
