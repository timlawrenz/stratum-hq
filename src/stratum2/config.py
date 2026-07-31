"""Configuration constants for stratum2 — Sapiens2 model IDs, artifact filenames.

SAPIENS2_SIZE controls which model variant is used across all tasks.
Set via SAPIENS2_SIZE env var (default: '5b').
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Model size (single parameter for all tasks) ---
SAPIENS2_SIZE = os.environ.get("SAPIENS2_SIZE", "5b")

# --- HF repos for task checkpoints ---
SAPIENS2_REPOS = {
    "seg": f"facebook/sapiens2-seg-{SAPIENS2_SIZE}",
    "normal": f"facebook/sapiens2-normal-{SAPIENS2_SIZE}",
    "pointmap": f"facebook/sapiens2-pointmap-{SAPIENS2_SIZE}",
    "pose": f"facebook/sapiens2-pose-{SAPIENS2_SIZE}",
    "matting": "facebook/sapiens2-matting-1b",  # Only 1B available for matting
}

# --- Checkpoint filenames ---
SAPIENS2_FILENAMES = {
    "seg": f"sapiens2_{SAPIENS2_SIZE}_seg.safetensors",
    "normal": f"sapiens2_{SAPIENS2_SIZE}_normal.safetensors",
    "pointmap": f"sapiens2_{SAPIENS2_SIZE}_pointmap.safetensors",
    "pose": f"sapiens2_{SAPIENS2_SIZE}_pose.safetensors",
    "matting": "sapiens2_1b_matting.safetensors",
}

# --- Cache directory for downloaded checkpoints ---
SAPIENS2_CACHE_DIR = Path(
    os.environ.get("SAPIENS2_CACHE_DIR", "/mnt/nas-ai-models/sapiens2")
)

# --- Person detector for pose ---
POSE_DETECTOR_REPO = "facebook/detr-resnet-101-dc5"

# --- New artifact filenames (stratum2-specific) ---
SEG2_FILE = "seg2.npy"
NORMAL2_FILE = "normal2.npy"
POINTMAP_FILE = "pointmap.npy"
POSE2_FILE = "pose2.npy"
MATTING_FILE = "matting.npy"
