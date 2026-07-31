"""Sapiens2 model loading — download checkpoints, init models via sapiens configs."""

from __future__ import annotations

from pathlib import Path

from stratum2.config import (
    SAPIENS2_CACHE_DIR,
    SAPIENS2_FILENAMES,
    SAPIENS2_REPOS,
    SAPIENS2_SIZE,
)


def _download_checkpoint(repo_id: str, filename: str) -> Path:
    """Download a safetensors checkpoint from HuggingFace if not cached.

    Returns the local path to the downloaded file.
    """
    cache_dir = SAPIENS2_CACHE_DIR / repo_id.replace("/", "--")
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


def _find_config_path(pattern: str, search_dir: str) -> Path | None:
    """Find a config file matching the given pattern in a directory tree."""
    import glob
    import os

    import sapiens

    base = Path(sapiens.__file__).parent
    search_path = base / search_dir
    matches = sorted(glob.glob(str(search_path / pattern), recursive=True))
    if matches:
        return Path(matches[0])
    return None


# Mapping from (task, size) → (config_subdir, glob_pattern)
# Uses the installed sapiens package config files.
_CONFIG_MAP: dict[str, tuple[str, str]] = {
    "seg": ("dense/configs/seg", "**/sapiens2_{size}_seg_*-1024x768.py"),
    "normal": ("dense/configs/normal", "**/sapiens2_{size}_normal_*-1024x768.py"),
    "pointmap": ("dense/configs/pointmap", "**/sapiens2_{size}_pointmap_*-1024x768.py"),
    "matting": ("dense/configs/matting", "**/sapiens2_1b_matting_*-1024x768.py"),
    "pose": ("pose/configs/keypoints308", "**/sapiens2_{size}_keypoints308_*-1024x768.py"),
}


def get_config_path(task: str, size: str) -> Path:
    """Resolve the installed sapiens config file for a given task and model size.

    Raises ValueError if the task is unknown or no config is found.
    """
    if task not in _CONFIG_MAP:
        raise ValueError(
            f"Unknown task: {task!r}. Valid tasks: {list(_CONFIG_MAP)}"
        )

    search_dir, pattern = _CONFIG_MAP[task]
    # matting always uses 1b regardless of size param
    effective_size = "1b" if task == "matting" else size
    glob_pattern = pattern.format(size=effective_size)

    path = _find_config_path(glob_pattern, search_dir)
    if path is None:
        raise FileNotFoundError(
            f"No config found for task={task!r} size={effective_size!r} "
            f"in sapiens/{search_dir}/{glob_pattern}"
        )
    return path


def load_sapiens2_model(task: str, device: str = "cpu"):
    """Download and load a Sapiens2 task checkpoint.

    Args:
        task: One of 'seg', 'normal', 'pointmap', 'pose', 'matting'.
        device: Torch device string (e.g. 'cpu', 'cuda:0').

    Returns:
        Loaded model with .pipeline(), .data_preprocessor(), and .__call__().
    """
    size = SAPIENS2_SIZE

    repo_id = SAPIENS2_REPOS[task]
    filename = SAPIENS2_FILENAMES[task]
    ckpt_path = _download_checkpoint(repo_id, filename)

    config_path = get_config_path(task, size)

    if task == "pose":
        from sapiens.pose.models import init_model
    else:
        from sapiens.dense.models import init_model

    model = init_model(str(config_path), str(ckpt_path), device=device)
    model.eval()
    return model
