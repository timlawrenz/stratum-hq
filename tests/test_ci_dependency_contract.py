"""Keep the hosted test workflow's dependency contract explicit."""

from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_test_extra_covers_non_torch_test_imports() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = config["project"]["optional-dependencies"]["test"]

    assert any(dependency.startswith("pytest") for dependency in dependencies)
    assert any(dependency.startswith("opencv-python-headless") for dependency in dependencies)


def test_ci_installs_cpu_torch_before_running_the_full_suite() -> None:
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")

    assert "download.pytorch.org/whl/cpu" in workflow
    assert ".[test]" in workflow
    assert "python -m pytest tests/ -q" in workflow
