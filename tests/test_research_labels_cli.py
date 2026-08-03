"""CLI behavior for the dry-run research-label planner."""

from __future__ import annotations

import json
import subprocess
import sys


def test_module_cli_plans_label_changes_without_network_access(tmp_path) -> None:
    desired = tmp_path / "desired.json"
    desired.write_text(
        json.dumps(
            [
                {"name": "research", "color": "5319E7", "description": "tree"},
                {"name": "research:hold", "color": "000000", "description": "hold"},
            ]
        )
    )
    current = tmp_path / "current.json"
    current.write_text(
        json.dumps(
            [
                {"name": "research", "color": "ffffff", "description": "old"},
                {"name": "unmanaged", "color": "123456", "description": "keep"},
            ]
        )
    )

    result = subprocess.run(
        [sys.executable, "-m", "research_harness", "plan-labels", str(desired), str(current)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout) == [
        {
            "action": "edit",
            "label": {"name": "research", "color": "5319E7", "description": "tree"},
        },
        {
            "action": "create",
            "label": {"name": "research:hold", "color": "000000", "description": "hold"},
        },
    ]


def test_module_cli_rejects_non_list_github_label_snapshot(tmp_path) -> None:
    desired = tmp_path / "desired.json"
    desired.write_text('[{"name":"research","color":"5319E7","description":"tree"}]')
    current = tmp_path / "current.json"
    current.write_text("{}")

    result = subprocess.run(
        [sys.executable, "-m", "research_harness", "plan-labels", str(desired), str(current)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "label snapshot" in result.stderr


def test_module_cli_rejects_invalid_utf8_label_snapshot_without_traceback(tmp_path) -> None:
    desired = tmp_path / "desired.json"
    desired.write_text('[{"name":"research","color":"5319E7","description":"tree"}]')
    current = tmp_path / "current.json"
    current.write_bytes(b"\xff\xfe")

    result = subprocess.run(
        [sys.executable, "-m", "research_harness", "plan-labels", str(desired), str(current)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "unable to decode label snapshot" in result.stderr
    assert "Traceback" not in result.stderr
