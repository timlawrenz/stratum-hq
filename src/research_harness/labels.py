"""Additive, idempotent planning for repository-managed GitHub labels."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import ContractError

_HEX_COLOR = re.compile(r"^[0-9A-F]{6}$")


def _normalize_spec(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ContractError("each label specification must be an object")
    name = value.get("name")
    description = value.get("description")
    color = value.get("color")
    if not isinstance(name, str) or not name.strip():
        raise ContractError("label name must be a non-empty string")
    if not isinstance(description, str):
        raise ContractError(f"label {name!r} description must be a string")
    if not isinstance(color, str):
        raise ContractError(f"label {name!r} color must be a string")
    normalized_color = color.removeprefix("#").upper()
    if not _HEX_COLOR.fullmatch(normalized_color):
        raise ContractError(f"label {name!r} color must be six hexadecimal digits")
    return {"name": name, "color": normalized_color, "description": description}


def load_label_specs(path: Path) -> list[dict[str, str]]:
    """Load and validate a tracked, repository-owned label specification."""
    def reject_constant(value: str) -> None:
        raise ContractError(f"invalid JSON in label specification {path}: non-standard JSON constant {value}")

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"unable to read label specification {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ContractError(f"unable to decode label specification {path} as UTF-8: {exc}") from exc
    try:
        value = json.loads(raw, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON in label specification {path}: {exc.msg}") from exc
    if not isinstance(value, list):
        raise ContractError("label specification must be a JSON list")
    labels = [_normalize_spec(item) for item in value]
    names = [label["name"] for label in labels]
    if len(names) != len(set(names)):
        raise ContractError("label specification contains duplicate names")
    return labels


def _current_label(value: Any) -> dict[str, str]:
    """Normalize a GitHub label response or test fixture for equality comparison."""
    return _normalize_spec(value)


def plan_label_sync(
    desired: Sequence[Mapping[str, Any]], current_by_name: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Plan additive create/edit operations; unmanaged labels are never deleted."""
    normalized_desired = [_normalize_spec(label) for label in desired]
    desired_names = [label["name"] for label in normalized_desired]
    if len(desired_names) != len(set(desired_names)):
        raise ContractError("label specification contains duplicate names")

    operations: list[dict[str, Any]] = []
    for label in normalized_desired:
        existing = current_by_name.get(label["name"])
        if existing is None:
            operations.append({"action": "create", "label": label})
            continue
        if _current_label(existing) != label:
            operations.append({"action": "edit", "label": label})
    return operations
