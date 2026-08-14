"""Tests for additive, idempotent GitHub research-label planning."""

from __future__ import annotations

import pytest

from research_harness import ContractError
from research_harness.labels import load_label_specs, plan_label_sync


def desired() -> list[dict]:
    return [
        {"name": "research", "color": "5319E7", "description": "Research tree"},
        {"name": "research:hold", "color": "000000", "description": "Global hold"},
    ]


def test_label_plan_creates_missing_labels() -> None:
    plan = plan_label_sync(desired(), {})

    assert [operation["action"] for operation in plan] == ["create", "create"]
    assert [operation["label"]["name"] for operation in plan] == ["research", "research:hold"]


def test_label_plan_updates_only_drifted_managed_labels() -> None:
    current = {
        "research": {"name": "research", "color": "ffffff", "description": "old"},
        "research:hold": {"name": "research:hold", "color": "000000", "description": "Global hold"},
        "unmanaged": {"name": "unmanaged", "color": "123456", "description": "keep"},
    }

    plan = plan_label_sync(desired(), current)

    assert plan == [{"action": "edit", "label": desired()[0]}]


def test_label_plan_is_idempotent_and_never_deletes() -> None:
    current = {
        "research": desired()[0],
        "research:hold": desired()[1],
        "unmanaged": {"name": "unmanaged", "color": "123456", "description": "keep"},
    }

    assert plan_label_sync(desired(), current) == []


def test_label_spec_rejects_duplicate_or_invalid_colors(tmp_path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('[{"name":"research","color":"123456","description":"a"},{"name":"research","color":"123456","description":"b"}]')
    with pytest.raises(ContractError, match="duplicate"):
        load_label_specs(duplicate)

    color = tmp_path / "color.json"
    color.write_text('[{"name":"research","color":"bad","description":"a"}]')
    with pytest.raises(ContractError, match="six hexadecimal"):
        load_label_specs(color)


def test_label_spec_normalizes_github_color_without_hash(tmp_path) -> None:
    path = tmp_path / "labels.json"
    path.write_text('[{"name":"research","color":"#5319e7","description":"tree"}]')

    assert load_label_specs(path) == [
        {"name": "research", "color": "5319E7", "description": "tree"}
    ]


def test_label_spec_rejects_invalid_utf8_and_nonstandard_json(tmp_path) -> None:
    invalid_utf8 = tmp_path / "invalid-utf8.json"
    invalid_utf8.write_bytes(b"\xff\xfe")
    with pytest.raises(ContractError, match="unable to decode"):
        load_label_specs(invalid_utf8)

    nonstandard = tmp_path / "nonstandard.json"
    nonstandard.write_text("[NaN]")
    with pytest.raises(ContractError, match="non-standard JSON constant"):
        load_label_specs(nonstandard)
