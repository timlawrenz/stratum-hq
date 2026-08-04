"""Command-line entry points for fail-closed research-contract validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .contracts import (
    ContractError,
    validate_comparison_parity_plan,
    validate_compression_bundle,
    validate_gpu_manifest,
    validate_program,
    validate_research_tree,
)
from .labels import load_label_specs, plan_label_sync


def _read_json_value(path: Path, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ContractError(f"invalid JSON in {label} {path}: non-standard JSON constant {value}")

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"unable to read {label} {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ContractError(f"unable to decode {label} {path} as UTF-8: {exc}") from exc
    try:
        return json.loads(raw, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON in {label} {path}: {exc.msg}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    value = _read_json_value(path, "JSON")
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return value


def _read_issue_tree_snapshot(path: Path) -> dict[str, Any]:
    """Accept a repository snapshot object or native ``gh issue list --json`` output."""
    value = _read_json_value(path, "issue-tree snapshot")
    if isinstance(value, list):
        return {"issues": value}
    if isinstance(value, dict):
        return value
    raise ContractError("issue-tree snapshot must be a JSON object or GitHub issue-list array")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="research-harness")
    sub = parser.add_subparsers(dest="command", required=True)

    program = sub.add_parser("validate-program", help="validate a program contract")
    program.add_argument("program", type=Path)

    tree = sub.add_parser("validate-tree", help="validate a GitHub issue-tree snapshot")
    tree.add_argument("program", type=Path)
    tree.add_argument("snapshot", type=Path)

    compression = sub.add_parser("validate-compression", help="validate a context bundle")
    compression.add_argument("program", type=Path)
    compression.add_argument("bundle", type=Path)

    comparison = sub.add_parser(
        "validate-comparison-plan", help="validate a frozen controlled-comparison plan"
    )
    comparison.add_argument("program", type=Path)
    comparison.add_argument("plan", type=Path)

    gpu = sub.add_parser("validate-gpu-manifest", help="validate a GPU job manifest")
    gpu.add_argument("program", type=Path)
    gpu.add_argument("manifest", type=Path)

    labels = sub.add_parser("plan-labels", help="plan additive GitHub label changes from JSON snapshots")
    labels.add_argument("desired", type=Path, help="tracked desired label-specification JSON")
    labels.add_argument("current", type=Path, help="read-only GitHub label-list JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "plan-labels":
            desired = load_label_specs(args.desired)
            def reject_constant(value: str) -> None:
                raise ContractError(
                    f"invalid JSON in label snapshot {args.current}: non-standard JSON constant {value}"
                )

            try:
                raw = args.current.read_text(encoding="utf-8")
            except OSError as exc:
                raise ContractError(f"unable to read label snapshot {args.current}: {exc}") from exc
            except UnicodeDecodeError as exc:
                raise ContractError(f"unable to decode label snapshot {args.current} as UTF-8: {exc}") from exc
            try:
                current = json.loads(raw, parse_constant=reject_constant)
            except json.JSONDecodeError as exc:
                raise ContractError(f"invalid JSON in label snapshot {args.current}: {exc.msg}") from exc
            if not isinstance(current, list):
                raise ContractError("label snapshot must be a JSON list")
            current_by_name = {
                label["name"]: label
                for label in current
                if isinstance(label, dict) and isinstance(label.get("name"), str)
            }
            if len(current_by_name) != len(current):
                raise ContractError("label snapshot contains invalid or duplicate label names")
            print(json.dumps(plan_label_sync(desired, current_by_name), indent=2))
            return 0

        program = _read_json(args.program)
        if args.command == "validate-program":
            validate_program(program)
        elif args.command == "validate-tree":
            validate_research_tree(_read_issue_tree_snapshot(args.snapshot), program)
        elif args.command == "validate-compression":
            validate_compression_bundle(_read_json(args.bundle), program)
        elif args.command == "validate-comparison-plan":
            validate_comparison_parity_plan(_read_json(args.plan), program)
        elif args.command == "validate-gpu-manifest":
            validate_gpu_manifest(_read_json(args.manifest), program)
        else:  # pragma: no cover - argparse enforces this branch set.
            raise ContractError(f"unknown command {args.command}")
    except ContractError as exc:
        print(f"research-harness: {exc}", file=sys.stderr)
        return 2
    print("valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
