"""Command-line entry points for fail-closed research-contract validation."""

from __future__ import annotations

import argparse
import json
import subprocess
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

    dims = sub.add_parser(
        "validate-dimension-registry", help="validate the evidence-dimension sweep registry"
    )
    dims.add_argument("registry", type=Path)

    sweep = sub.add_parser("dimension-sweep-status", help="print evidence-dimension sweep status")
    sweep.add_argument("registry", type=Path)

    sel = sub.add_parser(
        "autonomous-select",
        help="select the next highest-impact research arm from the registry",
    )
    sel.add_argument("registry", type=Path)

    verd = sub.add_parser(
        "autonomous-verdict",
        help="compute a better-or-not verdict for a measured comparison",
    )
    verd.add_argument("registry", type=Path, nargs="?", help="optional; validates registry context")
    verd.add_argument("--base-supported", type=int, required=True)
    verd.add_argument("--variant-supported", type=int, required=True)
    verd.add_argument("--base-unsupported", type=int, required=True)
    verd.add_argument("--variant-unsupported", type=int, required=True)
    verd.add_argument("--items", type=int, required=True)
    verd.add_argument("--p-supported", type=float, required=True)
    verd.add_argument("--method", choices=("claim-support", "reconstruction"), default="claim-support")
    verd.add_argument("--reconstruction-delta", type=float, default=None)

    tick = sub.add_parser(
        "autonomous-tick",
        help="run one autonomous-loop iteration (select/research/conclude/advance)",
    )
    tick.add_argument("registry", type=Path)
    tick.add_argument("--review-dir", type=Path, default=None,
                      help="review root for the active arm; omit to only select/activate")
    tick.add_argument("--write", action="store_true",
                      help="persist registry state changes back to the registry file")

    sync = sub.add_parser(
        "sync-issue-labels",
        help="reconcile GitHub issue state labels with the registry (idempotent)",
    )
    sync.add_argument("registry", type=Path)
    sync.add_argument("--apply", action="store_true",
                      help="actually run gh; otherwise print planned operations")
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

        if args.command in ("validate-dimension-registry", "dimension-sweep-status"):
            from .dimension_registry import load_registry, sweep_status

            registry = load_registry(args.registry)
            if args.command == "validate-dimension-registry":
                print("valid")
            else:
                print(json.dumps(sweep_status(registry), sort_keys=True))
            return 0

        if args.command == "autonomous-select":
            from .autonomous import AutonomousError, select_next_arm
            from .dimension_registry import load_registry

            try:
                selection = select_next_arm(load_registry(args.registry))
            except AutonomousError as exc:
                raise ContractError(str(exc)) from exc
            print(json.dumps(selection, sort_keys=True))
            return 0

        if args.command == "autonomous-verdict":
            from .autonomous import AutonomousError, better_or_not

            try:
                verdict = better_or_not(
                    supported_base=args.base_supported,
                    supported_variant=args.variant_supported,
                    unsupported_base=args.base_unsupported,
                    unsupported_variant=args.variant_unsupported,
                    items=args.items,
                    sign_test_p_supported=args.p_supported,
                    method=args.method,
                    reconstruction_delta=args.reconstruction_delta,
                )
            except AutonomousError as exc:
                raise ContractError(str(exc)) from exc
            print(json.dumps(verdict, sort_keys=True))
            return 0

        if args.command == "autonomous-tick":
            from .autonomous import AutonomousError, run_tick
            from .dimension_registry import load_registry

            registry = load_registry(args.registry)
            if args.review_dir is not None:
                review_dir = str(args.review_dir)
            else:
                review_dir = None
            try:
                outcome = run_tick(registry, review_dir=review_dir)
            except AutonomousError as exc:
                raise ContractError(str(exc)) from exc
            if args.write:
                args.registry.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({**outcome, "registry_written": bool(args.write)}, sort_keys=True))
            return 0

        if args.command == "sync-issue-labels":
            from .dimension_registry import load_registry
            from .issue_labels import IssueLabelError, plan_issue_label_sync

            registry = load_registry(args.registry)
            try:
                issue_numbers = sorted({
                    dim["arm_issue"] for dim in registry["dimensions"]
                })
                snapshot = subprocess.run(
                    ["gh", "issue", "list", "--state", "open", "--limit", "100",
                     "--json", "number,labels"],
                    capture_output=True, text=True, check=True, timeout=60,
                )
                issues = json.loads(snapshot.stdout)
            except subprocess.SubprocessError as exc:
                raise ContractError(f"gh issue list failed: {exc}") from exc
            current_by_issue: dict[int, set[str]] = {}
            for issue in issues:
                number = issue.get("number")
                if isinstance(number, int) and number in set(issue_numbers):
                    current_by_issue[number] = {
                        str(label.get("name")) for label in issue.get("labels", [])
                        if isinstance(label, dict) and isinstance(label.get("name"), str)
                    }
            try:
                operations = plan_issue_label_sync(registry, current_by_issue)
            except IssueLabelError as exc:
                raise ContractError(str(exc)) from exc
            if args.apply:
                for op in operations:
                    label = op["label"]
                    subprocess.run(
                        ["gh", "issue", "edit", str(op["issue"]),
                         "--add-label" if op["action"] == "add" else "--remove-label", label],
                        capture_output=True, text=True, check=True, timeout=60,
                    )
                print(json.dumps({"applied": len(operations), "operations": operations},
                                 indent=2, sort_keys=True))
            else:
                print(json.dumps({"planned": operations}, indent=2, sort_keys=True))
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
