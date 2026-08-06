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


def _resolve_review_dir(args: argparse.Namespace) -> str | None:
    """Resolve the review root from --review-dir or --review-dir-from marker.

    Marker resolution is deterministic: the wrapper emits a JSON marker with a
    `review_root` field after a completed review pass, so the tick never has to
    guess which root to conclude against. Fail-closed on ambiguity or an
    incomplete marker.
    """
    from .contracts import ContractError

    if args.review_dir is not None and args.review_dir_from is not None:
        raise ContractError(
            "autonomous-tick: pass either --review-dir or --review-dir-from, not both"
        )
    if args.review_dir_from is None:
        return str(args.review_dir) if args.review_dir is not None else None
    try:
        marker = json.loads(args.review_dir_from.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"unable to read tick-ready marker {args.review_dir_from}: {exc}") from exc
    if not isinstance(marker, dict):
        raise ContractError(f"tick-ready marker {args.review_dir_from} must be a JSON object")
    if marker.get("status") != "completed":
        raise ContractError(f"tick-ready marker {args.review_dir_from} is not completed: {marker.get('status')!r}")
    root = marker.get("review_root")
    if not isinstance(root, str) or not root.strip():
        raise ContractError(f"tick-ready marker {args.review_dir_from} has no review_root")
    return root


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
    tick.add_argument("--review-dir-from", type=Path, default=None,
                      help="JSON marker emitted by stratum_review_poll_wrapper.py; "
                           "the tick uses its review_root (deterministic, no path guessing)")
    tick.add_argument("--method", choices=("claim-support", "reconstruction"),
                      default="claim-support",
                      help="verdict method: claim-support (review root) or reconstruction (CLIP delta)")
    tick.add_argument("--reconstruction-delta", type=float, default=None,
                      help="CLIP similarity delta (variant minus base) for reconstruction method")
    tick.add_argument("--items", type=int, default=None,
                      help="override the item count used for the verdict")
    tick.add_argument("--write", action="store_true",
                      help="persist registry state changes back to the registry file atomically")

    propose = sub.add_parser(
        "propose-dimensions",
        help="gate-register N new candidate dimensions as proposals before selection",
    )
    propose.add_argument("registry", type=Path)
    propose.add_argument("--candidates", type=Path, required=True,
                         help="JSON array of candidate dimension objects with full declarations")
    propose.add_argument("--count", type=int, default=1,
                         help="number of NEW dimensions required to pass the gate (fail-closed below)")
    propose.add_argument("--require-new-evidence-part", action="store_true",
                         help="reject candidates that reuse only already-validated axes "
                              "(seed-diversity gate: must name new evidence part or new model class)")
    propose.add_argument("--write", action="store_true",
                         help="persist the augmented registry back to the registry file atomically")

    markb = sub.add_parser(
        "mark-blocked",
        help="classify a dimension as blocked: its gate is a policy/authority "
             "decision, not a measurement (excluded from selector scoring)",
    )
    markb.add_argument("registry", type=Path)
    markb.add_argument("dim_id", type=str)
    markb.add_argument("--reason", type=str, required=True,
                       help="human-readable reason the arm cannot advance by measurement")
    markb.add_argument("--issue", type=int, default=None,
                       help="issue number the arm is waiting on (e.g. the research:needs-human ruling)")
    markb.add_argument("--write", action="store_true",
                       help="persist the registry change atomically")

    marku = sub.add_parser(
        "mark-unblocked",
        help="return a blocked dimension to an actionable state (proposal by default)",
    )
    marku.add_argument("registry", type=Path)
    marku.add_argument("dim_id", type=str)
    marku.add_argument("--state", choices=("proposal", "active"), default="proposal")
    marku.add_argument("--write", action="store_true",
                       help="persist the registry change atomically")

    overview = sub.add_parser(
        "program-overview",
        help="program-state readout: validated evidence budget vs floor, goal-arm "
             "inputs validated, blocked arms, dependency frontier (strategist step-back)",
    )
    overview.add_argument("registry", type=Path)
    overview.add_argument("--program", type=Path, default=None,
                          help="program.json to cross-check goal_floors against")

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
            from .dimension_registry import (
                DimensionRegistryError,
                load_registry,
                registry_sha256,
                write_registry,
            )

            expected_sha = registry_sha256(args.registry) if args.write else None
            registry = load_registry(args.registry)
            review_dir = _resolve_review_dir(args)
            try:
                outcome = run_tick(
                    registry,
                    review_dir=review_dir,
                    method=args.method,
                    reconstruction_delta=args.reconstruction_delta,
                    items=args.items,
                )
            except AutonomousError as exc:
                raise ContractError(str(exc)) from exc
            if args.write:
                try:
                    write_registry(args.registry, registry, expected_sha256=expected_sha)
                except DimensionRegistryError as exc:
                    raise ContractError(str(exc)) from exc
            print(json.dumps({**outcome, "registry_written": bool(args.write)}, sort_keys=True))
            return 0

        if args.command == "propose-dimensions":
            from .dimension_registry import DimensionRegistryError, load_registry, registry_sha256, write_registry
            from .proposals import ProposalGateError, propose_dimensions

            registry = load_registry(args.registry)
            try:
                raw_candidates = args.candidates.read_text(encoding="utf-8")
                candidates = json.loads(raw_candidates)
            except json.JSONDecodeError as exc:
                raise ContractError(f"invalid JSON in candidates {args.candidates}: {exc.msg}") from exc
            expected_sha = registry_sha256(args.registry) if args.write else None
            try:
                result = propose_dimensions(
                    registry,
                    candidates,
                    count=args.count,
                    require_new_evidence_part=args.require_new_evidence_part,
                )
            except ProposalGateError as exc:
                raise ContractError(str(exc)) from exc
            if args.write:
                try:
                    write_registry(args.registry, registry, expected_sha256=expected_sha)
                except DimensionRegistryError as exc:
                    raise ContractError(str(exc)) from exc
            print(json.dumps({**result, "registry_written": bool(args.write)}, sort_keys=True))
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

        if args.command in ("mark-blocked", "mark-unblocked"):
            from .dimension_registry import (
                DimensionRegistryError,
                load_registry,
                mark_dimension_blocked,
                mark_dimension_unblocked,
                registry_sha256,
                write_registry,
            )

            expected_sha = registry_sha256(args.registry) if args.write else None
            registry = load_registry(args.registry)
            try:
                if args.command == "mark-blocked":
                    mark_dimension_blocked(registry, args.dim_id, args.reason, issue=args.issue)
                else:
                    mark_dimension_unblocked(registry, args.dim_id, state=args.state)
            except DimensionRegistryError as exc:
                raise ContractError(str(exc)) from exc
            if args.write:
                try:
                    write_registry(args.registry, registry, expected_sha256=expected_sha)
                except DimensionRegistryError as exc:
                    raise ContractError(str(exc)) from exc
            final_state = next(
                d["state"] for d in registry["dimensions"] if d["id"] == args.dim_id
            )
            print(json.dumps({
                "command": args.command,
                "dim_id": args.dim_id,
                "state": final_state,
                "registry_written": bool(args.write),
            }, sort_keys=True))
            return 0

        if args.command == "program-overview":
            from .dimension_registry import load_registry, program_overview

            registry = load_registry(args.registry)
            overview = program_overview(registry)
            if args.program is not None:
                program = _read_json(args.program)
                rep = program.get("representation") or {}
                program_floor = rep.get("expanded_dossier_min_tokens")
                declared = (registry.get("goal_floors") or {}).get("expanded_dossier_min_tokens")
                overview["program_floor_tokens"] = program_floor
                overview["program_floor_matches_registry"] = (
                    program_floor == declared if program_floor is not None else None
                )
            print(json.dumps(overview, sort_keys=True))
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
