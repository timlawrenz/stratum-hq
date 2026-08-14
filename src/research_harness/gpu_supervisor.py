"""Fail-closed observer for future scheduler-bound GPU research jobs.

This module intentionally does not claim a GPU or launch a process. A future
reviewed launcher can replace observer_only mode only after the job-manifest
protocol and host-specific launch verification have passed dedicated tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import ContractError, validate_gpu_manifest, validate_program


def inspect_manifests(program: Mapping[str, Any], manifest_dir: Path) -> list[Path]:
    """Return valid approved manifests, or fail closed on an invalid approval."""
    validate_program(program)
    if not manifest_dir.exists():
        return []
    if not manifest_dir.is_dir():
        raise ContractError(f"GPU manifest path is not a directory: {manifest_dir}")

    approved: list[Path] = []
    for path in sorted(manifest_dir.glob("*.json")):
        def reject_constant(value: str) -> None:
            raise ContractError(f"invalid JSON in GPU manifest {path}: non-standard JSON constant {value}")

        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ContractError(f"unable to read GPU manifest {path}: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise ContractError(f"unable to decode GPU manifest {path} as UTF-8: {exc}") from exc
        try:
            value = json.loads(raw, parse_constant=reject_constant)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid JSON in GPU manifest {path}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ContractError(f"GPU manifest {path} must contain a JSON object")
        if value.get("manifest_state") != "approved":
            continue
        validate_gpu_manifest(value, program)
        approved.append(path)
    return approved


def supervisor_message(program: Mapping[str, Any], manifests: list[Path]) -> str:
    """Return a human-visible hold message without performing any GPU mutation."""
    mode = program["gpu_scheduler"].get("execution_mode", "observer_only")
    if not manifests:
        return "[SILENT]"
    if mode != "observer_only":
        raise ContractError(f"unsupported GPU supervisor execution_mode: {mode}")
    names = ", ".join(path.name for path in manifests)
    return (
        "HOLD: approved GPU manifest(s) detected but the supervisor is in "
        f"{mode} mode and will not claim or launch work: {names}. "
        "A reviewed host-specific launcher and explicit activation are required."
    )


def main(argv: list[str] | None = None) -> int:
    """Run observer-only supervisor from a program JSON and manifest directory."""
    import argparse

    parser = argparse.ArgumentParser(prog="research-gpu-supervisor")
    parser.add_argument("program", type=Path)
    parser.add_argument("manifest_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        def reject_constant(value: str) -> None:
            raise ContractError(
                f"invalid JSON in GPU supervisor program {args.program}: "
                f"non-standard JSON constant {value}"
            )

        try:
            raw = args.program.read_text(encoding="utf-8")
        except OSError as exc:
            raise ContractError(f"unable to read GPU supervisor program {args.program}: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise ContractError(f"unable to decode GPU supervisor program {args.program} as UTF-8: {exc}") from exc
        try:
            program = json.loads(raw, parse_constant=reject_constant)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid JSON in GPU supervisor program {args.program}: {exc.msg}") from exc
        if not isinstance(program, dict):
            raise ContractError("program JSON must contain an object")
        print(supervisor_message(program, inspect_manifests(program, args.manifest_dir)))
        return 0
    except ContractError as exc:
        print(f"research-gpu-supervisor: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
