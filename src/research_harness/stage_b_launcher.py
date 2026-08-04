"""Registered scheduler launcher for one bounded local Stage-B comparison.

The launcher deliberately accepts no arbitrary shell command. It validates the
reviewed manifest, binds it to an exact frozen plan and model digest, then uses
the shared scheduler lifecycle:
request -> poll-and-claim -> launch -> verify -> activate -> heartbeat -> release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from .contracts import ContractError, validate_comparison_parity_plan, validate_gpu_manifest, validate_program
from .stage_b import StageBGenerationSettings


class StageBLaunchError(RuntimeError):
    """Raised when the scheduler-bound Stage-B launcher cannot fail closed."""


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_REQUIRED_OUTPUT_FILES = {
    "stage-b-plan.json",
    "run-provenance.json",
    "records.jsonl",
    "review-queue.jsonl",
    "review-guide.md",
    "outputs",
}

SchedulerCall = Callable[[str, list[str]], str]
LaunchRunner = Callable[[list[str], Path], subprocess.Popen[Any]]
VerifyRun = Callable[[], None]
VramUsed = Callable[[], float]
LaunchVerify = Callable[[subprocess.Popen[Any]], float]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageBLaunchError(f"{label} must be a mapping")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise StageBLaunchError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_safe_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise StageBLaunchError(f"{label} must be a non-empty normalized relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts) or path.as_posix() != value:
        raise StageBLaunchError(f"{label} must be a normalized relative POSIX path")
    return value


def _resolved_existing_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise StageBLaunchError(f"{label} must be an existing directory: {path}") from exc
    if not resolved.is_dir():
        raise StageBLaunchError(f"{label} must be an existing directory: {path}")
    return resolved


def _resolved_existing_file(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise StageBLaunchError(f"{label} must be an existing file: {path}") from exc
    if not resolved.is_file():
        raise StageBLaunchError(f"{label} must be an existing file: {path}")
    return resolved


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageBLaunchError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise StageBLaunchError(f"{label} must contain a JSON object")
    return value


def _runner_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runner_source_hash() -> str:
    return _sha256((Path(__file__).resolve().parent / "stage_b.py").read_bytes())


def _launcher_source_hash() -> str:
    return _sha256(Path(__file__).read_bytes())


def _execution(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return _require_mapping(manifest.get("execution"), "GPU manifest execution")


def validate_execution_binding(
    manifest: Mapping[str, Any],
    program: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    comparison_plan: Mapping[str, Any],
    settings: StageBGenerationSettings,
) -> None:
    """Bind a validated scheduler manifest to exact Stage-B inputs/settings."""
    try:
        validate_program(program)
        validate_gpu_manifest(manifest, program)
        validate_comparison_parity_plan(comparison_plan, program)
    except ContractError as exc:
        raise StageBLaunchError(f"invalid Stage-B launch contract: {exc}") from exc

    execution = _execution(manifest)
    if execution.get("runner_module") != "research_harness.stage_b":
        raise StageBLaunchError("GPU manifest execution.runner_module must be research_harness.stage_b")
    if execution.get("candidate_manifest_fingerprint") != candidate_manifest.get("manifest_fingerprint"):
        raise StageBLaunchError("GPU manifest execution.candidate_manifest_fingerprint does not bind the frozen candidate")
    if execution.get("comparison_plan_fingerprint") != comparison_plan.get("comparison_plan_fingerprint"):
        raise StageBLaunchError("GPU manifest execution.comparison_plan_fingerprint does not bind the expected comparison plan")
    if execution.get("comparison_plan_sha256") != _sha256(_canonical_json(comparison_plan).encode("utf-8")):
        raise StageBLaunchError("GPU manifest execution.comparison_plan_sha256 does not bind the expected plan bytes")
    if comparison_plan.get("candidate_manifest_fingerprint") != candidate_manifest.get("manifest_fingerprint"):
        raise StageBLaunchError("comparison plan does not bind the frozen candidate manifest")
    if execution.get("model_name") != settings.model_name:
        raise StageBLaunchError("GPU manifest execution.model_name does not match pinned settings")
    if execution.get("model_digest") != settings.model_digest:
        raise StageBLaunchError("GPU manifest execution.model_digest does not match pinned settings")
    if execution.get("generation_fingerprint") != settings.fingerprint:
        raise StageBLaunchError("GPU manifest execution.generation_fingerprint does not match pinned settings")
    if execution.get("runner_source_sha256") != _runner_source_hash():
        raise StageBLaunchError("GPU manifest execution.runner_source_sha256 does not match the local runner")
    if execution.get("launcher_source_sha256") != _launcher_source_hash():
        raise StageBLaunchError("GPU manifest execution.launcher_source_sha256 does not match the registered launcher")

    raw_items = candidate_manifest.get("items")
    raw_conditions = comparison_plan.get("conditions")
    expected_count = len(raw_items) * len(raw_conditions) if isinstance(raw_items, list) and isinstance(raw_conditions, list) else 0
    if execution.get("expected_record_count") != expected_count:
        raise StageBLaunchError("GPU manifest execution.expected_record_count does not match frozen items and conditions")

    candidate_path = execution.get("candidate_manifest_path")
    if not isinstance(candidate_path, str) or not Path(candidate_path).is_absolute():
        raise StageBLaunchError("GPU manifest execution.candidate_manifest_path must be absolute")
    plan_relative = _require_safe_relative_path(
        execution.get("comparison_plan_relative_path"), "GPU manifest execution.comparison_plan_relative_path"
    )
    if manifest.get("output_root") is None or not Path(manifest["output_root"]).is_absolute():
        raise StageBLaunchError("GPU manifest output_root must be an absolute path")
    if not isinstance(execution.get("git_commit"), str) or not _GIT_COMMIT_RE.fullmatch(execution["git_commit"]):
        raise StageBLaunchError("GPU manifest execution.git_commit must be a full lowercase Git commit")
    if plan_relative != execution["comparison_plan_relative_path"]:
        raise StageBLaunchError("GPU manifest comparison-plan path must be canonical")


def _ensure_clean_pinned_checkout(manifest: Mapping[str, Any], repo_root: Path) -> None:
    execution = _execution(manifest)
    try:
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
        head = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise StageBLaunchError(f"unable to establish a clean pinned Stage-B checkout: {exc}") from exc
    if status.strip():
        raise StageBLaunchError("Stage-B launcher requires a clean committed checkout")
    source_commit = execution["git_commit"]
    try:
        ancestry = subprocess.run(
            ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", source_commit, head],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise StageBLaunchError(f"unable to verify Stage-B implementation ancestry: {exc}") from exc
    if ancestry.returncode != 0:
        raise StageBLaunchError(
            f"Stage-B implementation commit is not an ancestor of checkout: {source_commit} -> {head}"
        )


def _snapshot_selected_inputs(candidate_manifest: Mapping[str, Any]) -> dict[str, dict[str, tuple[int, int, int, int]]]:
    """Capture metadata identities for the exact files a bounded runner may read."""
    source_root = _resolved_existing_directory(
        Path(candidate_manifest["canonical_source_root"]), "candidate canonical source root"
    )
    derived_root = _resolved_existing_directory(
        Path(candidate_manifest["derived_artifact_root"]), "candidate derived artifact root"
    )
    result: dict[str, dict[str, tuple[int, int, int, int]]] = {}
    for raw_item in candidate_manifest.get("items", []):
        item = _require_mapping(raw_item, "candidate item")
        image_id = item.get("image_id")
        relative = _require_safe_relative_path(item.get("source_relative_path"), "candidate source_relative_path")
        if not isinstance(image_id, str):
            raise StageBLaunchError("candidate image_id must be a string")
        paths = {
            "source": source_root / relative,
            "pose2": derived_root / image_id / "pose2.npy",
            "seg2": derived_root / image_id / "seg2.npy",
        }
        result[image_id] = {}
        for role, path in paths.items():
            try:
                resolved = path.resolve(strict=True)
                if not resolved.is_relative_to(source_root if role == "source" else derived_root):
                    raise StageBLaunchError(f"selected {role} path escapes its frozen root")
                stat = resolved.stat()
            except OSError as exc:
                raise StageBLaunchError(f"selected {role} input is unavailable for {image_id}: {exc}") from exc
            result[image_id][role] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    return result


def _assert_inputs_unchanged(before: Mapping[str, Mapping[str, tuple[int, int, int, int]]], candidate_manifest: Mapping[str, Any]) -> None:
    after = _snapshot_selected_inputs(candidate_manifest)
    if after != before:
        raise StageBLaunchError("selected source or derived-input metadata changed during Stage-B execution")


def _scheduler_subprocess(manifest: Mapping[str, Any], action: str, args: list[str]) -> str:
    command = _require_mapping(manifest.get("_launcher_program"), "internal launcher program")
    scheduler_path = command["gpu_scheduler_command"]
    try:
        # The NAS mount is noexec, so the scheduler must run through the Python
        # interpreter rather than as a direct executable.
        completed = subprocess.run(
            [sys.executable, scheduler_path, action, *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise StageBLaunchError(f"scheduler {action} invocation failed: {exc}") from exc
    output = completed.stdout.strip()
    if completed.returncode != 0:
        raise StageBLaunchError(f"scheduler {action} failed: {completed.stderr.strip() or output}")
    return output


def _build_runner_command(
    manifest: Mapping[str, Any], settings: StageBGenerationSettings, repo_root: Path
) -> list[str]:
    execution = _execution(manifest)
    plan_path = _resolved_existing_file(
        repo_root / execution["comparison_plan_relative_path"], "frozen comparison plan"
    )
    candidate_path = _resolved_existing_file(
        Path(execution["candidate_manifest_path"]), "frozen candidate manifest"
    )
    return [
        sys.executable,
        "-m",
        "research_harness.stage_b",
        str(repo_root / "research" / "program.json"),
        str(candidate_path),
        "--output",
        str(manifest["output_root"]),
        "--expected-plan",
        str(plan_path),
        "--endpoint",
        settings.endpoint,
        "--model",
        settings.model_name,
        "--model-digest",
        settings.model_digest,
        "--seed",
        str(settings.seed),
        "--num-predict",
        str(settings.num_predict),
        "--context-window",
        str(settings.context_window),
        "--timeout-seconds",
        str(settings.timeout_seconds),
    ]


def _launch_subprocess(command: list[str], log_path: Path) -> subprocess.Popen[Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(_runner_root() / "src")
    try:
        return subprocess.Popen(
            command,
            cwd=_runner_root(),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError:
        handle.close()
        raise


def _local_vram_used_gb() -> float:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        return float(completed.stdout.strip().splitlines()[0]) / 1024.0
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
        raise StageBLaunchError(f"unable to verify local GPU memory use: {exc}") from exc


def _wait_for_launch_gpu_activity(
    child: subprocess.Popen[Any], log_path: Path, *, minimum_vram_gb: float = 1.0, timeout_seconds: int = 120
) -> float:
    """Verify the launched child is alive, logging, and has loaded local GPU memory."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        exit_code = child.poll()
        if exit_code is not None:
            raise StageBLaunchError(
                f"Stage-B runner exited before scheduler activation with exit code {exit_code}"
            )
        if not log_path.exists():
            raise StageBLaunchError("Stage-B runner launch did not create an audit log")
        used = _local_vram_used_gb()
        if used >= minimum_vram_gb:
            return used
        time.sleep(2)
    raise StageBLaunchError(
        f"Stage-B runner did not demonstrate at least {minimum_vram_gb:.1f}GB local GPU memory activity before activation"
    )


def _verify_output_root(
    output_root: Path,
    candidate_manifest: Mapping[str, Any],
    comparison_plan: Mapping[str, Any],
) -> None:
    root = _resolved_existing_directory(output_root, "Stage-B output root")
    top_level = {path.name for path in root.iterdir()}
    permitted_record_sets = {
        frozenset(_REQUIRED_OUTPUT_FILES),
        frozenset(_REQUIRED_OUTPUT_FILES | {"scheduler-provenance.json"}),
    }
    if frozenset(top_level) not in permitted_record_sets:
        raise StageBLaunchError(f"Stage-B output root contains unexpected record set: {sorted(top_level)}")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise StageBLaunchError("Stage-B output root must not contain symlinks")
    actual_plan = _read_json(root / "stage-b-plan.json", "Stage-B output plan")
    if _canonical_json(actual_plan) != _canonical_json(comparison_plan):
        raise StageBLaunchError("published Stage-B plan differs from frozen expected plan")
    provenance = _read_json(root / "run-provenance.json", "Stage-B run provenance")
    if provenance.get("status") != "PENDING_INDEPENDENT_REVIEW":
        raise StageBLaunchError("Stage-B run provenance must remain PENDING_INDEPENDENT_REVIEW")
    if provenance.get("candidate_manifest_fingerprint") != candidate_manifest.get("manifest_fingerprint"):
        raise StageBLaunchError("Stage-B run provenance candidate binding drifted")
    if provenance.get("comparison_plan_fingerprint") != comparison_plan.get("comparison_plan_fingerprint"):
        raise StageBLaunchError("Stage-B run provenance plan binding drifted")

    expected_pairs = {
        (item["image_id"], condition["id"])
        for item in candidate_manifest["items"]
        for condition in comparison_plan["conditions"]
    }
    records: list[dict[str, Any]] = []
    try:
        for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("record is not an object")
            records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise StageBLaunchError(f"unable to validate Stage-B records: {exc}") from exc
    if {(record.get("image_id"), record.get("condition_id")) for record in records} != expected_pairs:
        raise StageBLaunchError("Stage-B records do not cover exactly the frozen item/condition matrix")
    if len(records) != len(expected_pairs) or provenance.get("record_count") != len(expected_pairs):
        raise StageBLaunchError("Stage-B record count does not match the frozen matrix")
    for record in records:
        relative = _require_safe_relative_path(record.get("output_relative_path"), "Stage-B record output_relative_path")
        output = _resolved_existing_file(root / relative, "Stage-B caption output")
        if not output.is_relative_to(root / "outputs"):
            raise StageBLaunchError("Stage-B caption output escapes outputs directory")
        caption = output.read_text(encoding="utf-8").strip()
        if _sha256(caption.encode("utf-8")) != record.get("caption_sha256"):
            raise StageBLaunchError("Stage-B caption hash drifted from its record")


def run_scheduler_bound_stage_b(
    manifest: Mapping[str, Any],
    program: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    comparison_plan: Mapping[str, Any],
    settings: StageBGenerationSettings,
    *,
    scheduler_call: SchedulerCall,
    launch_runner: LaunchRunner,
    verify_launch: LaunchVerify,
    verify_run: VerifyRun,
    vram_used_gb: VramUsed,
    heartbeat_interval_seconds: int,
    log_path: Path,
    request_if_missing: bool = True,
) -> dict[str, Any]:
    """Run one approved job after an atomic scheduler poll claim.

    The function never issues a separate ``claim`` command. Any exception after
    a successful claim releases only this manifest's job as failed.
    """
    validate_execution_binding(manifest, program, candidate_manifest, comparison_plan, settings)
    if heartbeat_interval_seconds <= 0:
        raise StageBLaunchError("heartbeat_interval_seconds must be positive")

    target_gpu = manifest["target_gpu"]
    job_id = manifest["job_id"]
    claimed = False
    gpu_activity_seen = False
    child: subprocess.Popen[Any] | None = None
    started_at = datetime.now(UTC)
    if request_if_missing:
        request_result = scheduler_call(
            "request",
            [
                "--gpu", target_gpu,
                "--project", manifest["scheduler_project"],
                "--vram", str(manifest["requested_vram_gb"]),
                "--duration", manifest["maximum_duration"],
                "--job-id", job_id,
            ],
        )
        if request_result != job_id:
            raise StageBLaunchError(f"scheduler request returned unexpected job identity: {request_result}")
    poll_result = scheduler_call("poll", ["--gpu", target_gpu, "--job-id", job_id])
    if poll_result != "claimed":
        return {"status": "queued", "poll_result": poll_result, "job_id": job_id}
    claimed = True

    try:
        inputs_before = _snapshot_selected_inputs(candidate_manifest)
        command = _build_runner_command(manifest, settings, _runner_root())
        child = launch_runner(command, log_path)
        initial_used = verify_launch(child)
        if initial_used < 1.0:
            raise StageBLaunchError("Stage-B launch verification returned insufficient local GPU memory activity")
        gpu_activity_seen = True
        activation = scheduler_call(
            "activate", ["--gpu", target_gpu, "--job-id", job_id, "--progress-unit", "condition"]
        )
        if activation != "activated":
            raise StageBLaunchError(f"scheduler activate did not activate the claimed job: {activation}")
        initial_heartbeat = scheduler_call(
            "heartbeat",
            [
                "--gpu", target_gpu,
                "--job-id", job_id,
                "--progress", "0",
                "--vram-used", f"{initial_used:.2f}",
            ],
        )
        if initial_heartbeat != "ok":
            raise StageBLaunchError(f"initial scheduler heartbeat failed: {initial_heartbeat}")

        progress = 0
        while True:
            try:
                exit_code = child.wait(timeout=heartbeat_interval_seconds)
            except subprocess.TimeoutExpired:
                progress += 1
                used = vram_used_gb()
                gpu_activity_seen = gpu_activity_seen or used >= 1.0
                heartbeat = scheduler_call(
                    "heartbeat",
                    [
                        "--gpu", target_gpu,
                        "--job-id", job_id,
                        "--progress", str(progress),
                        "--vram-used", f"{used:.2f}",
                    ],
                )
                if heartbeat != "ok":
                    raise StageBLaunchError(f"scheduler heartbeat failed: {heartbeat}")
                continue
            break

        if exit_code != 0:
            raise StageBLaunchError(f"Stage-B runner exited with exit code {exit_code}")
        if not gpu_activity_seen:
            raise StageBLaunchError("Stage-B runner completed without observable local GPU memory activity")
        verify_run()
        _assert_inputs_unchanged(inputs_before, candidate_manifest)
        release = scheduler_call("release", ["--gpu", target_gpu, "--job-id", job_id, "--status", "completed"])
        if release not in {"released", "already_released"}:
            raise StageBLaunchError(f"scheduler release did not complete: {release}")
        claimed = False
        return {
            "status": "completed",
            "job_id": job_id,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "gpu_activity_seen": gpu_activity_seen,
        }
    except Exception:
        if child is not None:
            try:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=15)
            except (AttributeError, OSError, subprocess.SubprocessError):
                pass
        if claimed:
            try:
                scheduler_call("release", ["--gpu", target_gpu, "--job-id", job_id, "--status", "failed"])
            except Exception:
                pass
        raise


def _settings_from_manifest(manifest: Mapping[str, Any]) -> StageBGenerationSettings:
    execution = _execution(manifest)
    generation = _require_mapping(execution.get("generation"), "GPU manifest execution.generation")
    try:
        return StageBGenerationSettings(
            endpoint=generation["endpoint"],
            model_name=execution["model_name"],
            model_digest=execution["model_digest"],
            temperature=generation["temperature"],
            seed=generation["seed"],
            num_predict=generation["num_predict"],
            top_k=generation["top_k"],
            top_p=generation["top_p"],
            context_window=generation["context_window"],
            timeout_seconds=generation["timeout_seconds"],
        )
    except KeyError as exc:
        raise StageBLaunchError(f"GPU manifest execution.generation omits {exc.args[0]}") from exc


def _load_bound_inputs(manifest: Mapping[str, Any], repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], StageBGenerationSettings]:
    program = _read_json(repo_root / "research" / "program.json", "program JSON")
    execution = _execution(manifest)
    candidate = _read_json(Path(execution["candidate_manifest_path"]), "candidate manifest JSON")
    plan = _read_json(repo_root / execution["comparison_plan_relative_path"], "comparison plan JSON")
    settings = _settings_from_manifest(manifest)
    return program, candidate, plan, settings


def _write_scheduler_provenance(output_root: Path, manifest: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    payload = {
        "schema_version": 1,
        "job_id": manifest["job_id"],
        "manifest_sha256": _sha256(_canonical_json(manifest).encode("utf-8")),
        "launcher_source_sha256": _launcher_source_hash(),
        "runner_source_sha256": _runner_source_hash(),
        "scheduler_result": dict(result),
        "status": "PENDING_INDEPENDENT_REVIEW",
    }
    (output_root / "scheduler-provenance.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="registered-research-launcher")
    parser.add_argument("--manifest", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--request", action="store_true")
    mode.add_argument("--poll-and-launch", action="store_true")
    parser.add_argument("--heartbeat-interval-seconds", type=int, default=30)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        repo_root = _runner_root()
        manifest = _read_json(args.manifest, "GPU manifest")
        program, candidate, plan, settings = _load_bound_inputs(manifest, repo_root)
        validate_execution_binding(manifest, program, candidate, plan, settings)
        _ensure_clean_pinned_checkout(manifest, repo_root)
        manifest_with_program = dict(manifest)
        manifest_with_program["_launcher_program"] = {"gpu_scheduler_command": program["gpu_scheduler"]["command"]}

        def scheduler(action: str, call_args: list[str]) -> str:
            return _scheduler_subprocess(manifest_with_program, action, call_args)

        if args.request:
            result = scheduler(
                "request",
                [
                    "--gpu", manifest["target_gpu"],
                    "--project", manifest["scheduler_project"],
                    "--vram", str(manifest["requested_vram_gb"]),
                    "--duration", manifest["maximum_duration"],
                    "--job-id", manifest["job_id"],
                ],
            )
            if result != manifest["job_id"]:
                raise StageBLaunchError(f"scheduler request returned unexpected job identity: {result}")
            print(json.dumps({"status": "queued", "job_id": result}, sort_keys=True))
            return 0

        output_root = Path(manifest["output_root"])
        log_path = Path("/mnt/nas-ai-models/gpu-scheduler/logs") / f"{manifest['job_id']}.log"

        def verify_launch(child: subprocess.Popen[Any]) -> float:
            return _wait_for_launch_gpu_activity(child, log_path)

        def verify() -> None:
            _verify_output_root(output_root, candidate, plan)

        result = run_scheduler_bound_stage_b(
            manifest_with_program,
            program,
            candidate,
            plan,
            settings,
            scheduler_call=scheduler,
            launch_runner=_launch_subprocess,
            verify_launch=verify_launch,
            verify_run=verify,
            vram_used_gb=_local_vram_used_gb,
            heartbeat_interval_seconds=args.heartbeat_interval_seconds,
            log_path=log_path,
            request_if_missing=args.request,
        )
        if result["status"] == "queued":
            return 0
        _write_scheduler_provenance(output_root, manifest, result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except StageBLaunchError as exc:
        print(f"registered-research-launcher: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
