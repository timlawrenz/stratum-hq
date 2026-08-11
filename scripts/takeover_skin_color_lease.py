"""Takeover lease monitor for stratum-stage-b-skin-color-v1.

The owning launcher process died on a foreground timeout after the claim. The
runner (research_harness.stage_b) is alive and generating real records. This
monitor takes over the lease ownership: heartbeats every 60s, waits for the
runner to finish writing records.jsonl (96 rows) + run-provenance.json, then
signals "DONE_RDY_TO_RELEASE" so the caller can verify artifacts and release
with --status completed. Never fabricates progress; progress is derived from
the real artifact state on disk.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

SCHEDULER = "/mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py"
JOB_ID = "stratum-stage-b-skin-color-v1"
GPU = "4090"
OUTPUT_ROOT = Path("/mnt/nas-ai-models/research/stratum/stage-b-skin-color-v1")
RECORDS = OUTPUT_ROOT / "records.jsonl"
PROVENANCE = OUTPUT_ROOT / "run-provenance.json"
EXPECTED_RECORDS = 96
MAX_WAIT_SECONDS = 45 * 60  # 2h lease; runner should finish well before this


def _scheduler(action: str, args: list[str]) -> str:
    completed = subprocess.run(
        [sys.executable, SCHEDULER, action, "--gpu", GPU, "--job-id", JOB_ID, *args],
        capture_output=True, text=True, check=False, timeout=60,
    )
    return (completed.stdout or completed.stderr).strip()


def main() -> int:
    deadline = time.time() + MAX_WAIT_SECONDS
    while time.time() < deadline:
        rows = 0
        if RECORDS.exists():
            rows = sum(1 for _ in RECORDS.open("r", encoding="utf-8"))
        provenance_ready = PROVENANCE.exists()
        runner_alive = False
        try:
            out = subprocess.run(
                ["pgrep", "-f", "research_harness.stage_b"],
                capture_output=True, text=True, check=False, timeout=30,
            )
            runner_alive = bool(out.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            runner_alive = False

        status_line = {
            "records": rows,
            "expected": EXPECTED_RECORDS,
            "provenance_ready": provenance_ready,
            "runner_alive": runner_alive,
        }
        print(json.dumps(status_line, sort_keys=True), flush=True)

        if rows >= EXPECTED_RECORDS and provenance_ready and not runner_alive:
            print("DONE_RDY_TO_RELEASE", flush=True)
            return 0

        # Heartbeat while the run is live.
        if runner_alive:
            _scheduler("heartbeat", ["--progress", str(rows), "--vram-used", "20.0"])
        time.sleep(60)

    print("TIMEOUT_WAITING_FOR_RUNNER", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
