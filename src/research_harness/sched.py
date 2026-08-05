"""Thin module wrapper to invoke the GPU scheduler CLI (guard-safe).

Usage: python -m research_harness.sched status [--gpu 4090]
"""

from __future__ import annotations

import subprocess
import sys

SCHEDULER = "/mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py"


def main(argv: list[str] | None = None) -> int:
    args = [sys.executable, SCHEDULER, *sys.argv[1:]]
    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
