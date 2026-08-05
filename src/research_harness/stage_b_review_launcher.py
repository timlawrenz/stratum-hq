"""Scheduler-bound driver for the independent Stage-B review pass.

Queues and holds a local GPU slot (qwen3-vl:32b reviewer), runs the reviewer
over the frozen run, then releases. Reads only the frozen run, the frozen
sources, and the candidate manifest; writes only a noncanonical review root.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from research_harness.stage_b_review import (
    ReviewSettings,
    StageBReviewError,
    build_review_plan,
    execute_review,
)

SCHEDULER = "/mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py"
REVIEW_ROOT = Path("/mnt/nas-ai-models/research/stratum/stage-b-first500-parity-v1-review")
RUN_ROOT = Path("/mnt/nas-ai-models/research/stratum/stage-b-first500-parity-v1")
SOURCE_ROOT = Path("/mnt/nas-ai-models/training-data/crawlr/approved")
CANDIDATE_MANIFEST = Path("/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json")
JOB_ID = "stratum-stage-b-adversarial-review-v1"


def _scheduler(action: str, args: list[str]) -> str:
    completed = subprocess.run([sys.executable, SCHEDULER, action, *args],
                               capture_output=True, text=True, check=False, timeout=60)
    output = completed.stdout.strip()
    if completed.returncode != 0:
        raise StageBReviewError(f"scheduler {action} failed: {completed.stderr.strip() or output}")
    return output


def _settings() -> ReviewSettings:
    return ReviewSettings(
        model_name="gemma4:e4b",
        digest="c6eb396dbd59",
        endpoint="http://127.0.0.1:11434/api/generate",
        temperature=0.0,
        seed=20260804,
        num_predict=2000,
        review_items="all",
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="stage-b-review-launcher")
    parser.add_argument("--run-root", default=str(RUN_ROOT))
    parser.add_argument("--review-root", default=str(REVIEW_ROOT))
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--job-id", default=JOB_ID)
    parser.add_argument("--request", action="store_true")
    args = parser.parse_args(argv)

    run_root = Path(args.run_root)
    review_root = Path(args.review_root)
    source_root = Path(args.source_root)
    job_id = args.job_id
    request_if_missing = args.request

    candidate = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
    settings = _settings()
    build_review_plan(settings, run_root, candidate["manifest_fingerprint"])

    claimed = False
    gpu_activity_seen = False
    try:
        if request_if_missing:
            result = _scheduler("request", [
                "--gpu", "4090", "--project", "stratum-contextual-specialist-research",
                "--vram", "22", "--duration", "1h", "--job-id", job_id,
            ])
            if result != job_id:
                raise StageBReviewError(f"scheduler request returned unexpected job identity: {result}")
            print(json.dumps({"status": "queued", "job_id": job_id}, sort_keys=True))
            return 0

        poll = _scheduler("poll", ["--gpu", "4090", "--job-id", job_id])
        if poll != "claimed":
            return 0  # queued / not_my_turn / busy — stay quiet
        claimed = True

        activation = _scheduler("activate", ["--gpu", "4090", "--job-id", job_id, "--progress-unit", "item"])
        if activation != "activated":
            raise StageBReviewError(f"activate returned {activation}")
        _scheduler("heartbeat", ["--gpu", "4090", "--job-id", job_id, "--progress", "0", "--vram-used", "20.0"])
        gpu_activity_seen = True

        result = execute_review(
            settings, run_root, source_root, candidate["manifest_fingerprint"], review_root,
        )
        _scheduler("release", ["--gpu", "4090", "--job-id", job_id, "--status", "completed"])
        claimed = False
        print(json.dumps({"status": "completed", "review_root": result["review_root"],
                          "record_count": result["record_count"], "gpu_activity_seen": gpu_activity_seen}, sort_keys=True))
        return 0
    except StageBReviewError as exc:
        if claimed:
            try:
                _scheduler("release", ["--gpu", "4090", "--job-id", job_id, "--status", "failed"])
            except Exception:
                pass
        print(f"stage-b-review: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
