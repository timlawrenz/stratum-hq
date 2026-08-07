#!/usr/bin/env python
"""Scheduler-owned lifecycle driver for the arm #47 VLM dense-description batch.

Owns the whole lifecycle for the Strix run (the generator script itself is
stateless/resumable and only writes under the noncanonical stage root):
  request (strix, 36GB) -> poll (atomic claim) -> activate ->
  launch via ssh (detached, log to NAS stage root) ->
  heartbeat (progress = completed item dirs, real artifacts) ->
  on vlm-done.json marker: evict ollama + release completed; else release failed.

Never run this in foreground with a short timeout — it owns the lease. Run it
detached (terminal background) and let it drive to release; block-check with
state.json / the stage marker.

Reads scheduler state only via the shared gpu_scheduler.py (run through the
venv python, never as a bare executable — NAS is noexec).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCHEDULER = "/mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py"
PY = "/home/tim/source/activity/stratum-hq/.venv/bin/python"
STAGE = Path("/mnt/nas-ai-models/research/stratum/stage-b-vlm-dense-v1")
DONE = STAGE / "vlm-done.json"
GEN_REMOTE = "/mnt/nas-ai-models/research/stratum/stage-b-vlm-dense-v1/vlm_dense_generate.py"

GPU = "strix"
JOB_ID = "stratum-vlm-dense-blocks-v1"
PROJECT = "stratum-contextual-specialist-research"
VRAM = "36"
DURATION = "5h"
WALL_CAP = 5 * 3600 + 600  # a little beyond the scheduler duration for tail
POLL_SLOT_MAX = 90 * 60
HEARTBEAT_SECONDS = 120


def _sched(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, SCHEDULER, *args],
                          capture_output=True, text=True, check=False, timeout=60)


def _ssh(args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["ssh", "max395", args], capture_output=True, text=True,
                          check=False, timeout=120)


def _blocks_done() -> int:
    if not (STAGE / "blocks").is_dir():
        return 0
    return sum(1 for p in (STAGE / "blocks").iterdir() if (p / "vlm-dense.json").is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-prefs", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps({"job_id": JOB_ID, "gpu": GPU, "stage": str(STAGE),
                      "generator": GEN_REMOTE, "vram_gb": VRAM, "duration": DURATION,
                      "wall_cap_s": WALL_CAP}, sort_keys=True), flush=True)
    if args.dry_prefs:
        return 0

    # 1. Queue (idempotent-ish: delete a stale job file, never another's).
    req = _sched(["request", "--gpu", GPU, "--project", PROJECT, "--vram", VRAM,
                  "--duration", DURATION, "--job-id", JOB_ID])
    if req.returncode != 0 and "already exists" not in req.stderr + req.stdout:
        print(f"REQUEST FAILED: {req.stderr.strip() or req.stdout.strip()}", flush=True)
        return 1
    print("request ok", flush=True)

    # 2. Poll until claimed (poll = atomic claim).
    claimed = False
    deadline = time.time() + POLL_SLOT_MAX
    while time.time() < deadline:
        if DONE.exists():
            print("DONE marker already present; nothing to run", flush=True)
            return 0
        p = _sched(["poll", "--gpu", GPU, "--job-id", JOB_ID])
        out = (p.stdout + p.stderr).strip()
        if "claimed" in out.lower() or "claim" in out.lower():
            claimed = True
            print(f"poll claimed: {out[:200]}", flush=True)
            break
        print(f"poll wait: {out[:120]}", flush=True)
        time.sleep(30)
    if not claimed:
        print("SLOT NOT CLAIMED within budget; releasing nothing, exiting", flush=True)
        return 1

    # 3. Activate.
    act = _sched(["activate", "--gpu", GPU, "--job-id", JOB_ID, "--progress-unit", "items"])
    print(f"activate: {(act.stdout + act.stderr).strip()[:200]}", flush=True)

    # 4. Launch via ssh (detached on Strix; log to NAS stage root so I can read it).
    launch = _ssh(f"mkdir -p {STAGE} && "
                  f"cd {STAGE} && "
                  "setsid nohup python3 " + GEN_REMOTE + f" > {STAGE}/run.log 2>&1 & "
                  f"sleep 3; echo STARTED")
    print(f"launch: {launch.stdout.strip()[:200] or launch.stderr.strip()[:200]}", flush=True)
    time.sleep(20)
    alive = _ssh("pgrep -f vlm_dense_generate || true")
    print(f"alive check: {alive.stdout.strip()[:120]}", flush=True)
    try:
        log_head = (STAGE / "run.log").read_text(encoding="utf-8", errors="replace")[:300]
    except OSError:
        log_head = "<no run.log yet>"
    print(f"run.log head: {log_head[:300]}", flush=True)

    if not alive.stdout.strip():
        print("WARN: generator process not detected after launch; will re-check", flush=True)

    # 5. Heartbeat loop to completion.
    start = time.time()
    last_hb = 0.0
    while time.time() - start < WALL_CAP:
        if DONE.exists():
            break
        done_now = _blocks_done()
        if time.time() - last_hb >= HEARTBEAT_SECONDS:
            hb = _sched(["heartbeat", "--gpu", GPU, "--job-id", JOB_ID,
                         "--progress", str(done_now), "--vram-used", VRAM])
            if hb.returncode != 0:
                print(f"heartbeat warn: {hb.stderr.strip()[:120]}", flush=True)
            last_hb = time.time()
            print(f"heartbeat progress={done_now} (elapsed {int(time.time()-start)}s)", flush=True)
        # If the process died early, surface it once.
        if done_now == 0 and time.time() - start > 300:
            alive2 = _ssh("pgrep -f vlm_dense_generate || true")
            if not alive2.stdout.strip() and not DONE.exists():
                print("GENERATOR DIED early: no process, no done marker", flush=True)
                print((STAGE / "run.log").read_text(encoding="utf-8", errors="replace")[-2000:], flush=True)
                _sched(["release", "--gpu", GPU, "--job-id", JOB_ID, "--status", "failed"])
                return 1
        time.sleep(25)

    if not DONE.exists():
        print("TIMED OUT waiting for vlm-done.json; releasing failed", flush=True)
        _sched(["release", "--gpu", GPU, "--job-id", JOB_ID, "--status", "failed"])
        return 1

    done = json.loads(DONE.read_text(encoding="utf-8"))
    # 6. Evict the resident model, then release completed.
    _ssh("python3 -c \"import urllib.request,json; urllib.request.urlopen("
         "urllib.request.Request('http://127.0.0.1:11434/api/generate',"
         "data=json.dumps({'model':'qwen3-vl:32b','keep_alive':0,'stream':False}).encode(),"
         "headers={'Content-Type':'application/json'}), timeout=30)\" || true")
    rel = _sched(["release", "--gpu", GPU, "--job-id", JOB_ID, "--status", "completed"])
    print(f"release: {(rel.stdout + rel.stderr).strip()[:200]}", flush=True)
    print(json.dumps({"status": "completed", "job_id": JOB_ID, "item_count": done.get("item_count"),
                      "attention": done.get("attention_summary"),
                      "leakers": done.get("leak_summary", {}).get("leaking_items"),
                      "done_marker": str(DONE)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
