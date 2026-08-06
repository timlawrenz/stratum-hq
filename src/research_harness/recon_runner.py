"""Scheduler-bound arm #37 reconstruction runner.

Owns the FULL scheduler lifecycle in one process (mirrors
stage_b_launcher): request -> poll (atomic claim) -> boot ComfyUI ->
verify VRAM -> activate -> heartbeat -> generate (24x2 + null) ->
stop ComfyUI -> CLIP ViT-L/14 score -> aggregate delta -> write run
artifacts -> release completed. Any exception releases failed.

Invocation (cron-guard-safe, module form):
  bash /tmp/recon_runner.sh   (exports PYTHONPATH=src; execs
  python -m research_harness.recon_runner --plan ...)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .recon import (
    BASELINE_PROMPT,
    CFG,
    CHECKPOINT_NAME,
    CHECKPOINT_SOURCE,
    HEIGHT,
    NEGATIVE_PROMPT,
    NULL_PROMPT,
    SCHEDULER_NAME,
    SAMPLER_NAME,
    STEPS,
    WIDTH,
    ReconError,
    build_frozen_plan,
    build_items,
    load_pilot_items,
)

COMFY_DIR = Path("/mnt/fscache/essdee/ComfyUI")
COMFY_PY = COMFY_DIR / ".venv" / "bin" / "python"
COMFY_MAIN = COMFY_DIR / "main.py"
COMFY_PORT = 8188
SCHEDULER = Path("/mnt/nas-ai-models/gpu-scheduler/gpu_scheduler.py")
PROJECT = "stratum-contextual-specialist-research"
DEFAULT_RUN_ROOT = Path("/mnt/nas-ai-models/research/stratum/stage-b-reconstruction-v1")
DEFAULT_JOB_ID = "stratum-stage-b-reconstruction-v1"
REQUESTED_VRAM_GB = 12
DURATION = "45m"
HEARTBEAT_SECONDS = 55
BOOT_TIMEOUT_SECONDS = 240
GEN_TIMEOUT_SECONDS = 600


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scheduler_call(cmd: str, args: list[str]) -> str:
    proc = subprocess.run(
        [sys.executable, str(SCHEDULER), cmd, *args],
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = (proc.stdout or "").strip()
    if proc.returncode != 0:
        raise ReconError(f"scheduler {cmd} failed rc={proc.returncode}: {out} {proc.stderr[-400:]}")
    return out


def local_vram_used_gb() -> float:
    try:
        raw = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        return float(raw.splitlines()[0]) / 1024.0
    except Exception:
        return 0.0


def comfy_alive() -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen(f"http://127.0.0.1:{COMFY_PORT}/system_stats", timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def boot_comfy(log_path: Path) -> subprocess.Popen[Any]:
    if comfy_alive():
        raise ReconError("ComfyUI already running on port {COMFY_PORT} — refusing to double-boot")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    with open(log_path, "a") as logf:
        proc = subprocess.Popen(
            [str(COMFY_PY), str(COMFY_MAIN), "--listen", "127.0.0.1", "--port", str(COMFY_PORT), "--disable-auto-launch"],
            cwd=str(COMFY_DIR),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    deadline = time.time() + BOOT_TIMEOUT_SECONDS
    while time.time() < deadline:
        if proc.poll() is not None:
            raise ReconError(f"ComfyUI exited early rc={proc.returncode} — see {log_path}")
        if comfy_alive():
            return proc
        time.sleep(3)
    raise ReconError(f"ComfyUI did not become ready within {BOOT_TIMEOUT_SECONDS}s — see {log_path}")


def _comfy_workflow(prompt: str, neg: str, seed: int, prefix: str) -> dict[str, Any]:
    return {
        "3": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT_NAME}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["3", 1]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": WIDTH, "height": HEIGHT, "batch_size": 1}},
        "10": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["3", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
                "seed": seed,
                "steps": STEPS,
                "cfg": CFG,
                "sampler_name": SAMPLER_NAME,
                "scheduler": SCHEDULER_NAME,
                "denoise": 1.0,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }


def comfy_generate(session: Any, workflow: dict[str, Any], gen_timeout: int = GEN_TIMEOUT_SECONDS) -> list[dict[str, str]]:
    import urllib.request

    body = json.dumps({"prompt": workflow, "client_id": session["client_id"]}).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{COMFY_PORT}/prompt", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=gen_timeout) as resp:
        pid = json.loads(resp.read().decode("utf-8"))["prompt_id"]
    deadline = time.time() + gen_timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{COMFY_PORT}/history/{pid}", timeout=30) as resp:
                hist = json.loads(resp.read().decode("utf-8"))
        except Exception:
            time.sleep(3)
            continue
        entry = hist.get(pid)
        if entry is not None and entry.get("status", {}).get("completed") is True:
            outputs = entry.get("outputs", {})
            for node in outputs.values():
                if node.get("images"):
                    return node["images"]
            # completed but no images -> failed silently
            raise ReconError(f"ComfyUI completed prompt {pid} without images: {json.dumps(entry)[:800]}")
        if entry is not None and entry.get("status", {}).get("status_str") == "error":
            raise ReconError(f"ComfyUI prompt {pid} errored: {json.dumps(entry)[:800]}")
        time.sleep(3)
    raise ReconError(f"ComfyUI prompt {pid} timed out after {gen_timeout}s")


def load_clip_model(device: str = "cpu"):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor


def clip_similarity(model: Any, processor: Any, image_a_path: Path, image_b_path: Path, device: str = "cpu") -> float:
    import torch
    from PIL import Image

    a = Image.open(image_a_path).convert("RGB")
    b = Image.open(image_b_path).convert("RGB")
    inputs = processor(images=[a, b], return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        sim = float((feats[0] * feats[1]).sum().item())
    return sim


def run() -> int:
    ap = argparse.ArgumentParser(description="arm #37 reconstruction runner (scheduler-bound)")
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--job-id", default=DEFAULT_JOB_ID)
    ap.add_argument("--plan", type=Path, default=Path("/home/tim/source/activity/stratum-hq-stage-b-experiment/research/stage-b-plans/stage-b-reconstruction-v1.json"))
    ap.add_argument("--request-if-missing", action="store_true")
    ap.add_argument("--no-request", action="store_true", help="skip scheduler request (job already queued)")
    args = ap.parse_args()

    run_root = args.run_root
    if run_root.exists():
        raise ReconError(f"run root already exists: {run_root} — must not pre-exist")

    # Frozen plan must already exist (pre-registered BEFORE this run).
    plan_path = args.plan
    if not plan_path.is_file():
        raise ReconError(f"frozen plan missing: {plan_path} — pre-register before running")
    plan = json.loads(plan_path.read_text())
    if plan.get("status") != "preregistered":
        raise ReconError("plan status must be 'preregistered'")

    checkpoint_src = Path(CHECKPOINT_SOURCE)
    if not checkpoint_src.is_file():
        raise ReconError(f"checkpoint missing: {checkpoint_src}")
    ckpt_link = COMFY_DIR / "models" / "checkpoints" / CHECKPOINT_NAME
    if not ckpt_link.exists():
        os.symlink(checkpoint_src, ckpt_link)
    if _sha256_file(checkpoint_src) != plan["generation_settings"]["checkpoint_sha256"]:
        raise ReconError("checkpoint sha256 changed since plan freeze")

    job_id = args.job_id
    target_gpu = "4090"
    claimed = False
    child_proc: subprocess.Popen[Any] | None = None
    started = _now()
    progress = [0]

    def _hb_loop(stop: threading.Event, run_root_p: Path) -> None:
        try:
            while not stop.is_set():
                stop.wait(HEARTBEAT_SECONDS)
                if stop.is_set():
                    return
                scheduler_call(
                    "heartbeat",
                    ["--gpu", target_gpu, "--job-id", job_id, "--progress", str(progress[0]),
                     "--vram-used", f"{local_vram_used_gb():.2f}"],
                )
        except Exception as e:
            print(f"heartbeat loop error: {e}", flush=True)

    try:
        if not args.no_request and args.request_if_missing:
            req = scheduler_call(
                "request",
                ["--gpu", target_gpu, "--project", PROJECT, "--vram", str(REQUESTED_VRAM_GB),
                 "--duration", DURATION, "--job-id", job_id],
            )
            if req != job_id:
                raise ReconError(f"scheduler request returned {req!r}")
        polled = scheduler_call("poll", ["--gpu", target_gpu, "--job-id", job_id])
        if polled != "claimed":
            # Still queued/behind; exit silently, re-invoked later.
            print(json.dumps({"status": "queued", "poll_result": polled, "job_id": job_id}))
            return 0
        claimed = True
        run_root.mkdir(parents=True, exist_ok=False)

        comfy_log = run_root / "comfy-boot.log"
        child_proc = boot_comfy(comfy_log)
        booted_at = _now()
        used = local_vram_used_gb()
        if used < 1.0:
            raise ReconError("ComfyUI booted without observable local GPU memory activity")
        act = scheduler_call("activate", ["--gpu", target_gpu, "--job-id", job_id, "--progress-unit", "image"])
        if act != "activated":
            raise ReconError(f"scheduler activate failed: {act}")

        stop_hb = threading.Event()
        hb_thread = threading.Thread(target=_hb_loop, args=(stop_hb, run_root), daemon=True)
        hb_thread.start()

        session = {"client_id": str(uuid.uuid4())}
        # Smoke: 2 steps, tiny, to validate the frozen checkpoint/VAE path.
        smoke_wf = _comfy_workflow("smoke", NEGATIVE_PROMPT, 1234, "recon-smoke")
        smoke_wf["10"]["inputs"]["steps"] = 2
        smoke_wf["5"]["inputs"] = {"width": 64, "height": 64, "batch_size": 1}
        comfy_generate(session, smoke_wf)
        progress[0] = 1

        outputs_dir = run_root / "outputs"
        rows: list[dict[str, Any]] = []
        items = build_items()
        images: list[dict[str, Any]] = []

        def _save_image(image_meta: dict[str, str], cond: str, image_id: str, seed: int) -> Path:
            src = COMFY_DIR / "output" / image_meta.get("subfolder", "") / image_meta["filename"]
            dst_dir = outputs_dir / cond
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{image_id}_{seed}.png"
            if not src.is_file():
                raise ReconError(f"generated image missing: {src}")
            shutil.copy2(src, dst)
            return dst

        for idx, item in enumerate(items):
            for cond in ("recon-ctx4k", "recon-baseline"):
                prompt_text = item["prompts"][cond]
                seed = item["seed"]
                wf = _comfy_workflow(prompt_text, NEGATIVE_PROMPT, seed, f"recon-{cond}")
                metas = comfy_generate(session, wf)
                dst = _save_image(metas[0], cond, item["image_id"], seed)
                images.append({"image_id": item["image_id"], "condition": cond, "seed": seed, "png": str(dst)})
                progress[0] += 1
                print(f"generated {cond} {item['image_id']} seed={seed}", flush=True)
        # Null calibration (2 images, fixed seed).
        for k in range(2):
            wf = _comfy_workflow(NULL_PROMPT, NEGATIVE_PROMPT, 9000 + k, "recon-null")
            metas = comfy_generate(session, wf)
            dst = _save_image(metas[0], "recon-null", f"null{k}", 9000 + k)
            images.append({"image_id": f"null{k}", "condition": "recon-null", "seed": 9000 + k, "png": str(dst)})
            progress[0] += 1

        stop_hb.set()
        hb_thread.join(timeout=5)

        # Tear down ComfyUI BEFORE the CLIP pass so VRAM is free.
        if child_proc is not None:
            child_proc.terminate()
            try:
                child_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child_proc.kill()
            child_proc = None

        print("generation complete; scoring with CLIP ViT-L/14", flush=True)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        model, processor = load_clip_model("cpu")
        src_by_id = {i["image_id"]: i["source_abs"] for i in items}
        sim_rows: list[dict[str, Any]] = []
        null_sims = []
        for img in images:
            if img["condition"] == "recon-null":
                null_sims.append(clip_similarity(model, processor, Path(img["png"]), Path(src_by_id[list(src_by_id)[0]])))
                continue
            sim = clip_similarity(model, processor, Path(img["png"]), Path(src_by_id[img["image_id"]]))
            sim_rows.append({**img, "sim": round(sim, 6)})
        ctx_map = {r["image_id"]: r["sim"] for r in sim_rows if r["condition"] == "recon-ctx4k"}
        base_map = {r["image_id"]: r["sim"] for r in sim_rows if r["condition"] == "recon-baseline"}
        if set(ctx_map) != set(base_map):
            raise ReconError("mismatched paired similarities")
        paired = [
            {"image_id": iid, "sim_ctx4k": ctx_map[iid], "sim_baseline": base_map[iid]}
            for iid in sorted(ctx_map)
        ]
        from .recon import aggregate_deltas

        agg = aggregate_deltas(paired)
        agg["null_floor_similarity_mean"] = round(sum(null_sims) / len(null_sims), 6) if null_sims else None

        records = [
            {
                "image_id": r["image_id"],
                "condition": r["condition"],
                "seed": r["seed"],
                "sim_clip_vitl14": r["sim"],
                "png": r["png"],
            }
            for r in sim_rows
        ]
        records += [
            {"image_id": f"null{k}", "condition": "recon-null", "seed": 9000 + k,
             "sim_clip_vitl14": null_sims[k] if k < len(null_sims) else None}
            for k in range(len(null_sims))
        ]
        provenance = {
            "status": "PENDING_INDEPENDENT_REVIEW",
            "verdict_stamp": "PENDING_HUMAN_SPOT_CHECK",
            "arm": "reconstruction",
            "arm_issue": 37,
            "plan_id": plan["plan_id"],
            "job_id": job_id,
            "gpu": target_gpu,
            "started_at": started,
            "generation_complete_at": _now(),
            "checkpoint_sha256": plan["generation_settings"]["checkpoint_sha256"],
            "sampler": plan["generation_settings"]["sampler_name"],
            "steps": plan["generation_settings"]["steps"],
            "cfg": plan["generation_settings"]["cfg"],
            "size": f"{WIDTH}x{HEIGHT}",
            "seed_rule": plan["generation_settings"]["seed_per_item"],
            "clip_model": "openai/clip-vit-large-patch14",
            "device": "cpu",
            "null_prompt": NULL_PROMPT,
            "aggregate": agg,
        }
        (run_root / "stage-b-plan.json").write_text(json.dumps(plan, indent=1))
        (run_root / "run-provenance.json").write_text(json.dumps(provenance, indent=1))
        (run_root / "delta.json").write_text(json.dumps({"aggregate": agg}, indent=1))
        (run_root / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")

        rel = scheduler_call("release", ["--gpu", target_gpu, "--job-id", job_id, "--status", "completed"])
        if rel not in {"released", "already_released"}:
            raise ReconError(f"scheduler release failed: {rel}")
        claimed = False
        print(json.dumps({"status": "completed", "job_id": job_id, "aggregate": agg}), flush=True)
        return 0
    except Exception as e:
        print(f"recon runner error: {e}", flush=True)
        if claimed:
            try:
                scheduler_call("release", ["--gpu", target_gpu, "--job-id", job_id, "--status", "failed"])
                claimed = False
            except Exception as rel_err:
                print(f"failed to release as failed: {rel_err}", flush=True)
        if child_proc is not None:
            try:
                child_proc.terminate()
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    sys.exit(run())