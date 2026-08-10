#!/usr/bin/env python3
"""Prose-caption2 -> ComfyUI SDXL demo runner (stdlib only, strix/ROCm).

Reproduces arm-37 generation settings (Juggernaut XL, 832x1216, dpmpp_2m,
karras, 28 steps, cfg 7.0, per-item seed = sha256(image_id)[:8]) with the
full-prose caption2 prompts as the ONLY changed axis.

Conditions per pilot item:
  - prose      : deterministic full-prose caption2 (renderer output)
  - prose-agg  : aggregator (gemma3:27b via local ollama) caption2
  - null       : 'zzzzzzzzzz' floor control (1 image)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

COMFY_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = Path(os.environ.get("PROSE_OUTPUT_DIR", "/home/tim/activity/ComfyUI/output"))
RUN_ROOT = Path("/mnt/nas-ai-models/research/stratum/prose-caption2-demo-v1")
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

CHECKPOINT = "Juggernaut_XL_v1759168.safetensors"
STEPS, CFG, WIDTH, HEIGHT = 28, 7.0, 832, 1216
SAMPLER, SCHEDULER = "dpmpp_2m", "karras"

CAPTION2_SYSTEM = (
    "You are an expert descriptive captioner for a text-to-image dataset. "
    "Write a single, rich, dense paragraph describing the image. "
    "The claims below are ground-truth determinations; you must never "
    "contradict them, never translate machine measurements into invented "
    "attributes, never add objects, and always keep the description strictly "
    "objective prose with no preamble like 'This image shows'. "
    "Start the description immediately."
)

PILOTS = [
    "03r1r5psuxprfkhomitcz5vh9w1c",
    "07hyx5wjk5rc339v2s3e2orcoorr",
    "0f3fdmh6iackl6049wmrm9n2p4i5",
    "058v0mg1bbxthvatdw324k8j4b0s",
    "08v25q5524t2u0zl0xtnzi6bd22f",
]
AGG_IDS = (
    set()
    if os.environ.get("PROSE_SKIP_AGG")
    else {"03r1r5psuxprfkhomitcz5vh9w1c", "07hyx5wjk5rc339v2s3e2orcoorr"}
)


def item_seed(image_id: str) -> int:
    seed = int(hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:8], 16)
    return seed if seed != 0 else 1


def workflow(prompt: str, seed: int, prefix: str) -> dict:
    return {
        "3": {"class_type": "KSampler", "inputs": {
            "cfg": CFG, "denoise": 1.0, "latent_image": ["5", 0],
            "model": ["7", 0], "negative": ["6", 0], "positive": ["4", 0],
            "sampler_name": SAMPLER, "scheduler": SCHEDULER,
            "seed": seed, "steps": STEPS}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": prompt}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"batch_size": 1, "height": HEIGHT, "width": WIDTH}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 1], "text": ""}},
        "7": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}},
        "8": {"class_type": "VAEDecodeTiled", "inputs": {
            "samples": ["3", 0], "vae": ["7", 2],
            "tile_size": 512, "overlap": 64,
            "temporal_size": 64, "temporal_overlap": 8}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["8", 0]}},
    }


def post_json(url: str, payload: dict, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get_json(url: str, timeout: int = 30) -> dict:
    """GET (ComfyUI /history is a GET endpoint; POST raises 405)."""
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def wait_server(timeout_s: int = 240) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(COMFY_URL + "/system_stats", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(3)
    raise SystemExit("ComfyUI did not become ready in time")


def generate(prompt: str, seed: int, prefix: str) -> Path:
    pid = post_json(COMFY_URL + "/prompt", {"prompt": workflow(prompt, seed, prefix)})["prompt_id"]
    deadline = time.time() + 1500  # ROCm first-pass kernel tuning can exceed 10 min
    while time.time() < deadline:
        try:
            hist = get_json(f"{COMFY_URL}/history/{pid}", timeout=30)
        except Exception:
            hist = {}
        if pid in hist:
            outs = hist[pid].get("outputs", {}).get("9", {}).get("images", [])
            if outs:
                img = outs[0]
                src = OUTPUT_DIR / img.get("subfolder", "") / img["filename"]
                return src
        time.sleep(2)
    raise SystemExit(f"generation timed out: {prefix}")


def aggregator_caption(image_id: str, claims_text: str) -> str:
    prompt = (
        f"{CAPTION2_SYSTEM}\n\nGROUND-TRUTH DETERMINATIONS:\n{claims_text}\n\nCaption:"
    )
    body = {"model": "gemma3:27b", "prompt": prompt, "stream": False,
            "num_predict": 500, "keep_alive": -1, "options": {"temperature": 0.4}}
    out = post_json(OLLAMA_URL, body, timeout=600)
    return out["response"].strip()


def main() -> None:
    wait_server()
    manifest = []
    for image_id in PILOTS:
        seed = item_seed(image_id)
        cap_path = RUN_ROOT / image_id / "caption2.txt"
        claims = cap_path.read_text().strip()
        jobs = [("prose", claims)]
        if image_id in AGG_IDS:
            agg = aggregator_caption(image_id, claims)
            (RUN_ROOT / image_id / "caption2-aggregator.txt").write_text(agg + "\n")
            jobs.append(("prose-agg", agg))
        for cond, prompt in jobs:
            out = generate(prompt, seed, f"prose-{image_id}-{cond}")
            dst = RUN_ROOT / image_id / f"recon-{cond}.png"
            shutil.copyfile(out, dst)
            manifest.append({"image_id": image_id, "condition": cond,
                             "seed": seed, "png": str(dst)})
            print(f"OK {image_id} {cond} -> {dst}", flush=True)
    # null floor control
    out = generate("zzzzzzzzzz", item_seed("null-control"), "prose-null")
    dst = RUN_ROOT / "_null.png"
    shutil.copyfile(out, dst)
    manifest.append({"image_id": None, "condition": "null", "png": str(dst)})
    (RUN_ROOT / "_recon-manifest.json").write_text(json.dumps(manifest, indent=2))
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()