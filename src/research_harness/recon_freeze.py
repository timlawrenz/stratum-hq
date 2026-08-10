"""Pre-register (freeze) the arm #37 reconstruction plan.

Computes and pins every frozen input (checkpoint sha256, pilot manifest,
context4k.md fingerprints) into
research/stage-b-plans/stage-b-reconstruction-v1.json BEFORE any
generation happens. Refuses to overwrite an existing preregistered plan.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from .recon import CHECKPOINT_NAME, CHECKPOINT_SOURCE, ReconError, build_frozen_plan, load_pilot_items, load_context4k_artifact

CHECKPOINT_SRC = Path(CHECKPOINT_SOURCE)
PLAN_OUT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment/research/stage-b-plans/stage-b-reconstruction-v1.json")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if PLAN_OUT.exists():
        existing = json.loads(PLAN_OUT.read_text())
        if existing.get("status") == "preregistered":
            raise ReconError(f"plan already preregistered at {PLAN_OUT} — refusing to overwrite")
    if not CHECKPOINT_SRC.is_file():
        raise ReconError(f"checkpoint missing: {CHECKPOINT_SRC}")
    # Validate every input artifact exists before freezing.
    for item in load_pilot_items():
        load_context4k_artifact(item["image_id"])
    ckpt_sha = _sha256_file(CHECKPOINT_SRC)
    plan = build_frozen_plan(Path("/home/tim/source/activity/stratum-hq-stage-b-experiment"), checkpoint_sha256=ckpt_sha)
    PLAN_OUT.write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps({"status": "preregistered", "plan": str(PLAN_OUT), "checkpoint_sha256": ckpt_sha,
                      "items": len(plan["pilot_manifest"]["items"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())