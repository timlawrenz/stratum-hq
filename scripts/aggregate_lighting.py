"""Post-review: aggregate claim support + paired sign test from the lighting review root."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

from research_harness.autonomous import aggregate_claim_support  # noqa: E402

REVIEW = "/mnt/nas-ai-models/research/stratum/stage-b-lighting-v2-review"


def main() -> int:
    agg = aggregate_claim_support(REVIEW)
    print(json.dumps(agg, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
