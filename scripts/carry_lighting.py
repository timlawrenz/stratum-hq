"""Deterministic cross-check: lighting-vocab carry in evidence captions vs matched baseline.

Independent of the LLM claim-support reviewer. For each frozen item, count whether the
context-raw-lighting caption carries >=1 declared lighting-evidence vocabulary trace that is
NOT present in the matched context-raw-no-evidence caption for the same item.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CANDIDATE = Path("/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json")
RUN = Path("/mnt/nas-ai-models/research/stratum/stage-b-lighting-v2/outputs")

# Declared-evidence vocabulary traces (luma band, DR band, shadow band, direction.
# Normalized lowercase; substring match on token stems).
TRACES = [
    "low-key", "lowkey", "dim", "bright", "moderately lit", "brightly lit",
    "contrast", "shadow", "lit from the front", "front-left", "front-right",
    "backlit", "rim-lit", "exposure", "evenly lit", "key light", "highlight",
]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def _carry(caption: str, baseline: str) -> list[str]:
    base = _norm(baseline)
    hits = []
    for tr in TRACES:
        if tr in _norm(caption) and tr not in base:
            # avoid substring-of-substring duplicates
            if not any(tr in other for other in hits):
                hits.append(tr)
    return hits


def main() -> int:
    cand = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    items = cand["items"]
    base_dir = RUN / "context-raw-no-evidence"
    ev_dir = RUN / "context-raw-lighting"
    n_carry = 0
    examples = []
    for it in items:
        iid = it["image_id"]
        base = (base_dir / f"{iid}.txt").read_text(encoding="utf-8")
        ev = (ev_dir / f"{iid}.txt").read_text(encoding="utf-8")
        hits = _carry(ev, base)
        if hits:
            n_carry += 1
            examples.append((iid, hits))
    print(f"items with NEW lighting-vocab trace beyond matched baseline: {n_carry}/{len(items)}")
    for iid, hits in examples:
        print(f"  {iid}: {hits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
