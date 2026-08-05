"""Compute the arm #31 skin-color evidence-only verdict from completed reviews.

Loads the Stage-B run (records.jsonl) + independent reviews (reviews.jsonl),
aggregates claim-support counts per condition, computes the evidence-only
(cond 3 -> cond 4) paired delta and a one-sided binomial sign test on
supported claims per item, and prints the numbers needed for
`autonomous-verdict`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

RUN_ROOT = Path("/mnt/nas-ai-models/research/stratum/stage-b-skin-color-v1")
REVIEW_ROOT = Path("/mnt/nas-ai-models/research/stratum/stage-b-skin-color-v1-review")

BASE_COND = "context-raw-no-evidence"
VARIANT_COND = "context-raw-skin-color"


def _binom_p(k: int, n: int, p: float = 0.5) -> float:
    """One-sided binomial tail P(X >= k | n, p)."""
    if n <= 0:
        return 1.0
    total = 0.0
    for x in range(k, n + 1):
        total += math.comb(n, x) * (p ** x) * ((1.0 - p) ** (n - x))
    return total


def main() -> int:
    records: list[dict] = []
    with (RUN_ROOT / "records.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    reviews: list[dict] = []
    with (REVIEW_ROOT / "reviews.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                reviews.append(json.loads(line))

    # Join review verdicts to records by (image_id, condition_id).
    by_key: dict[tuple[str, str], dict] = {}
    for review in reviews:
        key = (review["image_id"], review["condition_id"])
        by_key[key] = review

    def counts(cond: str) -> dict[str, int]:
        out = {"supported": 0, "unsupported": 0, "omissions": 0,
               "contradictions": 0, "abstentions": 0, "items": 0, "blank": 0}
        for record in records:
            if record["condition_id"] != cond:
                continue
            out["items"] += 1
            key = (record["image_id"], record["condition_id"])
            review = by_key.get(key)
            if review is None:
                continue
            for field in ("supported", "unsupported", "omissions", "contradictions", "abstentions"):
                out[field] += len(review.get(field, []))
            if not any(review.get(f, []) for f in ("supported", "unsupported", "omissions",
                                                   "contradictions", "abstentions")):
                out["blank"] += 1
        return out

    base = counts(BASE_COND)
    variant = counts(VARIANT_COND)

    # Paired per-item sign test on supported claims (cond 3 -> cond 4).
    pairs: list[tuple[int, int]] = []
    for record in records:
        if record["condition_id"] != BASE_COND:
            continue
        image_id = record["image_id"]
        bkey = (image_id, BASE_COND)
        vkey = (image_id, VARIANT_COND)
        b = by_key.get(bkey)
        v = by_key.get(vkey)
        if b is None or v is None:
            continue
        pairs.append((len(b.get("supported", [])), len(v.get("supported", []))))

    n_improve = sum(1 for b, v in pairs if v > b)
    n_worsen = sum(1 for b, v in pairs if v < b)
    n_tie = sum(1 for b, v in pairs if v == b)
    n_paired = n_improve + n_worsen
    p_supported = _binom_p(n_improve, n_paired) if n_paired else 1.0

    # Support ratio on supported vs unsupported (mirrors arms #4/#29/#32/#30).
    base_ratio = base["supported"] / (base["supported"] + base["unsupported"]) if (base["supported"] + base["unsupported"]) else 0.0
    variant_ratio = variant["supported"] / (variant["supported"] + variant["unsupported"]) if (variant["supported"] + variant["unsupported"]) else 0.0

    result = {
        "base_condition": BASE_COND,
        "variant_condition": VARIANT_COND,
        "base": base,
        "variant": variant,
        "evidence_only_delta": {
            "supported": variant["supported"] - base["supported"],
            "unsupported": variant["unsupported"] - base["unsupported"],
            "omissions": variant["omissions"] - base["omissions"],
            "contradictions": variant["contradictions"] - base["contradictions"],
            "abstentions": variant["abstentions"] - base["abstentions"],
        },
        "support_ratio": {"base": round(base_ratio, 4), "variant": round(variant_ratio, 4),
                          "delta": round(variant_ratio - base_ratio, 4)},
        "paired_sign_test": {"n_improve": n_improve, "n_worsen": n_worsen, "n_tie": n_tie,
                             "n_paired": n_paired, "p_supported_one_sided": round(p_supported, 6)},
        "items": len(pairs),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
