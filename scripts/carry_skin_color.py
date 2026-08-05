"""Deterministic carry cross-check for arm #31 (LLM-review-independent).

For each cohort item, check whether the context-raw-skin-color caption carries
skin-tone vocabulary beyond its matched context-raw-no-evidence baseline, and
whether declared evidence traces (tone name, coverage) appear. Mirrors the
hair arm's carry check.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

RUN_ROOT = Path("/mnt/nas-ai-models/research/stratum/stage-b-skin-color-v1")

BASE_COND = "context-raw-no-evidence"
VARIANT_COND = "context-raw-skin-color"

# Skin-tone vocabulary tokens the aggregator is expected to emit when the
# evidence is carried (neutral descriptive words, not claims of exact shade).
TONE_TRACE = re.compile(
    r"\b(skin|tone|complexion|fair|light|light medium|medium|tan|brown|dark brown|deep)\b",
    re.IGNORECASE,
)


def main() -> int:
    records: dict[tuple[str, str], dict] = {}
    with (RUN_ROOT / "records.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                records[(r["image_id"], r["condition_id"])] = r

    items = sorted({r[0] for r in records})
    carried = 0
    no_new = 0
    declared_trace_ok = 0
    details = []
    for image_id in items:
        base = records[(image_id, BASE_COND)]["caption"]
        variant = records[(image_id, VARIANT_COND)]["caption"]
        payload = records[(image_id, VARIANT_COND)]["evidence_payload"]
        base_has = bool(TONE_TRACE.search(base))
        variant_has = bool(TONE_TRACE.search(variant))
        new_trace = variant_has and not base_has
        if new_trace:
            carried += 1
        else:
            no_new += 1
        declared_ok = False
        tone = payload.get("skin_tone_name")
        if payload.get("exposed_skin_present") and tone and tone.lower() in variant.lower():
            declared_ok = True
        if declared_ok:
            declared_trace_ok += 1
        details.append({
            "image_id": image_id,
            "base_has_tone_vocab": base_has,
            "variant_has_tone_vocab": variant_has,
            "new_trace": new_trace,
            "declared_tone": tone,
            "declared_tone_in_caption": declared_ok,
        })

    print(json.dumps({
        "items": len(items),
        "carried_new_skin_tone_vocab": carried,
        "no_new_trace": no_new,
        "declared_tone_name_in_variant_caption": declared_trace_ok,
        "details": details,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
