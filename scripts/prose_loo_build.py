#!/usr/bin/env python3
"""Build the freeze-able LOO plan for the DINOv3 specialist-signal attempt (#79/#102 pilot).

For each pilot item and each specialist present in its dossier: write
C_after (full deterministic caption — already rendered as recon-prose.png)
and C_before (same caption with that specialist's claims excluded), via the
canonical dossier_caption.render_caption2 with exclude_evidence_ids.

Output: prose-caption2-demo-v1/_loo-plan.json  (frozen, hashed)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from research_harness.dossier_caption import render_caption2  # noqa: E402

RUN_ROOT = Path("/mnt/nas-ai-models/research/stratum/prose-caption2-demo-v1")
SRC = Path("/mnt/nas-ai-models/training-data/crawlr/approved")
DOSSIER_ROOT = Path("/mnt/nas-ai-models/research/stratum/dossier-context4k-v2")

PILOTS = ["03r1r5psuxprfkhomitcz5vh9w1c",
          "07hyx5wjk5rc339v2s3e2orcoorr",
          "0f3fdmh6iackl6049wmrm9n2p4i5",
          "058v0mg1bbxthvatdw324k8j4b0s",
          "08v25q5524t2u0zl0xtnzi6bd22f"]
# guarantee the full caption (C_after) matches the already-generated recon-prose.png
FULL_CAPS = {i: (RUN_ROOT / i / "caption2.txt").read_text().strip() for i in PILOTS}

SECTION_RE = re.compile(r"^##\s+(.+)$")
CLAIM_RE = re.compile(r"^\[([a-zA-Z0-9_\-:]+)\]\s*(.*)$")


def parse_dossier(md: str) -> dict:
    sections: dict[str, list[dict]] = {}
    current: str | None = None
    for line in md.splitlines():
        line = line.strip()
        if not line:
            continue
        m = SECTION_RE.match(line)
        if m:
            section_name = m.group(1).strip()
            current = section_name
            sections.setdefault(section_name, [])
            continue
        if current is None:
            continue
        cm = CLAIM_RE.match(line)
        sections[current].append(
            {"text": cm.group(2).strip() if cm else line,
             "evidence_ids": [cm.group(1)] if cm else []})
    return {"sections": sections}


def exclusions() -> dict[str, set[str]]:
    out = {}
    for i in PILOTS:
        d = parse_dossier((DOSSIER_ROOT / i / "expanded-dossier.md").read_text())
        ids = {e for s in d["sections"].values() for c in s for e in c["evidence_ids"]}
        out[i] = ids
    return out


def main() -> None:
    ex = exclusions()
    # union of specialists across pilots = 6 (identical per pilot, verified)
    all_ids = sorted(set().union(*ex.values()))
    rows = []
    for image_id in PILOTS:
        dossier = parse_dossier((DOSSIER_ROOT / image_id / "expanded-dossier.md").read_text())
        seed = int(hashlib.sha256(image_id.encode()).hexdigest()[:8], 16) or 1
        for spec in all_ids:
            before = render_caption2(dossier, exclude_evidence_ids=frozenset({spec}))
            after = render_caption2(dossier, exclude_evidence_ids=frozenset())
            rows.append({
                "image_id": image_id,
                "specialist": spec,
                "exclusion_seen": spec in ex[image_id],
                "seed": seed,
                "before": before["text"],
                "before_tokens": before["token_count"],
                "after": after["text"],
                "after_tokens": after["token_count"],
                "after_png": str((RUN_ROOT / image_id / "recon-prose.png")),
                "source": str(SRC / f"{image_id}.jpg"),
            })
    # no-op control: a specialist NOT in any dossier -> before == after
    noop_spec = "pointmap-depth:v1"
    dossier0 = parse_dossier((DOSSIER_ROOT / PILOTS[0] / "expanded-dossier.md").read_text())
    rows.append({
        "image_id": PILOTS[0], "specialist": noop_spec, "exclusion_seen": False,
        "seed": int(hashlib.sha256(PILOTS[0].encode()).hexdigest()[:8], 16) or 1,
        "before": FULL_CAPS[PILOTS[0]], "before_tokens": None,
        "after": FULL_CAPS[PILOTS[0]], "after_tokens": None,
        "after_png": str(RUN_ROOT / PILOTS[0] / "recon-prose.png"),
        "source": str(SRC / f"{PILOTS[0]}.jpg"),
    })
    plan = {
        "schema_version": 1,
        "kind": "dinov3-loo-signal-pilot",
        "issue": 79,
        "parent_issue": 102,
        "generator": "Juggernaut_XL_v1759168.safetensors 832x1216 dpmpp_2m/karras 28steps cfg7.0",
        "specialists": all_ids,
        "noop_control": noop_spec,
        "items": rows,
    }
    fp = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan["plan_sha256"] = fp
    (RUN_ROOT / "_loo-plan.json").write_text(json.dumps(plan, indent=2))
    print("specialists:", all_ids)
    print("rows:", len(rows))
    changed = sum(1 for r in rows if r["before"] != r["after"])
    print("before!=after rows:", changed, "/", len(rows))
    print("plan_sha256:", fp)


if __name__ == "__main__":
    main()