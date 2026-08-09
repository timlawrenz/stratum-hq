"""Demo: render full-prose caption2 from archived expanded-dossier.md (CPU-only).

Parses the archived dossier text (sections + [evidence-id] claim lines) into
the renderer's dossier shape and emits caption2.txt per item under an
approved noncanonical research root.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_harness.dossier_caption import render_caption2  # noqa: E402

DOSSIER_ROOT = Path("/mnt/nas-ai-models/research/stratum/dossier-context4k-v2")
OUT_ROOT = Path("/mnt/nas-ai-models/research/stratum/prose-caption2-demo-v1")
SECTION_RE = re.compile(r"^##\s+(.+)$")
CLAIM_RE = re.compile(r"^\[([a-zA-Z0-9_\-:]+)\]\s*(.*)$")


def parse_expanded_dossier(md: str) -> dict:
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
        if cm:
            sections[current].append(
                {"text": cm.group(2).strip(), "evidence_ids": [cm.group(1)]}
            )
        else:
            sections[current].append({"text": line, "evidence_ids": []})
    return {"image_id": "", "sections": sections}


def main(ids: list[str]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = []
    for image_id in ids:
        item_dir = DOSSIER_ROOT / image_id
        md_file = item_dir / "expanded-dossier.md"
        if not md_file.exists():
            print(f"MISSING dossier for {image_id}", file=sys.stderr)
            continue
        dossier = parse_expanded_dossier(md_file.read_text())
        rec = render_caption2(dossier)
        out = OUT_ROOT / image_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "caption2.txt").write_text(rec["text"] + "\n")
        (out / "caption2-provenance.json").write_text(
            json.dumps(
                {"image_id": image_id, "via": rec["via"],
                 "token_count": rec["token_count"],
                 "evidence_ids": dossier["sections"] and list(
                     {e for s in dossier["sections"].values() for c in s for e in c["evidence_ids"]}
                 )},
                indent=2,
            )
        )
        n_sections = len(dossier["sections"])
        n_claims = sum(len(v) for v in dossier["sections"].values())
        summary.append(
            {"image_id": image_id, "token_count": rec["token_count"],
             "sections": n_sections, "claims": n_claims}
        )
        print(f"== {image_id} | {rec['token_count']} tokens | {n_sections} sections | {n_claims} claims")
        print(rec["text"][:1200])
        print()
    (OUT_ROOT / "_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ids = sys.argv[1:] or [
        "03r1r5psuxprfkhomitcz5vh9w1c",
        "07hyx5wjk5rc339v2s3e2orcoorr",
        "0f3fdmh6iackl6049wmrm9n2p4i5",
        "058v0mg1bbxthvatdw324k8j4b0s",
        "08v25q5524t2u0zl0xtnzi6bd22f",
    ]
    main(ids)