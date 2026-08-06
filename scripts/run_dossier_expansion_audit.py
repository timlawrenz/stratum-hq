"""CPC honest-expansion ceiling audit over the frozen cohort (arm #36, CPU only).

Pre-GPU gate for the round-trip audit: run the deterministic evidence-bound expander over
the frozen candidate manifest and report, per item and in aggregate, how far honest
expansion gets toward the 100K expanded / 4K compact floors:

- expanded_prose_tokens        : evidence-bound elaboration of the base dossier
- payload_tokens               : full machine-readable evidence_payload JSON (dossier input)
- total_dossier_record_tokens  : prose + payload
- lm_verbosity_ceiling         : generous analytic LM bound over the same bounded fact set
- expanded_floor_gap / reached : honest verdict vs the 100K floor

No GPU, no scheduler, no corpus write. All outputs are additive under
/mnt/nas-ai-models/research/stratum/<RUN>/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the deterministic decode from the existing dossier runner (same frozen inputs).
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_dossier_context4k import _decode_item, _load  # noqa: E402

from research_harness.dossier import (  # noqa: E402
    assemble_dossier,
    build_evidence_payload,
    count_tokens,
    expanded_dossier_text,
)
from research_harness.dossier_expand import (  # noqa: E402
    expand_dossier,
    floor_gap_analysis,
)


def run(manifest_path: Path, program_path: Path, output_root: Path, *, report: Path | None) -> int:
    program = json.loads(program_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roots = _load(program, manifest)
    items = manifest["items"]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {"n_items": len(items), "runs": []}
    for item in items:
        calc = _decode_item(roots, item)
        image_id = calc["image_id"]
        lighting = calc.pop("lighting")
        if lighting is None:
            lighting = {"lighting_measurable": False, "abstention_reason": "normal2.npy unavailable for this item"}
        calc_dossier = {k: v for k, v in calc.items() if k != "source_sha256"}
        dossier = assemble_dossier(lighting=lighting, **calc_dossier)
        payload = build_evidence_payload(lighting=lighting, **calc)

        expanded = expand_dossier(dossier, payload)
        exp_dossier = expanded["expanded_dossier"]
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        payload_tokens = count_tokens(payload_json)  # consistent T5-based measure
        base_claim_count = sum(len(cl) for cl in (dossier.get("sections") or {}).values())

        gap = floor_gap_analysis(
            expanded_prose_tokens=expanded["token_count"],
            payload_tokens=payload_tokens,
            claim_count=base_claim_count,
        )

        item_dir = output_root / image_id
        item_dir.mkdir(parents=True, exist_ok=True)
        (item_dir / "expanded-dossier.md").write_text(expanded_dossier_text(exp_dossier) + "\n", encoding="utf-8")
        (item_dir / "expanded-dossier.json").write_text(
            json.dumps(exp_dossier, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (item_dir / "evidence_payload.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        summary["runs"].append({
            "image_id": image_id,
            "base_dossier_tokens": dossier["token_count"],
            "base_claim_count": base_claim_count,
            "expanded_prose_tokens": expanded["token_count"],
            "expansion_multiplier": round(expanded["token_count"] / dossier["token_count"], 3) if dossier["token_count"] else None,
            "payload_tokens": payload_tokens,
            "total_dossier_record_tokens": gap["total_dossier_record_tokens"],
            "lm_verbosity_ceiling": gap["lm_verbosity_ceiling"],
            "expanded_floor_gap": gap["expanded_floor_gap"],
            "expanded_floor_reached": gap["expanded_floor_reached"],
            "compact_floor_reached": gap["compact_floor_reached"],
            "note": gap["note"],
        })

    runs = summary["runs"]
    if runs:
        summary["expanded_prose_min"] = min(r["expanded_prose_tokens"] for r in runs)
        summary["expanded_prose_max"] = max(r["expanded_prose_tokens"] for r in runs)
        vals = sorted(r["expanded_prose_tokens"] for r in runs)
        mid = len(vals) // 2
        summary["expanded_prose_median"] = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2
        summary["total_record_min"] = min(r["total_dossier_record_tokens"] for r in runs)
        summary["total_record_max"] = max(r["total_dossier_record_tokens"] for r in runs)
        summary["lm_ceiling_min"] = min(r["lm_verbosity_ceiling"] for r in runs)
        summary["lm_ceiling_max"] = max(r["lm_verbosity_ceiling"] for r in runs)
        summary["any_expanded_floor_reached"] = any(r["expanded_floor_reached"] for r in runs)
        summary["any_compact_floor_reached"] = any(r["compact_floor_reached"] for r in runs)

    report_path = Path(report) if report else (output_root / "expansion-run-summary.json")
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dossier-expansion-audit")
    parser.add_argument("manifest", type=Path, help="frozen candidate manifest")
    parser.add_argument("--program", type=Path, default=Path("research/program.json"))
    parser.add_argument("--output-root", type=Path, required=True, help="approved noncanonical research root")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    return run(args.manifest, args.program, args.output_root, report=args.report)


if __name__ == "__main__":
    raise SystemExit(main())
