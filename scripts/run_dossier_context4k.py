"""Deterministic per-item dossier + context4k assembly over the frozen cohort.

Arm #36 (dossier-context4k) — CPU-only deterministic first stage. Reads only the
frozen candidate manifest's selected inputs (pose2/seg2/normal2 + source pixels
under the canonical source and derived roots), runs the five validated
deterministic specialists plus relational determinations, assembles the
claim-by-claim dossier, compresses to a context, and persists the three
context4k artifacts per item under an approved noncanonical research root.

Nothing is written to crawlr/approved or crawlr/stratum; outputs are purely
additive under /mnt/nas-ai-models/research/stratum/<RUN>. The run reports
honest token accounting (deterministic T5-based) for the expanded dossier and
compact context, surfacing whether the deterministic corpus reaches the program
floors or whether the aggregator expansion stage is required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from research_harness.clothing import compute_clothing
from research_harness.dossier import (
    assemble_dossier,
    build_compression_bundle,
    build_evidence_payload,
    build_item_context4k_artifacts,
    compress_dossier_to_context,
    expanded_dossier_text,
)
from research_harness.hair import compute_hair
from research_harness.lighting import compute_lighting
from research_harness.proportions import compute_proportions
from research_harness.skin_color import compute_skin_tone
from stratum2.pipeline.determinations import derive_determinations


def _load(program: dict, manifest: dict) -> dict:
    canonical = program["canonical_source"]
    return {
        "source_root": Path(canonical["path"]),
        "derived_root": Path(canonical["derived_tree"]),
    }


def _decode_item(roots: dict, item: dict):
    """Decode the frozen selected inputs for one item (mirrors stage_b's
    _load_selected_item but reads only what the dossier needs)."""
    image_id = item["image_id"]
    src_rel = item["source_relative_path"]
    image_path = roots["source_root"] / src_rel
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)

    def _arr(name: str):
        path = roots["derived_root"] / image_id / name
        return np.load(path, allow_pickle=False)

    pose2 = _arr("pose2.npy")
    if pose2.shape != (1, 308, 3):
        raise RuntimeError(f"{image_id}: pose2 not exactly one Goliath-308 detection")
    seg2 = _arr("seg2.npy")
    if seg2.ndim != 2:
        raise RuntimeError(f"{image_id}: seg2 must be 2D")
    if seg2.shape[:2] != rgb.shape[:2]:
        seg2 = np.asarray(Image.fromarray(seg2).resize((rgb.shape[1], rgb.shape[0]), Image.BICUBIC), dtype=np.uint8)
    normal2 = None
    try:
        normal2 = _arr("normal2.npy")
    except FileNotFoundError:
        pass

    proportions = compute_proportions(pose2)
    clothing = compute_clothing(seg2, rgb)
    hair = compute_hair(seg2, rgb)
    skin = compute_skin_tone(seg2, rgb)
    determinations = derive_determinations(pose2, seg2)
    lighting = compute_lighting(normal2, seg2, rgb) if normal2 is not None else None
    source_bytes = image_path.read_bytes()
    return {
        "image_id": image_id,
        "source_sha256": item.get("source_sha256"),
        "proportions": proportions,
        "clothing": clothing,
        "hair": hair,
        "skin": skin,
        "lighting": lighting,
        "determinations": determinations,
    }


def run(manifest_path: Path, program_path: Path, output_root: Path, *, report: Path | None, verify_bundles: bool) -> int:
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
            lighting = {
                "lighting_measurable": False,
                "abstention_reason": "normal2.npy unavailable for this item",
            }
        calc_dossier = {k: v for k, v in calc.items() if k != "source_sha256"}
        dossier = assemble_dossier(lighting=lighting, **calc_dossier)
        context = compress_dossier_to_context(dossier)
        payload = build_evidence_payload(lighting=lighting, **calc)

        item_dir = output_root / image_id
        artifacts = build_item_context4k_artifacts(bundle={
            "schema_version": 1,
            "image_id": image_id,
            "expanded_dossier": {"token_count": dossier["token_count"], "evidence_ids": dossier["evidence_ids"]},
            "compact_context": {"token_count": context["token_count"], "claims": context["claims"]},
            "artifacts": {"structured": "context4k.json", "human_readable": "context4k.md", "provenance": "compression.json"},
        }, target_dir=item_dir)
        # persist the machine-readable evidence payload alongside the context.
        (item_dir / "evidence_payload.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        # persist the expanded dossier text.
        (item_dir / "expanded-dossier.md").write_text(
            expanded_dossier_text(dossier) + "\n", encoding="utf-8"
        )

        if verify_bundles:
            try:
                build_compression_bundle(
                    image_id=image_id, dossier=dossier, context=context, program=program
                )
                contract_ok = True
            except Exception as exc:  # noqa: BLE001 - honest source enforcement surfaced to the report
                contract_ok = False
                contract_reason = str(exc)
        else:
            contract_ok = None
            contract_reason = None

        summary["runs"].append({
            "image_id": image_id,
            "expanded_dossier_tokens": dossier["token_count"],
            "compact_context_tokens": context["token_count"],
            "compact_under_budget": context["under_budget"],
            "claims": context["token_count"],
            "evidence_ids": dossier["evidence_ids"],
            "contract_ok": contract_ok,
            "contract_reason": contract_reason,
            "artifacts": artifacts,
        })

    summary["expanded_token_min"] = min(r["expanded_dossier_tokens"] for r in summary["runs"]) if summary["runs"] else None
    summary["expanded_token_max"] = max(r["expanded_dossier_tokens"] for r in summary["runs"]) if summary["runs"] else None
    summary["compacted_median"] = None
    if summary["runs"]:
        vals = sorted(r["compact_context_tokens"] for r in summary["runs"])
        mid = len(vals) // 2
        summary["compacted_median"] = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2

    report_path = Path(report) if report else (output_root / "dossier-run-summary.json")
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dossier-runner")
    parser.add_argument("manifest", type=Path, help="frozen candidate manifest")
    parser.add_argument("--program", type=Path, default=Path("research/program.json"))
    parser.add_argument("--output-root", type=Path, required=True, help="approved noncanonical research root")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--verify-bundles", action="store_true", help="attempt contract validation per bundle and report")
    args = parser.parse_args(argv)
    return run(args.manifest, args.program, args.output_root, report=args.report, verify_bundles=args.verify_bundles)


if __name__ == "__main__":
    raise SystemExit(main())
