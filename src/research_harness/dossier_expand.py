"""Honest, evidence-bound expansion of the per-asset dossier (arm #36 round-trip pre-gate).

After the deterministic stage (387-648 tokens/item expanded, ~298 compact) the program
contract floors are 100K expanded / 4K compact. Before committing GPU hours to the
scheduler-bound aggregator expansion + round-trip audit, this module measures HOW FAR an
*honest* expansion can go: every added line must trace to a base claim and/or a
machine-readable payload fact, and the same measurement semantics as `dossier.py` and
`proportions.py` hold:

- Only scale-invariant facts (ratios, bands, names, coverage fractions, relations) are
  verbalized as prose. Absolute pixel widths and raw camera-frame luminances remain
  machine-readable payload values and are NEVER verbalized as caption/description claims.
- No fabricated filler: the expander restates each claim with provenance + payload-bound
  detail + a boundary/abstention note. It does not invent new attributes.
- Abstention lines stay abstentions (they are honest facts about the measurement, and are
  preserved, never rescued into invented values).

It also reports the honest token ceiling THREE complementary ways so the floor decision is
explicit rather than a guess:

1. `expanded_prose_tokens` -- deterministic evidence-bound elaboration of the base dossier.
2. `payload_tokens` -- the full machine-readable `evidence_payload` JSON serialized (the
   contract's dossier/compressor input), counted separately because it is machine input,
   not verbalized description.
3. `lm_verbosity_ceiling` -- a conservative analytic upper bound for a grounded LM
   elaborating the same bounded fact set (facts x honest max tokens/fact). No stochastic
   LM can honestly exceed the evidence-bound fact count by unbounded prose; exceeding it
   would be fabrication, which the honesty gate forbids.

The audit's output is a raw floor-gap measurement: it does NOT weaken
`build_compression_bundle`'s under-budget refusal (that gate stays). It determines whether
the round-trip audit can run with contract-valid bundles or whether a floor/scope decision
is needed first.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from .dossier import (
    assemble_dossier,
    build_evidence_payload,
    count_tokens,
    expanded_dossier_text,
)

# ---------------------------------------------------------------------------
# Honesty constants
# ---------------------------------------------------------------------------
# Deterministic provenance note per evidence id (artifact + algorithm). Used to
# expand each claim with WHERE it came from without inventing new facts.
EVIDENCE_PROVENANCE: dict[str, str] = {
    "body-type-proportions:v1": "pose2 (Goliath-308) keypoints, plane-gated shoulder/hip + leg/torso ratios",
    "clothing-apparel:v1": "seg2 DOME-29 Apparel/Upper/Lower/Torso classes + source-pixel dominant color",
    "hair:v1": "seg2 Hair(4) region + source-pixel dominant color",
    "skin-color-tone:v1": "seg2 Face_Neck(3) + exposed-limb skin regions + source-pixel tone",
    "lighting:v1": "normal2 direction statistics + source luminance histogram / dynamic-range / shadow fractions",
    "relational-determinations:v1": "pose2+seg2 relational determinations (visibility, orientation, relations)",
}

# Conservative analytic upper bound: the number of tokens one honest LM sentence may
# spend elaborating a single bounded fact before it stops being an "elaboration" and
# becomes invented content. 500 tokens/fact is already extremely generous: a typical
# frozen-cohort item has ~20 claims.
LM_TOKENS_PER_CLAIM_HONEST_MAX = 500

# Absolute-pixel keys that must NEVER appear as verbalized description claims.
_PIXEL_KEYS = ("between_shoulders", "between_hips", "torso_length", "left_leg_length", "right_leg_length")
_HEX_TRIPLET = re.compile(r"#(?:[0-9a-fA-F]{3}){1,2}\b")

class ExpansionError(RuntimeError):
    """Raised when an honest expansion cannot be assembled safely."""


def provenance_note(evidence_ids: list[str]) -> str:
    seen: list[str] = []
    for eid in evidence_ids:
        note = EVIDENCE_PROVENANCE.get(eid, "deterministic evidence specialist")
        label = f"{eid}" if note == "deterministic evidence specialist" else f"{eid} ({note})"
        if label not in seen:
            seen.append(label)
    return "; ".join(seen)


def _payload_elaborations(section: str, payload: Mapping[str, Any]) -> list[str]:
    """Scale-invariant elaboration lines derived ONLY from payload facts."""
    lines: list[str] = []
    ep = (payload.get("evidence_payload") or {}) if isinstance(payload, Mapping) else {}

    if section == "body-type":
        ratios = ep.get("ratios") or {}
        shr = ratios.get("shoulder_hip_ratio")
        if isinstance(shr, (int, float)):
            band = "the declared human band [0.7, 2.4]" if 0.7 <= float(shr) <= 2.4 else "outside the declared human band (reported with the band as boundary)"
            lines.append(f"measurement boundary: shoulder:hip width ratio {float(shr):.2f} is a scale-invariant width ratio and falls in {band}; it is comparable across pictures only as a ratio.")
        abstained = ep.get("proportions_abstention")
        if abstained:
            lines.append(f"measurement boundary: shoulder:hip ratio abstained -- {abstained}.")
        # Absolute pixel widths live in the machine-readable payload and are NOT
        # verbalized (camera-frame dependent). State that boundary once.
        lines.append("measurement semantics: absolute keypoint pixel widths are recorded machine-readably in the evidence payload and are not caption claims.")

    elif section == "clothing":
        clothing = ep.get("clothing") or {}
        for name in sorted(clothing):
            g = clothing[name] or {}
            cov = g.get("coverage")
            color = g.get("dominant_color_name")
            bits = [f"garment class {name.replace('_', ' ')} present"]
            if color:
                bits.append(f"dominant color {color}")
            if isinstance(cov, (int, float)):
                bits.append(f"covers a scale-invariant fraction {float(cov):.2f} of subject foreground")
            lines.append("clothing: " + ", ".join(bits) + " (DOME-29 seg2 class gate).")

    elif section == "hair":
        hair = ep.get("hair") or {}
        cov = hair.get("coverage")
        color = hair.get("dominant_color_name")
        pos = hair.get("position")
        if isinstance(cov, (int, float)):
            lines.append(f"hair: measured coverage fraction {float(cov):.2f} of subject foreground (seg2 Hair class).")
        if color:
            lines.append(f"hair: dominant color name {color} (source-pixel quantization).")
        if pos:
            lines.append(f"hair: frame position {pos}.")

    elif section == "skin-color":
        skin = ep.get("skin") or {}
        tone = skin.get("skin_tone_name")
        ftone = skin.get("face_tone_name")
        cov = skin.get("skin_coverage")
        agree = skin.get("face_body_agree")
        if tone:
            lines.append(f"skin tone: dominant exposed-skin tone name {tone} (seg2 face/limb skin regions).")
        if ftone and agree is False:
            lines.append(f"skin tone: face tone name {ftone} distinct from body tone -- face/body disagree.")
        if isinstance(cov, (int, float)):
            lines.append(f"skin tone: exposed-skin coverage fraction {float(cov):.2f} of subject foreground.")

    elif section == "lighting":
        lit = ep.get("lighting") or {}
        # Raw luminances are camera-frame dependent; only bands/fractions that survive
        # cross-picture comparison are verbalized. Recapitulate the band claims instead.
        for key, label in (
            ("mean_luma", "background"),
            ("median_luma", "median"),
        ):
            pass  # raw values NOT verbalized; recorded machine-readably
        lines.append("lighting: raw luminance / dynamic-range numbers are recorded machine-readably; only relative bands are verbalized as claims.")

    elif section == "relational":
        rel = ep.get("relational") or {}
        parts = rel.get("body_parts_visible") or []
        if parts:
            lines.append(f"relational: visible body regions {', '.join(sorted(parts))} (detector agreement gate).")
        orient = rel.get("orientation_upright_deg")
        if isinstance(orient, (int, float)):
            lines.append(f"relational: torso upright angle {float(orient):.1f} degrees.")
        for r in rel.get("relations") or []:
            lines.append(f"relational: {r}.")

    return lines


def expand_dossier(
    dossier: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    target_tokens: int | None = None,
) -> dict[str, Any]:
    """Evidence-bound expansion of a dossier into a longer, honest dossier record.

    Every expanded line references at least one evidence id (the base claim's, or the
    section's). Never verbalizes absolute pixels or hex triplets. `target_tokens` is
    informational (a soft ceiling for compacting later); the expanded dossier is NOT padded
    if it is under budget -- that is the honesty gate.
    """
    sections: dict[str, list[dict[str, Any]]] = {}
    details: dict[str, list[str]] = {}
    evidence_ids: list[str] = []
    for section, claims in (dossier.get("sections") or {}).items():
        sec_evidence: list[str] = []
        expanded_claims: list[dict[str, Any]] = []
        for claim in claims:
            text = (claim.get("text") or "").strip()
            if not text:
                continue
            eids = list(claim.get("evidence_ids") or [])
            base = {"text": text, "evidence_ids": eids, "kind": "claim"}
            prov_line = {"text": f"Evidence source: {provenance_note(eids)}.", "evidence_ids": eids, "kind": "provenance"}
            expanded_claims.extend([base, prov_line])
            for eid in eids:
                if eid not in sec_evidence:
                    sec_evidence.append(eid)
        # payload-bound elaboration for the whole section (scale-invariant only)
        added = _payload_elaborations(section, payload)
        if added:
            sec_evidence_cpy = list(sec_evidence)
            for line in added:
                expanded_claims.append(
                    {"text": line, "evidence_ids": sec_evidence_cpy, "kind": "payload-bound"}
                )
        sections[section] = expanded_claims
        for eid in sec_evidence:
            if eid not in evidence_ids:
                evidence_ids.append(eid)
        details[section] = [c["text"] for c in expanded_claims]

    exp_dossier: dict[str, Any] = {
        "schema_version": 1,
        "image_id": dossier.get("image_id"),
        "sections": sections,
        "evidence_ids": evidence_ids,
        "token_count": None,
        "expanded": True,
    }
    exp_text = expanded_dossier_text(exp_dossier)
    exp_dossier["token_count"] = count_tokens(exp_text)
    # Sanity: the expander must never emit absolute pixel keys / hex triplets in prose.
    violations = honesty_check(exp_text)
    if violations:
        raise ExpansionError(f"honesty check failed on expanded dossier: {violations!r}")

    result = {
        "expanded_dossier": exp_dossier,
        "expanded_text": exp_text,
        "token_count": exp_dossier["token_count"],
        "base_token_count": int(dossier.get("token_count") or 0),
        "expansion_multiplier": (exp_dossier["token_count"] / dossier["token_count"]) if dossier.get("token_count") else None,
        "claim_count": sum(len(cl) for cl in sections.values()),
        "evidence_ids": evidence_ids,
        "target_tokens": target_tokens,
    }
    if target_tokens:
        result["under_budget"] = exp_dossier["token_count"] < target_tokens
    return result


def honesty_check(text: str) -> list[str]:
    """Return violations of the verbalization honesty rule, else [].

    Forbids:
    - the machine snake_case absolute-pixel key NAMES (e.g. ``between_shoulders``),
    - pixel-unit magnitudes (``240 px`` / ``300 pixels``),
    - raw hex triplet colors,
    in *verbalized description* text. They must stay in the machine-readable evidence
    payload. Human phrase variants (e.g. ``leg:torso length ratio``, ``shoulders and
    hips``) are scale-invariant ratio vocabulary and are deliberately NOT flagged.
    """
    violations: list[str] = []
    lower = text.lower()
    for key in _PIXEL_KEYS:
        # Only the exact machine key (underscore form) is a leak; the spaced human
        # phrase is ratio vocabulary and must remain allowed.
        if re.search(rf"\b{re.escape(key)}\b", lower):
            violations.append(f"machine absolute-pixel key verbalized: {key}")
    if re.search(r"\b\d+(?:\.\d+)?\s*px\b", lower):
        violations.append("pixel-unit magnitude verbalized")
    if re.search(r"\b\d+(?:\.\d+)?\s*pixel(?:s)?\b", lower):
        violations.append("pixel-unit magnitude verbalized")
    for m in _HEX_TRIPLET.finditer(text):
        violations.append(f"hex color verbalized: {m.group(0)}")
    return violations


def floor_gap_analysis(
    *,
    expanded_prose_tokens: int,
    payload_tokens: int,
    claim_count: int,
    expanded_floor: int = 100_000,
    compact_floor: int = 4_000,
) -> dict[str, Any]:
    """Compare an item's honest token ceiling against the program floors."""
    total_record = expanded_prose_tokens + payload_tokens
    # Honest analytic upper bound for verbose-but-grounded LM elaboration: each of the
    # `claim_count` bounded facts may receive at most LM_TOKENS_PER_CLAIM_HONEST_MAX tokens
    # of prose before elaboration stops being a restatement of that fact and becomes
    # invented/duplicated content. The deterministic total_record is the measured floor of
    # that same corpus; the analytic ceiling is what an honest LM could reach on the SAME
    # fact set (so it can exceed total_record, but is still bounded by facts x max/fact).
    lm_ceiling = claim_count * LM_TOKENS_PER_CLAIM_HONEST_MAX
    return {
        "expanded_prose_tokens": expanded_prose_tokens,
        "payload_tokens": payload_tokens,
        "total_dossier_record_tokens": total_record,
        "lm_verbosity_ceiling": lm_ceiling,
        "expanded_floor": expanded_floor,
        "compact_floor": compact_floor,
        "expanded_floor_gap": expanded_floor - total_record,
        "expanded_floor_reached": total_record >= expanded_floor,
        "max_honest_floor_reached": lm_ceiling >= expanded_floor,
        "compact_floor_reached": expanded_prose_tokens >= compact_floor,
        "note": (
            f"the {expanded_floor // 1000}K/{compact_floor // 1000}K floors cannot be met by "
            f"honest elaboration of the bounded evidence fact set: measured deterministic + "
            f"payload record is {total_record} tokens/item, and the generous honest LM "
            f"ceiling ({claim_count} facts x {LM_TOKENS_PER_CLAIM_HONEST_MAX}) is {lm_ceiling} "
            f"tokens/item -- {expanded_floor - lm_ceiling} tokens short. Exceeding that "
            "ceiling would require fabricating content, which the honesty gate forbids."
        ),
    }
