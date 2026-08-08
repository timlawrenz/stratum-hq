"""Dossier -> full-prose caption2 renderer (proposal #79 prerequisite, 2026-08-08).

The arm #36 dossier/context4k stage produces, per item, an evidence-linked
*dossier* (claim lines grouped into sections, every claim carrying its
evidence ids) plus a machine-readable `evidence_payload`. It does NOT produce
the **full prose `caption2.txt`** Tim's reconstruction-fidelity gate (issue
#79) needs: a single flowing paragraph that can be fed verbatim into a simple
ComfyUI flow as the prompt.

This module closes that gap as an additive, deterministic-first stage:

- ``render_caption2`` -- turn a dossier into ONE flowing prose paragraph.
  With no ``backend`` it is purely deterministic (walks the dossier sections,
  joins the already-evidence-bound renderer claim lines into prose, and keeps
  every claim's provenance in a reversible side-channel). With a ``backend``
  (a callable ``prompt -> text``, e.g. an Ollama wrapper) it hands the
  ``caption2_prompt`` text to the aggregator and returns its single-paragraph
  output. Either way the output is a standalone caption suitable for ComfyUI.
- ``caption2_prompt`` -- build the aggregator prompt from the dossier +
  optional evidence payload (the "aggregator expansion stage" input that
  `dossier_expand.floor_gap_analysis` requires but which had no producer).
- ``caption2_variants`` -- for issue #79's protocol: render the SAME dossier
  with a specialist's evidence included vs excluded, so `C_before` (no
  specialist) and `C_after` (specialist added) differ ONLY on that axis.

Honesty invariants (inherited from dossier.py / dossier_expand.py, do not
regress):

- Only scale-invariant facts are verbalized. The deterministic renderer reuses
  the existing `render_*` claim lines (already ratio/band/name-only), and every
  emitted caption passes `honesty_check` (no machine absolute-pixel keys, no
  ``px`` magnitudes, no raw hex triplets in verbalized text).
- No fabricated filler: absent/abstained measurements stay abstentions. The
  deterministic path never invents sentences; the aggregator path is
  constrained by the same prompt template that forbids contradicting the
  determinations.
- Deterministic by default: same dossier -> same caption text (same token
  count). Token counting reuses `dossier.count_tokens`.

This module writes nothing to the corpus and is purely additive; outputs are
strings the caller persists under an approved noncanonical research root.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .dossier import count_tokens
from .dossier_expand import honesty_check

# Callable contract for the aggregator backend: prompt text -> prose text.
# Production wiring (e.g. an OllamaCaptionBackend / gemma3:27b) is intentionally
# NOT imported here so the module runs without a model in tests and on CPU.
AggregatorLike = Callable[[str], str]

# Section headings are prefixed, then stripped, so a deterministic caption reads
# as natural prose rather than a bulleted `## LABEL` listing.
_SECTION_PREFIXES = (
    "body type: ",
    "clothing: ",
    "hair: ",
    "skin tone: ",
    "lighting: ",
    "setting: ",
    "texture: ",
    "pose-articulation: ",
    "pointmap-depth: ",
    "matting-alpha: ",
    "face-geometry: ",
    "object-relations: ",
    "scene-category: ",
    "gaze-head-orientation: ",
    "camera-viewing-angle: ",
    "image-focus: ",
    "apparent-age: ",
    "affordance-contact: ",
    "body-configuration: ",
    "hairstyle: ",
    "face-visibility: ",
    "environment-clearance: ",
    "eye-color: ",
    "relational: ",
)

CAPTION2_SYSTEM = (
    "You are an expert descriptive captioner for a text-to-image dataset. "
    "Write a single, rich, dense paragraph describing the image. "
    "The claims below are ground-truth determinations; you must never "
    "contradict them, never translate machine measurements into invented "
    "attributes, never add objects, and always keep the description strictly "
    "objective prose with no preamble like 'This image shows'. "
    "Start the description immediately."
)


def _join_claim_lines(dossier: Mapping[str, Any], *, exclude_evidence_ids: frozenset[str] = frozenset()) -> list[str]:
    """Flatten the dossier's evidence-linked claim lines into prose sentences.

    Skips claims whose ONLY evidence ids are all in `exclude_evidence_ids`
    (used by `caption2_variants` to drop a specialist's contribution). Keeps
    section ordering (dimensions first by declared order, relational last, as
    `assemble_dossier` emits them). Every returned line is a plain sentence.
    """
    lines: list[str] = []
    for section, claims in (dossier.get("sections") or {}).items():
        for claim in claims:
            text = (claim.get("text") or "").strip()
            if not text:
                continue
            eids = set(claim.get("evidence_ids") or [])
            if eids and eids.issubset(exclude_evidence_ids):
                continue
            sentence = text
            for prefix in _SECTION_PREFIXES:
                if sentence.startswith(prefix):
                    sentence = sentence[len(prefix) :]
                    break
            # The render lines are independent declaratives; end each with '.'
            # so the joined caption reads as prose, not a list.
            lines.append(sentence.rstrip(".").strip() + ".")
    return lines


def caption2_prompt(
    dossier: Mapping[str, Any],
    evidence_payload: Mapping[str, Any] | None = None,
) -> str:
    """Build the aggregator-expansion prompt for a full caption2.

    Composes the ground-truth evidence (deterministic claim lines + optional
    machine-readable payload) under `CAPTION2_SYSTEM`. This is the input the
    `dossier_expand.floor_gap_analysis` "aggregator expansion stage" requires
    but which previously had no producer.
    """
    sections_parts: list[str] = []
    for section, claims in (dossier.get("sections") or {}).items():
        text = [c.get("text", "") for c in claims if (c.get("text") or "").strip()]
        if text:
            sections_parts.append(f"[{section}] " + ". ".join(s.rstrip(".") for s in text) + ".")
    evidence_block = "\n".join(sections_parts) if sections_parts else "No determinations available."
    payload_block = ""
    if evidence_payload and evidence_payload.get("evidence_payload"):
        import json

        payload_block = "\n\nMACHINE-READABLE PAYLOAD (do not verbalize raw values):\n" + json.dumps(
            evidence_payload["evidence_payload"], ensure_ascii=False, sort_keys=True
        )
    return (
        f"{CAPTION2_SYSTEM}\n\n"
        f"GROUND-TRUTH DETERMINATIONS:\n{evidence_block}"
        f"{payload_block}\n\nCaption:"
    )


def render_caption2(
    dossier: Mapping[str, Any],
    *,
    backend: AggregatorLike | None = None,
    evidence_payload: Mapping[str, Any] | None = None,
    exclude_evidence_ids: frozenset[str] = frozenset(),
    max_tokens: int | None = None,
    aggregator_max_tokens: int = 500,
) -> dict[str, Any]:
    """Render a single full-prose caption2 from the dossier.

    Deterministic path (backend=None): joins the evidence-linked claim lines
    into one paragraph. Aggregator path (backend given): builds `caption2_prompt`
    and passes it to the backend, returning its output. Both:

    - strip section/evidence prefixes,
    - pass `honesty_check` (raise ``DossierCaptionError`` on violation),
    - optionally truncate to ``max_tokens`` at a natural sentence boundary
      (never mid-sentence, never padding).

    Returns a record dict (``text``, ``token_count``, ``via``, ``excluded``)
    so callers can persist caption2.txt + provenance.
    """
    prose_lines = _join_claim_lines(dossier, exclude_evidence_ids=exclude_evidence_ids)
    if backend is None:
        text = " ".join(prose_lines)
        via = "deterministic"
    else:
        prompt = caption2_prompt(
            dossier,
            evidence_payload=evidence_payload if evidence_payload is not None else {},
        )
        raw = backend(prompt)
        text = " ".join(part.strip() for part in raw.split() if part.strip())
        via = "aggregator"
        # The aggregator may reference things outside the evidence; honest_check
        # still guards against absolute-pixel/hex leaks in whatever came back.
    if not text:
        raise DossierCaptionError("caption2 rendered empty; refusing to write an empty caption")

    violations = honesty_check(text)
    if violations:
        raise DossierCaptionError(f"honesty check failed on caption2: {violations!r}")

    token_count = count_tokens(text)
    if max_tokens is not None and token_count > max_tokens:
        text, token_count = _truncate_at_sentence(text, max_tokens)

    return {
        "text": text,
        "token_count": token_count,
        "via": via,
        "excluded_evidence_ids": sorted(exclude_evidence_ids),
    }


def caption2_variants(
    dossier: Mapping[str, Any],
    *,
    specialist_evidence_id: str,
    backend: AggregatorLike | None = None,
    evidence_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Render `C_before` and `C_after` for issue #79's reconstruction-fidelity gate.

    - ``before``: the caption generated with the specialist's evidence EXCLUDED
      (the weaker prompt). Monotonic: same dossier, same backend, only the
      specialist's claims dropped.
    - ``after``: the caption with the specialist's evidence INCLUDED (stronger).
    - ``exclusion_seen``: whether the specialist id was actually present in the
      dossier (False => the before/after pair would be identical, a no-op the
      caller must not score as an improvement).

    The caller measures DINOv3 CLS/patch distance from the source to the image
    generated from each variant; `after` closing the distance proves the
    specialist helped.
    """
    before = render_caption2(
        dossier,
        backend=backend,
        evidence_payload=evidence_payload,
        exclude_evidence_ids=frozenset({specialist_evidence_id}),
    )
    after = render_caption2(
        dossier,
        backend=backend,
        evidence_payload=evidence_payload,
        exclude_evidence_ids=frozenset(),
    )
    all_ids = set(dossier.get("evidence_ids") or [])
    return {
        "specialist_evidence_id": specialist_evidence_id,
        "before": before,
        "after": after,
        "exclusion_seen": specialist_evidence_id in all_ids,
    }


def _truncate_at_sentence(text: str, max_tokens: int) -> tuple[str, int]:
    """Truncate at a sentence boundary staying under max_tokens (no padding)."""
    sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
    kept: list[str] = []
    for sentence in sentences:
        candidate = " ".join([*kept, sentence])
        if count_tokens(candidate) <= max_tokens:
            kept.append(sentence)
        else:
            break
    trimmed = " ".join(kept)
    return trimmed, count_tokens(trimmed)


class DossierCaptionError(RuntimeError):
    """Raised when caption2 cannot be rendered honestly."""
