"""Frozen-cohort source verification.

Read-only diagnostic for the frozen Stage-B cohort contract: every Stage-B arm
and the reconstruction check preflight ALL of the frozen plan's canonical
sources (``stage_b_launcher._snapshot_selected_inputs`` and ``recon.py``
resolve ``approved/<image_id>.jpg`` strictly), so a source disappearing from
the canonical tree (e.g. a crawl-r re-sync purging blocked photos, hold #132)
gates the whole Stage-B menu.

``verify_plan_sources`` checks each ``pilot_manifest.items`` entry for
presence at ``source_root / source_relative_path`` and, when the plan pins a
``source_sha256``, byte-exact SHA-256 match. It performs NO writes and never
touches the derived tree; it is the cheap pre-launch (and pre-human-decision)
way to ask "is the frozen cohort intact right now?".

Exit semantics: ``all_intact`` is true only when every item is present and
SHA-matching. Absence or mismatch is reported per-item with the expected
digest so a future restore-from-raw decision can be verified against the
manifest before any copy is made.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class CohortVerifyError(Exception):
    """The plan is not a verifiable frozen-cohort plan."""


def _require_mapping(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CohortVerifyError(f"{what} must be a JSON object")
    return value


def _checked_resolve(root: Path, relative: str, image_id: str) -> Path:
    """Resolve ``root / relative`` and reject escapes outside the root."""
    if not relative or relative != relative.strip():
        raise CohortVerifyError(
            f"item {image_id}: source_relative_path must be a non-empty, "
            f"non-blank relative path"
        )
    candidate = root / relative
    try:
        resolved = candidate.resolve(strict=False)
    except OSError as exc:
        raise CohortVerifyError(
            f"item {image_id}: unable to resolve source path {candidate}: {exc}"
        ) from exc
    if not resolved.is_relative_to(root):
        raise CohortVerifyError(
            f"item {image_id}: source path {relative!r} escapes the source root"
        )
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_plan_sources(plan_path: Path) -> dict[str, Any]:
    """Verify a frozen comparison plan's cohort sources against the canonical root.

    Returns a machine-readable report:

    .. code-block:: json

        {
          "plan": "<path>",
          "source_root": "<root>",
          "checked": 24,
          "present": 22,
          "missing": 2,
          "sha_mismatch": 0,
          "all_intact": false,
          "items": [
            {
              "image_id": "...",
              "relative_path": "....jpg",
              "status": "present-match" | "present-mismatch" | "missing",
              "expected_sha256": "..." | null,
              "actual_sha256": "..." | null
            }
          ]
        }

    Raises :class:`CohortVerifyError` when the plan is malformed or an item
    path escapes the source root. Never writes anything.
    """
    plan_path = Path(plan_path)
    try:
        raw = plan_path.read_text(encoding="utf-8")
        plan = json.loads(raw)
    except OSError as exc:
        raise CohortVerifyError(f"unable to read plan {plan_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CohortVerifyError(f"invalid JSON in plan {plan_path}: {exc.msg}") from exc

    plan = _require_mapping(plan, "plan")
    pilot = _require_mapping(plan.get("pilot_manifest"), "plan.pilot_manifest")
    items = pilot.get("items")
    if not isinstance(items, list):
        raise CohortVerifyError("plan.pilot_manifest.items must be a JSON array")
    source_root_raw = pilot.get("source_root")
    if not isinstance(source_root_raw, str) or not source_root_raw:
        raise CohortVerifyError("plan.pilot_manifest.source_root must be a non-empty string")
    source_root = Path(source_root_raw).resolve(strict=False)

    report_items: list[dict[str, Any]] = []
    present = missing = mismatch = 0
    for raw_item in items:
        item = _require_mapping(raw_item, "plan.pilot_manifest.items entry")
        image_id = item.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise CohortVerifyError("plan item missing non-empty image_id")
        relative = item.get("source_relative_path")
        if not isinstance(relative, str):
            raise CohortVerifyError(
                f"item {image_id}: source_relative_path must be a string"
            )
        expected = item.get("source_sha256")
        if expected is not None and not isinstance(expected, str):
            raise CohortVerifyError(
                f"item {image_id}: source_sha256 must be a string when present"
            )

        resolved = _checked_resolve(source_root, relative, image_id)
        entry: dict[str, Any] = {
            "image_id": image_id,
            "relative_path": relative,
            "status": "missing",
            "expected_sha256": expected,
            "actual_sha256": None,
        }
        if resolved.is_file():
            actual = _sha256_file(resolved)
            entry["actual_sha256"] = actual
            if expected is None or actual == expected:
                entry["status"] = "present-match"
                present += 1
            else:
                entry["status"] = "present-mismatch"
                mismatch += 1
        else:
            missing += 1
        report_items.append(entry)

    checked = len(report_items)
    return {
        "plan": str(plan_path),
        "source_root": str(source_root),
        "checked": checked,
        "present": present,
        "missing": missing,
        "sha_mismatch": mismatch,
        "all_intact": checked > 0 and present == checked,
        "items": report_items,
    }