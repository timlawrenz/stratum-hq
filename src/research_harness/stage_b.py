"""Bounded, noncanonical Stage-B caption-parity execution.

This module intentionally keeps the experiment separate from production corpus
artifacts: it reads only the frozen candidate manifest's selected inputs and
publishes all generated outputs under a new noncanonical research root.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image

from stratum.config import CAPTION_PROMPT, DEFAULT_ASPECT_BUCKETS
from stratum.pipeline.bucket import assign_aspect_bucket, parse_bucket_dims
from stratum.pipeline.caption import ensure_single_paragraph
from stratum2.pipeline.caption2 import CAPTION2_PROMPT_TEMPLATE, build_prompt
from stratum2.pipeline.determinations import derive_determinations

from .contracts import ContractError, validate_comparison_parity_plan, validate_program


class StageBRunError(RuntimeError):
    """Raised when a Stage-B run cannot preserve its frozen safety contract."""


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_OUTPUT_SEGMENT = re.compile(r"[A-Za-z0-9._-]+")
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
_REQUIRED_REVIEW_FIELDS = (
    "supported_claims",
    "unsupported_claims",
    "omissions",
    "contradictions",
    "abstentions",
)
_CONTEXT_PROMPT_TEMPLATE = """You are an expert descriptive captioner for a text-to-image dataset.
Your task is to write a single, rich, dense paragraph describing the provided image.

DECLARED SPECIALIST EVIDENCE:
{evidence_text}

Use declared specialist evidence only as bounded support; do not turn absent evidence into a claim.
Add what declared evidence omits from the image itself: mood, lighting quality, color palette,
fabric, texture, skin details, expression, background, and any posture/activity that is visually
obvious and not contradicted by the declared evidence.

Write strictly objective prose. No conversational filler, no preambles like "This image shows".
Start the description immediately.
"""


@dataclass(frozen=True)
class StageBGenerationSettings:
    """Pinned local Ollama request settings for every comparison condition."""

    endpoint: str
    model_name: str
    model_digest: str
    temperature: float
    seed: int
    num_predict: int
    top_k: int
    top_p: float
    context_window: int
    timeout_seconds: int

    def __post_init__(self) -> None:
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in _LOCAL_HOSTS:
            raise StageBRunError("Stage-B aggregator endpoint must be a local loopback HTTP endpoint")
        if parsed.path != "/api/generate":
            raise StageBRunError("Stage-B aggregator endpoint must target /api/generate")
        if not self.model_name or self.model_name != self.model_name.strip() or any(
            character.isspace() for character in self.model_name
        ):
            raise StageBRunError("Stage-B model_name must be a canonical non-empty identifier")
        if not _SHA256_RE.fullmatch(self.model_digest):
            raise StageBRunError("Stage-B model_digest must be a lowercase SHA-256 digest")
        if self.temperature != 0.0 or self.top_k != 1 or self.top_p != 1.0:
            raise StageBRunError(
                "Stage-B requires deterministic sampling: temperature=0, top_k=1, top_p=1"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise StageBRunError("Stage-B seed must be a non-negative integer")
        if isinstance(self.num_predict, bool) or not isinstance(self.num_predict, int) or self.num_predict <= 0:
            raise StageBRunError("Stage-B num_predict must be a positive integer")
        if isinstance(self.context_window, bool) or not isinstance(self.context_window, int) or self.context_window < self.num_predict:
            raise StageBRunError("Stage-B context_window must be an integer no smaller than num_predict")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, int) or self.timeout_seconds <= 0:
            raise StageBRunError("Stage-B timeout_seconds must be a positive integer")

    @property
    def fingerprint(self) -> str:
        return _sha256(_canonical_json(asdict(self)).encode("utf-8"))

    def request_options(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "seed": self.seed,
            "num_predict": self.num_predict,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "num_ctx": self.context_window,
        }


Generator = Callable[[Image.Image, str, StageBGenerationSettings], str]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_fingerprint(value: Mapping[str, Any], field: str) -> str:
    payload = {key: item for key, item in value.items() if key != field}
    return _sha256(_canonical_json(payload).encode("utf-8"))


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageBRunError(f"{label} must be a mapping")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise StageBRunError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _safe_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise StageBRunError(f"{label} must be a non-empty normalized POSIX path")
    if "\\" in value or "\x00" in value:
        raise StageBRunError(f"{label} must be a normalized POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise StageBRunError(f"{label} must stay under its declared root")
    if path.as_posix() != value:
        raise StageBRunError(f"{label} must be normalized")
    return value


def _safe_output_segment(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_OUTPUT_SEGMENT.fullmatch(value):
        raise StageBRunError(f"{label} must be a safe output path segment")
    return value


def _resolved_existing_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise StageBRunError(f"{label} must be an existing directory: {path}") from exc
    if not resolved.is_dir():
        raise StageBRunError(f"{label} must be an existing directory: {path}")
    return resolved


def _require_contained(path: Path, root: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise StageBRunError(f"{label} is unavailable: {path}") from exc
    if not resolved.is_relative_to(root):
        raise StageBRunError(f"{label} escapes its declared root")
    return resolved


def _component(component_id: str, descriptor: Mapping[str, Any]) -> dict[str, str]:
    payload = {"id": component_id, **dict(descriptor)}
    return {"id": component_id, "fingerprint": _sha256(_canonical_json(payload).encode("utf-8"))}


def _evidence_fingerprint(evidence: Mapping[str, Any]) -> str:
    return _canonical_fingerprint(evidence, "fingerprint")


def _no_specialist_evidence() -> dict[str, Any]:
    evidence: dict[str, Any] = {"kind": "none", "id": "no-specialist-evidence-v1"}
    evidence["fingerprint"] = _evidence_fingerprint(evidence)
    return evidence


def _geometry_evidence() -> dict[str, Any]:
    module_path = Path(derive_determinations.__code__.co_filename)
    code_hash = _sha256(module_path.read_bytes())
    evidence: dict[str, Any] = {
        "kind": "specialist_bundle",
        "id": "in-memory-geometry-determinations-v1",
        "specialists": [
            {
                "id": "in-memory-geometry-determinations-v1",
                "scope": "Pose2 and segmentation-derived continuous geometry and open-set relations only; never posture/activity labels.",
                "inputs": "Frozen selected-item pose2.npy and seg2.npy only; pointmap and all other derived artifacts are out of scope for this first evidence contrast.",
                "output_semantics": "Provenance-bearing measurements and natural-language relations, not semantic ground truth or caption claims.",
                "provenance": (
                    "stratum2.pipeline.determinations.derive_determinations "
                    f"SHA-256 {code_hash}; computed in memory during this bounded run with no crawlr/stratum write."
                ),
                "abstention_policy": "Abort the selected item before model generation if required artifacts are missing, unreadable, or detector count is not exactly one; detector disagreement remains a quality anomaly, never prompt content.",
                "known_failure_modes": "Tight crops, segmentation errors, pose errors, monocular pointmap scale limits, and incomplete body visibility can limit or omit relations.",
                "qualification_gate": "Candidate evidence only; no effectiveness claim is permitted until the frozen comparison receives completed rubric and adversarial reviews.",
            }
        ],
    }
    evidence["fingerprint"] = _evidence_fingerprint(evidence)
    return evidence


def _candidate_items(candidate: Mapping[str, Any], program: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if candidate.get("kind") != "first500-coverage-balanced-candidate-manifest":
        raise StageBRunError("candidate manifest kind is not a first500 coverage-balanced manifest")
    if candidate.get("program_id") != program["program_id"]:
        raise StageBRunError("candidate manifest program_id does not match the Stage-B program")
    if candidate.get("status") != "PENDING_PRE_COMPUTE_NON_EXECUTING":
        raise StageBRunError("candidate manifest is not the frozen pre-compute Stage-B input")
    observed_fingerprint = _require_sha256(candidate.get("manifest_fingerprint"), "candidate manifest fingerprint")
    if _canonical_fingerprint(candidate, "manifest_fingerprint") != observed_fingerprint:
        raise StageBRunError("candidate manifest fingerprint does not bind its current content")

    canonical = _require_mapping(program.get("canonical_source"), "program canonical_source")
    if candidate.get("canonical_source_root") != canonical.get("path"):
        raise StageBRunError("candidate manifest canonical source root does not match the program")
    if candidate.get("derived_artifact_root") != canonical.get("derived_tree"):
        raise StageBRunError("candidate manifest derived root does not match the program")

    raw_items = candidate.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise StageBRunError("candidate manifest must contain at least one frozen item")
    seen: set[str] = set()
    for raw_item in raw_items:
        item = _require_mapping(raw_item, "candidate manifest item")
        image_id = _safe_output_segment(item.get("image_id"), "candidate item image_id")
        if image_id in seen:
            raise StageBRunError("candidate manifest item image_ids must be unique")
        seen.add(image_id)
        _safe_relative_path(item.get("source_relative_path"), "candidate item source_relative_path")
        _require_sha256(item.get("source_sha256"), "candidate item source_sha256")
        quality = _require_mapping(item.get("quality_status"), "candidate item quality_status")
        if quality.get("detector_disagreement") is not False or quality.get("pose2_detection_count") != 1:
            raise StageBRunError("Stage-B candidate items must be one-pose non-anomaly rows")
        availability = _require_mapping(item.get("artifact_availability"), "candidate item artifact_availability")
        for required in ("pose2.npy", "seg2.npy"):
            if availability.get(required) is not True:
                raise StageBRunError(f"candidate item lacks required readable {required}")
    return list(raw_items)


def _freeze_evidence_input_artifact_hashes(
    candidate_manifest: Mapping[str, Any], program: Mapping[str, Any], items: list[Mapping[str, Any]]
) -> dict[str, dict[str, str]]:
    """Hash only the two selected evidence inputs for a frozen Stage-B plan."""
    canonical = _require_mapping(program["canonical_source"], "program canonical_source")
    derived_root = _resolved_existing_directory(Path(canonical["derived_tree"]), "derived artifact root")
    hashes: dict[str, dict[str, str]] = {}
    for item in items:
        image_id = _safe_output_segment(item.get("image_id"), "candidate item image_id")
        artifact_dir = _require_contained(
            derived_root / image_id, derived_root, "selected derived artifact directory"
        )
        hashes[image_id] = {}
        for name in ("pose2.npy", "seg2.npy"):
            artifact_path = _require_contained(artifact_dir / name, artifact_dir, f"selected {name}")
            try:
                payload = artifact_path.read_bytes()
                np.load(io.BytesIO(payload), allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise StageBRunError(f"selected {name} is unreadable for {image_id}") from exc
            hashes[image_id][name] = _sha256(payload)
    return hashes


def freeze_stage_b_plan(
    program: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    settings: StageBGenerationSettings,
) -> dict[str, Any]:
    """Bind the exact selected geometry-input bytes before any model execution."""
    plan = build_stage_b_plan(program, candidate_manifest, settings)
    items = _candidate_items(candidate_manifest, program)
    plan["evidence_input_artifact_sha256"] = _freeze_evidence_input_artifact_hashes(
        candidate_manifest, program, items
    )
    plan["comparison_plan_fingerprint"] = _canonical_fingerprint(plan, "comparison_plan_fingerprint")
    try:
        validate_comparison_parity_plan(plan, program)
    except ContractError as exc:
        raise StageBRunError(f"frozen Stage-B plan violates the comparison contract: {exc}") from exc
    return plan


def build_stage_b_plan(
    program: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    settings: StageBGenerationSettings,
) -> dict[str, Any]:
    """Build a fully pinned four-condition comparison plan without inference."""
    try:
        validate_program(program)
    except ContractError as exc:
        raise StageBRunError(f"invalid Stage-B program: {exc}") from exc
    items = _candidate_items(candidate_manifest, program)
    manifest_id = candidate_manifest.get("manifest_id")
    if not isinstance(manifest_id, str) or not manifest_id.strip() or manifest_id != manifest_id.strip():
        raise StageBRunError("candidate manifest_id must be a canonical non-empty string")

    legacy_view = _component(
        "legacy-bucketed-crop-view-v1",
        {
            "renderer": "stage_b.resize_to_cover_center_crop",
            "buckets": list(DEFAULT_ASPECT_BUCKETS),
            "resample": "Pillow.BICUBIC",
        },
    )
    raw_view = _component(
        "raw-source-view-v1",
        {"renderer": "stage_b.decoded_source_rgb", "crop": "none", "resize": "none"},
    )
    legacy_prompt = _component(
        "legacy-caption-prompt-v1",
        {"renderer": "stratum.config.CAPTION_PROMPT", "text": CAPTION_PROMPT},
    )
    context_prompt = _component(
        "context-grounded-prompt-v1",
        {
            "renderer": "research_harness.stage_b._context_prompt + stratum2.pipeline.caption2.build_prompt",
            "template": _CONTEXT_PROMPT_TEMPLATE,
            "evidence_renderer_template": CAPTION2_PROMPT_TEMPLATE,
        },
    )
    no_evidence = _no_specialist_evidence()
    geometry_evidence = _geometry_evidence()
    aggregator = {
        "model_id": f"local-ollama-{settings.model_name.replace(':', '-').replace('/', '-')}-{settings.model_digest[:12]}",
        "provenance": (
            f"Already-installed local Ollama model {settings.model_name} digest {settings.model_digest}; "
            f"loopback endpoint {settings.endpoint}; fixed request settings fingerprint {settings.fingerprint}."
        ),
        "generation_fingerprint": settings.fingerprint,
        "local_only": True,
    }

    def condition(
        condition_id: str,
        input_view: Mapping[str, Any],
        prompt: Mapping[str, Any],
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "id": condition_id,
            "pilot_manifest_id": manifest_id,
            "input_view": dict(input_view),
            "prompt": dict(prompt),
            "evidence": dict(evidence),
            "aggregator": dict(aggregator),
        }

    plan: dict[str, Any] = {
        "schema_version": 1,
        "kind": "comparison-parity-plan",
        "comparison_plan_id": "stage-b-first500-parity-v1",
        "program_id": program["program_id"],
        "status": "PENDING",
        "parent_issue": candidate_manifest.get("parent_issue", 4),
        "candidate_manifest_fingerprint": candidate_manifest["manifest_fingerprint"],
        "hypothesis": (
            "For the frozen coverage-balanced first-500 cohort, declared in-memory deterministic geometry "
            "may improve supported contextual coverage without increasing unsupported or contradictory claims "
            "when the source item, view, prompt template, local model, and generation settings are controlled."
        ),
        "falsified_if": (
            "The evidence-only contrast does not improve the pre-registered human claim-support rubric over its "
            "matched no-specialist condition, or an apparent difference is attributable to an uncontrolled view, "
            "prompt, aggregator, generation, or evaluation change."
        ),
        "metric_version": "claim-support-rubric-v1",
        "pilot_manifest": {
            "id": manifest_id,
            "source_root": program["canonical_source"]["path"],
            "frozen": True,
            "selection_rationale": (
                "The pre-frozen first-500 coverage-balanced 12 portrait / 6 squareish / 6 landscape cohort "
                "removes missing core geometry availability as a confound without claiming population representativeness."
            ),
            "coverage_notes": (
                "All frozen rows have readable existing core artifacts; existing determinations/caption2/t52 files and pointmap "
                "are not used as evidence inputs. Geometry is recomputed in memory from selected existing pose2/seg2 inputs only."
            ),
            "items": [
                {
                    "image_id": item["image_id"],
                    "source_relative_path": item["source_relative_path"],
                    "source_sha256": item["source_sha256"],
                    "artifact_availability": dict(item["artifact_availability"]),
                }
                for item in items
            ],
        },
        "conditions": [
            condition("legacy-bucketed-no-evidence", legacy_view, legacy_prompt, no_evidence),
            condition("legacy-raw-no-evidence", raw_view, legacy_prompt, no_evidence),
            condition("context-raw-no-evidence", raw_view, context_prompt, no_evidence),
            condition("context-raw-geometry", raw_view, context_prompt, geometry_evidence),
        ],
        "contrasts": [
            {
                "id": "input-view-only",
                "baseline_condition": "legacy-bucketed-no-evidence",
                "variant_condition": "legacy-raw-no-evidence",
                "changed_axes": ["input_view"],
            },
            {
                "id": "prompt-only",
                "baseline_condition": "legacy-raw-no-evidence",
                "variant_condition": "context-raw-no-evidence",
                "changed_axes": ["prompt"],
            },
            {
                "id": "evidence-only",
                "baseline_condition": "context-raw-no-evidence",
                "variant_condition": "context-raw-geometry",
                "changed_axes": ["evidence"],
            },
        ],
        "review_protocol": {
            "human_review_required": True,
            "sequence": [
                "selected_input_view",
                "provenance_evidence",
                "candidate_output_or_context",
                "decision_rubric",
            ],
            "fields": list(_REQUIRED_REVIEW_FIELDS),
            "detector_disagreement_handling": "quality_anomaly_not_caption_content",
        },
        "metric_self_audit": {
            "before_comparative_inference": True,
            "known_case_item_id": items[0]["image_id"],
            "null_output_id": "empty-caption-null-v1",
            "evaluator_version": "claim-support-rubric-v1",
        },
        "adversarial_review": {
            "planned": True,
            "checks": [
                "metric_definition_stable",
                "fresh_process_or_second_review",
                "edge_case_inspection",
            ],
        },
        "representation_boundary": {
            "legacy_text_encoder_max_tokens": program["representation"]["legacy_text_encoder_max_tokens"],
            "compact_context_routing": "out_of_scope",
            "no_silent_legacy_routing": True,
        },
    }
    plan["comparison_plan_fingerprint"] = _canonical_fingerprint(plan, "comparison_plan_fingerprint")
    try:
        validate_comparison_parity_plan(plan, program)
    except ContractError as exc:
        raise StageBRunError(f"Stage-B plan violates the comparison contract: {exc}") from exc
    return plan


def _validate_frozen_execution_plan(
    expected_plan: Mapping[str, Any],
    program: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    settings: StageBGenerationSettings,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Validate a persisted plan and extract its exact selected evidence hashes."""
    plan = dict(_require_mapping(expected_plan, "expected comparison plan"))
    observed_fingerprint = _require_sha256(
        plan.get("comparison_plan_fingerprint"), "expected comparison plan fingerprint"
    )
    if _canonical_fingerprint(plan, "comparison_plan_fingerprint") != observed_fingerprint:
        raise StageBRunError("expected comparison plan fingerprint does not bind its current content")
    try:
        validate_comparison_parity_plan(plan, program)
    except ContractError as exc:
        raise StageBRunError(f"expected comparison plan violates the comparison contract: {exc}") from exc

    rebuilt = build_stage_b_plan(program, candidate_manifest, settings)
    expected_core = {
        key: value
        for key, value in plan.items()
        if key not in {"comparison_plan_fingerprint", "evidence_input_artifact_sha256"}
    }
    rebuilt_core = {
        key: value
        for key, value in rebuilt.items()
        if key != "comparison_plan_fingerprint"
    }
    if _canonical_json(expected_core) != _canonical_json(rebuilt_core):
        raise StageBRunError("expected comparison plan does not exactly match frozen manifest and settings")

    raw_hashes = _require_mapping(
        plan.get("evidence_input_artifact_sha256"), "expected comparison plan evidence_input_artifact_sha256"
    )
    frozen_items = _candidate_items(candidate_manifest, program)
    item_ids = {_safe_output_segment(item.get("image_id"), "candidate item image_id") for item in frozen_items}
    if set(raw_hashes) != item_ids:
        raise StageBRunError("expected comparison plan evidence artifact hashes must cover exactly the frozen items")
    hashes: dict[str, dict[str, str]] = {}
    for image_id in sorted(item_ids):
        row = _require_mapping(raw_hashes[image_id], f"frozen evidence hashes for {image_id}")
        if set(row) != {"pose2.npy", "seg2.npy"}:
            raise StageBRunError("frozen evidence hashes must contain exactly pose2.npy and seg2.npy")
        hashes[image_id] = {
            name: _require_sha256(row[name], f"frozen evidence hash {image_id}/{name}")
            for name in ("pose2.npy", "seg2.npy")
        }
    return plan, hashes


def _decode_source(payload: bytes, item: Mapping[str, Any]) -> Image.Image:
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            width, height = opened.size
            source_format = opened.format
            image = opened.convert("RGB")
    except (OSError, ValueError, SyntaxError) as exc:
        raise StageBRunError(f"selected source cannot be decoded: {item['source_relative_path']}") from exc
    dimensions = _require_mapping(item.get("source_dimensions"), "candidate item source_dimensions")
    if dimensions.get("width") != width or dimensions.get("height") != height:
        raise StageBRunError(f"source dimensions drifted for {item['source_relative_path']}")
    if item.get("source_format") != source_format:
        raise StageBRunError(f"source format drifted for {item['source_relative_path']}")
    return image


def _load_selected_item(
    item: Mapping[str, Any],
    source_root: Path,
    derived_root: Path,
    expected_evidence_hashes: Mapping[str, str],
) -> dict[str, Any]:
    relative_path = _safe_relative_path(item.get("source_relative_path"), "candidate item source_relative_path")
    source_path = _require_contained(source_root / relative_path, source_root, "selected source")
    payload = source_path.read_bytes()
    observed_sha = _sha256(payload)
    if observed_sha != _require_sha256(item.get("source_sha256"), "candidate item source_sha256"):
        raise StageBRunError(f"source SHA-256 drifted for {relative_path}")
    image = _decode_source(payload, item)

    image_id = _safe_output_segment(item.get("image_id"), "candidate item image_id")
    artifact_dir = _require_contained(derived_root / image_id, derived_root, "selected derived artifact directory")
    availability = _require_mapping(item.get("artifact_availability"), "candidate item artifact_availability")

    def artifact(name: str, *, required: bool) -> np.ndarray | None:
        declared = availability.get(name)
        path = artifact_dir / name
        if required and declared is not True:
            raise StageBRunError(f"frozen manifest lacks required {name} for {image_id}")
        if declared is not True:
            return None
        resolved = _require_contained(path, artifact_dir, f"selected {name}")
        expected_sha = _require_sha256(expected_evidence_hashes.get(name), f"frozen evidence hash {image_id}/{name}")
        try:
            payload = resolved.read_bytes()
            observed_sha = _sha256(payload)
            if observed_sha != expected_sha:
                raise StageBRunError(f"selected {name} artifact SHA-256 drifted for {image_id}")
            return np.load(io.BytesIO(payload), allow_pickle=False)
        except StageBRunError:
            raise
        except (OSError, ValueError) as exc:
            raise StageBRunError(f"selected {name} is unreadable for {image_id}") from exc

    pose2 = artifact("pose2.npy", required=True)
    seg2 = artifact("seg2.npy", required=True)
    assert pose2 is not None and seg2 is not None
    if pose2.ndim != 3 or pose2.shape[0] != 1 or pose2.shape[1:] != (308, 3):
        raise StageBRunError(f"selected pose2.npy no longer has one Goliath-308 detection for {image_id}")
    if seg2.ndim != 2:
        raise StageBRunError(f"selected seg2.npy must be two-dimensional for {image_id}")
    determinations = derive_determinations(pose2, seg2)
    if determinations["subject"]["n_detections"] != 1:
        raise StageBRunError(f"detector disagreement emerged for frozen selected item {image_id}")
    return {
        "item": dict(item),
        "image": image,
        "source_sha256": observed_sha,
        "determinations": determinations,
        "evidence_input_artifact_sha256": dict(expected_evidence_hashes),
        "source_byte_read_count": 1,
        "derived_reads": ["pose2.npy", "seg2.npy"],
    }


def _bucketed_view(image: Image.Image) -> Image.Image:
    bucket = assign_aspect_bucket(*image.size)
    dims = parse_bucket_dims(bucket)
    if dims is None:
        raise StageBRunError(f"unable to parse selected legacy bucket {bucket}")
    bucket_w, bucket_h = dims
    width, height = image.size
    scale = max(bucket_w / width, bucket_h / height)
    new_width = int(np.ceil(width * scale))
    new_height = int(np.ceil(height * scale))
    resampling = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    resized = image.resize((new_width, new_height), resample=resampling)
    left = max(0, (new_width - bucket_w) // 2)
    top = max(0, (new_height - bucket_h) // 2)
    return resized.crop((left, top, left + bucket_w, top + bucket_h))


def _context_prompt(evidence_text: str) -> str:
    """Render the context prompt without smuggling geometry into a baseline."""
    return _CONTEXT_PROMPT_TEMPLATE.format(evidence_text=evidence_text)


def _render_condition(
    condition: Mapping[str, Any], prepared: Mapping[str, Any]
) -> tuple[Image.Image, str, dict[str, Any] | None]:
    condition_id = condition["id"]
    raw = prepared["image"]
    if condition_id == "legacy-bucketed-no-evidence":
        return _bucketed_view(raw), CAPTION_PROMPT, None
    if condition_id == "legacy-raw-no-evidence":
        return raw.copy(), CAPTION_PROMPT, None
    if condition_id == "context-raw-no-evidence":
        return raw.copy(), _context_prompt("- no specialist evidence declared"), None
    if condition_id == "context-raw-geometry":
        determinations = prepared["determinations"]
        rendered = build_prompt(determinations)
        evidence_text = rendered.split("DETERMINATIONS:\n", 1)[-1].strip()
        return raw.copy(), _context_prompt(evidence_text), determinations
    raise StageBRunError(f"unsupported Stage-B condition: {condition_id}")


def _verify_installed_ollama_digest(settings: StageBGenerationSettings) -> None:
    """Bind the local tag to the pre-registered digest before model generation."""
    parsed = urlparse(settings.endpoint)
    tags_endpoint = f"{parsed.scheme}://{parsed.netloc}/api/tags"
    try:
        response = requests.get(tags_endpoint, timeout=settings.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise StageBRunError(f"unable to inspect local Ollama tag metadata: {exc}") from exc
    models = payload.get("models") if isinstance(payload, Mapping) else None
    if not isinstance(models, list):
        raise StageBRunError("local Ollama tag metadata did not contain a models list")
    matching = [model for model in models if isinstance(model, Mapping) and model.get("name") == settings.model_name]
    if len(matching) != 1:
        raise StageBRunError(f"local Ollama model tag is unavailable or ambiguous: {settings.model_name}")
    observed = matching[0].get("digest")
    if observed != settings.model_digest:
        raise StageBRunError(
            f"local Ollama model digest drift for {settings.model_name}: expected {settings.model_digest}, observed {observed}"
        )


def _unload_local_ollama_model(settings: StageBGenerationSettings) -> None:
    """Request local model eviction before a scheduler lease is released."""
    parsed = urlparse(settings.endpoint)
    unload_endpoint = f"{parsed.scheme}://{parsed.netloc}/api/generate"
    try:
        response = requests.post(
            unload_endpoint,
            json={"model": settings.model_name, "keep_alive": 0, "stream": False},
            timeout=settings.timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise StageBRunError(f"unable to unload local Ollama model before scheduler release: {exc}") from exc


def _default_generate(image: Image.Image, prompt: str, settings: StageBGenerationSettings) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, subsampling=0)
    payload = {
        "model": settings.model_name,
        "prompt": prompt,
        "images": [base64.b64encode(buffer.getvalue()).decode("ascii")],
        "stream": False,
        "keep_alive": "10m",
        "options": settings.request_options(),
    }
    try:
        response = requests.post(settings.endpoint, json=payload, timeout=settings.timeout_seconds)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise StageBRunError(f"local Ollama generation failed: {exc}") from exc
    caption = ensure_single_paragraph(data.get("response", ""))
    if not caption:
        raise StageBRunError("local Ollama generation returned an empty caption")
    return caption


def _review_template(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record["record_id"],
        "image_id": record["image_id"],
        "condition_id": record["condition_id"],
        "output_relative_path": record["output_relative_path"],
        "review_status": "unreviewed",
        **{field: [] for field in _REQUIRED_REVIEW_FIELDS},
        "verdict": "PENDING",
    }


def _metric_self_audit_template(known_case_item_id: str) -> dict[str, Any]:
    """Pre-register review calibration without fabricating a completed judgment."""
    return {
        "status": "PENDING_HUMAN_SELF_AUDIT",
        "known_case_item_id": known_case_item_id,
        "null_output_id": "empty-caption-null-v1",
        "required_checks": [
            "Score one generated known-case output against its original selected source and declared evidence.",
            "Score the declared empty-caption null output and confirm it is recorded as an abstention rather than a supported claim.",
            "Confirm every claim-support field accepts explicit evidence-linked entries and empty lists where appropriate.",
        ],
        "limitation": "The runner validates record structure only. It does not fabricate a reviewer judgment or a metric PASS.",
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_canonical_json(dict(record)) + "\n")


def _prepare_output_root(program: Mapping[str, Any], output_root: Path) -> tuple[Path, Path]:
    if not output_root.is_absolute():
        raise StageBRunError("Stage-B output_root must be absolute")
    canonical = _require_mapping(program.get("canonical_source"), "program canonical_source")
    source_root = _resolved_existing_directory(Path(canonical["path"]), "canonical source root")
    derived_root = _resolved_existing_directory(Path(canonical["derived_tree"]), "derived artifact root")
    output = output_root.resolve(strict=False)
    if output.exists() or os.path.lexists(output):
        raise StageBRunError("Stage-B output_root already exists; choose a new versioned run root")
    if output.is_relative_to(source_root) or output.is_relative_to(derived_root):
        raise StageBRunError("Stage-B output_root must not be inside a protected corpus root")
    policy = _require_mapping(program.get("artifact_policy"), "program artifact_policy")
    raw_roots = policy.get("approved_output_roots")
    if not isinstance(raw_roots, list) or not raw_roots:
        raise StageBRunError("program must declare at least one approved noncanonical output root")
    approved_roots = tuple(_resolved_existing_directory(Path(root), "approved output root") for root in raw_roots)
    if not any(output.is_relative_to(root) for root in approved_roots):
        raise StageBRunError("Stage-B output_root must be under an approved noncanonical output root")
    parent = _resolved_existing_directory(output.parent, "Stage-B output parent")
    return output, parent


def execute_stage_b(
    program: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    settings: StageBGenerationSettings,
    *,
    output_root: Path,
    expected_plan: Mapping[str, Any] | None = None,
    generate: Generator | None = None,
) -> dict[str, Any]:
    """Execute the frozen four-condition run and publish only noncanonical outputs.

    All selected sources/artifacts are preflighted before the first model call;
    a changed input fails closed and leaves no final run root behind.
    """
    try:
        validate_program(program)
    except ContractError as exc:
        raise StageBRunError(f"invalid Stage-B program: {exc}") from exc
    output, parent = _prepare_output_root(program, output_root)
    if expected_plan is None:
        # Direct callers are retained for synthetic fixture compatibility only;
        # the CLI/registered launcher always supplies a persisted frozen plan.
        plan = build_stage_b_plan(program, candidate_manifest, settings)
        evidence_hashes = _freeze_evidence_input_artifact_hashes(
            candidate_manifest, program, _candidate_items(candidate_manifest, program)
        )
        plan["evidence_input_artifact_sha256"] = evidence_hashes
        plan["comparison_plan_fingerprint"] = _canonical_fingerprint(plan, "comparison_plan_fingerprint")
    else:
        plan, evidence_hashes = _validate_frozen_execution_plan(
            expected_plan, program, candidate_manifest, settings
        )
    items = _candidate_items(candidate_manifest, program)
    canonical = _require_mapping(program["canonical_source"], "program canonical_source")
    source_root = _resolved_existing_directory(Path(canonical["path"]), "canonical source root")
    derived_root = _resolved_existing_directory(Path(canonical["derived_tree"]), "derived artifact root")

    # Preflight all frozen inputs before model invocation so an input epoch cannot
    # silently split a paired comparison halfway through the cohort.
    prepared = [
        _load_selected_item(
            item,
            source_root,
            derived_root,
            evidence_hashes[_safe_output_segment(item.get("image_id"), "candidate item image_id")],
        )
        for item in items
    ]
    if generate is None:
        _verify_installed_ollama_digest(settings)
    generator = generate or _default_generate
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-b-", dir=parent))
    try:
        outputs_dir = temporary / "outputs"
        outputs_dir.mkdir()
        records: list[dict[str, Any]] = []
        review_queue: list[dict[str, Any]] = []
        for prepared_item in prepared:
            item = prepared_item["item"]
            image_id = _safe_output_segment(item["image_id"], "candidate item image_id")
            for condition in plan["conditions"]:
                condition_id = _safe_output_segment(condition["id"], "Stage-B condition id")
                view, prompt, evidence_payload = _render_condition(condition, prepared_item)
                caption = ensure_single_paragraph(generator(view, prompt, settings))
                if not caption:
                    raise StageBRunError(f"generator returned an empty caption for {image_id}/{condition_id}")
                condition_dir = outputs_dir / condition_id
                condition_dir.mkdir(exist_ok=True)
                output_relative = Path("outputs") / condition_id / f"{image_id}.txt"
                output_path = temporary / output_relative
                output_path.write_text(caption + "\n", encoding="utf-8")
                record = {
                    "schema_version": 1,
                    "record_id": f"{condition_id}:{image_id}",
                    "image_id": image_id,
                    "source_relative_path": item["source_relative_path"],
                    "source_sha256": prepared_item["source_sha256"],
                    "condition_id": condition_id,
                    "input_view": condition["input_view"],
                    "prompt": {
                        **condition["prompt"],
                        "rendered_sha256": _sha256(prompt.encode("utf-8")),
                        "rendered_text": prompt,
                    },
                    "evidence": condition["evidence"],
                    "evidence_payload": evidence_payload,
                    "generation_fingerprint": settings.fingerprint,
                    "output_relative_path": output_relative.as_posix(),
                    "caption_sha256": _sha256(caption.encode("utf-8")),
                    "caption": caption,
                    "caption_character_count": len(caption),
                    "caption_word_count": len(caption.split()),
                    "selected_source_byte_read_count": prepared_item["source_byte_read_count"],
                    "selected_derived_reads": prepared_item["derived_reads"],
                    "selected_evidence_input_artifact_sha256": prepared_item[
                        "evidence_input_artifact_sha256"
                    ],
                }
                records.append(record)
                review_queue.append(_review_template(record))

        self_audit = _metric_self_audit_template(plan["metric_self_audit"]["known_case_item_id"])
        _write_json(temporary / "stage-b-plan.json", plan)
        _write_jsonl(temporary / "records.jsonl", records)
        _write_jsonl(temporary / "review-queue.jsonl", review_queue)
        _write_json(
            temporary / "run-provenance.json",
            {
                "schema_version": 1,
                "status": "PENDING_INDEPENDENT_REVIEW",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "candidate_manifest_id": candidate_manifest["manifest_id"],
                "candidate_manifest_fingerprint": candidate_manifest["manifest_fingerprint"],
                "comparison_plan_id": plan["comparison_plan_id"],
                "comparison_plan_fingerprint": plan["comparison_plan_fingerprint"],
                "generation": {**asdict(settings), "generation_fingerprint": settings.fingerprint},
                "record_count": len(records),
                "selected_source_byte_read_count": sum(item["source_byte_read_count"] for item in prepared),
                "selected_derived_reads_only": True,
                "metric_self_audit": self_audit,
                "semantic_verdict": "PENDING — human claim-support and adversarial reviews are not fabricated by the runner.",
                "non_authorizations": [
                    "no canonical source mutation",
                    "no derived-tree mutation or backfill",
                    "no legacy artifact overwrite",
                    "no context4k or t52 substitution claim",
                    "no empirical PASS or FAIL verdict",
                ],
            },
        )
        (temporary / "review-guide.md").write_text(
            "# Stage-B sequential review guide\n\n"
            "For each `review-queue.jsonl` row, inspect in this order: selected source view, "
            "in-memory determinations/evidence payload (when present), rendered prompt, and candidate caption. "
            "Record supported claims, unsupported claims, omissions, contradictions, and abstentions. "
            "Do not convert detector anomalies into caption semantics. A PASS remains prohibited until an "
            "independent adversarial review is complete.\n",
            encoding="utf-8",
        )
        if output.exists() or os.path.lexists(output):
            raise StageBRunError("Stage-B output_root appeared during execution; refusing to overwrite it")
        os.rename(temporary, output)
        temporary = None
    finally:
        if generate is None:
            # Best-effort local model eviction before the scheduler lease is
            # released; never mask the primary result with an unload error.
            try:
                _unload_local_ollama_model(settings)
            except StageBRunError:
                pass
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)

    return {
        "status": "PENDING_INDEPENDENT_REVIEW",
        "output_root": str(output),
        "record_count": len(records),
        "comparison_plan_fingerprint": plan["comparison_plan_fingerprint"],
        "candidate_manifest_fingerprint": candidate_manifest["manifest_fingerprint"],
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageBRunError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise StageBRunError(f"{label} must contain a JSON object")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="research-stage-b",
        description="Run the frozen first-500 Stage-B caption parity comparison into a noncanonical root",
    )
    parser.add_argument("program", type=Path)
    parser.add_argument("candidate_manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-plan", required=True, type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--num-predict", type=int, default=384)
    parser.add_argument("--context-window", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = StageBGenerationSettings(
            endpoint=args.endpoint,
            model_name=args.model,
            model_digest=args.model_digest,
            temperature=0.0,
            seed=args.seed,
            num_predict=args.num_predict,
            top_k=1,
            top_p=1.0,
            context_window=args.context_window,
            timeout_seconds=args.timeout_seconds,
        )
        result = execute_stage_b(
            _read_json(args.program, "program JSON"),
            _read_json(args.candidate_manifest, "candidate manifest JSON"),
            settings,
            output_root=args.output,
            expected_plan=_read_json(args.expected_plan, "expected comparison plan JSON"),
        )
    except StageBRunError as exc:
        print(f"research-stage-b: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
