"""Independent adversarial reviewer pass for the frozen Stage-B run.

Reads the finished run's records + outputs (noncanonical, local), and uses a
*separate* installed local vision model (default qwen3-vl:32b) to score each
output against the pre-registered claim-support rubric fields. It does not
grade with the same generator family, and it writes only to a noncanonical
review root. No corpus mutation, no model download, no merges.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from urllib import request as urlrequest


class StageBReviewError(RuntimeError):
    pass


_REQUIRED_FIELDS = ("supported_claims", "unsupported_claims", "omissions", "contradictions", "abstentions")


@dataclass(frozen=True)
class ReviewSettings:
    model_name: str
    digest: str
    endpoint: str
    temperature: float
    seed: int
    num_predict: int
    review_items: str = "all"

    @property
    def fingerprint(self) -> str:
        payload = {
            "model_name": self.model_name,
            "model_digest": self.digest,
            "endpoint": self.endpoint,
            "temperature": self.temperature,
            "seed": self.seed,
            "num_predict": self.num_predict,
            "review_items": self.review_items,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def model_id(self) -> str:
        safe = self.model_name.replace(":", "-").replace("/", "-")
        return f"local-ollama-{safe}-{self.digest[:12]}"


REVIEW_PROMPT_TEMPLATE = """You are an independent adversarial reviewer for a text-to-image captioning experiment.

Given:
- IMAGE: the source image for one caption.
- CAPTION: the model-generated caption to evaluate.
- DECLARED EVIDENCE: optional deterministic specialist evidence (pose2/seg2 geometry) that was available to the captioner.

Score the caption against the source image and the declared evidence. Reply with ONLY a JSON object with exactly these five list fields:
  {{"supported_claims": [...], "unsupported_claims": [...], "omissions": [...], "contradictions": [...], "abstentions": []}}

Rules:
- supported_claims: statements in the caption that are true of the image / consistent with declared evidence.
- unsupported_claims: statements the image or evidence do NOT support (hallucinations, wrong limb/side, invented objects).
- omissions: important visible facts the prompt-model should have captured but did not.
- contradictions: statements contradicting the declared evidence (e.g. evidence says left leg raised, caption says both legs down).
- abstentions: leave empty in this pass.

Be strict and specific. No prose, only the JSON object.

IMAGE: <image>
DECLARED EVIDENCE:
{evidence}
CAPTION:
{caption}
"""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_json(path: Path, label: str) -> dict:
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageBReviewError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise StageBReviewError(f"{label} must be a JSON object")
    return value


def _parse_review_json(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise StageBReviewError("reviewer response must be a JSON object")
    normalized: dict[str, list[str]] = {}
    for key in _REQUIRED_FIELDS:
        items = value.get(key)
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            raise StageBReviewError(f"reviewer response field {key!r} must be a list of strings")
        normalized[key.replace("_claims", "").replace("_", "")] = list(items)
    return normalized


def _aggregate(records: list[dict]) -> dict[str, dict]:
    by_condition: dict[str, dict] = {}
    for record in records:
        condition = record["condition_id"]
        row = by_condition.setdefault(condition, {
            "supported": [], "unsupported": [], "omissions": [], "contradictions": [], "abstentions": [],
            "counts": {"items": 0, "supported": 0, "unsupported": 0, "omissions": 0, "contradictions": 0, "abstentions": 0},
        })
        row["counts"]["items"] += 1
        for key, value in record.items():
            if key in {"supported", "unsupported", "omissions", "contradictions", "abstentions"}:
                row[key].extend(value)
                row["counts"][key] += len(value)
    return by_condition


def _load_run(run_root: Path) -> tuple[list[dict], dict]:
    records = [
        json.loads(line)
        for line in (run_root / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    plan = _read_json(run_root / "stage-b-plan.json", "stage-b-plan.json")
    return records, plan


def build_review_plan(settings: ReviewSettings, run_root: Path, candidate_fingerprint: str) -> dict:
    records, plan = _load_run(run_root)
    items = {record["image_id"] for record in records}
    conditions = sorted({record["condition_id"] for record in records})
    return {
        "review_plan_id": "stage-b-adversarial-review-v1",
        "reviewer_model_id": settings.model_id,
        "reviewer_fingerprint": settings.fingerprint,
        "independent_of_generator": True,
        "generator_model_id": "local-ollama-gemma3-27b-a418f5838eaf",
        "source_run_comparison_plan_fingerprint": plan.get("comparison_plan_fingerprint"),
        "candidate_manifest_fingerprint": candidate_fingerprint,
        "item_count": len(items),
        "condition_count": len(conditions),
        "target_n": len(records),
        "fields": list(_REQUIRED_FIELDS),
        "limitation": "Reviewer is a separate local vision model; verdicts remain PENDING until a human spot-check confirms calibration on known/null cases.",
    }


def _image_payload(image_path: Path) -> dict:
    try:
        payload = image_path.read_bytes()
    except OSError as exc:
        raise StageBReviewError(f"unable to read source image {image_path}: {exc}") from exc
    return {"image_sha256": hashlib.sha256(payload).hexdigest(),
            "base64": base64.b64encode(payload).decode("ascii")}


def _call_reviewer(settings: ReviewSettings, prompt: str, image_path: Path) -> dict:
    from PIL import Image
    # Normalize to RGB JPEG so the reviewer gets exactly the same view family as the generator.
    with Image.open(io.BytesIO(image_path.read_bytes())) as opened:
        buffer = io.BytesIO()
        opened.convert("RGB").save(buffer, format="JPEG", quality=95, subsampling=0)
    body = {
        "model": settings.model_name,
        "prompt": prompt,
        "images": [base64.b64encode(buffer.getvalue()).decode("ascii")],
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": settings.temperature, "seed": settings.seed,
                    "num_predict": settings.num_predict, "top_k": 1, "top_p": 1.0, "num_ctx": 4096},
    }
    request = urlrequest.Request(settings.endpoint, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urlrequest.urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # network/timeout
        raise StageBReviewError(f"reviewer model call failed: {exc}") from exc
    text = result.get("response", "")
    try:
        parsed = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except (ValueError, json.JSONDecodeError) as exc:
        raise StageBReviewError(f"reviewer returned non-JSON: {text[:200]!r}") from exc
    return _parse_review_json(parsed)


def execute_review(settings: ReviewSettings, run_root: Path,
                   source_root: Path, candidate_fingerprint: str,
                   review_root: Path, expected_plan: Mapping[str, Any] | None = None) -> dict:
    records, run_plan = _load_run(run_root)
    plan = build_review_plan(settings, run_root, candidate_fingerprint)
    if expected_plan is not None:
        expected = dict(expected_plan)
        # Allow the runtime-only source-run fingerprint to be re-read from the frozen plan.
        expected = {key: value for key, value in expected.items() if key != "source_run_comparison_plan_fingerprint"}
        actual = {key: value for key, value in plan.items() if key != "source_run_comparison_plan_fingerprint"}
        if _canonical_json(expected) != _canonical_json(actual):
            raise StageBReviewError("expected review plan does not match current settings")

    plan_path = review_root / "review-plan.json"
    if plan_path.exists():
        raise StageBReviewError(f"review root already exists: {review_root}")
    review_root.mkdir(parents=True, exist_ok=False)

    results: list[dict] = []
    for record in records:
        image_path = source_root / record["source_relative_path"]
        evidence = record.get("evidence_payload") or {}
        evidence_text = _canonical_json(evidence)
        prompt = REVIEW_PROMPT_TEMPLATE.replace("{evidence}", evidence_text).replace("{caption}", record["caption"])
        # <image> marker must be present in the prompt text; the image is attached via payload.
        prompt = prompt.replace("IMAGE: <image>", "Consider the attached source image.")
        score = _call_reviewer(settings, prompt, image_path)
        results.append({
            "image_id": record["image_id"],
            "condition_id": record["condition_id"],
            "model": "reviewer-qwen3vl-32b",
            **score,
        })

    aggregate = _aggregate(results)
    run_record = {
        "schema_version": 1,
        "review_plan_id": plan["review_plan_id"],
        "reviewer_fingerprint": settings.fingerprint,
        "status": "PENDING_HUMAN_SPOT_CHECK",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "record_count": len(results),
        "aggregate": aggregate,
    }
    review_root.joinpath("review-plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    review_root.joinpath("reviews.jsonl").write_text("\n".join(json.dumps(r, sort_keys=True) for r in results) + "\n", encoding="utf-8")
    review_root.joinpath("review-run.json").write_text(json.dumps(run_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "PENDING_HUMAN_SPOT_CHECK", "review_root": str(review_root), "record_count": len(results)}
