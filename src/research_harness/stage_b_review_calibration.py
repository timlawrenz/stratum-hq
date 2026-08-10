"""Reviewer calibration (metric self-audit) for the Stage-B adversarial pass.

Two synthetic cases exercise the reviewer rubric mechanics without using the
real cohort:
- KNOWN case: a caption that faithfully verbalizes the declared evidence -> the
  reviewer must return supported_claims for that statement and no
  contradiction.
- NULL case: an empty caption -> the reviewer must abstain (supported==0,
  unsupported==0) rather than fabricate an agreement.

Writes a noncanonical calibration JSON under the review root. No corpus
mutation; local models only; scheduler-bound (GPU).
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from PIL import Image

from research_harness.stage_b_review import (
    ReviewSettings,
    StageBReviewError,
    _build_review_prompt,
    _call_reviewer_with_retry,
    _parse_review_json,
)

CALIBRATION_ROOT = Path("/mnt/nas-ai-models/research/stratum/stage-b-first500-parity-v1-review")
JOB_ID = "stratum-stage-b-adversarial-review-calib-v1"


def _synthetic_image() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (512, 512), (140, 110, 90)).save(buf, "JPEG")
    return buf.getvalue()


KNOWN_CAPTION = (
    "A person stands upright facing the camera. The left arm is extended downward "
    "and the right arm is extended downward. The torso is upright."
)
NULL_CAPTION = ""
EVIDENCE = "subject: 1 detections\ntorso upright 0 deg\nvisible: face, torso\nrelations: left arm extended downward, right arm extended downward"


def _settings() -> ReviewSettings:
    return ReviewSettings(
        model_name="gemma4:e4b",
        digest="c6eb396dbd59",
        endpoint="http://127.0.0.1:11434/api/generate",
        temperature=0.0,
        seed=20260804,
        num_predict=2000,
        review_items="calibration",
    )


def _score(settings: ReviewSettings, image_path: Path, prompt: str) -> dict:
    return _call_reviewer_with_retry(settings, prompt, image_path, attempts=2)


def main(argv: list[str] | None = None) -> int:
    settings = _settings()
    CALIBRATION_ROOT.mkdir(parents=True, exist_ok=True)
    image_path = CALIBRATION_ROOT / "_calib_image.jpg"
    image_path.write_bytes(_synthetic_image())

    known = _score(settings, image_path, _build_review_prompt(EVIDENCE, KNOWN_CAPTION))
    null = _score(settings, image_path, _build_review_prompt(EVIDENCE, NULL_CAPTION))

    known_ok = bool(known["supported"]) and not known["contradictions"]
    null_ok = (not null["supported"]) and (not null["unsupported"]) and (not null["contradictions"])

    record = {
        "schema_version": 1,
        "calibration_id": "stage-b-reviewer-calibration-v1",
        "reviewer_fingerprint": settings.fingerprint,
        "known_case": {"caption": KNOWN_CAPTION, "result": known, "passed": known_ok},
        "null_case": {"caption": NULL_CAPTION, "result": null, "passed": null_ok},
        "verdict": "reviewer_calibration_passed" if (known_ok and null_ok) else "reviewer_calibration_failed",
        "note": (
            "Synthetic rubric calibration only. It validates that the reviewer "
            "rewards faithful evidence verbalization and abstains on empty input; "
            "it does not score the real run."
        ),
    }
    (CALIBRATION_ROOT / "reviewer-calibration.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ok = known_ok and null_ok
    print(json.dumps({"calibration_passed": ok, "known": known_ok, "null": null_ok}, sort_keys=True))
    image_path.unlink(missing_ok=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
