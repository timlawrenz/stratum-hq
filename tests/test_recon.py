"""Unit tests for arm #37 reconstruction measurement surface (recon.py)."""

import hashlib
import json

import pytest

from research_harness.recon import (
    BASELINE_PROMPT,
    NULL_PROMPT,
    CANONICAL_SOURCE_ROOT,
    DOSSIER_V2_ROOT,
    ReconError,
    aggregate_deltas,
    build_frozen_plan,
    build_items,
    item_seed,
    load_pilot_items,
)

ITEMS = load_pilot_items()

# Arm #37's reconstruction measurement reads the live evidence-linked
# context4k compact artifacts and canonical source images from the
# owned-hardware corpus (/mnt/nas-ai-models). That corpus is not present on
# the neutral GitHub runner, so the tests that genuinely consume it are
# skipped there while still running locally on owned hardware. The pure
# unit tests (frozen manifest, seeds, delta math) run everywhere.
_OWNED_HW_CORPUS = CANONICAL_SOURCE_ROOT.is_dir() and DOSSIER_V2_ROOT.is_dir()
requires_owned_hardware_corpus = pytest.mark.skipif(
    not _OWNED_HW_CORPUS,
    reason="requires owned-hardware corpus under /mnt/nas-ai-models (not present in CI)",
)


def test_pilot_manifest_is_frozen_24():
    assert len(ITEMS) == 24
    for item in ITEMS:
        assert item["source_relative_path"].lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        assert len(item["source_sha256"]) == 64


def test_item_seed_deterministic_and_nonzero():
    a = item_seed("0yo0gxbfflugqp205k128kktigl5")
    b = item_seed("0yo0gxbfflugqp205k128kktigl5")
    assert a == b
    assert a != item_seed("another-image-id")
    assert 1 <= a < 2**32


@requires_owned_hardware_corpus
def test_build_items_pairs_conditions_and_same_seed():
    rows = build_items()
    assert len(rows) == 24
    for r in rows:
        assert set(r["prompts"]) == {"recon-ctx4k", "recon-baseline"}
        assert r["prompts"]["recon-baseline"] == BASELINE_PROMPT
        assert r["prompts"]["recon-ctx4k"].startswith("# context4k")
        # same seed; baseline identical across items by construction
        assert r["seed"] == item_seed(r["image_id"])
        # prompt hashes are stable, lowercase hex
        assert len(r["prompt_sha256"]["recon-ctx4k"]) == 64


@requires_owned_hardware_corpus
def test_context4k_artifact_is_evidence_linked():
    rows = build_items()
    for r in rows:
        assert "[body-type-proportions:v1]" in r["prompts"]["recon-ctx4k"] or \
               "[clothing-apparel:v1]" in r["prompts"]["recon-ctx4k"] or \
               "[hair:v1]" in r["prompts"]["recon-ctx4k"]


def test_aggregate_deltas_math():
    rows = [
        {"image_id": f"id{i:02d}", "sim_ctx4k": 0.60, "sim_baseline": 0.50}
        for i in range(24)
    ]
    agg = aggregate_deltas(rows)
    assert agg["items"] == 24
    assert agg["reconstruction_delta"] == 0.1
    assert agg["paired_positive"] == 24
    assert agg["paired_negative"] == 0
    assert agg["median_delta"] == 0.1
    # half positive, half negative -> delta 0.0 and 12/12
    rows2 = []
    for i in range(24):
        if i < 12:
            rows2.append({"image_id": f"id{i:02d}", "sim_ctx4k": 0.60, "sim_baseline": 0.50})
        else:
            rows2.append({"image_id": f"id{i:02d}", "sim_ctx4k": 0.50, "sim_baseline": 0.60})
    agg2 = aggregate_deltas(rows2)
    assert agg2["reconstruction_delta"] == 0.0
    assert agg2["paired_positive"] == 12


def test_aggregate_deltas_requires_24():
    with pytest.raises(ReconError):
        aggregate_deltas([{"image_id": "x", "sim_ctx4k": 0.5, "sim_baseline": 0.4}])


@requires_owned_hardware_corpus
def test_build_frozen_plan_pins_everything(tmp_path):
    plan = build_frozen_plan(tmp_path, checkpoint_sha256="a" * 64)
    assert plan["status"] == "preregistered"
    assert plan["kind"] == "reconstruction"
    assert plan["parent_issue"] == 37
    assert len(plan["pilot_manifest"]["items"]) == 24
    g = plan["generation_settings"]
    assert g["checkpoint_name"] == "Juggernaut_XL_v1759168.safetensors"
    assert len(g["checkpoint_sha256"]) == 64
    assert g["sampler_name"] and g["scheduler_name"]
    assert g["width"] == 832 and g["height"] == 1216
    assert g["seed_per_item"].startswith("sha256")
    assert plan["scoring"]["model"] == "openai/clip-vit-large-patch14"
    assert plan["null_case"]["prompt"] == NULL_PROMPT
    assert plan["contrasts"][0]["delta_rule"].startswith("mean over 24 items")
    # every frozen input is hashed into the plan -> tamper-evident
    assert plan["conditions"][0]["prompt_text_sha256_fingerprint"] == plan["conditions"][0].get("prompt_text_sha256_fingerprint")


@requires_owned_hardware_corpus
def test_plan_roundtrips_json():
    plan = build_frozen_plan(__import__("pathlib").Path("/tmp"), checkpoint_sha256="b" * 64)
    blob = json.dumps(plan, sort_keys=True)
    assert json.loads(blob)["metric_version"] == "reconstruction-clip-v1"
    assert "reconstruction" in blob