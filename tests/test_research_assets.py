"""Guard durable project-control assets against silent contract drift."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_harness import validate_compression_bundle, validate_program

ROOT = Path(__file__).resolve().parents[1]

_STAGE_A_ALLOWED_ACTIONS = (
    "Select no more than `<maximum item count>` canonical-source pilot candidates using the stated selection protocol.",
    "Read and SHA-256 hash only the selected canonical-source pilot images.",
    "Read only the selected items' existing derived-artifact availability/readability facts; do not mutate `crawlr/stratum`.",
    "Write only the pilot manifest, preparation log, and review record under the approved preparation output root.",
)

_STAGE_B_ALLOWED_ACTIONS = (
    "Read the exact frozen canonical-source pilot items for the approved `<bounded N>` comparison conditions only.",
    "Read the exact frozen existing derived artifacts for those pilot items only; do not mutate `crawlr/stratum`.",
    "Invoke `<already-installed local model>` locally for exactly `<bounded N>` pilot items under the frozen plan's fixed conditions.",
    "Write comparison outputs, evaluation records, and adversarial-review artifacts only under `<approved noncanonical root>` within `/mnt/nas-ai-models/research`.",
    "Use GPU scheduling under separate reviewed manifest `<manifest path / ID>` for the exact frozen plan. The current supervisor is `observer_only`; this requires a separately reviewed registered launcher and does not authorize the observer to claim, launch, heartbeat, release, or kill work.",
    "Generate specifically named additive artifacts for only the frozen pilot items: `<pass list, model size, output root>`. This requires separately checked data/GPU authority.",
)

_EXPLICIT_NON_AUTHORIZATIONS = (
    "No merge or direct push to `main`.",
    "No canonical-source mutation.",
    "No corpus-wide or derived-tree backfill.",
    "No external image model.",
    "No model installation or download unless separately named and approved.",
    "No scheduler operation without the separately approved manifest and registered launcher.",
    "No overwrite of `caption.txt`, `t5_*`, `pose.npy`, or other Stratum1 artifacts.",
    "No empirical PASS claim merely because a plan validates or an inference job completes.",
)


def _section(text: str, start: str, end: str) -> str:
    assert start in text
    assert end in text
    return text.split(start, 1)[1].split(end, 1)[0]


def _checked_actions(text: str) -> tuple[str, ...]:
    return tuple(
        line.removeprefix("- [ ] ").strip()
        for line in text.splitlines()
        if line.startswith("- [ ] ")
    )


def _assert_stage_a_is_bounded(template: str) -> None:
    stage_a = _section(
        template,
        "## Stage A — preparation authorization",
        "## Freeze and validate after Stage A",
    )
    authority = _section(
        stage_a,
        "### Stage A requested authority",
        "**Stage A non-authorizations:**",
    )
    assert _checked_actions(authority) == _STAGE_A_ALLOWED_ACTIONS
    assert (
        "No model invocation, GPU scheduling, additive artifact generation, corpus mutation, or backfill."
        in stage_a
    )
    assert "Stage A preparation approval does not authorize Stage B execution." in stage_a


def _assert_freeze_and_stage_b_are_bound(template: str) -> None:
    freeze = _section(
        template,
        "## Freeze and validate after Stage A",
        "## Stage B — execution authorization",
    )
    assert "**Frozen pilot manifest:**" in freeze
    assert "**Immutable manifest identity/digest:**" in freeze
    assert "**Comparison-plan identity/digest:**" in freeze
    assert "research-harness validate-comparison-plan" in freeze

    stage_b = _section(
        template,
        "## Stage B — execution authorization",
        "## Explicit non-authorizations",
    )
    assert "fresh owner decision" in stage_b
    assert "individually checked" in stage_b
    authority = _section(
        stage_b,
        "### Stage B requested execution authority",
        "**Requested accelerator / resource envelope, if any:**",
    )
    assert _checked_actions(authority) == _STAGE_B_ALLOWED_ACTIONS
    assert all(
        line.startswith("- [ ] ")
        for line in authority.splitlines()
        if line.startswith("- ")
    )
    decision = stage_b.split("### Stage B owner decision", 1)[1]
    assert "**Immutable manifest identity/digest:** <copy exact value from freeze record>" in decision
    assert "**Comparison-plan identity/digest:** <copy exact value from freeze record>" in decision
    assert (
        "A Stage B approval is invalid unless it records both fields above and the individually checked execution authorities."
        in decision
    )

    denials = _section(
        template,
        "## Explicit non-authorizations",
        "## Verification before Stage B execution",
    )
    assert tuple(
        line.removeprefix("- ").strip()
        for line in denials.splitlines()
        if line.startswith("- ")
    ) == _EXPLICIT_NON_AUTHORIZATIONS


def test_project_program_contract_validates() -> None:
    program = json.loads((ROOT / "research" / "program.json").read_text())
    validate_program(program)


def test_generic_program_template_validates_without_stratum_hardware_or_subject_assumptions() -> None:
    template = json.loads((ROOT / "research" / "templates" / "program.template.json").read_text())
    validate_program(template)
    assert template["canonical_source"]["subject_invariant"] != "exactly_one_curated_woman"
    assert set(template["gpu_scheduler"]["resources"]) == {"gpu-a", "gpu-b"}


def test_stratum_program_keeps_its_specific_corpus_and_accelerator_contract() -> None:
    program = json.loads((ROOT / "research" / "program.json").read_text())
    source = program["canonical_source"]
    assert source["path"] == "/mnt/nas-ai-models/training-data/crawlr/approved"
    assert source["subject_invariant"] == "exactly_one_curated_woman"
    resources = program["gpu_scheduler"]["resources"]
    assert resources["4090"] == {
        "host_route": "local",
        "total_vram_gb": 24,
        "usable_vram_gb": 24,
        "notes": "Use the shared scheduler lifecycle before every GPU action.",
    }
    assert resources["strix"] == {
        "host_route": "ssh:max395",
        "total_vram_gb": 110,
        "usable_vram_gb": 100,
        "evergreen_reserved_vram_gb": 10,
        "notes": "The Crawlr labeling process is evergreen. Scheduler state, not a transient utilization snapshot, determines availability.",
    }
    assert program["policy_profile"] == "stratum-single-woman-v1"
    assert "known_failure_modes" in program["specialists"]["required_declaration_fields"]
    assert program["artifact_policy"] == {
        "approved_output_roots": ["/mnt/nas-ai-models/research"],
        "canonical_source_write_allowed": False,
    }
    assert program["research_tree"] == {
        "require_program_root": True,
        "require_parent_issue": True,
        "require_selection_rationale": True,
    }


def test_label_spec_is_unique_and_contains_hold_controls() -> None:
    labels = json.loads((ROOT / "research" / "labels.json").read_text())
    names = [label["name"] for label in labels]
    assert len(names) == len(set(names))
    assert {"research", "research:active", "research:hold", "research:postmortem"}.issubset(names)


def test_control_plane_documents_and_templates_exist() -> None:
    required = [
        ROOT / "AGENTS.md",
        ROOT / "PROJECT_STATUS.md",
        ROOT / "RESEARCH_CONTRACT.md",
        ROOT / "docs" / "EXPERIMENT_TREE.md",
        ROOT / "docs" / "EXPERIMENTS_AND_RESULTS.md",
        ROOT / "research" / "templates" / "program.template.json",
        ROOT / "research" / "templates" / "RESEARCH_CONTRACT.template.md",
        ROOT / "research" / "templates" / "research-arm-body.md",
        ROOT / "research" / "templates" / "pilot-authorization-proposal.md",
        ROOT / ".github" / "ISSUE_TEMPLATE" / "research-arm.yml",
        ROOT / ".github" / "ISSUE_TEMPLATE" / "research-postmortem.yml",
        ROOT / ".github" / "ISSUE_TEMPLATE" / "research-harness-gap.yml",
        ROOT / ".github" / "workflows" / "test.yml",
    ]
    assert all(path.is_file() for path in required)
    forms = {path.name: path.read_text() for path in required[-4:-1]}
    assert "research-harness:" in forms["research-arm.yml"]
    assert all(key in forms["research-arm.yml"] for key in ("parent_issue", "selection_rationale", "surveyed_issue_numbers"))
    assert all(key in forms["research-postmortem.yml"] for key in ("parent_issue", "decision", "evidence_summary"))
    assert all(key in forms["research-harness-gap.yml"] for key in ("trigger", "risk", "decision"))
    pilot_authorization = (ROOT / "research" / "templates" / "pilot-authorization-proposal.md").read_text()
    normalized_pilot_authorization = " ".join(
        line.removeprefix("> ").strip() for line in pilot_authorization.splitlines()
    )
    assert all(
        phrase in normalized_pilot_authorization
        for phrase in (
            "# Pilot Authorization Proposal",
            "## Stage A — preparation authorization",
            "## Freeze and validate after Stage A",
            "## Stage B — execution authorization",
            "may be filled without selected item identities or source SHA-256 values",
            "Stage A preparation approval does not authorize Stage B execution.",
            "`/mnt/nas-ai-models/training-data/crawlr/approved` (read-only)",
            "`/mnt/nas-ai-models/research`",
            "`research-harness validate-comparison-plan`",
            "immutable manifest identity/digest",
            "local_only: true",
            "exactly one axis",
            "caption_max_tokens",
            "Detector disagreement",
            "observer_only",
            "registered launcher",
            "No model invocation, GPU scheduling, additive artifact generation, corpus mutation, or backfill.",
            "No overwrite of `caption.txt`, `t5_*`, `pose.npy`, or other Stratum1 artifacts.",
        )
    )

    _assert_stage_a_is_bounded(pilot_authorization)
    _assert_freeze_and_stage_b_are_bound(pilot_authorization)


@pytest.mark.parametrize(
    "unsafe_action",
    (
        "Invoke `<already-installed local model>` during Stage A.",
        "Use GPU scheduling during Stage A.",
        "Generate additive artifacts during Stage A.",
    ),
)
def test_stage_a_template_rejects_execution_authority_mutation(unsafe_action: str) -> None:
    template = (ROOT / "research" / "templates" / "pilot-authorization-proposal.md").read_text()
    anchor = _STAGE_A_ALLOWED_ACTIONS[-1]
    mutated = template.replace(anchor, f"{anchor}\n- [ ] {unsafe_action}", 1)

    with pytest.raises(AssertionError):
        _assert_stage_a_is_bounded(mutated)


@pytest.mark.parametrize(
    "removed_requirement",
    (
        "**Frozen pilot manifest:**",
        "**Immutable manifest identity/digest:** <versioned manifest ID and SHA-256 or other immutable content identity>.",
        "**Comparison-plan identity/digest:** <versioned filled plan path and immutable identity>.",
        "research-harness validate-comparison-plan",
    ),
)
def test_freeze_template_rejects_missing_provenance_or_validation_record(
    removed_requirement: str,
) -> None:
    template = (ROOT / "research" / "templates" / "pilot-authorization-proposal.md").read_text()
    mutated = template.replace(removed_requirement, "", 1)

    with pytest.raises(AssertionError):
        _assert_freeze_and_stage_b_are_bound(mutated)


@pytest.mark.parametrize(
    "removed_requirement",
    (
        "**Immutable manifest identity/digest:** <copy exact value from freeze record>",
        "**Comparison-plan identity/digest:** <copy exact value from freeze record>",
        "A Stage B approval is invalid unless it records both fields above and the individually checked execution authorities.",
    ),
)
def test_stage_b_template_rejects_unbound_execution_authorization(
    removed_requirement: str,
) -> None:
    template = (ROOT / "research" / "templates" / "pilot-authorization-proposal.md").read_text()
    mutated = template.replace(removed_requirement, "", 1)

    with pytest.raises(AssertionError):
        _assert_freeze_and_stage_b_are_bound(mutated)


def test_stage_b_template_rejects_plain_execution_authority() -> None:
    template = (ROOT / "research" / "templates" / "pilot-authorization-proposal.md").read_text()
    checked_model_action = (
        "- [ ] Invoke `<already-installed local model>` locally for exactly "
        "`<bounded N>` pilot items under the frozen plan's fixed conditions."
    )
    mutated = template.replace(checked_model_action, checked_model_action.removeprefix("- [ ] "), 1)

    with pytest.raises(AssertionError):
        _assert_freeze_and_stage_b_are_bound(mutated)


def test_stage_b_template_rejects_extra_unchecked_execution_authority() -> None:
    template = (ROOT / "research" / "templates" / "pilot-authorization-proposal.md").read_text()
    anchor = (
        "- [ ] Generate specifically named additive artifacts for only the frozen pilot items: "
        "`<pass list, model size, output root>`. This requires separately checked data/GPU authority."
    )
    mutated = template.replace(anchor, f"{anchor}\n- [ ] Download an unapproved model for the pilot.", 1)

    with pytest.raises(AssertionError):
        _assert_freeze_and_stage_b_are_bound(mutated)


@pytest.mark.parametrize(
    "removed_denial",
    (
        "- No merge or direct push to `main`.",
        "- No canonical-source mutation.",
        "- No corpus-wide or derived-tree backfill.",
        "- No external image model.",
        "- No model installation or download unless separately named and approved.",
        "- No scheduler operation without the separately approved manifest and registered launcher.",
        "- No overwrite of `caption.txt`, `t5_*`, `pose.npy`, or other Stratum1 artifacts.",
        "- No empirical PASS claim merely because a plan validates or an inference job completes.",
    ),
)
def test_stage_b_template_rejects_missing_explicit_global_denial(removed_denial: str) -> None:
    template = (ROOT / "research" / "templates" / "pilot-authorization-proposal.md").read_text()
    mutated = template.replace(removed_denial, "", 1)

    with pytest.raises(AssertionError):
        _assert_freeze_and_stage_b_are_bound(mutated)


def test_resumption_documents_preserve_the_sole_active_arm_and_two_stage_boundary() -> None:
    status = (ROOT / "PROJECT_STATUS.md").read_text()
    tree = (ROOT / "docs" / "EXPERIMENT_TREE.md").read_text()
    ledger = (ROOT / "docs" / "EXPERIMENTS_AND_RESULTS.md").read_text()
    research_readme = (ROOT / "research" / "README.md").read_text()

    assert "#4 is the sole `research:active`" in status
    assert "The `stratum-ffhq` strategist is paused" in status
    assert "strategist and observer records remain enabled" not in status
    assert "[ACTIVE / METRIC-RISK / PRE-COMPUTE] #4" in tree
    assert "## Arm 0 — Geometry-grounded captioning prototype — `[PROPOSAL — PENDING]`" in ledger
    assert "## Arm 0 — Geometry-grounded captioning prototype — `[ACTIVE — PENDING]`" not in ledger
    assert "Stage A" in research_readme
    assert "Stage B" in research_readme


def test_program_keeps_compact_context_separate_from_legacy_t5() -> None:
    program = json.loads((ROOT / "research" / "program.json").read_text())
    representation = program["representation"]
    assert representation["compact_context_target_tokens"] > representation["legacy_text_encoder_max_tokens"]
    assert representation["compact_artifacts"] == {
        "structured": "context4k.json",
        "human_readable": "context4k.md",
        "provenance": "compression.json",
    }


def test_stratum_context_bundle_template_matches_the_project_contract() -> None:
    program = json.loads((ROOT / "research" / "program.json").read_text())
    bundle = json.loads((ROOT / "research" / "templates" / "context4k-bundle.json").read_text())
    validate_compression_bundle(bundle, program)
