"""Guard durable project-control assets against silent contract drift."""

from __future__ import annotations

from collections.abc import Callable
import json
import re
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from research_harness import validate_compression_bundle, validate_program

ROOT = Path(__file__).resolve().parents[1]
_STAGE_A_PROPOSAL_PATH = (
    ROOT / "research" / "proposals" / "stage-a-caption-context-parity-preparation.md"
)

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

_STAGE_A_PROPOSAL_ALLOWED_ACTIONS = (
    "Select no more than 24 canonical-source candidates using the selection protocol above.",
    "Read and SHA-256 hash only the selected canonical-source images once each.",
    "Read only the selected candidates’ existing derived-artifact availability/readability facts; do not mutate `crawlr/stratum`.",
    "Write only the listed manifest, preparation log, review record, and non-executing comparison-plan draft beneath the preparation output root.",
)

_STAGE_A_PROPOSAL_AUTHORITY_INTRO = (
    "The owner may approve only the checked items below. Any unchecked item remains denied."
)

_STAGE_A_PROPOSAL_NONAUTHORIZATION_SUMMARY = (
    "This request explicitly denies model invocation, model download/installation, GPU scheduling, "
    "GPU claims, additive artifact generation, corpus mutation, derived-tree mutation, backfill, "
    "external image-model use, merge, direct `main` push, and any Stage-B execution."
)

_STAGE_A_PROPOSAL_NONAUTHORIZATION_BULLETS = (
    "- no inference or caption generation;",
    "- no repair or invocation of `caption2`;",
    "- no `context4k` production or consumption;",
    "- no new specialist qualification claim;",
    "- no empirical result or PASS/FAIL verdict;",
    "- no scheduler request, poll, claim, launch, activate, heartbeat, release, or kill action;",
    "- no modification of `caption.txt`, `t5_*`, `pose.npy`, `pose2.npy`, `seg2.npy`, "
    "`determinations.json`, `caption2.txt`, `t52_*`, or any other corpus artifact.",
)

_STAGE_A_PROPOSAL_STAGE_B_DENIAL = (
    "**Stage B execution is not requested or authorized by this document.**"
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


def test_program_model_sourcing_is_open_world_while_sensitive_hosting_stays_gated() -> None:
    """(Owner directive 2026-08-05) Model SOURCING is open-world: the loop may
    discover/download/install/qualify new candidate models when local options
    are exhausted, and may scan literature/arXiv for better or new-part models.
    This must NOT reopen hosted third-party inference of the sensitive canonical
    corpus, which stays gated. The validator must accept installation=True while
    still rejecting external image-hosting authorization."""
    program = json.loads((ROOT / "research" / "program.json").read_text())
    assert program["content_policy"]["model_sourcing"] == "open_world"
    assert program["content_policy"]["model_execution"] == "local_first"
    assert program["content_policy"]["autonomous_external_image_model_allowed"] is False
    assert program["autonomy"]["autonomous_model_installation_allowed"] is True
    assert program["autonomy"]["requires_hold"] == [
        "GPU request, claim, launch, heartbeat, or release",
        "dataset mutation or corpus-wide backfill",
        "hosted third-party inference of the sensitive canonical corpus",
        "sending canonical source images to an external hosted image service",
        "metric or policy uncertainty",
        "scope-changing specialist or architecture opportunity",
        "merge or direct push to main",
    ]
    assert any("arXiv" in s or "literature" in s for s in program["autonomy"]["authorized_without_new_human_approval"])
    assert program["content_policy"]["external_model_requirement"] != (
        "A non-local image model requires an explicit reviewed policy and qualification issue before use."
    )


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

    # The resumption documents must name exactly one active arm and it must
    # agree across status + tree (currently lighting #33, the selector pick
    # after skin-color #31 concluded BETTER; the completed skin-color run is
    # recorded in the ledger).
    assert "is the sole `research:active`" in status
    assert "#33 lighting" in status
    assert "The `stratum-ffhq` strategist is re-engaged for autonomous research" in status
    assert "The `stratum-ffhq` strategist is paused" not in status
    assert "#33 lighting" in tree
    assert "## Arm #31 — skin-color/tone evidence — `[EMPIRICAL RUN COMPLETE — VERDICT: BETTER]`" in ledger
    assert "## Arm 0 — Geometry-grounded captioning prototype — `[PROPOSAL — PENDING]`" in ledger
    assert "## Arm 0 — Geometry-grounded captioning prototype — `[ACTIVE — PENDING]`" not in ledger
    assert "Stage A" in research_readme
    assert "Stage B" in research_readme


def _assert_stage_a_proposal_is_bounded(proposal: str) -> None:
    assert "**Arm:** #4" in proposal
    assert "**Parent program:** #2" in proposal
    assert "`DRAFT / STAGE A REQUEST / NO EXECUTION AUTHORITY`" in proposal
    assert "maximum 24 candidates" in proposal
    assert not re.search(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", proposal, flags=re.IGNORECASE)
    assert "source file names" not in proposal
    assert "image IDs" not in proposal
    assert "no candidate has been selected, no source has been read or hashed" in proposal
    assert "**Canonical root:** `/mnt/nas-ai-models/training-data/crawlr/approved`" in proposal

    selection = _section(
        proposal,
        "### Selection protocol after approval",
        "## Stage-A requested authority",
    )
    assert "Selection happens only after Stage-A approval" in selection
    assert "no entry is opened or decoded before selection" in selection
    assert "six equal ordinal slices" in selection
    assert "[floor(j*N/6), floor((j+1)*N/6))" in selection
    assert "No source content, dimensions, or derived artifacts are read for unselected candidates" in selection

    authority = _section(
        proposal,
        "## Stage-A requested authority",
        "## Stage-A non-authorizations",
    )
    assert tuple(line for line in authority.splitlines() if line) == (
        _STAGE_A_PROPOSAL_AUTHORITY_INTRO,
        *(f"- [ ] {action}" for action in _STAGE_A_PROPOSAL_ALLOWED_ACTIONS),
    )
    assert _checked_actions(authority) == _STAGE_A_PROPOSAL_ALLOWED_ACTIONS

    non_authorizations = _section(
        proposal,
        "## Stage-A non-authorizations",
        "## Required freeze before any Stage-B request",
    )
    assert tuple(line for line in non_authorizations.splitlines() if line) == (
        _STAGE_A_PROPOSAL_NONAUTHORIZATION_SUMMARY,
        "In particular:",
        *_STAGE_A_PROPOSAL_NONAUTHORIZATION_BULLETS,
        _STAGE_A_PROPOSAL_STAGE_B_DENIAL,
    )

    freeze = _section(
        proposal,
        "## Required freeze before any Stage-B request",
        "## Owner decision — unfilled",
    )
    assert "repair and test the prototype backend forwarding of `caption_max_tokens`" in freeze


def test_stage_a_caption_context_parity_proposal_is_preparation_only() -> None:
    proposal = _STAGE_A_PROPOSAL_PATH.read_text()
    status = (ROOT / "PROJECT_STATUS.md").read_text()
    tree = (ROOT / "docs" / "EXPERIMENT_TREE.md").read_text()
    research_readme = (ROOT / "research" / "README.md").read_text()

    _assert_stage_a_proposal_is_bounded(proposal)
    assert "research/proposals/stage-a-caption-context-parity-preparation.md" in status
    assert "research/proposals/stage-a-caption-context-parity-preparation.md" in tree
    assert "- `proposals/` — filled, draft-only owner-decision requests." in research_readme


@pytest.mark.parametrize(
    "mutate",
    (
        lambda text: text.replace(
            "- [ ] Write only the listed manifest, preparation log, review record, and non-executing comparison-plan draft beneath the preparation output root.",
            "- [ ] Write only the listed manifest, preparation log, review record, and non-executing comparison-plan draft beneath the preparation output root.\n"
            "- [ ] Invoke an already-installed local model for the selected candidates.",
            1,
        ),
        lambda text: text.replace(
            "- [ ] Write only the listed manifest, preparation log, review record, and non-executing comparison-plan draft beneath the preparation output root.",
            "- [ ] Write only the listed manifest, preparation log, review record, and non-executing comparison-plan draft beneath the preparation output root.\n"
            "- [ ] Request GPU scheduling for the selected candidates.",
            1,
        ),
        lambda text: text.replace(
            "- [ ] Write only the listed manifest, preparation log, review record, and non-executing comparison-plan draft beneath the preparation output root.",
            "- [ ] Write only the listed manifest, preparation log, review record, and non-executing comparison-plan draft beneath the preparation output root.\n"
            "- [ ] Generate additive artifacts for the selected candidates.",
            1,
        ),
        lambda text: text.replace(
            "The owner may approve only the checked items below. Any unchecked item remains denied.",
            "The owner may approve only the checked items below. Any unchecked item remains denied. "
            "The owner may also invoke an already-installed local model for selected candidates.",
            1,
        ),
        lambda text: text.replace(
            _STAGE_A_PROPOSAL_NONAUTHORIZATION_SUMMARY,
            _STAGE_A_PROPOSAL_NONAUTHORIZATION_SUMMARY
            + " However, model invocation is permitted for selected candidates.",
            1,
        ),
        lambda text: text.replace(
            "Selection happens only after Stage-A approval",
            "Selection happens before Stage-A approval",
            1,
        ),
        lambda text: text.replace(
            "4. repair and test the prototype backend forwarding of `caption_max_tokens` before any comparison using the prototype path;\n",
            "",
            1,
        ),
    ),
    ids=(
        "extra_stage_a_local_model_authority",
        "extra_stage_a_gpu_authority",
        "extra_stage_a_artifact_authority",
        "plain_prose_stage_a_model_authority",
        "keyword_preserving_nonauthorization_rewrite_with_model_permission",
        "selection_before_approval",
        "removed_caption_max_tokens_prerequisite",
    ),
)
def test_stage_a_proposal_rejects_unsafe_public_entrypoint_mutations(
    mutate: Callable[[str], str],
) -> None:
    original_read_text = cast(Callable[..., str], Path.read_text)

    def patched_read_text(path: Path, *args: object, **kwargs: object) -> str:
        text = original_read_text(path, *args, **kwargs)
        return mutate(text) if path == _STAGE_A_PROPOSAL_PATH else text

    with patch.object(Path, "read_text", patched_read_text):
        with pytest.raises(AssertionError):
            test_stage_a_caption_context_parity_proposal_is_preparation_only()


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


def test_first_500_core_coverage_design_is_precompute_and_not_a_stage_a_substitute() -> None:
    report_path = ROOT / "research" / "coverage" / "first-500-core-coverage-v1.json"
    design_path = ROOT / "docs" / "FIRST_500_CORE_COHORT_PILOT_DESIGN.md"
    report = json.loads(report_path.read_text())
    design = design_path.read_text()

    assert report["kind"] == "core-artifact-coverage-audit"
    assert report["status"] == "PRE_COMPUTE_READ_ONLY"
    assert report["source_content_read_count"] == 0
    cohort = report["cohort"]
    assert cohort["eligible_source_count"] == 11825
    assert cohort["membership_sha256"] == "4e9f8ca775a6e62e308afcccb1e36cce2a5d0bf1f5579631c4a76af0bc80f57c"
    assert cohort["requested_limit"] == 500
    assert cohort["selected_count"] == 500
    assert cohort["selection_rule"] == "first limit eligible flat source filenames in bytewise POSIX relative-path order"
    assert len(cohort["source_relative_paths"]) == 500
    assert report["summary"]["core_complete_count"] == 500
    assert report["summary"]["later_chain_complete_count"] == 10
    assert report["summary"]["legacy_chain_complete_count"] == 500
    assert report["detail_provenance"]["item_details_included"] is False
    assert "items" not in report

    assert "It is **not** a replacement, interpretation, or extension of the immutable Stage-A 24-item ordinal-slice manifest" in design
    assert "The Stage-A manifest's six global ordinal slices are not the first-500 cohort." in design
    assert "**Only 10 / 500**" in design
    assert "No Stage-B action is authorized." in design


def test_coverage_balanced_candidate_freeze_is_nonexecuting_and_preserves_stage_a() -> None:
    freeze_path = ROOT / "docs" / "FIRST_500_COVERAGE_BALANCED_CANDIDATE_FREEZE.md"
    freeze = freeze_path.read_text()

    for expected in (
        "PENDING_PRE_COMPUTE_NON_EXECUTING",
        "first-500-coverage-balanced-candidate-manifest-v1.json",
        "8684c6e38c90b12898135235164677d780a4c897122f26a4b386f07283a9c5e0",
        "b18843c759a8b93165a1261350ac46feea7cc62df787d44d4beb0ef9bc4b132d",
        "12 portrait, 6 squareish, and 6 landscape",
        "0 / 24",
        "immutable Stage-A",
        "Stage-B execution",
    ):
        assert expected in freeze
