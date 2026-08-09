# Design Proposal — Reframe the Dossier Objective from Absolute Floors to a Directional Target

**Status:** DRAFT for owner review. No code or contract changed by this document.
**Resolves:** the `research:needs-human` stall on issue #46 (arm #36 `dossier-context4k` parked 4+ cycles).
**Decision it encodes:** issue #46 **Option A** (reframe to honest attainable scale), generalized into the program contract so it stops being a per-cycle human gate.
**Author:** design proposal prepared for Tim's review; route as a `research:harness-gap` discussion or straight to a draft PR.

---

## 1. Problem statement

The program goal is convex and correct: **expand** each image into a rich evidence dossier, then **compress** it into a bounded context a text-to-image training path can consume. Convexity comes from the *relationship* (many signals → one bounded context), not from any absolute magnitude.

Two budget numbers are currently conflated:

- **`compact_context` = 4000 tokens** is a **real downstream constraint** (the T2I ingestion budget). If the compact context exceeds what training can feed, it is useless. **Keep it as a hard ceiling.**
- **`expanded_dossier` = 100000 tokens** is a **paper-borrowed aspiration**, not a consumer requirement. Nothing downstream breaks at 13.5K. It is currently wired as a **pass/fail floor** (`expanded_dossier_min_tokens`), and that is the defect.

The strategist's honest-expansion ceiling audit (commit `2bd3292`, run `/mnt/nas-ai-models/research/stratum/dossier-expansion-audit-v1/`) measured the honest evidence ceiling at **~13.5K tokens/item** (17–27 facts × ~500 tok/fact) against the 100K floor — a 7–50× structural gap. The harness correctly refused to fabricate content to fill the gap, then treated "100K not reached" as **arm failure**, which forced the #46 human decision. The arm has been parked ever since.

**Root cause:** the 100K floor is doing the job of a *metric* when it should be doing the job of a *direction*. "Reach exactly 100K" is a brittle absolute target that stalls when reality doesn't match the paper. "Maximize honest evidence density, then verify it survives compression to ≤4K better than a plain summary" is a robust directional target that keeps producing signal at any scale.

## 2. What is preserved (the honesty gate is NOT weakened)

This proposal changes **only** the budget semantics. Every honesty invariant in `validate_compression_bundle` stays exactly as-is:

- `expanded_dossier.evidence_ids` must be a **non-empty, de-duplicated** list — the dossier must carry provenance.
- Every `compact_context.claim` must carry **non-empty `evidence_ids`** that resolve into the dossier's evidence set — the claim→evidence path is the actual anti-fabrication gate and is untouched.
- `compact_context.token_count` must be **≤ `compact_context_target_tokens` (4000)** — the real consumer ceiling stays a hard gate.
- `compact_context.token_count` must be **> `legacy_text_encoder_max_tokens` (512)** — the context4k must remain a first-class long-context artifact, never silently truncated into the legacy T5 path.
- `compact_context.token_count` must be **≤ `expanded_dossier.token_count`** — compression cannot exceed its source.
- `build_compression_bundle`'s under-budget refusal behavior is preserved; nothing about the honesty check in `dossier_expand.py` is relaxed.

The only number that moves is `expanded_dossier_min_tokens`, and it moves from a **blocking floor** to a **floor-signal** (see §4).

## 3. Contract changes (`research/program.json` → `representation` block)

Bump `schema_version` 1 → 2 and re-shape `representation`:

```json
"representation": {
  "expanded_dossier_target_tokens": 100000,
  "expanded_dossier_min_tokens": 4001,
  "expanded_dossier_target_role": "aspiration",
  "compact_context_target_tokens": 4000,
  "compact_context_min_tokens": 4000,
  "legacy_text_encoder_max_tokens": 512,
  "saturation": {
    "enabled": true,
    "window": 2,
    "min_delta_support_ratio": 0.02
  },
  "compact_artifacts": {
    "structured": "context4k.json",
    "human_readable": "context4k.md",
    "provenance": "compression.json"
  },
  "rule": "The compact context is a first-class provenance-bearing artifact bounded by the downstream consumer budget (<=4000 tokens, >512 legacy). The expanded dossier grows honestly with the evidence supply; 100000 is an aspiration metadata target, not a pass gate. Evidence supply is saturated when a window of consecutive evidence-growth arms each yield < min_delta_support_ratio on the <=4K compression verdict."
}
```

Field semantics:

- `expanded_dossier_target_tokens: 100000` — **kept, but demoted to `aspiration` metadata.** Recorded, reported, never a gate.
- `expanded_dossier_target_role: "aspiration"` — explicit machine-readable flag so no future reader mistakes it for a floor.
- `expanded_dossier_min_tokens: 4001` — the new floor is **structural, not aspirational**: the dossier must be larger than the compact context it compresses into (`> compact_context_target_tokens`), otherwise "expand-then-compress" is vacuous. `4001` is the minimal honest value; it is **not** a target to hit, just the convexity invariant.
- `saturation` — the **honest, convex replacement for "did I collect enough?"** (see §5). 100K's only useful signal is preserved as a diminishing-returns test, not an absolute number.

## 4. Validator changes (`src/research_harness/contracts.py`)

`validate_program` (representation block, ~lines 482–510):

- Require `expanded_dossier_target_tokens`, `compact_context_target_tokens`, `compact_context_min_tokens`, `legacy_text_encoder_max_tokens` as today.
- **Accept `expanded_dossier_min_tokens` as optional**, defaulting to `compact_context_target_tokens + 1` when absent.
- Replace the invariant `expanded_min > expanded` check: now require `expanded_dossier_min_tokens > compact_context_target_tokens` (dossier strictly larger than the compact ceiling) and `expanded_dossier_target_tokens >= expanded_dossier_min_tokens` (aspiration not below the floor).
- Accept `expanded_dossier_target_role` as an optional enum `{"aspiration", "gate"}`; **default `"aspiration"`**. When `"gate"`, restore the old strict floor behavior (escape hatch, documented as not recommended).
- Accept optional `saturation` object: `enabled` (bool), `window` (positive int), `min_delta_support_ratio` (number in [0,1]).

`validate_compression_bundle` (~lines 933–956):

- Change the expanded-dossier check at line 936 from `expanded_tokens < expanded_dossier_min_tokens` to enforce the **structural floor**: `expanded_tokens` must be `> compact_context_target_tokens` (dossier larger than the compact ceiling) **and** `>= expanded_dossier_min_tokens` (which now defaults to 4001). **No 100K comparison anywhere.**
- Keep every other check byte-for-byte: evidence_ids non-empty/dedup, compact ≤ target, compact > legacy, compact ≤ expanded, claim→evidence path.

`dossier.py` `_COMPACT_TARGET` rail and `build_compression_bundle`: unchanged (already keyed to the 4000 compact target).

## 5. Saturation signal (replaces "reach 100K" as the stopping rule)

Add a deterministic saturation readout to `dimension-sweep-status` (and surface it in the strategist cycle):

- Track the `delta_support_ratio` (or reconstruction delta) each **evidence-growth** arm produces on the ≤4K compression verdict.
- When `saturation.window` consecutive evidence-growth arms each yield `< min_delta_support_ratio`, emit `evidence_saturated: true`.
- `evidence_saturated` is the honest "we collected enough" signal: adding more specialists stops helping the ≤4K compression, **regardless of whether the dossier is 13K or 100K.** It routes to `brainstorm-new-data` (new evidence *parts* / new data sources / new model classes), not to a human gate.

This is the convex stopping rule the 100K number was gesturing at, expressed as a gradient (`Δ < ε over a window`) instead of an absolute.

## 6. Strategist prompt changes (small, targeted)

- Remove any instruction that treats the 100K expanded floor as a pass/fail gate. State the objective as: **maximize honest evidence density, then verify the ≤4K evidence-linked compact context beats a plain ≤4K summary at equal budget.**
- Add: the dossier grows with the evidence supply; **growing evidence (option B) is the default climb direction and is authorized autonomous work** — renegotiating a budget number is the only thing that needs a human.
- Add: report `expanded_dossier_target_tokens` as **aspiration metadata** ("reached X of 100K aspiration"), never as a missed gate.
- Add: read `evidence_saturated` from `dimension-sweep-status`; when true, widen (new parts/models/data) rather than re-running the same arm pattern.

## 7. Migration & compatibility

- **Registry schema:** no change to `dimension_registry.py` states; `dossier-context4k` stays `active` and can now conclude at honest scale.
- **Existing artifacts:** none mutated. This is a contract-semantics + validator change only; no corpus, no legacy `caption.txt`/`t5_*`/`pose.npy`, no backfill.
- **Schema version:** bump `program.json` `schema_version` 1→2; `validate_program` accepts both (v1 = legacy strict floor, v2 = directional) so in-flight v1 references don't break, but the program moves to v2.
- **Tests to add/update:**
  - `validate_compression_bundle` passes a 13.5K-token honest dossier → ≤4K compact bundle (currently fails the 100K floor).
  - `validate_compression_bundle` still **rejects** a bundle missing claim→evidence_ids (honesty gate intact).
  - `validate_compression_bundle` still **rejects** compact > 4000 and compact ≤ 512.
  - `validate_program` rejects `expanded_dossier_min_tokens <= compact_context_target_tokens`.
  - `validate_program` defaults `expanded_dossier_min_tokens` to 4001 when omitted.
  - `dimension-sweep-status` emits `evidence_saturated: true` after `window` sub-threshold arms (synthetic fixture).
  - Update the two existing tests that assert the 100K floor behavior.

## 8. Acceptance criteria

1. A frozen-cohort honest dossier (~13.5K tokens) compresses to a ≤4K bundle that **passes** `validate_compression_bundle`.
2. The claim→evidence honesty gate, the ≤4K ceiling, and the >512 legacy floor all still **reject** violations (no regression).
3. `dimension-sweep-status` reports `evidence_saturated` correctly on a synthetic sub-threshold window.
4. `pytest tests/ -q` green; `validate-program research/program.json` valid at schema v2.
5. Arm #36 can run its pre-registered round-trip (evidence-linked ≤4K vs plain ≤4K summary) at honest scale and reach a BETTER/NOT_BETTER verdict via `autonomous-tick` — **no human A/B ruling required to proceed**, because the absolute floor no longer exists to be missed.

## 9. Non-goals / explicitly out of scope

- Does **not** authorize GPU claims, model installs, corpus mutation, backfill, merges, or `main` pushes. All existing hold conditions remain in force.
- Does **not** weaken the honesty gate, the ≤4K consumer ceiling, or the >512 legacy-encoder floor.
- Does **not** pick option B over A — it makes the A/B distinction moot by removing the absolute floor, so evidence-growth (B) becomes normal autonomous progress and the floor-renegotiation (A) is no longer a blocking human decision.
- Does **not** implement the `blocked`-state / dependency-graph selector improvements from the earlier review; those are separable follow-ups (this reframe shrinks the stall they were designed to escape, but they remain worth doing on their own).

---

## Review focus for Tim

1. Is `expanded_dossier_min_tokens = 4001` (dossier strictly larger than the compact ceiling) the right structural floor, or do you want a different convexity invariant?
2. Is `saturation.window = 2` / `min_delta_support_ratio = 0.02` the right sensitivity for the "collected enough" signal, or should it be more conservative (window 3)?
3. Should `expanded_dossier_target_role` even keep a `"gate"` escape hatch, or is aspiration-only cleaner?
4. Route: `research:harness-gap` issue for discussion first, or straight to a draft PR against `feat/autonomous-research-harness`?
