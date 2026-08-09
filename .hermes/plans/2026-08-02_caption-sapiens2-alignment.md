# Aligning Text Captions with Sapiens2 Conditioning (Brainstorm Findings → Plan)

> **Status:** DRAFT for review. Nothing in this document has been implemented.
> **Session type:** Brainstorming / information-gathering. No code shipped as a result of the discussion itself; this plan captures the findings and proposes the next actionable artifact (a `determinations.json` schema + extraction pass) for explicit approval before any pipeline change.

**Goal:** Make stratum text captions *consistent with and complementary to* the deterministic Sapiens2 conditioning signals, instead of letting the VLM re-derive (and hallucinate) geometry we already measure exactly.

**Core insight (the one that framed the whole session):** The current caption pass asks gemma3:27b to *infer* pose, depth, framing, and anatomy **from pixels alone** — the same pixels Sapiens2 has already measured deterministically. The caption is therefore a **noisy re-derivation of ground truth we already hold.** The fix is not primarily "a better prompt" — it is restructuring *who says what*: geometry comes from the artifacts, language/mood/color/texture comes from the VLM.

---

## 1. The four levers (from the brainstorm)

| Lever | What | Effort | Verdict |
|---|---|---|---|
| **1. Prompt restructure** | Rewrite `CAPTION_PROMPT` around *relations* (subject↔frame, limb↔limb, subject↔background, depth layering) per MiniMax H3's "Contextual Omni Representation" — captions describe relationships, not just elements. | Hours | Do now, nearly free, testable on ~20 images. |
| **2. Conditioned captioning** | Feed the VLM a block of **deterministic facts** extracted from stratum2 artifacts; caption verbalizes measured truth instead of guessing. | Days | **The structural win.** This plan's focus. |
| **3. Fine-tune a captioner** | QLoRA a local open VLM (Qwen2.5-VL-7B / Gemma3-12B class) on (image + facts → caption) silver data from Lever 2, per the DALL-E 3 recipe. | Weeks | Later, if Lever 2 pilot justifies it. Local only (content constraint). |
| **4. Structured-text serialization** | Skip prose; emit templated text from the structured signal. | — | **Complement, not replacement.** We need rich prose (mood, color, light) for a T2I model — structured text can't supply that. Keep as a separate debug/documentation artifact. |

**Content constraint (decides the platform):** Source material includes swimwear/nude subjects. Cloud vision APIs are a ToS/consistency risk (silent refusals poison the dataset). **All captioning stays local** (Ollama gemma3:27b now; local fine-tune later). seg2 skin-fraction doubles as a free content flagger for dataset hygiene.

---

## 2. Empirical validation on a real sample

All of Lever 2 was validated end-to-end against **one fully-enriched image**:
`/mnt/nas-ai-models/training-data/crawlr/stratum/0jkbuyws5tk2x9bxo5ui24h1o8b9/`
(936×1778 vertical, full stratum + stratum2 artifact set, plus visual ground truth from direct inspection).

**Ground truth:** Frontal nude woman, head tilted back, eyes closed, hair blowing. Both arms reach down/inward; both hands grip a white open-face motorcycle helmet held at **pelvis/upper-thigh level, in front of the body**. Frame crops **mid-thigh**. Camera at **hip height, slightly low**. Grey rock background.

### 2a. Caption vs. geometry (the existing caption was ~90% right — the 10% is the point)

| Caption claim | Geometry | Verdict |
|---|---|---|
| "facing the viewer" | nose (453,554) centered between ears (541/350), both conf 0.99 | ✓ trivially confirmed |
| "slight upward angle" (low camera) | pointmap camera-frame: nose Y=−0.18, shoulder Y=+0.04, wrist Y=+0.48 → camera at chin/chest level | ✓ **derived, not guessed** |
| "holds helmet in both hands" | wrists (448,1401)/(558,1457) conf 0.93/0.97; fingertip centroids 13px apart → both hands on one object | ✓ confirmed |
| "at **waist** level" | fingertips at hip_y **+128px** → hands at **pelvis/upper-thigh**, not waist | ✗ imprecise — and geometry-checkable |
| "black bars framing either side" | fg spans x∈[94,935] of W=936 → no bars | ✗ wrong — but letterboxing lives in `pixel.npy`, not pose/seg (see 2d) |
| "blurred light-grey rock" | — | not checkable from pose/seg; VLM's job (background semantics) |

**Takeaway:** the failure classes are *systematic*, not random — vertical-level vocabulary drift (waist/hip/thigh), invented framing claims, hedged camera claims where a measurement exists. Structured facts fix the **variance**, and pin the dimensions where VLMs hallucinate most (which hand, cropped where, how high the camera is).

### 2b. `pointmap.npy` is the sleeper artifact
Camera-frame metric XYZ; convention **confirmed empirically** (corr(image_y, Y)=0.999, +Y down, +Z forward), so **the camera is at the origin**. Camera height relative to subject = a subtraction (nose Y −0.18 vs shoulder +0.04 → camera at chin/chest level). Subject distance = median fg Z (2.50m, tight p5–p95 2.45–2.71 → single body on a plane ~2.5m out). Highest value-per-effort of the whole set — **and currently at 0% on FFHQ.**

### 2c. `pose2.npy` hand keypoints — CORRECTED finding
Initial read ("no hand keypoints") was **wrong**. The official Goliath-308 layout (from `facebookresearch/sapiens` `lite/demo/classes_and_palettes.py`) is:
- **0–16 body** (0 nose, 1 leye, 2 reye, 3 lear, 4 rear, 5 lsho, 6 rsho, 7 lelb, 8 relb, **9 lhip, 10 rhip**, 11 lkne, 12 rkne, 13 lank, 14 rank, 15–20 toes/heels)
- **21–41 RIGHT hand** (41 = right_wrist), **42–62 LEFT hand** (62 = left_wrist)
- **63–69 body extras** (olecranon, acromion, **69 = neck**), **70–307 dense face** (238 pts)

My first pass used a COCO-WholeBody guess (9/10 = wrists) and misread hips as wrists. Corrected: wrists are **#41/#62** (conf 0.93/0.97), both hands resolve cleanly with fingertip spread 33/69px. **pose2 alone is sufficient for hand-level facts; DWPose `pose.npy` is a cross-check, not a requirement.** (On this nude sample both estimators agree to within ~15px on wrists.) Note the mapping must be frozen from the official table, never assumed — this exact error is the cost of assuming.

### 2d. `seg2` true class list — CORRECTED finding
seg2 is **DOME_CLASSES_29** (from the installed `sapiens` pkg `seg_utils.py`), *not* the Sapiens1 28-class list. Classes: Background, Apparel, Eyeglass, Face_Neck, Hair, L/R Foot, L/R Hand, L/R Lower_Arm, L/R Lower_Leg, L/R Shoe, L/R Sock, L/R Upper_Arm, L/R Upper_Leg, Lower_Clothing, Torso, Upper_Clothing, Lower_Lip, Upper_Lip, Lower_Teeth, Upper_Teeth, Tongue. **There are no "armor" classes** — my earlier "armor hallucination" finding was a wrong-class-list artifact. What actually fired on this nude: Lips/Teeth/Tongue (correct on the face), Apparel (5px, noise), and **zero clothing classes** (correct for a nude). **The helmet is invisible to seg2** (a held prop is not a body part) — it appears only as dense fg between the wrists. So prop detection needs a **size + persistence + wrist-adjacency gate**, not a class read.

### 2e. Framing has a two-tier answer
- **Crop point** (structured): lowest confident keypoint vs. frame bottom + which seg classes reach the edge. Here knees conf 0.84/0.81 at y≈1770, ankles conf 0.10/0.12 (collapsed at frame bottom) → "cropped mid-thigh" ✓.
- **Letterbox/black bars** (pixels): not visible to pose/seg (fg is subject-masked). ~10 lines against `pixel.npy` (edge-column intensity histogram) if we want framing facts complete. Optional.

---

## 3. Proposed `determinations.json` schema (the reviewable artifact)

Emitted per-image as its own idempotent, CPU-only, embarrassingly-parallel stratum pass. **Every field carries a confidence; uncertain measurements are omitted, never guessed.** Continuous values use the *subject's own body* as the reference frame (kills waist/hip drift); relations are **natural-language phrases**, not enum tokens (the consumer is the caption2 prompt / T5, both of which read prose natively).

**Design rule (user-directed):** determinations contains *only what geometry measures* — raw continuous values + open-set limb/part relations. **Posture and activity are excluded and left to caption2** (the interpreter). A closed taxonomy of poses/crops would be routinely escaped by the approved image set (eye closeup, hand, knee; kneeling, cartwheel, ballet, high jump, floating) and would emit confident-wrong labels that poison caption2. Measurements scale to unseen configurations; labels don't.

### Layer 1 — raw measurements (always emitted, no thresholds)

```json
{
  "schema_version": 2,
  "subject": {
    "n_detections": 1,
    "detector_anomaly": "none | no_detection | extra_detections(k)",
    "note": "exactly one real subject guaranteed by curation; N!=1 is a quality flag, not content"
  },
  "subject_extent": {
    "frame_frac": 0.38,
    "bbox_px": [94, 416, 935, 1777],
    "frame_px": [936, 1778],
    "h_position": "left | left_of_center | center | right_of_center | right",
    "letterbox_bars": "none | left_right | top_bottom"
  },
  "body_parts_visible": [
    {"part": "face",         "pixel_frac": 0.041, "kp_conf": 0.97},
    {"part": "torso",        "pixel_frac": 0.166, "kp_conf": 0.95},
    {"part": "left_arm",     "pixel_frac": 0.062, "kp_conf": 0.90},
    {"part": "left_hand",    "pixel_frac": 0.012, "kp_conf": 0.88},
    {"part": "left_leg",     "pixel_frac": 0.089, "kp_conf": 0.84}
  ],
  "orientation": {
    "upright_deg": 8.0,
    "definition": "angle of neck->hip axis vs image-down; 0=upright, 90=horizontal, 180=inverted"
  },
  "camera": {
    "distance_m": 2.5,
    "height_rel_shoulder_m": -0.14,
    "note": "from pointmap (camera frame, origin at camera); reserved until pointmap backfill"
  },
  "content": {
    "skin_fraction": 0.31,
    "clothing_classes_present": []
  }
}
```

- `body_parts_visible` **replaces the `crop_point` enum entirely.** A crop *is* the visible-part list: eye closeup → `[{face-region…}]`; hand closeup → `[{left_hand…}]`; knee → `[{left_lower_leg…}, {left_upper_leg…}]`. No special-casing, works for crops nobody anticipated. (Coarse ~8-bucket parts here; the full DOME-29 fractions stay available in raw seg2 if finer granularity is ever needed.)
- `orientation.upright_deg` handles cartwheel (≈180°), floating (≈90°), kneeling, standing — *without naming any of them*. caption2 reads `upright_deg: 152` + the image and says "inverted, mid-cartwheel" far more reliably than geometry alone could.
- `camera.height_rel_shoulder_m` (pointmap) replaces the `camera_height: head|chest|hip|…` enum with a continuous value ("14 cm below shoulder level").

### Layer 2 — open-set limb/part relations (natural-language phrases, sparse, combinatorial)

```json
{
  "relations": [
    "face turned toward camera",
    "left arm extended downward",
    "right arm extended downward",
    "hands together in front of body",
    "hands gripping an object at pelvis level"
  ]
}
```

- Generated from joint-angle / relative-position rules, but an **open set** — novel combinations *add* rather than collide. A ballet arabesque and a high jump yield different relation sets; nothing forces them into a shared label.
- This is where "dozens of new hand positions" land: not a closed `hands: free|holding|…` enum, but phrases like `hands together`, `left hand on hip`, `left hand above head`, `hands gripping an object at {level}` — emitted sparsely, extensible forever.
- **Natural language, not snake_case.** The only consumers are the caption2 prompt and T5; `"left arm extended upward"` drops straight into the prompt as a bullet with no de-tokenizing step. (User flag: the underscore format was enum-thinking leaking into an open-ended design.)
- **Facing is a relation, not an enum**: `face turned toward camera` / `face turned away from camera` / `face in profile` — derived from ear/eye/shoulder symmetry, phrased as prose.
- `held_object` collapses into relations: `hands gripping an object at pelvis level` (hand geometry + seg2 fg density between wrists, wrist-adjacency gate). No separate `present/hand_count/height_level` block — the level is rendered against the subject's own joint levels in the phrase.

### Layer 3 — (none)

Posture and activity labels are **deliberately absent**. `standing/seated/kneeling/cartwheel/ballet/floating` are semantic classifications; the VLM names them from pixels, *grounded* by Layers 1–2. Determinations never guesses them, so it can never confidently contradict the image.

**Field → source artifact:**
- `subject.n_detections`, `detector_anomaly` — pose2 shape + DETR behavior
- `subject_extent.*` — pose2 keypoint bbox vs frame; `letterbox_bars` — pixel.npy (optional)
- `body_parts_visible` — seg2 class fractions (grouped to coarse parts) + pose2 per-part kp confidence
- `orientation.upright_deg` — pose2 neck(#69)→hip(#9/#10) axis vs image-down
- `camera.*` — **pointmap** (reserved until backfill; omitted when absent)
- `relations` — pose2 joints (wrists #41/#62, hips #9/#10, shoulders #5/#6, neck #69, fingertips 21–62) + seg2 fg density
- `content.*` — seg2 class pixel fractions

**Graceful degradation:** `camera.*` reserved for pointmap; everything else runs on pose2+seg2+pose.npy+pixel.npy **today**, matching current backfill coverage (pose2 ~17%, seg2 ~4%, pointmap 0%).

---

## 3b. Architecture decision (user-directed, supersedes the §5 open question)

**Backwards compatibility to stratum is a hard requirement.** Hence the naming pattern `pose2.npy` (new) vs. `pose.npy` (legacy) — stratum2 *adds* artifacts, never breaks stratum1 consumers.

Applied to captions:

- **`caption.txt` is kept as-is.** The existing rich gemma3:27b prose caption. Untouched. Remains the universal baseline consumed by prx-tg, the T5 pass, and the published stratum-ffhq dataset contract. No recaption of the existing 70k.
- **A NEW captioning pass is introduced**, depending on `pose2`, `seg2`, (later `pointmap`). It produces a **separate artifact** that pairs the deterministic `determinations` block (§3) with a **rich, large-vocabulary LLM caption** — the VLM verbalizes over measured truth (mood, color, light, texture, fabric, expression), grounded by and never contradicting the determinations.
- **The new pass is stratum2-only.** Because it depends on pose2/seg2, it cannot run on a stratum1-only tree. This is consistent with backwards compatibility: `caption.txt` stays the stratum1/universal path; the new artifact is the stratum2 upgrade. A stratum1 tree simply lacks the new artifact (same as it lacks `pose2.npy`).

**Implication for the §5 open question (preamble vs. separate artifact):** resolved as **separate artifact**. The geometric determinations are not a throwaway prompt preamble — they are persisted as their own artifact (consumed alongside the new rich caption at training time), AND they double as the grounding input to the new captioning pass. One extraction, two uses.

**Naming (DECIDED):** the deterministic block is **NOT** called `facts` ("a stretch"). It's its own first-class step that `caption2` depends on.
- **`determinations.json`** (pass name `determinations`) — **DECIDED.** Exact: what the pipeline *determined* about the image from measured signal. Noun form mirrors `metadata.json`.
- Rejected: `conclusions.json` (too inferential — these are measured, not reasoned), `observations.json` (reads as raw sensor readings; these are derived), `analysis.json` (generic; collides with stratum-api `analyze()`).
- T5 set: **`t52_hidden.npy` / `t52_mask.npy`** (pass name `t52`) — **DECIDED.** Mirrors the `pose2`/`seg2`/`normal2` suffix convention. Rejected: `t5_v2_*`.

**Artifact & pass chain (all additive; nothing renames/overwrites stratum1):**

| Pass | Depends on | Artifact(s) | Notes |
|---|---|---|---|
| `determinations` | pose2, seg2 (later pointmap) | `determinations.json` | deterministic, CPU, numpy-only |
| `caption2` | determinations | `caption2.txt` | rich, determinations-grounded LLM caption |
| `t52` | caption2 | `t52_hidden.npy`, `t52_mask.npy` | T5-Large embeddings of `caption2.txt` — **the new t5 set** |

stratum1 chain (`caption.txt` → `t5_hidden.npy`/`t5_mask.npy`) is untouched and remains the universal baseline.

---

## 4. Phased plan (execution order)

Execution order, user-directed: **determinations first** (so the caption2 prompt is written against *validated* measurements, not an assumed schema), **pointmap exploration next** (may extend the determinations schema with pointmap-derived fields), **then** the caption2 prompt + pass. Clean 0–6 renumbering in execution order.

### TDD mandate (applies to every phase)

**Iron Law: NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST** (per the `test-driven-development` skill). Every phase below is RED-GREEN-REFACTOR: write the failing test for the phase's acceptance criteria, watch it fail, write minimal code to pass, watch it pass, then run the full suite for regressions. Tests run against **synthetic fixtures and `tmp_path` only** — never against the live 70k dataset or the CIFS mount (the "don't clobber live data" rule). Geometry tests use **synthetic keypoint arrays** (hand-built `(308,3)` arrays with known joint positions) so expected measurements are computable by hand; no GPU, no Sapiens2 model, no real image needed in unit tests. Model-integration smoke tests (if any) are separate, marked, and skippable.

Each phase has **objective, testable gates** — a phase is done only when its gates pass. Gates are phrased as assertions a test can check, not vibes.

---

**Phase 0 — freeze the index tables (prerequisite, ~1 hr, read-only)**
Dump and freeze into `src/stratum2/config.py`: (a) Goliath-308 keypoint names/indices from the official `classes_and_palettes.py`; (b) DOME_CLASSES_29 seg names from the installed `sapiens` pkg. Rationale: both of my corrections came from assuming instead of reading these tables. Never derive anatomy from a guessed index again. Prerequisite for Phase 1 geometry.

*Gates (objective):*
- `GOLIATH_308` has exactly 308 entries; indices 0–16 body, 41 = `right_wrist`, 62 = `left_wrist`, 9/10 = `left_hip`/`right_hip`, 69 = `neck` (spot-asserted against the official table, not transcribed by hand).
- `DOME_29` has exactly 29 entries; 0 = `Background`, 22 = `Torso`, 3 = `Face_Neck`.
- A unit test asserts both tables load from config and these sentinel indices match — so a future table edit breaks loudly.
- *Note:* this phase is config/constants, a TDD "configuration file" exception in spirit — but the sentinel-index test is still required, because wrong indices are exactly the failure mode this phase exists to prevent.

**Phase 1 — `determinations` extraction pass (Lever 2 core, ~2 days)**
Build `src/stratum2/pipeline/determinations.py` (CPU, numpy-only, idempotent, per-image). Layer 1 raw measurements (subject extent, body_parts_visible, upright_deg, detector anomaly, content/skin) + Layer 2 open-set relation-phrase generator (arm/leg/hand orientations, hand relations, facing-as-relation, held-object phrase). Reserve `camera.*` for pointmap (filled in Phase 2). **Threshold-tuning loop: eyeball ~50 images *deliberately spanning the hard set* (eye/hand/knee closeups, kneeling, cartwheel, ballet, high jump, floating) to set the relation-rule thresholds and confirm measurements behave where enums would have broken.** Emit `determinations.json`.

*Gates (objective):*
- **Schema gate:** output validates against the §3 schema — required keys present, `body_parts_visible` entries each have `part`/`pixel_frac`/`kp_conf`, `relations` is a list of non-empty strings. (JSON-schema or explicit asserts.)
- **Geometry gate (synthetic fixtures):** a synthetic upright skeleton yields `upright_deg ≈ 0 ± 5`; a synthetic inverted skeleton yields `upright_deg ≈ 180 ± 5`; a synthetic horizontal skeleton yields `upright_deg ≈ 90 ± 5`. A skeleton with wrists raised above shoulders yields a `"left arm extended upward"` relation. Hand-built arrays, hand-computable expectations.
- **Crop gate:** a synthetic `body_parts_visible` with only `face` present yields a tight `subject_extent` and **no** limb relations (degenerate-crop silence, not a confident-wrong label).
- **Single-subject gate:** `pose2.shape[0] == 2` → `detector_anomaly == "extra_detections(2)"` and relations are computed from the highest-confidence box only; `pose2.shape[0] == 0` → `detector_anomaly == "no_detection"`.
- **Idempotency gate:** running the pass twice on the same fixture dir produces byte-identical `determinations.json` and does not recompute (second run is a no-op skip).
- **No-enum gate:** `relations` contains no snake_case tokens and no posture/activity labels (assert no entry matches a banned list: `standing|seated|lying|kneeling|cartwheel|ballet|…`).

**Phase 2 — pointmap exploration → extend determinations (~1–2 days, exploration)**
Explore what pointmap reliably measures for determinations — camera distance/height, body depth extent, and body-relative volumetric/depth relations. *(Informed by a separate experiment Tim is running elsewhere on pointmap-derived body-volume ratios — not part of this plan, but its findings on what pointmap can/can't measure feed this exploration.)* Outcome: extend the `determinations.json` schema's `camera.*` (and any new pointmap-derived measurement fields) with validated semantics; backfill `camera.*` into determinations where pointmap exists. Decides whether pointmap jumps the FFHQ backfill queue (currently 0%).

*Gates (objective):*
- **Convention gate (synthetic pointmap):** a synthetic pointmap with known camera-frame geometry yields `camera.height_rel_shoulder_m` and `camera.distance_m` within tolerance of hand-computed values (validates the +Y-down/+Z-forward/origin-at-camera convention against a fixture, not just the one real sample).
- **Schema-extension gate:** any new pointmap-derived field is added to the §3 schema with a definition + unit + source, and a test asserts it round-trips.
- **Exploration is TDD-exempt in the RED-GREEN sense** (it's measurement R&D), but every *conclusion* that becomes a schema field gets a synthetic-fixture test before it's accepted — no field lands on the strength of one real image.

**Phase 3 — `caption2` prompt (Lever 1, ~half day)**
Write the prompt for the new `caption2` pass against the *validated* determinations from Phases 1–2: relational structure (subject→pose/relations→depth/space→light/camera) + explicit instruction to verbalize mood, color, light quality, texture, fabric, expression — grounded by the determinations block, "never contradict, add what determinations omit." `caption.txt`'s prompt is untouched. Pilot on ~20 crawlr images; score factual agreement against pose2/seg2 ground truth. No pipeline change.

*Gates (objective):*
- **Template gate (pure function, no LLM):** the prompt-builder renders a complete prompt from a fixture `determinations.json` — asserts the determinations block is present, relations appear as bullets, and the "never contradict / add what determinations omit" instruction is present. (Tests the string assembly, not the model.)
- **No-touch gate:** `CAPTION_PROMPT` (stratum1) is byte-identical before/after — a test diffs it.
- **Pilot-agreement gate (reported, not a hard assert):** on the ~20-image pilot, report caption↔artifact factual-agreement rate (spot-check + heuristics). This is a *measured* outcome to inform Phase 4, not a pass/fail unit test — flagged as such so nobody mistakes a subjective eyeball for a gate.

**Phase 4 — `caption2` pass + pilot (Lever 2 integration, ~1 day)**
Build `src/stratum2/pipeline/caption2.py`: loads `determinations.json` (dependency), calls Ollama with the Phase 3 prompt + determinations block, writes `caption2.txt`. New pass registered in the stratum2 CLI/dependency graph (`depends on: determinations` → transitively pose2, seg2). Pilot on **500 images**; measure caption↔artifact agreement (spot-check + heuristics, e.g. "caption says arm raised" vs pose2 shoulder/wrist heights). **Decision gate before any wider caption2 rollout.** `caption.txt` is never regenerated.

*Gates (objective):*
- **Dependency gate:** running `caption2` on a fixture dir *without* `determinations.json` fails/skips with a clear dependency error (does not silently caption ungrounded). Running with it proceeds.
- **Pass-through gate (mocked Ollama):** with the Ollama HTTP call mocked, the pass (a) sends a prompt containing the determinations block, and (b) writes the model's returned text verbatim to `caption2.txt`. Tests the wiring, not the model.
- **Idempotency gate:** existing `caption2.txt` → pass is a no-op skip.
- **Isolation gate:** `caption.txt` and `t5_hidden/t5_mask.npy` in the fixture dir are untouched (byte-identical) after the pass.
- **500-image pilot → decision gate (human review, reported):** caption↔artifact agreement measured and *reviewed by Tim* before any wider rollout. This is the explicit go/no-go; it is a reviewed measurement, not an automated assert.

**Phase 5 — `t52` pass (the new t5 set, ~half day)**
Build `src/stratum2/pipeline/t52.py`: loads `caption2.txt` (dependency), runs T5-Large (same model as stratum1 `t5`), writes `t52_hidden.npy` + `t52_mask.npy`. Registered as `depends on: caption2`. Idempotent. This is the training-time text-conditioning tensor for the stratum2 chain.

*Gates (objective):*
- **Dependency gate:** missing `caption2.txt` → skip/fail with clear dependency error.
- **Shape/dtype gate (synthetic or tiny T5):** `t52_hidden.npy` is `(512, 1024)` float16, `t52_mask.npy` is `(512,)` uint8 — matching the stratum1 `t5_hidden/t5_mask` contract exactly. (Runs against the real T5-Large on CPU/tiny input, or a shape-mocked encoder if load is too heavy for CI.)
- **Content-sensitivity gate:** two different fixture captions produce *different* `t52_hidden` arrays (guards against a pass that writes a constant/empty tensor).
- **Idempotency + isolation gates:** existing `t52_*` → no-op; `t5_hidden/t5_mask` untouched.

**Phase 6 — (later, gated on Phase 4) fine-tune**
QLoRA a local 7B-class VLM on the best `caption2` outputs (DALL-E 3 recipe: pretrain grounding + few-hundred curated style examples). Uses unsloth/axolotl/TRL. Local-only per content constraint.

*Gates:* deferred — defined when Phase 6 is scoped (needs its own data-quality and eval gates; out of scope for this plan's TDD mandate until then).

---

## 5. Risks, tradeoffs, open questions

- **caption2 rollout cost.** `caption.txt` (70k) is untouched, so no *re*caption risk — but `caption2` is a fresh Ollama pass over whatever subset we enable it on, subject to the same CIFS I/O bottleneck as pose2/seg2. Hence the 500-image pilot gate before committing to a full run.
- **Threshold subjectivity (relation rules).** Layer 2 relation phrases still need thresholds (arm-raised angle, hand-together distance, facing symmetry). The tuning set must *deliberately span the hard cases* (eye/hand/knee closeups, kneeling, cartwheel, ballet, high jump, floating) — that's the real cost of Phase 2, not the code. The open-set design means a missed relation is silence (caption2 fills it), not a confident-wrong label.
- **Prop detection is heuristic.** seg2 can't see held props; the wrist-adjacency + fg-density gate behind `hands gripping an object at {level}` will have false positives (e.g., hands clasped with nothing held). Emitting it as a phrase with the grip qualifier is honest; the VLM still names/omits the object from pixels.
- **Extreme-crop degenerate cases.** On an eye/hand/knee closeup, most body keypoints are absent or collapsed — `upright_deg` and limb relations may be underdetermined. Layer 1 still emits `body_parts_visible` + `subject_extent` (the useful signal there); relations degrade to silence. Acceptable: the crop *is* the description.
- **Black-bar framing** needs pixel.npy, adding a pixel read to an otherwise pose/seg pass. Optional; decide if framing facts must be complete.
- **Two captions to keep in sync?** `caption.txt` and `caption2.txt` describe the same image with different grounding. They're intentionally *not* merged (backwards compat), but downstream consumers must pick one (or fuse at T5 time). Decision deferred to the prx-tg integration; not a Phase 1–5 blocker.
- **~~Open question — preamble vs. separate artifact~~** → **RESOLVED (§3b):** separate artifact. `determinations.json` persisted + `caption2.txt` as the new rich caption + `t52_hidden/t52_mask.npy` as its T5 set; `caption.txt`/`t5_*` untouched.

## 6. Next step

The reviewable deliverables are the **`determinations.json` schema (§3)** and the **architecture decision (§3b)** — every field, its source artifact, its enum vocabulary, its confidence rule, its missing-pass fallback, and the three-artifact additive chain (`determinations.json` → `caption2.txt` → `t52_hidden/t52_mask.npy`), with `caption.txt`/`t5_*` untouched. Naming is **decided**: `determinations` (pass + `.json`), `t52` (pass + `t52_hidden/t52_mask.npy`). On approval, Phase 0 (freeze index tables) is the first concrete, read-only step.

**Do I have your approval to (a) finalize the schema + architecture as written, and (b) run Phase 0 (extract + freeze the Goliath-308 and DOME-29 index tables into config)?** Phase 0 is read-only against the checkpoints/package; no pipeline behavior changes.
