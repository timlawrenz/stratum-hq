# Arm #73 apparent-age round-trip — NEW MODEL CLASS (2026-08-07, in flight)

**Registry:** apparent-age → active (sole `research:active` arm, selection 15
via EXPLOIT after image-focus #75 validated cycle 14); affordance-contact #76
→ proposal (next explore slot). This is the arm being researched THIS cycle.

## What this arm is
NEW-MODEL-CLASS evidence specialist: open-weight **MiVOLO-V2** age+gender
transformer (Apache-2.0, arXiv 2307.04616 / 2403.02302, face+body multi-input)
run on owned hardware over the frozen 24-item cohort. Emits a coarse
SCALE-INVARIANT apparent-age band only:
- late-teens-to-early-twenties / early-twenties / mid-twenties /
  late-twenties-or-older (4 bands, re-calibrated 2026-08-07);
the raw floating age estimate + gender probe stay in `evidence_payload` JSON
and are NEVER prose claims (measurement-semantics directive).

## Model asset
- `iitolstykh/mivolo_v2` open-weight HF transformers remote-code model
  (`MiVOLOForImageClassification`, face+body, `model.safetensors`), downloaded
  to `/mnt/nas-ai-models/research/stratum/models/apparent-age/`
  (115 MB), model.safetensors sha256
  `96efb47051c038ebeec74b73b4253c5fd000433e5afcab7deee0bd8f3fa7bf18`.
- Vendored minimal `mivolo_src/mivolo` package (create_timm_model,
  mivolo_model, cross_bottleneck_attn, data/misc — the remote code only needs
  four modules; `__init__` files are empty so no ultralytics/yolo drag).

## Capability probe (qualification gate step 2) — PASS
Synthetic non-sensitive image: age 26.173 reproduced identically (diff 0.0),
gender `female` p=0.733, MiVOLO-V2 output bounded to [0,122] by construction.

## Infrastructure learnings (each cost a real run/design error)
1. **HF transformers remote-code imports the `mivolo` pip package** — the pip
   build failed on `pkg_resources` (setuptools build-env) under Python 3.14.
   Rather than fight pip, VENDOR the four needed modules and add `mivolo_src`
   to `sys.path` before `from_pretrained(trust_remote_code=True)`. The repo's
   `mivolo/model/__init__.py` and `mivolo/__init__.py` are EMPTY, so the
   minimal subset imports cleanly with only timm/torch/scipy/cv2.
2. **timm 1.0.27 API drift (two renames/collisions)** — (a)
   `remap_checkpoint` was removed from `timm.models._helpers` →
   `remap_state_dict`; shim in create_timm_model.py. (b) `split_model_name_tag`
   moved from `timm.models._pretrained` to `timm.models._factory`; import from
   the new home. (c) **POSITIONAL super().__init__() mis-binds**: timm 1.0.27
   added `pos_drop_rate` to `VOLO.__init__`, so MiVOLO's all-positional call
   shifted `norm_layer` to receive the `post_layers` tuple →
   `TypeError: 'tuple' object is not callable` at `self.norm1 = norm_layer(...)`.
   Fix: pass every VOLO arg as a KEYWORD. This is the same legacy-timm-code-on-
   modern-timm class of problem as arm #60's MediaPipe (1.0.0 dropped
   `solutions`), arm #61's GroundingDINO `threshold`/`text_labels`, arm #69's
   `pooler_output`: newer versions break old pinned pipelines.
3. **The 4090 is local and the model is small** (115 MB) — the CPU probe from
   the harness venv resolves the SAME staged NAS assets the scheduler run
   uses, so capability + band-calibration probes are representative.

## Band calibration (probe 24/24 measured, 0 abstained) — DEGENERATE first pass
The first 3-band scheme (teens/twenties/thirties) was NOT discriminating: on
this homogeneous portrait cohort 21/24 items fell in "twenties" (87.5% —
breaks the no-band-≥75% rule). Measured ages cluster 24-29 (median 26.2,
range 19.8-32.9). Honest re-probe (band-degeneracy rule arm #34/#35/#58/#59/
#60): cut the 4 bands at the measured distribution gaps →
2/6/12/4 (2 late-teens-to-early-twenties, 6 early-twenties, 12 mid-twenties,
4 late-twenties-to-thirties), max share 50.0%. Via: 23/24 seg2_face_crop,
1/24 full_frame fallback. 0 abstentions (portrait cohort, faces present).
A deliberately-honest note: the group skews genuinely young-mid-20s, so the
bands express the cohort rather than the full human range — that is the honest
calibration, NOT a forced spread. (If the axis were truly homogeneous we would
silence it payload-only per #74; here it discriminates 4 ways cleanly.)

## Evidence-kind surface (full touchpoint checklist — follows arm #34/#60)
- New module `src/research_harness/apparent_age.py`: `compute_apparent_age`
  (seg2 + RGB, `--model_dir` injectable), union face-crop policy (seg2
  Face_Neck crop margin-1× + full-frame fallback) + subject body crop from
  the seg2 != 0 union; `_age_band` (4 calibrated bands); `render_apparent_age`
  (no-claim when not measured). Lazy `_MiVOLORuntime` singleton.
- `stage_b.py`: import; `_apparent_age_evidence()` (binds module SHA + model
  SHA); `_serialize_apparent_age()`; `_EVIDENCE_INPUT_NAMES["apparent-age"] =
  ("seg2.npy",)`; allowed tuple; `elif evidence_kind == "apparent-age"`
  branch (`context-raw-apparent-age`); `_validate_frozen_execution_plan`
  rebuild kind; `_load_selected_item` gated `compute_apparent_age`
  (`include_apparent_age` flag, only the apparent-age run pays model cost);
  `_render_condition` branch.
- `dossier.py`: `DIMENSION_EVIDENCE_IDS` + `render_apparent_age` +
  `_apparent_age_payload` + `build_evidence_payload`/`assemble_dossier` params
  + factories row. **No-claim pattern**: `render_apparent_age({})` → [] so the
  context4k `dossier_evidence_ids` test stays green (the #74/#75 pattern) —
  apparent-age only claims when measured.

## Tests
`tests/test_apparent_age.py` (10 tests): `_age_band` four-way + the 75%-share
invariant replayed over the probe's 24 measured ages; validators; crop bbox;
render band / abstain / not-measured-no-claim; compute abstains on model
failure + implausible age (monkeypatched `_infer_age` so no GPU/model needed);
mismatched shapes. Plus the existing roundtrip context4k tests (must stay
green).

## Round-trip execution (in flight 2026-08-07)
- Freeze: `scripts/freeze_apparent_age_manifest.py` → plan
  `stage-b-first500-apparent-age-v1`, manifest on 4090, 96 records, gemma3:27b
  digest a418f5838eaf, identical arm-#4 settings; `validate-gpu-manifest` valid.
- Generation: `stage_b_launcher --request` then `--poll-and-launch` (running).
- Review: pre-queue `stage_b_review_launcher --request --job-id
  stratum-stage-b-adversarial-review-apparent-age-v1` (DISTINCT from the gen
  job id — arm #75 pitfall), then the wrapper with INLINE PYTHONPATH=src.
- Tick: `autonomous-tick --review-dir-from <marker> --write` → verdict.

## Pitfalls to remember (do not regress)
- VENDOR the mivolo package; don't pip-install it (pkg_resources build fail).
- timm 1.0.27: keyword-args to VOLO super(), split_model_name_tag in _factory,
  remap_state_dict.
- Band thresholds MUST be cut from the measured cohort distribution, not set on
  paper — the 3-band paper scheme was 87.5% degenerate (21/24 twenties).
- Keep the raw age float payload-only; band only in prose.
- The 4090's resident Ollama model can consume VRAM after a killed run —
  `curl .../api/ps` to check, evict via keep_alive:0.
