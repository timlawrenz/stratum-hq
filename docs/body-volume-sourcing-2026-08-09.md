# Body-volume #96 — open-world model sourcing verification (2026-08-09)

Status: **[SOURCING VERIFIED — NOT RUN]** (a measurement of candidate-model availability
on owned hardware; NOT a qualification PASS and NOT a round-trip verdict).

Prepared by the autonomous strategist (cron) while arm #97 garment-type is GPU-queued
behind the user's manual gaming block (4090 busy until ~2026-08-10 13:24Z). Builds on
the sourcing pre-scan comment on issue #96 (2026-08-09T00:2xZ).

## Candidate per registry declaration (arm_issue #96)

- **HMR2.0 / 4D-Humans** (`shubham-goel/4D-Humans`, arXiv "Humans in 4D", ICCV 2023).
  Whole-body SMPL mesh regression from a single image: ViTDet person detection +
  transformerized HMR2 regressor. Scale-invariant normalized volume/bulk band can be
  derived deterministically from the SMPL mesh vertices (e.g. convex-hull/PCA bulk),
  no renderer needed.
- **PyMAF-X** (registered alternative): whole-body SMPL-X regression; requires
  mmhuman3d/mmcv stack — heavier install, kept as fallback only.

## License verification (2026-08-09, fetched live)

| Asset | License | Verified |
|---|---|---|
| 4D-Humans code (`LICENSE.md` @ main) | **MIT** (UC Regents, Shubham Goel) | ✅ fetched raw, text read |
| HMR2.0 checkpoint / `hmr2_data.tar.gz` | Distributed from the repo's own host (`https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz`), NOT HF-gated; no click-through | ✅ URL pinned in `hmr2/models/__init__.py` `download_models()` |
| SMPL body asset (`SMPL_NEUTRAL.pkl`) | SMPL is licensed for **non-commercial scientific research** (MPI/ps-iz; ~mpg.de). The qualification gate requires license verification for research use before freezing — this program's frozen-cohort measurement round-trips are non-commercial research, consistent. | ⏳ content check pending: the official tarball may bundle `data/smpl/SMPL_NEUTRAL.pkl` (that path is `check_smpl_exists`'s first candidate) — if so, NO registration click-through is needed and the 00:2xZ caveat is downgraded from "owner decision" to "record license note at freeze" |

## Local-asset recon (owned hardware)

- `SMPL_NEUTRAL.pkl` absent on local + Strix (pre-scan, 00:2xZ) — the question is
  whether `hmr2_data.tar.gz` provides it (checking as part of provisioning).
- Strix (owned, ssh:max395): system Python 3.14; working `comfyui-gfx1151-venv`
  (Python 3.12.12, torch 2.12.0a0+rocm7.13.0); 311 GB free disk.

## Capability-probe plan (runs when provisioning completes; next cron ticks)

1. **Host**: Strix, isolated venv `~/4dhumans-probe-venv` (Python 3.12), **CPU torch**
   (no scheduler GPU claim needed for a capability measurement; zero interference with
   the 10GB evergreen labeling reservation). A scheduler-managed GPU manifest is only
   required if/when the arm's round-trip is frozen.
2. **Inputs**: frozen-cohort source RGB (SHA-bound via `source_sha256`) cropped via
   the EXISTING deterministic seg2 Subject mask / pose2 bbox (no detectron2 dependency
   in the probe; ViTDet becomes an optional production-path decision at arm freeze).
3. **Measurements**: coverage (items with plausible mesh / N), per-item runtime,
   mesh sanity (vertex bounds, mesh-face count, SMPL params finite), and an initial
   normalized-volume scatter (band degeneracy check vs the 75% rule is a LATER
   calibration step — the probe only establishes capability + coverage floor).
4. **Pre-registered coverage floor (draft)**: ≥ 18/24 frozen items (75%) with a
   plausible whole-body mesh before the arm can proceed to band calibration;
   below that → capability-fail → mark-blocked/needs-human with the measured numbers.
5. **Abstention policy** (declared in registry): abstain on mesh-regression failure,
   degenerate mesh, or model/input failure — never fabricate a body-volume class.

## Provisioning state (2026-08-09 ~18:24Z)

- Detached job on Strix: `nohup bash /home/tim/4dhumans-probe-setup.sh` →
  log `/home/tim/4dhumans-probe-setup.log`; marker `PROBE_ENV_READY` at the end.
- Steps: venv create → CPU torch/torchvision → smplx/pytorch-lightning/yacs/
  scikit-image/einops/timm/dill/pandas/opencv-headless/pyrender/pyopengl →
  `git clone --depth 1` 4D-Humans → `hmr2_data.tar.gz` download + sha256 pin.
- Pitfall noted: `hmr2.models` imports `hmr2.utils` which imports pyrender at module
  level even when `init_renderer=False`; if headless GL libs are missing at import
  time, the probe adapter stubs the three renderer classes (never instantiated in a
  measurement-only probe) — documented, additive, no corpus write.

## Routing

- Arm #96 stays `proposal`; **no registry change** in this cycle.
- Next decision point: when the selector reaches body-volume (after hair-texture #94),
  the capability-probe result determines activate-vs-mark-blocked. If the probe covers
  ≥18/24 items, proceed to band calibration; else open `research:needs-human` with the
  measured coverage and the exact decision needed (model swap vs arm falsification).