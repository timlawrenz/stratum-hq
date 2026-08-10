"""CPU probe: candidate DISCRIMINATORS for the arm-#84 face-visibility axis.

The on-paper occlusion-overlap measure (Face_Neck & occluding-class overlap)
is DEGENERATE on hard-label seg2: one class per pixel means Face_Neck can
NEVER overlap Hair/Hand/Arm at a pixel, so occlusion_fraction is 0.000 for
23/23 measured items (max_share = 1.00 -> band-degeneracy rule fires).
Per the re-probe-discriminators rule (arm #34/#35/#59/#75/#82/#83), this
probe measures genuine ALTERNATIVE scale-invariant signals on the frozen
cohort so the arm can be honestly re-cut or honestly silenced:

- face_share_of_local_head: Face_Neck px / (Face_Neck+Hair) px inside the
  Face_Neck-bbox dilated to the local head window. A face covered by hair /
  bangs has a shrunken Face_Neck -> smaller share.
- face_share_with_occluders: Face_Neck / (Face_Neck+Hair+Hand+Arm) in the
  same local window.
- face_bbox_aspect: Face_Neck bbox height/width (only covers part of the face
  when a side strand covers the cheek -> narrower/shorter bbox).

Read-only, no GPU claim, no corpus write, no new model.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path("/home/tim/source/activity/stratum-hq-stage-b-experiment")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from stratum2.config import DOME_29

_FACE = DOME_29.index("Face_Neck")
_HAIR = DOME_29.index("Hair")
_OCC = tuple(DOME_29.index(n) for n in (
    "Left_Hand", "Right_Hand", "Left_Upper_Arm", "Right_Upper_Arm",
    "Left_Lower_Arm", "Right_Lower_Arm",
))

MANIFEST = "/mnt/nas-ai-models/research/stratum/first-500-coverage-balanced-candidate-manifest-v1.json"
DERIVED = "/mnt/nas-ai-models/training-data/crawlr/stratum"

MIN_PX = 200
DILATE = 20  # local head window = Face_Neck bbox grown by this margin


def _window(mask: np.ndarray, margin: int) -> tuple[int, int, int, int] | None:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    h, w = mask.shape
    return (max(0, r0 - margin), min(h, r1 + margin),
            max(0, c0 - margin), min(w, c1 + margin))


def main() -> int:
    manifest = json.loads(Path(MANIFEST).read_text())
    items = manifest["items"]
    rows = []
    for item in items:
        image_id = item["image_id"]
        segp = Path(DERIVED) / image_id / "seg2.npy"
        try:
            seg2 = np.load(segp, allow_pickle=False)
        except FileNotFoundError:
            rows.append({"image_id": image_id, "face_present": False})
            continue
        face = seg2 == _FACE
        face_px = int(face.sum())
        if face_px < MIN_PX:
            rows.append({"image_id": image_id, "face_present": False})
            continue
        win = _window(face, DILATE)
        if win is None:
            rows.append({"image_id": image_id, "face_present": False})
            continue
        r0, r1, c0, c1 = win
        local = seg2[r0:r1, c0:c1]
        hair_px = int((local == _HAIR).sum())
        occ_px = int(np.isin(local, list(_OCC)).sum())
        face_local = int((local == _FACE).sum())
        denom_head = face_local + hair_px
        denom_all = face_local + hair_px + occ_px
        rows.append({
            "image_id": image_id,
            "face_present": True,
            "face_px": face_px,
            "hair_px_local": hair_px,
            "occ_px_local": occ_px,
            "face_share_of_local_head": round(face_local / denom_head, 4) if denom_head > 0 else None,
            "face_share_with_occluders": round(face_local / denom_all, 4) if denom_all > 0 else None,
        })

    det = [r for r in rows if r.get("face_present")]
    print(f"face present: {len(det)}/{len(rows)}\n")
    for r in det:
        print(
            f"{r['image_id'][:12]}  share_head={r['face_share_of_local_head']}  "
            f"share_all={r['face_share_with_occluders']}  "
            f"hair_px={r['hair_px_local']}  occ_px={r['occ_px_local']}"
        )

    print("\n=== distribution (measured) ===")
    for ax in ("face_share_of_local_head", "face_share_with_occluders"):
        vals = sorted(r.get(ax) for r in det if r.get(ax) is not None)
        if vals:
            q = statistics.quantiles(vals, n=4)
            print(f"{ax}: n={len(vals)} min={vals[0]:.3f} p25={q[0]:.3f} "
                  f"median={statistics.median(vals):.3f} p75={q[2]:.3f} max={vals[-1]:.3f}")

    Path("/mnt/nas-ai-models/research/stratum/face-visibility-discriminators-probe.json").write_text(
        json.dumps({"rows": rows}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
