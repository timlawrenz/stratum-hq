"""Verify recalibrated matting bands distribute across the frozen cohort."""
import json
import os

import numpy as np

from research_harness.matting_alpha import compute_matting_alpha, MattingAlphaError

items = json.load(
    open('/mnt/nas-ai-models/research/stratum/stage-b-pointmap-depth-v1/stage-b-plan.json')
)['pilot_manifest']['items']
base = "/mnt/nas-ai-models/training-data/crawlr/stratum"

rows = []
for it in items:
    iid = it['image_id']
    d = os.path.join(base, iid)
    alpha = np.load(os.path.join(d, 'matting.npy'))
    seg2 = np.load(os.path.join(d, 'seg2.npy'))
    try:
        r = compute_matting_alpha(alpha, seg2)
    except MattingAlphaError as exc:  # noqa: PERF203
        print(iid, "ERROR", exc)
        rows.append(None)
        continue
    if r.get('abstained'):
        print(iid, "ABSTAIN", (r.get('abstention_reason') or '')[:70])
        rows.append(None)
        continue
    rows.append(r)
    print(
        f"{iid[:16]} cov={r['coverage_ratio']:.3f} {r['coverage_band']:>10} "
        f"crisp={r['boundary_crispness']} {r['boundary_crisp_band']:>8} "
        f"hairshare={r['hair_soft_share']} {r['soft_edge_band']:>12} "
        f"sil_share={r['silhouette_closedness']} open={r['border_open_fraction']}"
    )

meas = [r for r in rows if r is not None]
from collections import Counter

for k in ('coverage_band', 'boundary_crisp_band', 'soft_edge_band'):
    c = Counter(r[k] for r in meas)
    shares = {b: round(v / len(meas), 3) for b, v in c.items()}
    print(f"\n{k}: {dict(c)}  max_share={max(shares.values()) if shares else 0:.3f}")
print(f"\ntotal measured {len(meas)}/24")
