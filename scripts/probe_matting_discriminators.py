"""CPU-only matting discriminators probe for arm #59 band calibration."""
import json
import os

import numpy as np
from scipy import ndimage

from research_harness.matting_alpha import _DOME_INDEX, _SEMI, _OPAQUE, _SUBJECT

items = json.load(
    open('/mnt/nas-ai-models/research/stratum/stage-b-pointmap-depth-v1/stage-b-plan.json')
)['pilot_manifest']['items']
base = "/mnt/nas-ai-models/training-data/crawlr/stratum"
HAIR = _DOME_INDEX["Hair"]


def crispness(alpha, subject):
    interior = ndimage.binary_erosion(subject, structure=np.ones((3, 3)))
    ring = subject & ~interior
    if ring.sum() < 50:
        return None
    ay, ax = np.gradient(alpha.astype(np.float64))
    mag = np.sqrt(ax * ax + ay * ay)
    return float(np.percentile(mag[ring], 50))


def hair_soft_share(alpha, seg2):
    semi = (alpha >= _SEMI) & (alpha < _OPAQUE)
    hair = seg2 == HAIR
    hs = int((semi & hair).sum())
    ss = int((semi & ~hair).sum())
    if (hs + ss) == 0:
        return None
    return float(hs) / (hs + ss)


def hair_detail(alpha, seg2):
    interior = ndimage.binary_erosion(seg2 == HAIR, structure=np.ones((3, 3)))
    band = interior & (alpha >= _SEMI) & (alpha < _OPAQUE)
    if band.sum() < 50:
        return None
    ay, ax = np.gradient(alpha.astype(np.float64))
    mag = np.sqrt(ax * ax + ay * ay)
    return float(np.mean(mag[band]))


def run():
    for it in items:
        iid = it['image_id']
        d = os.path.join(base, iid)
        alpha = np.load(os.path.join(d, 'matting.npy')).astype(np.float64)
        seg2 = np.load(os.path.join(d, 'seg2.npy'))
        subject = alpha >= _SUBJECT
        c = crispness(alpha, subject)
        hs = hair_soft_share(alpha, seg2)
        hd = hair_detail(alpha, seg2)
        cov = float((alpha >= _OPAQUE).sum()) / alpha.size
        print(
            f"{iid[:16]} cov={cov:.3f} crisp={'-' if c is None else round(c,3)} "
            f"hairsoft={'-' if hs is None else round(hs,3)} "
            f"hairdetail={'-' if hd is None else round(hd,3)}"
        )


if __name__ == "__main__":
    run()
