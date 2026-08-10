"""Deterministic eye-color / iris-hue measurement from `pose2` + source RGB.

Arm #80. NEW deterministic evidence part (no new model). Reads the existing
`pose2.npy` (GOLIATH-308 iris/pupil center + border keypoints, [x, y, conf])
and the already-decoded source RGB (SHA-bound via source_sha256) and emits a
coarse scale-invariant eye-color band:

- eye_color_band: brown / dark / blue / green-hazel / gray (or abstain).

The band is the only verbalized fact; raw sampled RGB/HSV statistics stay in
the machine-readable `evidence_payload`.

Measurement:
- For each eye with reliable iris-center, iris-border, and pupil-border
  keypoints, derive the iris radius (median distance from iris center to iris
  border) and pupil radius (median distance from iris center to pupil border).
- Sample the ANNULUS between pupil radius and iris radius (the iris body,
  avoiding the dark pupil and the specular glare hotspot), clamped to the
  source frame.
- A robust (trimmed/p90) mean of the annulus RGB -> HSV. Classify onto the
  frozen closed hue/saturation/value set.

Abstention: abstains when (a) neither eye's iris keypoints are reliable
(conf < 0.5), (b) the annulus is degenerate/tiny (fewer than a pixel floor),
(c) the specimen is too dark to resolve hue (low value AND low saturation —
heavy shadow), or (d) the coordinate falls outside the decoded frame. Never
fabricate an eye color; detector disagreement is a quality anomaly, never
caption content. CPU-only, in-memory, no corpus write, no new model.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from stratum2.config import GOLIATH_308

_GOLIATH_INDEX = {name: i for i, name in enumerate(GOLIATH_308)}

CORE_MIN_CONF = 0.5

# Annulus pixel floor: fewer pixels => degenerate, abstain.
MIN_ANNULUS_PX = 8

# Number of sampling rays around the iris annulus.
N_RAYS = 12
# Radial steps within the annulus.
N_STEPS = 3


class EyeColorError(RuntimeError):
    pass


def validate_pose2_array(pose: np.ndarray) -> None:
    if not isinstance(pose, np.ndarray):
        raise EyeColorError("pose2 must be a numpy array")
    if pose.shape == (1, 308, 3):
        pose = pose[0]
    if pose.shape != (308, 3):
        raise EyeColorError(
            f"pose2 must be shape (308,3) or (1,308,3), got {pose.shape}"
        )


def _normalize_pose(pose: np.ndarray) -> np.ndarray:
    validate_pose2_array(pose)
    if pose.shape == (1, 308, 3):
        return pose[0]
    return pose


def _pt(pose: np.ndarray, name: str) -> tuple[float, float] | None:
    idx = _GOLIATH_INDEX[name]
    x, y, conf = float(pose[idx, 0]), float(pose[idx, 1]), float(pose[idx, 2])
    if x < 0 or y < 0 or conf < CORE_MIN_CONF:
        return None
    return (x, y)


def _annulus_rgb(
    rgb: np.ndarray,
    center: tuple[float, float],
    iris_border: list[tuple[float, float]],
    pupil_border: list[tuple[float, float]],
) -> np.ndarray | None:
    """Sample the iris annulus (between pupil + iris borders) -> RGB pixels."""
    cx, cy = center
    iris_rad = float(np.median([math.hypot(bx - cx, by - cy) for bx, by in iris_border]))
    pupil_rad = float(np.median([math.hypot(bx - cx, by - cy) for bx, by in pupil_border]))
    if not (math.isfinite(iris_rad) and math.isfinite(pupil_rad)):
        return None
    if iris_rad < 2.0:
        return None  # tiny iris, cannot sample an annulus reliably
    pupil_rad = max(0.0, min(pupil_rad, iris_rad * 0.85))
    h, w = rgb.shape[:2]
    r0 = (pupil_rad + iris_rad) / 2.0 if pupil_rad >= iris_rad * 0.5 else pupil_rad + (iris_rad - pupil_rad) * 0.35
    r1 = iris_rad * 0.95
    samples: list[tuple[int, int]] = []
    for step in range(N_STEPS):
        r = r0 + (r1 - r0) * ((step + 0.5) / N_STEPS)
        for ray in range(N_RAYS):
            angle = 2.0 * math.pi * ray / N_RAYS
            sx = int(round(cx + r * math.cos(angle)))
            sy = int(round(cy + r * math.sin(angle)))
            if 0 <= sx < w and 0 <= sy < h:
                samples.append((sy, sx))
    if len(samples) < MIN_ANNULUS_PX:
        return None
    rows, cols = zip(*samples)
    return rgb[np.array(rows), np.array(cols)]


def _classify(rgb_pixels: np.ndarray) -> tuple[str, dict[str, Any]]:
    """Classify sampled RGB onto the frozen closed eye-color set."""
    px = rgb_pixels.astype(float)
    # Trim the hottest 5% (specular glare) + darkest 5% before aggregating.
    luma = px.mean(axis=1)
    lo = np.percentile(luma, 5)
    hi = np.percentile(luma, 95)
    keep = (luma >= lo) & (luma <= hi)
    px = px[keep] if keep.any() else px
    mean = px.mean(axis=0)
    mx, mn = mean.max(), mean.min()
    value = float(mean.max() / 255.0)
    sat = float((mx - mn) / mx) if mx > 0 else 0.0
    # Hue from the max channel convention.
    r_, g_, b_ = [float(c) / 255.0 for c in mean]
    mx2, mn2 = max(r_, g_, b_), min(r_, g_, b_)
    delta = mx2 - mn2
    hue = 0.0
    if delta > 0:
        if mx2 == r_:
            hue = 60.0 * (((g_ - b_) / delta) % 6)
        elif mx2 == g_:
            hue = 60.0 * (((b_ - r_) / delta) + 2)
        else:
            hue = 60.0 * (((r_ - g_) / delta) + 4)

    stats = {
        "mean_rgb": [round(c, 1) for c in mean],
        "hue_deg": round(hue, 1),
        "saturation": round(sat, 4),
        "value": round(value, 4),
    }
    # Sequence — re-cut 2026-08-08 from the frozen-cohort probe: the first
    # scheme's broad `value < 0.30 or sat < 0.18 -> dark` rule mislabeled
    # light low-saturation eyes (a hue-250.7/val-0.45 eye and a val-0.59/
    # sat-0.08 eye both fell into dark). The genuinely-discriminating axes on
    # this cohort are VALUE (shadow-dark brown/black vs well-lit brown) and
    # HUE (warm brown vs cool blue/green). A low-sat eye with moderate value
    # reads gray/light-brown, not dark.
    if value < 0.24 and sat < 0.35:
        return "dark", stats  # shadowed brown/black — cannot resolve a chromatic hue
    if hue < 45 or hue > 340:
        # warm hue range (red-brown) — brown when lit enough to have chroma
        return "brown", stats
    if 170 <= hue <= 300 and value >= 0.30 and sat >= 0.12:
        return "blue", stats
    if 60 <= hue < 160 and value >= 0.28 and sat >= 0.15:
        return "green-hazel", stats
    if value >= 0.32:
        return "gray", stats  # light, low-chroma eye
    return "dark", stats


def compute_eye_color(pose2: np.ndarray, image_rgb: np.ndarray) -> dict[str, Any]:
    """Compute the deterministic eye-color band with honest abstention.

    Args:
        pose2: (308,3) or (1,308,3) GOLIATH-308 keypoints.
        image_rgb: (H, W, 3) uint8 RGB source, aligned to the pose frame.

    Returns a dict with scale-invariant facts only:
    - abstained / abstention_reason
    - eye_color_band (brown / dark / blue / green-hazel / gray)
    - per-eye samples + aggregate HSV stats (payload)
    """
    pose = _normalize_pose(pose2)
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise EyeColorError("image_rgb must be an (H, W, 3) numpy array")

    out: dict[str, Any] = {
        "abstained": False,
        "abstention_reason": None,
        "eye_color_band": None,
        "sample_count": 0,
        "per_eye": {},
        "mean_rgb": None,
        "hue_deg": None,
        "saturation": None,
        "value": None,
    }
    pooled: np.ndarray | None = None
    eyes = []
    for side in ("l", "r"):
        center = _pt(pose, f"{side}_center_of_iris") or _pt(pose, f"{side}_center_of_pupil")
        if center is None:
            continue
        iris_border = [
            p for p in (_pt(pose, f"{side}_border_of_iris_3"),
                        _pt(pose, f"{side}_border_of_iris_6"),
                        _pt(pose, f"{side}_border_of_iris_9"),
                        _pt(pose, f"{side}_border_of_iris_12")) if p is not None
        ]
        pupil_border = [
            p for p in (_pt(pose, f"{side}_border_of_pupil_3"),
                        _pt(pose, f"{side}_border_of_pupil_6"),
                        _pt(pose, f"{side}_border_of_pupil_9"),
                        _pt(pose, f"{side}_border_of_pupil_12")) if p is not None
        ]
        if len(iris_border) < 2:
            continue
        samples = _annulus_rgb(image_rgb, center, iris_border, pupil_border)
        if samples is None or len(samples) < MIN_ANNULUS_PX:
            continue
        eyes.append((side, samples))
        pooled = samples if pooled is None else np.concatenate([pooled, samples])

    if not eyes:
        out.update({
            "abstained": True,
            "abstention_reason": "no reliable iris/pupil keypoints or usable iris annulus (eyes closed, cropped, or heavily occluded)",
        })
        return out
    out["sample_count"] = int(len(pooled))
    band, stats = _classify(pooled)
    out["eye_color_band"] = band
    out["mean_rgb"] = stats["mean_rgb"]
    out["hue_deg"] = stats["hue_deg"]
    out["saturation"] = stats["saturation"]
    out["value"] = stats["value"]
    out["per_eye"] = {side: len(s) for side, s in eyes}
    return out


def render_eye_color(config: Mapping[str, Any]) -> list[str]:
    """Scale-invariant eye-color claim for the dossier (arm #80).

    Verbalizes ONLY the coarse closed-set band. Raw RGB/HSV stats stay in the
    machine-readable payload.
    """
    if not config:
        # Dimension not measured for this item (e.g. non-eye-color runs) —
        # emit no claim, never a fabricated eye-color statement.
        return []
    if config.get("abstained"):
        reason = config.get("abstention_reason") or "eye color not measurable"
        return [f"eye-color: abstain ({reason})"]
    band = config.get("eye_color_band")
    if band in ("brown", "dark", "blue", "green-hazel", "gray"):
        return [f"eye-color: eyes are {band}"]
    return []
