"""Deterministic lighting measurements from `normal2` + `seg2` + source pixels.

Arm #33. Reads the existing camera-space normal map `normal2.npy` (H, W, 3)
float16/float32 unit surface normals), the `seg2.npy` DOME-29 semantic labels,
and the source RGB pixels, and emits continuous, scale-invariant lighting
statistics:

- subject_present: is there a foreground subject (seg2 > 0)?
- lighting_measurable: do the foreground normals clear a raw-pixel floor and a
  validity gate so that statistics are stable?
- luminance level: quantized mean foreground luma band (low-key / moderate /
  bright) plus continuous `mean_luma` / `median_luma`.
- dynamic range: (p98-p2) robust luma span over the foreground, quantized to a
  low / medium / high `dynamic_range_band`, plus continuous `dynamic_range`.
- shadow fraction: fraction of foreground pixels below a deep-shadow luma
  threshold.
- backlight proxy: ratio of background mean luma to foreground mean luma
  (subject darker than its surround => rim/backlit); verbalized as a coarse
  band with the continuous ratio kept in the machine-readable payload.
- dominant light direction: a least-squares Lambertian fit ``L = argmin ||lum -
  N.L||`` over validated foreground normals, normalized; verbalized as a coarse
  direction name (frontal / left / right / backlit). The signed cosine vector
  and the fit residual stay in the machine-readable payload (camera-frame).

Every measurement honors the exactly-one-subject invariant and abstains (emits
None/False with a surfaced reason) when the supporting foreground or normal
support is degenerate. Only scale-invariant, caption-relevant facts are
verbalized (bands, fractions, directions); absolute pixel counts and raw RGB /
normal values are deliberately NOT (camera/size/white-balance dependent) and
remain in the machine-readable `evidence_payload` JSON only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# DOME-29 class indices (authoritative in stratum2.config.DOME_29). Values are
# pinned here and cross-checked in tests so this module can never silently drift
# from the real seg2 layout.
FACE_NECK = 3          # "Face_Neck"
HAIR = 4               # "Hair"

# Presence / support floors (mirror skin_color.py / hair.py / clothing.py).
MIN_FG_PX = 400        # raw foreground pixel floor for stable statistics
MIN_VALID_NORMAL_PX = 400   # valid camera-facing normal floor for the direction fit
MIN_COVERAGE = 0.01    # foreground fraction of the frame floor
MIN_NORMAL_FRACTION = 0.5   # fraction of foreground that must carry valid normals

# Quantization thresholds (deterministic, camel-case band names the caption
# vocab already uses).
_LUMA_DARK = 0.25
_LUMA_MODERATE = 0.50
_DR_FLAT = 0.30
_DR_MEDIUM = 0.55
_SHADOW_LOW = 0.08
_SHADOW_HEAVY = 0.25
_SHADOW_LUMA = 0.15
_BACKLIT_RATIO = 1.25   # background mean luma / foreground mean luma

# Direction fit gates.
_NORMAL_MIN_NORM = 0.5       # valid normals must be unit-vector length within tol
_NORMAL_NORM_TOL = 0.35
_LATERAL_SIGN_THRESHOLD = 0.30   # |lateral cosine| above this names a left/right key
_FRONTAL_THRESHOLD = 0.70        # nz above this is a frontal key


class LightingError(RuntimeError):
    pass


def validate_normal2_array(normal2: np.ndarray) -> None:
    if not isinstance(normal2, np.ndarray):
        raise LightingError("normal2 must be a numpy array")
    if normal2.ndim != 3 or normal2.shape[2] != 3:
        raise LightingError(
            f"normal2 must be (H, W, 3) unit normals, got shape {normal2.shape}"
        )


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise LightingError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise LightingError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise LightingError(f"seg2 must be uint8/integer class labels, got dtype {seg2.dtype}")


def _luma(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luma in [0, 1] from (H, W, 3) uint8 RGB. Deterministic, scale-invariant."""
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def _quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q * 100.0))


def _luma_band(mean_luma: float) -> str:
    if mean_luma < _LUMA_DARK:
        return "low-key dim"
    if mean_luma < _LUMA_MODERATE:
        return "moderately lit"
    return "brightly lit"


def _dr_band(dr: float) -> str:
    if dr < _DR_FLAT:
        return "low flat"
    if dr < _DR_MEDIUM:
        return "medium contrast"
    return "high contrast"


def _shadow_band(frac: float) -> str:
    if frac >= _SHADOW_HEAVY:
        return "heavy shadow"
    if frac >= _SHADOW_LOW:
        return "some shadow"
    return "little shadow"


def _dz_band(ratio: float) -> str:
    if ratio >= _BACKLIT_RATIO:
        return "backlit rim-lit"
    if ratio <= 1.0 / _BACKLIT_RATIO:
        return "subject brighter than background"
    return "balanced surround"


def _direction_name(lx: float, lz: float) -> str:
    """Coarse, caption-relevant key-light direction from the fitted vector."""
    if lz <= 0.0:
        return "from behind backlit"
    if abs(lx) >= _LATERAL_SIGN_THRESHOLD:
        side = "left" if lx > 0 else "right"
        if lz >= _FRONTAL_THRESHOLD:
            return f"from the front-{side}"
        return f"from the {side}"
    if lz >= _FRONTAL_THRESHOLD:
        return "from the front"
    return "from the front"


def compute_lighting(
    normal2: np.ndarray,
    seg2: np.ndarray,
    image_rgb: np.ndarray,
    *,
    min_fg_px: int = MIN_FG_PX,
    min_valid_normal_px: int = MIN_VALID_NORMAL_PX,
    min_coverage: float = MIN_COVERAGE,
    min_normal_fraction: float = MIN_NORMAL_FRACTION,
) -> dict[str, Any]:
    """Compute deterministic lighting statistics with explicit abstention.

    Args:
        normal2: (H, W, 3) camera-space unit surface normals (float16/32).
        seg2: (H, W) uint8 DOME-29 class labels aligned with normal2.
        image_rgb: (H, W, 3) uint8 RGB source pixels aligned with normal2.
        floors: presence / support gates.

    Returns a dict with scale-invariant lighting facts only; every caption-facing
    fact is a named band / fraction / direction — the continuous values and the
    fit residual live in the machine-readable payload for the dossier, never as
    caption claims.
    """
    validate_normal2_array(normal2)
    validate_seg2_array(seg2)
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise LightingError("image_rgb must be an (H, W, 3) numpy array")
    h, w = seg2.shape
    if normal2.shape[0] != h or normal2.shape[1] != w:
        raise LightingError(
            f"normal2 {normal2.shape} must be pixel-aligned with seg2 {seg2.shape}"
        )
    if image_rgb.shape[0] != h or image_rgb.shape[1] != w:
        raise LightingError(
            f"image_rgb {image_rgb.shape} must be pixel-aligned with seg2 {seg2.shape}"
        )

    fg_pixels = int((seg2 > 0).sum())
    total_pixels = int(seg2.size)
    if fg_pixels <= 0:
        return {
            "subject_present": False,
            "abstained": True,
            "abstention_reason": "no foreground subject detected",
            "lighting_measurable": False,
            "luma_band": None,
            "mean_luma": None,
            "median_luma": None,
            "dynamic_range": None,
            "dynamic_range_band": None,
            "shadow_fraction": None,
            "shadow_band": None,
            "surround_ratio": None,
            "surround_band": None,
            "light_direction": None,
            "light_vector": None,
            "light_residual": None,
            "measured_fg_px": 0,
            "valid_normal_px": 0,
        }

    fg_mask = seg2 > 0
    luma = _luma(image_rgb)
    fg_luma = luma[fg_mask]

    fg_frac = fg_pixels / max(total_pixels, 1)
    if fg_pixels < min_fg_px or fg_frac < min_coverage:
        return {
            "subject_present": True,
            "abstained": True,
            "abstention_reason": "foreground too small relative to gates for stable lighting statistics",
            "lighting_measurable": False,
            "luma_band": None,
            "mean_luma": None,
            "median_luma": None,
            "dynamic_range": None,
            "dynamic_range_band": None,
            "shadow_fraction": None,
            "shadow_band": None,
            "surround_ratio": None,
            "surround_band": None,
            "light_direction": None,
            "light_vector": None,
            "light_residual": None,
            "measured_fg_px": fg_pixels,
            "valid_normal_px": 0,
        }

    mean_luma = float(fg_luma.mean())
    median_luma = float(np.median(fg_luma))
    p2 = _quantile(fg_luma, 0.02)
    p98 = _quantile(fg_luma, 0.98)
    dynamic_range = float(max(0.0, p98 - p2))
    shadow_frac = float((fg_luma < _SHADOW_LUMA).mean())

    bg_luma = luma[~fg_mask]
    surround_ratio = float(bg_luma.mean() / mean_luma) if bg_luma.size and mean_luma > 0.0 else None

    # Direction fit over validated camera-facing foreground normals.
    normals = normal2.astype(np.float64)
    n_norm = np.linalg.norm(normals, axis=2)
    valid = fg_mask & (np.abs(n_norm - 1.0) <= _NORMAL_NORM_TOL) & (n_norm >= _NORMAL_MIN_NORM)
    valid_normals = normals[valid]
    valid_luma = luma[valid]
    valid_px = int(valid.sum())
    normal_fraction = valid_px / fg_pixels

    result: dict[str, Any] = {
        "subject_present": True,
        "abstained": False,
        "abstention_reason": None,
        "lighting_measurable": True,
        "luma_band": _luma_band(mean_luma),
        "mean_luma": round(mean_luma, 4),
        "median_luma": round(median_luma, 4),
        "dynamic_range": round(dynamic_range, 4),
        "dynamic_range_band": _dr_band(dynamic_range),
        "shadow_fraction": round(shadow_frac, 4),
        "shadow_band": _shadow_band(shadow_frac),
        "surround_ratio": round(surround_ratio, 4) if surround_ratio is not None else None,
        "surround_band": _dz_band(surround_ratio) if surround_ratio is not None else None,
        "light_direction": None,
        "light_vector": None,
        "light_residual": None,
        "measured_fg_px": fg_pixels,
        "valid_normal_px": valid_px,
    }

    if valid_px < min_valid_normal_px or normal_fraction < min_normal_fraction:
        result["light_direction"] = "undetermined"
        result["abstention_reason"] = (
            "insufficient valid camera-facing normals for a stable light-direction fit"
        )
        return result

    # Lambertian least-squares light fit: L = argmin ||lum - N.L||^2.
    coeffs, _, _, _ = np.linalg.lstsq(valid_normals, valid_luma, rcond=None)
    lv = np.asarray(coeffs, dtype=np.float64)
    norm = float(np.linalg.norm(lv))
    if norm < 1e-6:
        result["light_direction"] = "undetermined"
        result["abstention_reason"] = "degenerate light fit (no directional shading gradient)"
        return result
    unit = lv / norm
    residual = float(np.sqrt(np.mean((valid_normals @ lv - valid_luma) ** 2))) if valid_px else None
    lx = float(unit[0])
    lz = float(unit[2])
    result["light_vector"] = [round(float(v), 4) for v in unit]
    result["light_residual"] = round(residual, 4) if residual is not None else None
    result["light_direction"] = _direction_name(lx, lz)
    return result
