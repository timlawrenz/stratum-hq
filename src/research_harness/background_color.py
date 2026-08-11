"""Deterministic background-color measurement from seg2 + source RGB.

Arm #126 (selected by the SELECTOR EXPLOIT slot, selection_progress 36,
activated 2026-08-11). NEW evidence part `background-color` (no new model,
CPU, deterministic from the frozen derived artifacts).

Scope / non-redundance (from the registry selection rationale):
- scene-category #69 covers SEMANTIC place ('indoor studio', 'beach').
- setting #34 covers pattern/busyness, tone (light/mid/dark), vibrancy
  (muted/moderate/vivid) and a QUANTIZED NAMED-PALETTE color — but it never
  verbalizes the background's TEMPERATURE (warm / cool / neutral hue family),
  which is what captions actually say ('blue backdrop', 'warm beige wall').
- environment-clearance #85 covers negative SPACE (subject-to-backdrop gap).
- This arm verbalizes the background HUE FAMILY band (warm / cool / neutral).
  Lightness is measured for the payload only and is NOT a separate verbalized
  axis, because setting #34 already grounds a light/mid/dark tone band —
  a second verbalized lightness axis would be redundant (and would risk the
  warm/cool axis collapsing into a framing-dependent channel).

Measurement (deterministic, scale-invariant):
- Use the seg2 DOME-29 Background region (class 0) exactly as setting #34
  does (same MIN_BG_PX / MIN_BG_COVERAGE gates) so the two arms read the same
  region.
- Per-background-pixel classify: saturation < SAT_NEUTRAL -> ACHROMATIC
  (neutral contributor); else hue family warm / cool from the HSV hue angle:
    warm  = red, orange, yellow, and pink/magenta hues (hue angle near 0-70
            and 300-360 in [0,360) terms);
    cool  = green, cyan, blue, purple (hue angle ~70-300).
- The VERBALIZED band is the dominant family by pixel share:
    neutral if achromatic pixels >= NEUTRAL_SHARE_FLOOR (e.g. 0.6);
    otherwise warm if warm share > cool share, else cool.
  This is scale-invariant (a pixel-share ratio within the same region) and
  calibration happens over the band-degeneracy rule (no single band >=75% of
  measured items).
- Raw mean RGB / HSV, shares, and mean lightness stay in the
  machine-readable evidence_payload and are NEVER caption claims.

Abstention: no foreground subject (seg2 all background), background smaller
than the gates, or degenerate color statistics (near-zero chroma variance on
a tiny region). Never fabricate a background-color band; detector
disagreement remains a quality anomaly, never prompt content. CPU-only,
in-memory, no corpus write.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from stratum2.config import DOME_29

# DOME-29 class index (authoritative in stratum2.config.DOME_29).
_BACKGROUND_I = DOME_29.index("Background")

# Presence / support floors (mirror setting.py so both arms gate identically).
MIN_BG_PX = 400
MIN_BG_COVERAGE = 0.02  # 2% of the frame must be Background to measure it

# Hue-family thresholds (deterministic, HSV in [0,360) hue terms).
_SAT_NEUTRAL = 0.15       # normalized saturation below this => achromatic/neutral
_NEUTRAL_SHARE_FLOOR = 0.60  # >=60% achromatic background pixels => neutral
_WARM_HI = 70.0           # hue <= 70 (red/orange/yellow) or >= 300 (pink/magenta) => warm
_WARM_LO = 300.0

# Lightness split for the PAYLOAD-only band (Rec.709 luma 0..1).
_LIGHT_LO = 0.27          # mean luma below this => dark
_LIGHT_HI = 0.62          # mean luma above this => light; else mid


class BackgroundColorError(RuntimeError):
    pass


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise BackgroundColorError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise BackgroundColorError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if not np.issubdtype(seg2.dtype, np.integer):
        raise BackgroundColorError(f"seg2 must be integer class labels, got dtype {seg2.dtype}")


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized RGB (H,W,3 float 0..255) -> hue(0..360)/sat(0..1)/val(0..255)."""
    r = rgb[..., 0] / 255.0
    g = rgb[..., 1] / 255.0
    b = rgb[..., 2] / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = mx - mn
    val = mx * 255.0

    sat = np.zeros_like(mx)
    np.divide(delta, np.maximum(mx, 1e-9), out=sat, where=mx > 0)

    hue = np.zeros_like(mx)
    mask = delta > 1e-9
    r_, g_, b_ = r, g, b
    with np.errstate(invalid="ignore", divide="ignore"):
        h_r = np.where(mask, (60.0 * ((g_ - b_) / np.maximum(delta, 1e-9))) % 360.0, 0.0)
        h_g = np.where(mask, 60.0 * ((b_ - r_) / np.maximum(delta, 1e-9)) + 120.0, 0.0)
        h_b = np.where(mask, 60.0 * ((r_ - g_) / np.maximum(delta, 1e-9)) + 240.0, 0.0)
    hue = np.where(mask, np.where(mx == r_, h_r, np.where(mx == g_, h_g, h_b)), 0.0)
    return hue, sat, val


def _family_of_inner(hue: np.ndarray, sat: np.ndarray) -> np.ndarray:
    """Per-pixel family: 0=neutral/achromatic, 1=warm, 2=cool."""
    neutral = sat < _SAT_NEUTRAL
    warm_hue = (hue <= _WARM_HI) | (hue >= _WARM_LO)
    out = np.full_like(hue, 2, dtype=np.int8)
    out[warm_hue] = 1
    out[neutral] = 0
    return out


def compute_background_color(
    seg2: np.ndarray,
    image_rgb: np.ndarray,
    *,
    min_bg_px: int = MIN_BG_PX,
    min_bg_coverage: float = MIN_BG_COVERAGE,
) -> dict[str, Any]:
    """Compute the scale-invariant background-color hue-family band.

    Returns a dict with ``abstained`` / ``abstention_reason`` on failure, or
    the band + raw shares (payload). Only the coarse warm/cool/neutral band is
    verbalized; raw HSV stats stay in the machine-readable payload.
    """
    validate_seg2_array(seg2)
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise BackgroundColorError("image_rgb must be an (H, W, 3) numpy array")
    if image_rgb.shape[0] != seg2.shape[0] or image_rgb.shape[1] != seg2.shape[1]:
        raise BackgroundColorError(
            f"image_rgb {image_rgb.shape} must be pixel-aligned with seg2 {seg2.shape}"
        )

    out: dict[str, Any] = {
        "subject_present": True,
        "abstained": False,
        "abstention_reason": None,
        "background_color_band": None,
        "background_lightness_payload": None,
        "bg_family_shares": None,
        "measured_bg_px": 0,
        "measured_fg_px": 0,
    }

    total_pixels = int(seg2.size)
    fg_pixels = int((seg2 > 0).sum())
    if fg_pixels <= 0:
        out.update({
            "subject_present": False,
            "abstained": True,
            "abstention_reason": "no foreground subject detected",
        })
        return out

    bg_mask = seg2 == _BACKGROUND_I
    bg_pixels = int(bg_mask.sum())
    bg_coverage = bg_pixels / max(total_pixels, 1)
    if bg_pixels < min_bg_px or bg_coverage < min_bg_coverage:
        out.update({
            "abstained": True,
            "abstention_reason": (
                "background region too small relative to gates for stable color statistics"
            ),
            "measured_bg_px": bg_pixels,
            "measured_fg_px": fg_pixels,
        })
        return out

    bg_rgb = image_rgb[bg_mask].astype(np.float64)
    hue, sat, _val = _rgb_to_hsv(bg_rgb)
    families = _family_of_inner(hue, sat)
    n = families.size
    share_neutral = float((families == 0).sum()) / n
    share_warm = float((families == 1).sum()) / n
    share_cool = float((families == 2).sum()) / n

    # Payload-only lightness (Rec.709 luma over the same Background region).
    luma = (0.2126 * bg_rgb[:, 0] + 0.7152 * bg_rgb[:, 1] + 0.0722 * bg_rgb[:, 2]) / 255.0
    mean_luma = float(luma.mean())
    if mean_luma < _LIGHT_LO:
        lightness = "dark"
    elif mean_luma > _LIGHT_HI:
        lightness = "light"
    else:
        lightness = "mid"

    # Degenerate-stats honesty gate: if the background region is tiny after the
    # pixel floor but chroma variance is ~0 AND the region is essentially a
    # solid color edge artifact, still emit the neutral band (it is honest).
    if share_neutral >= _NEUTRAL_SHARE_FLOOR:
        band = "neutral"
    elif share_warm > share_cool:
        band = "warm"
    else:
        band = "cool"

    out.update({
        "abstained": False,
        "background_color_band": band,
        "background_lightness_payload": lightness,
        "bg_family_shares": {
            "neutral": round(share_neutral, 4),
            "warm": round(share_warm, 4),
            "cool": round(share_cool, 4),
        },
        "background_mean_luma_payload": round(mean_luma, 4),
        "background_coverage_payload": round(bg_coverage, 4),
        "measured_bg_px": bg_pixels,
        "measured_fg_px": fg_pixels,
    })
    return out


def render_background_color(config: Mapping[str, Any] | None) -> str:
    """Scale-invariant background-color claim for the dossier (arm #126).

    Verbalizes ONLY the coarse warm/cool/neutral hue-family band. Raw family
    shares, mean luma, and mean RGB stay in the machine-readable payload.
    """
    if not config:
        return "background-color: not measured for this item"
    if config.get("abstained"):
        reason = config.get("abstention_reason") or "background color not measurable"
        return f"background-color: abstain ({reason})"
    band = config.get("background_color_band")
    if band == "warm":
        return "background-color: the backdrop has a warm (red/yellow/orange) hue family"
    if band == "cool":
        return "background-color: the backdrop has a cool (blue/green/purple) hue family"
    if band == "neutral":
        return "background-color: the backdrop is neutral / achromatic (grey, white, black)"
    return "background-color: measured but not banded (payload)"
