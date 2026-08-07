"""Deterministic setting/environment measurements from `seg2` + source pixels.

Arm #34. Reads the existing `seg2.npy` DOME-29 semantic labels (class 0 ==
Background, the non-subject surround) and the source RGB pixels and emits
continuous, scale-invariant setting/environment statistics:

- subject_present: is there a foreground subject (seg2 > 0)?
- setting_measurable: does the Background (class 0) region clear a raw-pixel
  floor and a frame-coverage gate so the statistics are stable?
- background_coverage: fraction of the whole frame classified as Background —
  a scale-invariant share of the scene, caption-relevant ("studio backdrop
  fills most of the frame" vs a tight crop with little surround).
- dominant background color name + hex derived from the source pixels under
  the Background mask (deterministic, quantized to a fixed named palette).
- background_tone_band: light / mid / dark (mean background luma bands).
- background_vibrancy_band: muted / moderate / vivid (mean background
  saturation bands).
- background_pattern_band: solid / some variation / busy — the fraction of
  Background pixels that lie far from the mean background color, a proxy for
  a plain backdrop vs a structured/busy scene.

Every measurement honors the exactly-one-subject invariant and abstains (emits
None/False with a surfaced reason) when the Background region is degenerate or
absent. Only scale-invariant, caption-relevant facts are verbalized (coverage
ratio, color names, tone/vibrancy/pattern bands); absolute pixel counts and raw
RGB values are deliberately NOT (camera/size/white-balance dependent) and
remain in the machine-readable `evidence_payload` JSON only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# DOME-29 class indices (authoritative in stratum2.config.DOME_29). Values are
# pinned here and cross-checked in tests so this module can never silently drift
# from the real seg2 layout.
BACKGROUND = 0  # "Background" — the non-subject surround

# Presence / support floors (mirror skin_color.py / hair.py / clothing.py /
# lighting.py): a region must clear a raw pixel count AND a frame fraction.
MIN_BG_PX = 400
MIN_BG_COVERAGE = 0.02  # 2% of the frame must be Background to measure it

# Quantization thresholds (deterministic, camel-case band names the caption
# vocab already uses).
_TONE_DARK = 0.25
_TONE_MID = 0.55
_SAT_MUTED = 0.12
_SAT_VIVID = 0.32
_PATTERN_ZERO = 0.10   # fraction of deviant bg pixels below this => solid
_PATTERN_BUSY = 0.35   # above this => busy/structured

# Fixed named-color palette for backgrounds (deterministic quantization
# target). Scene-appropriate neutrals and common backdrop hues first; names
# match the caption vocabulary the aggregator already uses.
_NAMED_BG_COLORS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("white", (245, 245, 245)),
    ("off-white", (228, 224, 214)),
    ("beige", (210, 190, 160)),
    ("tan", (170, 140, 100)),
    ("light grey", (200, 200, 200)),
    ("grey", (140, 140, 140)),
    ("dark grey", (80, 80, 80)),
    ("black", (15, 15, 15)),
    ("brown", (100, 70, 45)),
    ("dark brown", (55, 40, 28)),
    ("light blue", (180, 205, 225)),
    ("blue", (70, 110, 165)),
    ("navy", (40, 52, 96)),
    ("teal", (60, 140, 135)),
    ("green", (95, 135, 70)),
    ("olive", (120, 115, 70)),
    ("red", (150, 60, 52)),
    ("brick", (185, 95, 70)),
    ("orange", (205, 130, 60)),
    ("yellow", (215, 190, 90)),
    ("pink", (220, 160, 165)),
    ("purple", (120, 90, 150)),
    ("lavender", (195, 180, 215)),
)


class SettingError(RuntimeError):
    pass


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise SettingError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise SettingError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise SettingError(
            f"seg2 must be uint8/integer class labels, got dtype {seg2.dtype}"
        )


def _dominant_color(pixels: np.ndarray) -> tuple[str, str, float]:
    """Map masked RGB pixels to (palette_color_name, hex, deviant_fraction).

    Deterministic: nearest named anchor to the mean pixel, exact mean encoded
    as hex, and the fraction of masked pixels that lie far from that mean (a
    busy/patterned proxy). Deviant distance is scale-invariant in the sense of
    being a relative threshold on the same camera-space pixel values — only
    the resulting fraction (a ratio) is ever verbalized, never the raw RGB.
    """
    mean = pixels.mean(axis=0)
    best_name, best_dist = "grey", float("inf")
    for name, rgb in _NAMED_BG_COLORS:
        dist = float(np.hypot(mean[0] - rgb[0], np.hypot(mean[1] - rgb[1], mean[2] - rgb[2])))
        if dist < best_dist:
            best_name, best_dist = name, dist
    hex_str = "#{:02x}{:02x}{:02x}".format(
        int(round(float(mean[0]))), int(round(float(mean[1]))), int(round(float(mean[2])))
    )
    # Deviant = more than ~24% of the full 0-255 scale away from the mean in
    # euclidean RGB distance; the resulting fraction is the caption-relevant
    # uniform-vs-busy ratio, never the raw colors.
    deviant = float(
        (np.linalg.norm(pixels.astype(np.float64) - mean[None, :], axis=1) > 60.0).mean()
    ) if pixels.shape[0] else 0.0
    return best_name, hex_str, deviant


def _tone_band(mean_luma: float) -> str:
    if mean_luma < _TONE_DARK:
        return "dark"
    if mean_luma < _TONE_MID:
        return "mid"
    return "light"


def _vibrancy_band(mean_sat: float) -> str:
    if mean_sat < _SAT_MUTED:
        return "muted"
    if mean_sat < _SAT_VIVID:
        return "moderate"
    return "vivid"


def _pattern_band(deviant_fraction: float) -> str:
    if deviant_fraction < _PATTERN_ZERO:
        return "solid"
    if deviant_fraction < _PATTERN_BUSY:
        return "some variation"
    return "busy"


def compute_setting(
    seg2: np.ndarray,
    image_rgb: np.ndarray,
    *,
    min_bg_px: int = MIN_BG_PX,
    min_bg_coverage: float = MIN_BG_COVERAGE,
) -> dict[str, Any]:
    """Compute deterministic setting statistics with explicit abstention.

    Args:
        seg2: (H, W) uint8 DOME-29 class labels aligned with image_rgb.
        image_rgb: (H, W, 3) uint8 RGB source pixels aligned with seg2.
        floors: presence / support gates.

    Returns a dict with scale-invariant setting facts only; every caption-facing
    fact is a named band / coverage ratio / color name — the continuous values,
    mean RGB hex, and deviant fraction live in the machine-readable payload for
    the dossier, never as caption claims.
    """
    validate_seg2_array(seg2)
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise SettingError("image_rgb must be an (H, W, 3) numpy array")
    if image_rgb.shape[0] != seg2.shape[0] or image_rgb.shape[1] != seg2.shape[1]:
        raise SettingError(
            f"image_rgb {image_rgb.shape} must be pixel-aligned with seg2 {seg2.shape}"
        )

    total_pixels = int(seg2.size)
    fg_pixels = int((seg2 > 0).sum())
    if fg_pixels <= 0:
        return _abstain(
            "no foreground subject detected",
            fg_pixels=fg_pixels,
            subject_present=False,
        )

    bg_mask = seg2 == BACKGROUND
    bg_pixels = int(bg_mask.sum())
    bg_coverage = bg_pixels / max(total_pixels, 1)
    if bg_pixels < min_bg_px or bg_coverage < min_bg_coverage:
        return _abstain(
            "background region too small relative to gates for stable setting statistics",
            fg_pixels=fg_pixels,
            bg_pixels=bg_pixels,
            bg_coverage=bg_coverage,
        )

    bg_rgb = image_rgb[bg_mask].astype(np.float64)
    mean_bg = bg_rgb.mean(axis=0)
    # Rec.709 luma and approximate HSV saturation from the same masked pixels.
    luma = (0.2126 * bg_rgb[:, 0] + 0.7152 * bg_rgb[:, 1] + 0.0722 * bg_rgb[:, 2]) / 255.0
    mean_luma = float(luma.mean())
    s = mean_bg.max() / 255.0 if mean_bg.max() > 0 else 0.0
    v = mean_bg.max() / 255.0
    mean_sat = float(0.0 if v == 0.0 else (v - mean_bg.min() / 255.0) / v)

    color_name, hex_str, deviant_fraction = _dominant_color(bg_rgb.astype(np.uint8))

    return {
        "subject_present": True,
        "abstained": False,
        "abstention_reason": None,
        "setting_measurable": True,
        "background_coverage": round(bg_coverage, 4),
        "dominant_background_color": color_name,
        "dominant_background_hex": hex_str,
        "background_tone_band": _tone_band(mean_luma),
        "background_vibrancy_band": _vibrancy_band(mean_sat),
        "background_pattern_band": _pattern_band(deviant_fraction),
        "background_deviant_fraction": round(deviant_fraction, 4),
        "background_mean_luma": round(mean_luma, 4),
        "measured_bg_px": bg_pixels,
        "measured_fg_px": fg_pixels,
    }


def _abstain(reason: str, **counts: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "subject_present": True,
        "abstained": True,
        "abstention_reason": reason,
        "setting_measurable": False,
        "background_coverage": None,
        "dominant_background_color": None,
        "dominant_background_hex": None,
        "background_tone_band": None,
        "background_vibrancy_band": None,
        "background_pattern_band": None,
        "background_deviant_fraction": None,
        "background_mean_luma": None,
    }
    # subject_present may be explicitly overridden (no foreground at all).
    result["subject_present"] = counts.pop("subject_present", True)
    result.update(counts)
    result.setdefault("measured_bg_px", 0)
    result.setdefault("measured_fg_px", 0)
    return result