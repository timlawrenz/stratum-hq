"""Deterministic hair measurements from `seg2` masks + source pixels.

Arm #30. Reads an existing `seg2.npy` (DOME-29 semantic labels, uint8, at full
source resolution) plus the source RGB image pixels and emits continuous
hair measurements:

- subject_present: is there a foreground subject (seg2 > 0)?
- hair_present: does the Hair region clear a raw-pixel floor and a
  foreground-coverage gate?
- hair_coverage: fraction of subject foreground pixels classified as Hair.
- hair_frame_coverage: fraction of the whole frame classified as Hair.
- dominant color name + hex derived from the source pixels under the Hair mask
  (deterministic, quantized to a fixed named palette).
- hair_position: which vertical band of the frame the Hair centroid occupies,
  as a scale-invariant normalized fact (top / middle / bottom).
- hair_face_extent_ratio: vertical span of Hair relative to the vertical span
  of the Face_Neck region — a scale-invariant hair-length proxy.

Every measurement honors the single-subject invariant and abstains (emits
None/False rather than fabricating) when the supporting region is degenerate or
absent. Only scale-invariant, caption-relevant facts are emitted; absolute
pixel counts and coordinates are deliberately NOT among them (camera-frame
dependent) and stay in the machine-readable payload only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# DOME-29 class indices (authoritative in stratum2.config.DOME_29). Values are
# pinned here and cross-checked in tests so this module can never silently drift
# from the real seg2 layout.
HAIR = 4          # "Hair"
FACE_NECK = 3     # "Face_Neck"

# Floor thresholds (mirror determinations.py / clothing.py): a class must clear
# a raw pixel count AND a foreground fraction before it is treated as present.
MIN_CLASS_PX = 200
MIN_COVERAGE = 0.01

# Fixed named-color palette for hair (deterministic quantization target).
# Hair-appropriate hues first; names match the same caption vocabulary the
# aggregator already uses.
_NAMED_HAIR_COLORS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("black", (20, 18, 20)),
    ("dark brown", (70, 45, 30)),
    ("brown", (110, 70, 40)),
    ("auburn", (140, 60, 40)),
    ("red", (180, 60, 40)),
    ("ginger", (200, 110, 60)),
    ("blonde", (215, 185, 130)),
    ("light blonde", (235, 215, 170)),
    ("gray", (165, 165, 170)),
    ("silver", (205, 205, 210)),
    ("white", (240, 240, 240)),
    ("dark", (40, 35, 42)),
)


class HairError(RuntimeError):
    pass


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise HairError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise HairError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise HairError(f"seg2 must be uint8/integer class labels, got dtype {seg2.dtype}")


def _dominant_color(pixels: np.ndarray) -> tuple[str, str]:
    """Map masked RGB pixels to (palette_color_name, hex). Deterministic."""
    mean = pixels.mean(axis=0)
    best_name, best_dist = "brown", float("inf")
    for name, rgb in _NAMED_HAIR_COLORS:
        dist = float(np.hypot(mean[0] - rgb[0], np.hypot(mean[1] - rgb[1], mean[2] - rgb[2])))
        if dist < best_dist:
            best_name, best_dist = name, dist
    hex_str = "#{:02x}{:02x}{:02x}".format(int(round(float(mean[0]))), int(round(float(mean[1]))),
                                           int(round(float(mean[2]))))
    return best_name, hex_str


def _vertical_band(row_frac: float) -> str | None:
    """Map a normalized centroid row (0=top, 1=bottom) to a caption band."""
    if row_frac is None:
        return None
    if row_frac < 1 / 3:
        return "top"
    if row_frac < 2 / 3:
        return "middle"
    return "bottom"


def compute_hair(seg2: np.ndarray, image_rgb: np.ndarray, *, min_px: int = MIN_CLASS_PX,
                 min_coverage: float = MIN_COVERAGE) -> dict[str, Any]:
    """Compute deterministic hair measurements with per-region abstention.

    Args:
        seg2: (H, W) uint8 DOME-29 class labels at full source resolution.
        image_rgb: (H, W, 3) uint8 RGB source pixels, aligned to seg2.
        min_px / min_coverage: presence floors for a region to be measured.

    Returns a dict with scale-invariant hair facts only:
    - subject_present / abstained
    - hair_present, hair_coverage, hair_frame_coverage
    - hair_dominant_color_name / hair_dominant_hex (only when present)
    - hair_position (vertical band), hair_face_extent_ratio (length proxy)
    """
    validate_seg2_array(seg2)
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise HairError("image_rgb must be an (H, W, 3) numpy array")
    if image_rgb.shape[0] != seg2.shape[0] or image_rgb.shape[1] != seg2.shape[1]:
        raise HairError(f"image_rgb {image_rgb.shape} must be pixel-aligned with seg2 {seg2.shape}")

    fg_pixels = int((seg2 > 0).sum())
    total_pixels = int(seg2.size)
    if fg_pixels <= 0:
        return {
            "subject_present": False,
            "abstained": True,
            "hair_present": False,
            "hair_coverage": 0.0,
            "hair_frame_coverage": 0.0,
            "hair_dominant_color_name": None,
            "hair_dominant_hex": None,
            "hair_position": None,
            "hair_face_extent_ratio": None,
        }
    denom = max(fg_pixels, 1)
    frame_denom = max(total_pixels, 1)

    hair_mask = seg2 == HAIR
    hair_px = int(hair_mask.sum())
    coverage = hair_px / denom
    present = hair_px >= min_px and coverage > min_coverage

    result: dict[str, Any] = {
        "subject_present": True,
        "abstained": False,
        "hair_present": present,
        "hair_coverage": round(coverage, 4),
        "hair_frame_coverage": round(hair_px / frame_denom, 4),
        "hair_dominant_color_name": None,
        "hair_dominant_hex": None,
        "hair_position": None,
        "hair_face_extent_ratio": None,
    }

    if present:
        color_name, hex_str = _dominant_color(image_rgb[hair_mask])
        result["hair_dominant_color_name"] = color_name
        result["hair_dominant_hex"] = hex_str

        rows, cols = np.nonzero(hair_mask)
        if rows.size:
            result["hair_position"] = _vertical_band(float(rows.mean()) / seg2.shape[0])

        face_mask = seg2 == FACE_NECK
        face_rows = np.nonzero(face_mask)[0]
        if face_rows.size:
            hair_span = float(rows.max() - rows.min() + 1)
            face_span = float(face_rows.max() - face_rows.min() + 1)
            if face_span > 0:
                result["hair_face_extent_ratio"] = round(hair_span / face_span, 3)

    return result
