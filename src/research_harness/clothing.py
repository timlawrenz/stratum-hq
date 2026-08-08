"""Deterministic clothing / apparel measurements from `seg2` masks + source pixels.

Arm #29. Reads an existing `seg2.npy` (DOME-29 semantic labels, uint8, at full
source resolution) plus the source RGB image pixels and emits continuous
clothing/apparel measurements:

- subject_present: is there a foreground subject (seg2 > 0)?
- per garment class: coverage (fraction of subject foreground pixels) and
  dominance, plus a deterministic dominant color name + hex derived from the
  source pixels under that class mask.
- skin exposure: torso/face skin coverage so a caption never claims a covered
  body part or an absent garment when there is no computing support.

Every measurement honors the single-subject invariant and abstains (emits
None/False rather than fabricating) when the supporting belong to a degenerate
or absent region. Colors are quantized to a fixed named palette so the same
source pixels always map to the same caption-relevant color.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# DOME-29 class indices (authoritative in stratum2.config.DOME_29). Values are
# pinned here and cross-checked in tests so this module can never silently drift
# from the real seg2 layout.
APPAREL = 1          # "Apparel"
FACE_NECK = 3        # "Face_Neck"
TORSO = 22           # "Torso" (skin)
UPPER_CLOTHING = 23  # "Upper_Clothing"
LOWER_CLOTHING = 13  # "Lower_Clothing"
LEFT_SOCK = 10
RIGHT_SOCK = 19
LEFT_SHOE = 9
RIGHT_SHOE = 18
LEFT_UPPER_LEG = 12
RIGHT_UPPER_LEG = 21
LEFT_LOWER_ARM = 7
RIGHT_LOWER_ARM = 16
LEFT_UPPER_ARM = 11
RIGHT_UPPER_ARM = 20

# Garment / apparel classes we declare. Each maps to a human caption concept.
GARMENT_CLASSES: dict[str, int] = {
    "apparel": APPAREL,
    "upper_clothing": UPPER_CLOTHING,
    "lower_clothing": LOWER_CLOTHING,
    "left_sock": LEFT_SOCK,
    "right_sock": RIGHT_SOCK,
    "left_shoe": LEFT_SHOE,
    "right_shoe": RIGHT_SHOE,
}

# Skin classes measured for exposure/abstention (never a semantic claim).
SKIN_CLASSES: dict[str, int] = {
    "torso_skin": TORSO,
    "face_skin": FACE_NECK,
}

# Floor thresholds (mirror determinations.py): a class must clear a raw pixel
# count AND a foreground fraction before it is treated as present/measurable.
MIN_CLASS_PX = 200
MIN_COVERAGE = 0.01

# Fixed named-color palette (deterministic quantization target).
_NAMED_COLORS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("black", (15, 15, 15)),
    ("white", (240, 240, 240)),
    ("gray", (128, 128, 128)),
    ("silver", (188, 190, 196)),
    ("red", (190, 30, 40)),
    ("maroon", (128, 30, 45)),
    ("orange", (225, 120, 25)),
    ("gold", (210, 175, 60)),
    ("yellow", (235, 220, 45)),
    ("olive", (120, 120, 40)),
    ("green", (35, 140, 60)),
    ("teal", (30, 135, 135)),
    ("blue", (35, 70, 190)),
    ("navy", (22, 38, 100)),
    ("purple", (120, 55, 165)),
    ("pink", (235, 150, 175)),
    ("brown", (120, 75, 40)),
    ("beige", (205, 185, 150)),
    ("tan", (190, 155, 95)),
)


class ClothingError(RuntimeError):
    pass


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise ClothingError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise ClothingError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise ClothingError(f"seg2 must be uint8/integer class labels, got dtype {seg2.dtype}")


def _dominant_color(pixels: np.ndarray) -> tuple[str, str]:
    """Map masked RGB pixels to (palette_color_name, hex). Deterministic.

    The mean RGB of the masked pixels is quantized by nearest-neighbor distance
    to the fixed `_NAMED_COLORS` palette. Same pixels -> same name/hex every
    time. ``pixels`` is (N, 3) uint8.
    """
    mean = pixels.mean(axis=0)
    best_name, best_dist = "gray", float("inf")
    for name, rgb in _NAMED_COLORS:
        dist = float(np.hypot(mean[0] - rgb[0], np.hypot(mean[1] - rgb[1], mean[2] - rgb[2])))
        if dist < best_dist:
            best_name, best_dist = name, dist
    hex_str = "#{:02x}{:02x}{:02x}".format(int(round(float(mean[0]))), int(round(float(mean[1]))),
                                           int(round(float(mean[2]))))
    return best_name, hex_str


def compute_clothing(seg2: np.ndarray, image_rgb: np.ndarray, *, min_px: int = MIN_CLASS_PX,
                     min_coverage: float = MIN_COVERAGE) -> dict[str, Any]:
    """Compute deterministic clothing measurements with per-class abstention.

    Args:
        seg2: (H, W) uint8 DOME-29 class labels at full source resolution.
        image_rgb: (H, W, 3) uint8 RGB source pixels, aligned to seg2.
        min_px / min_coverage: presence floors for a class to be measured.

    Returns a dict:
    - subject_present: True when the foreground (seg2 > 0) is non-degenerate.
    - classes: per class id -> {present, coverage, dominant_color_name, dominant_hex}
      (dominant color only when present).
    - garments: list of present garment class names with coverage + color.
    - skin: per skin class id -> coverage (for exposure/abstention notes).
    - abstained: True when subject absent (all measurements abstained).
    """
    validate_seg2_array(seg2)
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ClothingError("image_rgb must be an (H, W, 3) numpy array")
    if image_rgb.shape[0] != seg2.shape[0] or image_rgb.shape[1] != seg2.shape[1]:
        raise ClothingError(
            f"image_rgb {image_rgb.shape} must be pixel-aligned with seg2 {seg2.shape}"
        )

    fg_pixels = int((seg2 > 0).sum())
    total_pixels = int(seg2.size)
    if fg_pixels <= 0:
        return {
            "subject_present": False,
            "abstained": True,
            "classes": {},
            "garments": [],
            "skin": {},
        }
    denom = max(fg_pixels, 1)
    frame_denom = max(total_pixels, 1)

    classes: dict[int, dict[str, Any]] = {}
    garments: list[dict[str, Any]] = []
    for name, cls in GARMENT_CLASSES.items():
        mask = seg2 == cls
        px = int(mask.sum())
        coverage = px / denom
        present = px >= min_px and coverage > min_coverage
        entry: dict[str, Any] = {
            "present": present,
            "coverage": round(coverage, 4),
            "dominant_color_name": None,
            "dominant_hex": None,
        }
        if present:
            color_name, hex_str = _dominant_color(image_rgb[mask])
            entry["dominant_color_name"] = color_name
            entry["dominant_hex"] = hex_str
            garments.append({
                "class": name,
                "coverage": round(coverage, 4),
                "dominant_color_name": color_name,
                "dominant_hex": hex_str,
            })
        classes[cls] = entry
        classes[cls]["class_name"] = name
        classes[cls]["frame_coverage"] = round(px / frame_denom, 4)

    skin: dict[int, dict[str, Any]] = {}
    for name, cls in SKIN_CLASSES.items():
        mask = seg2 == cls
        skin[cls] = {
            "class_name": name,
            "coverage": round(int(mask.sum()) / denom, 4),
        }

    return {
        "subject_present": True,
        "abstained": False,
        "classes": classes,
        "garments": garments,
        "skin": skin,
    }
