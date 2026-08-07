"""Deterministic skin-tone measurements from `seg2` masks + source pixels.

Arm #31. Reads an existing `seg2.npy` (DOME-29 semantic labels, uint8, at full
source resolution) plus the source RGB image pixels and emits continuous
skin-tone measurements:

- subject_present: is there a foreground subject (seg2 > 0)?
- exposed_skin_present: do the declared skin regions clear a raw-pixel floor
  and a foreground-coverage gate in aggregate?
- skin_coverage: fraction of subject foreground classified as exposed skin
  (Face_Neck + Torso + limb skin regions). A scale-invariant exposure fact.
- skin_tone_name / skin_tone_hex: deterministic dominant skin tone derived
  from the masked source pixels, quantized to a fixed named tonal palette.
- face_tone / body_tone: per-region dominant tones (face/neck vs the rest) so
  the caption can reflect where the tone was measured and whether regions
  agree (e.g. tan lines, makeup, or lighting drift).

Every measurement honors the single-subject invariant and abstains (emits
None/False rather than fabricating) when the supporting skin region is
degenerate or absent. Only scale-invariant, caption-relevant facts are emitted
(tone name, exposure fraction); absolute pixel counts and raw RGB are
deliberately NOT (camera/size/white-balance dependent) and stay in the
machine-readable payload only.

Skin tone is a sensitive descriptor: the palette names are neutral, purely
descriptive tone labels (not social categories), the measurement is the mean
of masked pixels quantized to that palette, and the specialist never infers a
tone when no skin region clears the gate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# DOME-29 class indices (authoritative in stratum2.config.DOME_29). Values are
# pinned here and cross-checked in tests so this module can never silently drift
# from the real seg2 layout.
FACE_NECK = 3          # "Face_Neck"
HAIR = 4               # "Hair"
LEFT_FOOT = 5
LEFT_HAND = 6
LEFT_LOWER_ARM = 7
LEFT_LOWER_LEG = 8
LEFT_UPPER_ARM = 11
LEFT_UPPER_LEG = 12
RIGHT_FOOT = 14
RIGHT_HAND = 15
RIGHT_LOWER_ARM = 16
RIGHT_LOWER_LEG = 17
RIGHT_UPPER_ARM = 20
RIGHT_UPPER_LEG = 21
TORSO = 22             # "Torso" (skin, when not covered)

# Declared exposed-skin classes (DOME-29 skin regions). This is what skin tone
# is measured from. Lips/teeth/tongue and hair/eyeglass are excluded.
SKIN_CLASSES: dict[str, int] = {
    "face_neck": FACE_NECK,
    "torso": TORSO,
    "left_upper_arm": LEFT_UPPER_ARM,
    "left_lower_arm": LEFT_LOWER_ARM,
    "left_hand": LEFT_HAND,
    "right_upper_arm": RIGHT_UPPER_ARM,
    "right_lower_arm": RIGHT_LOWER_ARM,
    "right_hand": RIGHT_HAND,
    "left_upper_leg": LEFT_UPPER_LEG,
    "left_lower_leg": LEFT_LOWER_LEG,
    "left_foot": LEFT_FOOT,
    "right_upper_leg": RIGHT_UPPER_LEG,
    "right_lower_leg": RIGHT_LOWER_LEG,
    "right_foot": RIGHT_FOOT,
}

# Floor thresholds (mirror determinations.py / clothing.py / hair.py): a region
# must clear a raw pixel count AND a foreground fraction before it is measured.
MIN_CLASS_PX = 200
MIN_COVERAGE = 0.01

# Fixed named skin-tone palette (deterministic quantization target). Neutral,
# purely-descriptive tone labels ordered light -> deep. Names match the same
# caption vocabulary the aggregator already uses.
_NAMED_SKIN_TONES: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("very fair", (250, 226, 214)),
    ("fair", (240, 208, 188)),
    ("light", (228, 190, 164)),
    ("light medium", (214, 170, 138)),
    ("medium", (198, 148, 112)),
    ("tan", (178, 126, 88)),
    ("brown", (150, 100, 66)),
    ("dark brown", (118, 74, 50)),
    ("deep", (90, 55, 38)),
)


class SkinColorError(RuntimeError):
    pass


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise SkinColorError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise SkinColorError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise SkinColorError(f"seg2 must be uint8/integer class labels, got dtype {seg2.dtype}")


def _dominant_tone(pixels: np.ndarray) -> tuple[str, str]:
    """Map masked RGB pixels to (palette_tone_name, hex). Deterministic."""
    mean = pixels.mean(axis=0)
    best_name, best_dist = "medium", float("inf")
    for name, rgb in _NAMED_SKIN_TONES:
        dist = float(np.hypot(mean[0] - rgb[0], np.hypot(mean[1] - rgb[1], mean[2] - rgb[2])))
        if dist < best_dist:
            best_name, best_dist = name, dist
    hex_str = "#{:02x}{:02x}{:02x}".format(int(round(float(mean[0]))), int(round(float(mean[1]))),
                                           int(round(float(mean[2]))))
    return best_name, hex_str


def compute_skin_tone(seg2: np.ndarray, image_rgb: np.ndarray, *, min_px: int = MIN_CLASS_PX,
                      min_coverage: float = MIN_COVERAGE) -> dict[str, Any]:
    """Compute deterministic skin-tone measurements with per-region abstention.

    Args:
        seg2: (H, W) uint8 DOME-29 class labels at full source resolution.
        image_rgb: (H, W, 3) uint8 RGB source pixels, aligned to seg2.
        min_px / min_coverage: presence floors for a region to be measured.

    Returns a dict with scale-invariant skin facts only:
    - subject_present / abstained
    - exposed_skin_present, skin_coverage, skin_frame_coverage
    - skin_tone_name / skin_tone_hex (overall, when exposed skin clears the gate)
    - face_tone_name / face_tone_hex (Face_Neck region alone, if it clears)
    - body_tone_name (all non-Face_Neck skin regions), face_body_agree bool
    """
    validate_seg2_array(seg2)
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise SkinColorError("image_rgb must be an (H, W, 3) numpy array")
    if image_rgb.shape[0] != seg2.shape[0] or image_rgb.shape[1] != seg2.shape[1]:
        raise SkinColorError(f"image_rgb {image_rgb.shape} must be pixel-aligned with seg2 {seg2.shape}")

    fg_pixels = int((seg2 > 0).sum())
    total_pixels = int(seg2.size)
    if fg_pixels <= 0:
        return {
            "subject_present": False,
            "abstained": True,
            "exposed_skin_present": False,
            "skin_coverage": 0.0,
            "skin_frame_coverage": 0.0,
            "skin_tone_name": None,
            "skin_tone_hex": None,
            "face_tone_name": None,
            "face_tone_hex": None,
            "body_tone_name": None,
            "face_body_agree": None,
            "measured_regions": [],
        }
    denom = max(fg_pixels, 1)
    frame_denom = max(total_pixels, 1)

    skin_ids = list(SKIN_CLASSES.values())
    skin_mask = np.isin(seg2, skin_ids)
    skin_px = int(skin_mask.sum())
    coverage = skin_px / denom
    present = skin_px >= min_px and coverage > min_coverage

    result: dict[str, Any] = {
        "subject_present": True,
        "abstained": False,
        "exposed_skin_present": present,
        "skin_coverage": round(coverage, 4),
        "skin_frame_coverage": round(skin_px / frame_denom, 4),
        "skin_tone_name": None,
        "skin_tone_hex": None,
        "face_tone_name": None,
        "face_tone_hex": None,
        "body_tone_name": None,
        "face_body_agree": None,
        "measured_regions": [],
    }

    if not present:
        return result

    result["skin_tone_name"], result["skin_tone_hex"] = _dominant_tone(image_rgb[skin_mask])

    # Per-region measurement: which declared regions actually clear the gate.
    measured: list[str] = []
    for name, cls in SKIN_CLASSES.items():
        region_mask = seg2 == cls
        region_px = int(region_mask.sum())
        if region_px >= min_px and (region_px / denom) > min_coverage:
            measured.append(name)
    result["measured_regions"] = sorted(measured)

    face_mask = seg2 == FACE_NECK
    face_px = int(face_mask.sum())
    if face_px >= min_px and (face_px / denom) > min_coverage:
        result["face_tone_name"], result["face_tone_hex"] = _dominant_tone(image_rgb[face_mask])

    body_mask = skin_mask & (seg2 != FACE_NECK)
    body_px = int(body_mask.sum())
    if body_px >= min_px:
        result["body_tone_name"], _ = _dominant_tone(image_rgb[body_mask])

    if result["face_tone_name"] is not None and result["body_tone_name"] is not None:
        result["face_body_agree"] = result["face_tone_name"] == result["body_tone_name"]

    return result
