"""Deterministic texture/material measurements from `seg2` masks + source pixels.

Arm #35. Reads the existing `seg2.npy` DOME-29 semantic labels and the source
RGB pixels and emits continuous, scale-invariant texture/material statistics
PER REGION CLASS (never pooled across classes — pooling a white top with black
leggings produces a mean far from every pixel and a degenerate "busy" band):

- subject_present: is there a foreground subject (seg2 > 0)?
- fabric (garment) region: the dominant measurable garment class (Apparel /
  Upper_Clothing / Lower_Clothing) reports its own coverage, edge_fraction
  (fraction of pixels whose normalized gradient magnitude clears a fixed
  gate — a roughness/edge-density proxy), deviant_fraction (fraction of
  pixels whose normalized color lies far from that class's own mean — a
  print/pattern proxy), texture_band (smooth / some texture / textured), and
  pattern_band (solid / some variation / busy). Abstains per-class when the
  class does not clear the raw-pixel floor and coverage gate.
- skin region: the dominant measurable skin class (Torso / Upper_Leg /
  Lower_Leg / Upper_Arm / Lower_Arm / Face_Neck / Hand / Foot) reports its
  coverage, edge_fraction, texture_band, and a mean_gradient — but NO pattern
  band (tattoos/moles are identity-adjacent and out of scope for a material
  claim).
- texture_measurable: true when at least one region class (fabric or skin)
  cleared the gates — the frozen portrait cohort includes topless items with
  no garment classes, so skin keeps the arm honest and measurable there;
  fabric claims abstain (never fabricate) when no garment class is present.

Every measurement honors the exactly-one-subject invariant and abstains
(emits None/False with a surfaced reason) when the region class is degenerate
or absent. Only scale-invariant, caption-relevant facts are verbalized (bands
and fractions); absolute gradient values, raw RGB, and pixel counts are
camera/size/white-balance dependent and remain in the machine-readable
`evidence_payload` JSON only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# DOME-29 class indices (authoritative in stratum2.config.DOME_29). Values are
# pinned here and cross-checked in tests so this module can never silently drift
# from the real seg2 layout.
APPAREL = 1          # "Apparel"
FACE_NECK = 3        # "Face_Neck"
LEFT_FOOT = 5
LEFT_HAND = 6
LEFT_LOWER_ARM = 7
LEFT_LOWER_LEG = 8
LEFT_SHOE = 9
LEFT_SOCK = 10
LEFT_UPPER_ARM = 11
LEFT_UPPER_LEG = 12
LOWER_CLOTHING = 13  # "Lower_Clothing"
RIGHT_FOOT = 14
RIGHT_HAND = 15
RIGHT_LOWER_ARM = 16
RIGHT_LOWER_LEG = 17
RIGHT_SHOE = 18
RIGHT_SOCK = 19
RIGHT_UPPER_ARM = 20
RIGHT_UPPER_LEG = 21
TORSO = 22           # "Torso" (skin)
UPPER_CLOTHING = 23  # "Upper_Clothing"

# Garment / fabric classes we measure texture over. (Shoes and socks are
# excluded: small, specular, and not the "fabric" the caption vocabulary
# describes as material.)
FABRIC_CLASSES: dict[str, int] = {
    "apparel": APPAREL,
    "upper_clothing": UPPER_CLOTHING,
    "lower_clothing": LOWER_CLOTHING,
}

# Skin classes measured for the surface band (dominant class only). Face_Neck
# is included; lips/teeth/tongue (24-28) are excluded as non-surface.
SKIN_CLASSES: dict[str, int] = {
    "torso": TORSO,
    "face_neck": FACE_NECK,
    "left_upper_leg": LEFT_UPPER_LEG,
    "right_upper_leg": RIGHT_UPPER_LEG,
    "left_lower_leg": LEFT_LOWER_LEG,
    "right_lower_leg": RIGHT_LOWER_LEG,
    "left_upper_arm": LEFT_UPPER_ARM,
    "right_upper_arm": RIGHT_UPPER_ARM,
    "left_lower_arm": LEFT_LOWER_ARM,
    "right_lower_arm": RIGHT_LOWER_ARM,
    "left_hand": LEFT_HAND,
    "right_hand": RIGHT_HAND,
    "left_foot": LEFT_FOOT,
    "right_foot": RIGHT_FOOT,
}

# Presence / support floors (mirror clothing.py / setting.py): a region class
# must clear a raw pixel count AND a frame fraction before it is measured.
MIN_REGION_PX = 400
MIN_COVERAGE = 0.01

# Gradient edge gate on the normalized luma (0..1 after per-channel
# normalization): pixels whose gradient magnitude exceeds this are "edges".
# The image is normalized per channel by its own 99.9th percentile inside the
# region class, so this gate is exposure/white-balance invariant. Band cut
# points are CALIBRATED on the real frozen cohort by scripts/probe_texture.py
# (a band holding >=75% of items is not discriminating — recalibrate).
_EDGE_GATE = 0.06
_TEXTURE_ZERO = 0.06    # fabric edge_fraction below this => smooth (calibrated on cohort: 4/10 fabric items smooth, bimodal low cluster)
_TEXTURE_BUSY = 0.20    # fabric edge_fraction above this => textured

# Skin-specific texture bands (skin gradients are naturally lower than fabric;
# a shared threshold set collapses 19/24 items into "smooth"). Calibrated on
# the cohort: 15 smooth / 7 some texture / 2 textured.
_SKIN_ZERO = 0.03
_SKIN_BUSY = 0.08

# Deviant-fraction pattern gate (per-class normalized color distance from that
# class's own mean; shading gradients are typical 0.1-0.25 in normalized
# space, prints sit well above).
_DEV_DIST = 0.25        # normalized-color distance gate (0..1 space)
_PATTERN_ZERO = 0.15   # deviant_fraction below this => solid
_PATTERN_BUSY = 0.45   # above this => busy/printed

# Normalization percentile per channel (robust to a few blown highlights).
_PCT = 99.9


class TextureError(RuntimeError):
    pass


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise TextureError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise TextureError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise TextureError(
            f"seg2 must be uint8/integer class labels, got dtype {seg2.dtype}"
        )


def _erode1(mask: np.ndarray) -> np.ndarray:
    """1-px 4-neighborhood erosion (mask interior only).

    Gradient statistics must be measured on the region's INTERIOR, not its
    silhouette boundary: the source pixel plane is 0-filled outside the mask,
    so boundary pixels see a spurious step to 0 that inflates edge_fraction
    (a pure outline artifact, not material roughness). Eroding by 1 px keeps
    only pixels whose 4 neighbors are all inside the mask; their gradient is
    computed from real neighbors only.
    """
    eroded = np.zeros_like(mask)
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return eroded
    eroded[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    return eroded


def _region_stats(image_rgb: np.ndarray, mask: np.ndarray) -> dict[str, float] | None:
    """Per-class normalized gradient/edge/deviance statistics.

    Returns None when the mask is empty. All statistics are computed on a
    per-channel-p99.9-normalized image so exposure/white balance cannot
    inflate them, and every caption-facing fact is a fraction/band (ratio),
    never an absolute pixel value. Normalization uses the full region; the
    gradient/deviance sample is the 1-px-eroded interior so the silhouette
    boundary never counts as texture.
    """
    px = image_rgb[mask].astype(np.float64)
    if px.shape[0] == 0:
        return None
    scale = np.percentile(px, _PCT, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    norm_full = np.clip(image_rgb.astype(np.float64) / scale[None, None, :], 0.0, 1.0)
    norm = norm_full[mask]
    luma = 0.2126 * norm[:, 0] + 0.7152 * norm[:, 1] + 0.0722 * norm[:, 2]

    plane = np.zeros(image_rgb.shape[:2], dtype=np.float64)
    plane[mask] = luma
    gy, gx = np.gradient(plane)

    interior = _erode1(mask)
    if int(interior.sum()) == 0:
        interior = mask  # tiny region below floors anyway; fall back
    grad = np.sqrt(gx * gx + gy * gy)[interior]
    interior_norm = norm_full[interior]

    mean_color = interior_norm.mean(axis=0)
    dist = np.linalg.norm(interior_norm - mean_color[None, :], axis=1)
    return {
        "edge_fraction": float((grad > _EDGE_GATE).mean()),
        "mean_gradient": float(grad.mean()),
        "deviant_fraction": float((dist > _DEV_DIST).mean()),
    }


def _texture_band(edge_fraction: float) -> str:
    if edge_fraction < _TEXTURE_ZERO:
        return "smooth"
    if edge_fraction < _TEXTURE_BUSY:
        return "some texture"
    return "textured"


def _pattern_band(deviant_fraction: float) -> str:
    if deviant_fraction < _PATTERN_ZERO:
        return "solid"
    if deviant_fraction < _PATTERN_BUSY:
        return "some variation"
    return "busy"


def _skin_texture_band(edge_fraction: float) -> str:
    if edge_fraction < _SKIN_ZERO:
        return "smooth"
    if edge_fraction < _SKIN_BUSY:
        return "some texture"
    return "textured"


def _dominant_measurable(
    seg2: np.ndarray,
    image_rgb: np.ndarray,
    classes: dict[str, int],
    min_region_px: int,
    min_coverage: float,
) -> tuple[str, int, float, dict[str, float]] | None:
    """Pick the single dominant class among `classes` that clears the gates."""
    best: tuple[str, int, float, dict[str, float]] | None = None
    for name, cls in classes.items():
        mask = seg2 == cls
        n = int(mask.sum())
        coverage = n / max(int(seg2.size), 1)
        if n < min_region_px or coverage < min_coverage:
            continue
        stats = _region_stats(image_rgb, mask)
        if stats is None:
            continue
        if best is None or n > best[1]:
            best = (name, n, coverage, stats)
    return best


def compute_texture(
    seg2: np.ndarray,
    image_rgb: np.ndarray,
    *,
    min_region_px: int = MIN_REGION_PX,
    min_coverage: float = MIN_COVERAGE,
) -> dict[str, Any]:
    """Compute deterministic texture/material statistics with explicit abstention.

    Args:
        seg2: (H, W) uint8 DOME-29 class labels aligned with image_rgb.
        image_rgb: (H, W, 3) uint8 RGB source pixels aligned with seg2.
        floors: presence / support gates.

    Returns a dict with scale-invariant texture facts only; every caption-facing
    fact is a named band / fraction ratio — the continuous gradient values, raw
    RGB, and pixel counts live in the machine-readable payload for the dossier,
    never as caption claims.
    """
    validate_seg2_array(seg2)
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise TextureError("image_rgb must be an (H, W, 3) numpy array")
    if image_rgb.shape[0] != seg2.shape[0] or image_rgb.shape[1] != seg2.shape[1]:
        raise TextureError(
            f"image_rgb {image_rgb.shape} must be pixel-aligned with seg2 {seg2.shape}"
        )

    fg_pixels = int((seg2 > 0).sum())
    if fg_pixels <= 0:
        return _abstain(
            "no foreground subject detected",
            fg_pixels=fg_pixels,
            subject_present=False,
        )

    fabric = _dominant_measurable(
        seg2, image_rgb, FABRIC_CLASSES, min_region_px, min_coverage
    )
    skin = _dominant_measurable(
        seg2, image_rgb, SKIN_CLASSES, min_region_px, min_coverage
    )

    if fabric is None and skin is None:
        return _abstain(
            "no fabric or skin region cleared the measurement gates for stable texture statistics",
            fg_pixels=fg_pixels,
        )

    result: dict[str, Any] = {
        "subject_present": True,
        "abstained": False,
        "abstention_reason": None,
        "texture_measurable": True,
        "measured_fg_px": fg_pixels,
    }
    if fabric is not None:
        name, n, coverage, stats = fabric
        result.update({
            "fabric_class": name,
            "fabric_coverage": round(coverage, 4),
            "fabric_edge_fraction": round(stats["edge_fraction"], 4),
            "fabric_deviant_fraction": round(stats["deviant_fraction"], 4),
            "fabric_texture_band": _texture_band(stats["edge_fraction"]),
            "fabric_pattern_band": _pattern_band(stats["deviant_fraction"]),
            "measured_fabric_px": n,
        })
    else:
        result.update({
            "fabric_class": None,
            "fabric_coverage": None,
            "fabric_edge_fraction": None,
            "fabric_deviant_fraction": None,
            "fabric_texture_band": None,
            "fabric_pattern_band": None,
            "measured_fabric_px": 0,
        })
    if skin is not None:
        name, n, coverage, stats = skin
        result.update({
            "skin_class": name,
            "skin_coverage": round(coverage, 4),
            "skin_edge_fraction": round(stats["edge_fraction"], 4),
            "skin_mean_gradient": round(stats["mean_gradient"], 4),
            "skin_texture_band": _skin_texture_band(stats["edge_fraction"]),
            "measured_skin_px": n,
        })
    else:
        result.update({
            "skin_class": None,
            "skin_coverage": None,
            "skin_edge_fraction": None,
            "skin_mean_gradient": None,
            "skin_texture_band": None,
            "measured_skin_px": 0,
        })
    return result


def _abstain(reason: str, **counts: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "subject_present": True,
        "abstained": True,
        "abstention_reason": reason,
        "texture_measurable": False,
        "fabric_class": None,
        "fabric_coverage": None,
        "fabric_edge_fraction": None,
        "fabric_deviant_fraction": None,
        "fabric_texture_band": None,
        "fabric_pattern_band": None,
        "skin_class": None,
        "skin_coverage": None,
        "skin_edge_fraction": None,
        "skin_mean_gradient": None,
        "skin_texture_band": None,
        "measured_fabric_px": 0,
        "measured_skin_px": 0,
    }
    # subject_present may be explicitly overridden (no foreground at all).
    result["subject_present"] = counts.pop("subject_present", True)
    result.update(counts)
    result.setdefault("measured_fg_px", 0)
    return result