"""Deterministic hair-texture / curl-waviness measurement.

Arm #94. NEW deterministic evidence part (no new model). Shows hairstyle #82
(length + arrangement) and hair #30 (color + coverage) do NOT cover texture:
'curly / wavy / straight hair' is a recurring caption claim with no texture
support. This specialist reads the existing `seg2.npy` (DOME-29 Hair mask)
plus the already-decoded source RGB pixels and emits a coarse scale-invariant
hair-texture band:

- straight / wavy / curly  (texture state of the hair region)

Signal: the magnitude-weighted circular concentration of the hair-interior
luminance-gradient orientation, folded with doubled angles so a strand line
(theta and theta+pi are the same orientation). Straight hair has one dominant
strand direction -> high concentration (R2 near 1). Curly hair has dispersed
local directions -> low concentration (R2 near 0). R2 is rotation-invariant
(camera roll rotates all angles by a constant, |mean(e^{i*2*theta})| is
unchanged) and scale-invariant (no absolute pixels in the band).

ONLY the coarse scale-invariant band is verbalized. Raw gradient statistics
(R2, orientation histogram entropy, mean gradient, pixel counts) stay in the
machine-readable `evidence_payload` and are never caption claims
(measurement-semantics directive).

Abstention: abstains when the Hair region is absent/tiny, the eroded interior
is empty, or the texture signal is unresolvable (interior mean gradient below
a floor — hair region too smooth/cropped to measure); never fabricate a curl
state. Detector disagreement is a quality anomaly, never caption content.
CPU-only, in-memory, no corpus write, no new model.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from stratum2.config import DOME_29

# DOME-29 class indices (authoritative in stratum2.config.DOME_29). Cross-checked
# in tests so this module can never silently drift from the real seg2 layout.
HAIR = DOME_29.index("Hair")

# Presence floor: a hair region must clear a raw pixel count before it is
# treated as measured (mirror hair.py / hairstyle.py).
MIN_CLASS_PX = 200

# Interior mean-gradient floor: below this the hair region is too smooth /
# texture-unresolvable (e.g. soft-focus, tiny crop, flat lighting) -> honest
# abstention rather than a fabricated curl state. Normalized luminance
# gradients are typically 0.05-0.35 on real hair; shading-only regions sit
# near 0.02-0.05. Calibrated from the frozen-cohort probe (2026-08-10).
MIN_INTERIOR_GRADIENT = 0.04

# Orientation histogram resolution (degrees per bin). 12-degree bins.
HIST_BINS = 30


class HairTextureError(RuntimeError):
    pass


def validate_seg2_array(seg2: np.ndarray) -> None:
    if not isinstance(seg2, np.ndarray):
        raise HairTextureError("seg2 must be a numpy array")
    if seg2.ndim != 2:
        raise HairTextureError(f"seg2 must be two-dimensional, got shape {seg2.shape}")
    if seg2.dtype != np.uint8 and not np.issubdtype(seg2.dtype, np.integer):
        raise HairTextureError(f"seg2 must be integer class labels, got dtype {seg2.dtype}")


def _erode1(mask: np.ndarray) -> np.ndarray:
    """1-px erosion so the silhouette boundary never counts as strand texture."""
    eroded = mask.copy()
    eroded[1:, :] &= mask[:-1, :]
    eroded[:-1, :] &= mask[1:, :]
    eroded[:, 1:] &= mask[:, :-1]
    eroded[:, :-1] &= mask[:, 1:]
    return eroded


def compute_hair_texture(
    seg2: np.ndarray,
    image_rgb: np.ndarray,
    *,
    min_px: int = MIN_CLASS_PX,
    min_interior_gradient: float = MIN_INTERIOR_GRADIENT,
) -> dict[str, Any]:
    """Compute the deterministic hair-texture band with honest abstention.

    Args:
        seg2: (H, W) DOME-29 class labels at full source resolution.
        image_rgb: (H, W, 3) uint8 RGB source pixels, pixel-aligned to seg2.
        min_px: raw-pixel floor for the Hair region.
        min_interior_gradient: interior mean-gradient floor; below it the
            region is texture-unresolvable -> abstain.

    Returns a dict with scale-invariant facts only:
    - subject_present / hair_present / abstained / abstention_reason
    - hair_texture_band (straight / wavy / curly) or None
    - raw payload: r2_concentration, orientation_entropy (nats), mean_gradient,
      interior_pixel_count
    """
    validate_seg2_array(seg2)
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise HairTextureError("image_rgb must be an (H, W, 3) numpy array")
    if image_rgb.shape[0] != seg2.shape[0] or image_rgb.shape[1] != seg2.shape[1]:
        raise HairTextureError(
            f"image_rgb {image_rgb.shape} must be pixel-aligned with seg2 {seg2.shape}"
        )

    out: dict[str, Any] = {
        "subject_present": True,
        "abstained": False,
        "abstention_reason": None,
        "hair_present": False,
        "hair_texture_band": None,
        "r2_concentration": None,
        "orientation_entropy": None,
        "mean_gradient": None,
        "interior_pixel_count": 0,
    }

    fg_pixels = int((seg2 > 0).sum())
    if fg_pixels <= 0:
        out.update({
            "subject_present": False,
            "abstained": True,
            "abstention_reason": "no foreground subject present",
        })
        return out

    hair_mask = seg2 == HAIR
    hair_px = int(hair_mask.sum())
    if hair_px < min_px:
        out.update({
            "abstained": True,
            "abstention_reason": "Hair region absent or below the raw-pixel floor",
        })
        return out
    out["hair_present"] = True

    interior = _erode1(hair_mask)
    interior_px = int(interior.sum())
    out["interior_pixel_count"] = interior_px
    if interior_px < 50:
        out.update({
            "abstained": True,
            "abstention_reason": "Hair region too thin to measure interior texture",
        })
        return out

    # Luminance gradient on the hair interior (silhouette-halo-free).
    lum = np.asarray(image_rgb, dtype=np.float32).mean(axis=2)
    lum = (lum - lum.min()) / max(float(lum.max() - lum.min()), 1e-6)
    gy, gx = np.gradient(lum)
    mag = np.hypot(gx, gy)
    interior_mag = mag[interior]
    mean_grad = float(interior_mag.mean())
    out["mean_gradient"] = round(mean_grad, 4)
    if mean_grad < min_interior_gradient:
        out.update({
            "abstained": True,
            "abstention_reason": "hair interior texture unresolvable (too smooth / cropped)",
        })
        return out

    # Orientation concentration: fold with doubled angles (strand line
    # orientation, not arrow direction), weight by gradient magnitude.
    ang = np.arctan2(gy[interior], gx[interior])
    weights = interior_mag
    total_w = float(weights.sum())
    if total_w <= 0:
        out.update({
            "abstained": True,
            "abstention_reason": "hair interior gradient degenerate",
        })
        return out
    r2 = float(
        np.hypot(
            (weights * np.cos(2.0 * ang)).sum(),
            (weights * np.sin(2.0 * ang)).sum(),
        )
        / total_w
    )
    out["r2_concentration"] = round(r2, 4)

    # Orientation histogram entropy (nats) as a payload-only second signal.
    hist, _ = np.histogram(np.degrees(ang) % 180.0, bins=HIST_BINS, range=(0.0, 180.0))
    prob = hist / float(hist.sum())
    prob = prob[prob > 0]
    out["orientation_entropy"] = round(float(-(prob * np.log(prob)).sum()), 4)

    # Band calibration (from the frozen-cohort probe, 2026-08-10):
    # straight -> high concentration, curly -> low concentration. Cuts are
    # scale-invariant and rotation-invariant; they will be re-audited by the
    # band-degeneracy rule (no band >= 75% of measured items).
    if r2 >= 0.62:
        out["hair_texture_band"] = "straight"
    elif r2 >= 0.40:
        out["hair_texture_band"] = "wavy"
    else:
        out["hair_texture_band"] = "curly"
    return out


def render_hair_texture(config: Mapping[str, Any]) -> list[str]:
    """Scale-invariant hair-texture claims for the dossier (arm #94).

    Verbalizes ONLY the coarse texture band. Raw gradient statistics (R2,
    entropy, mean gradient) stay in the machine-readable payload.
    """
    if not config:
        # Dimension not measured for this item — emit no claim, never
        # fabricate a curl state.
        return []
    if config.get("abstained"):
        reason = config.get("abstention_reason") or "hair texture not measurable"
        return [f"hair-texture: abstain ({reason})"]
    if not config.get("hair_present"):
        return ["hair-texture: abstain (no hair region present)"]
    band = config.get("hair_texture_band")
    if band == "straight":
        return ["hair-texture: hair is straight (single dominant strand direction)"]
    if band == "wavy":
        return ["hair-texture: hair is wavy (moderately dispersed strand directions)"]
    if band == "curly":
        return ["hair-texture: hair is curly (strongly dispersed strand directions)"]
    return []