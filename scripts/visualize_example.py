#!/usr/bin/env python3
"""Generate overlay visualizations of stratum dataset layers on source images.

Usage:
    python scripts/visualize_example.py \\
        --image /path/to/raw/00028.png \\
        --stratum-dir /path/to/stratum/00028 \\
        --output examples/00028_combined.png

Produces a 2×2 panel: pose overlay, caption panel, DINOv3 heatmap, T5 mask chart.
Individual overlays can also be saved with --save-individual.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# COCO-WholeBody skeleton definitions
# ---------------------------------------------------------------------------

BODY_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (5, 6),                                   # shoulders
    (5, 7), (7, 9),                           # left arm
    (6, 8), (8, 10),                          # right arm
    (5, 11), (6, 12),                         # torso
    (11, 12),                                 # hips
    (11, 13), (13, 15),                       # left leg
    (12, 14), (14, 16),                       # right leg
]

FEET_SKELETON = [
    (15, 17), (15, 18), (15, 19), (17, 18),  # left foot
    (16, 20), (16, 21), (16, 22), (20, 21),  # right foot
]

HAND_SKELETON_OFFSETS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Color palette for different body parts
COLOR_BODY = (0, 255, 128)
COLOR_HEAD = (255, 200, 0)
COLOR_FEET = (0, 200, 255)
COLOR_FACE = (200, 180, 255)
COLOR_HAND_L = (255, 128, 128)
COLOR_HAND_R = (128, 200, 255)

CONF_THRESHOLD = 0.3


def denormalize(pose: np.ndarray, w: int, h: int) -> np.ndarray:
    """Convert [-1, 1] normalized coords to pixel coords."""
    out = pose.copy()
    out[:, 0] = (pose[:, 0] + 1.0) * w / 2.0
    out[:, 1] = (pose[:, 1] + 1.0) * h / 2.0
    return out


def draw_skeleton(draw: ImageDraw.Draw, kpts: np.ndarray, connections: list[tuple[int, int]],
                  color: tuple, width: int = 2, point_r: int = 3):
    """Draw skeleton connections and keypoints."""
    for i, j in connections:
        if i >= len(kpts) or j >= len(kpts):
            continue
        if kpts[i, 2] < CONF_THRESHOLD or kpts[j, 2] < CONF_THRESHOLD:
            continue
        x1, y1 = float(kpts[i, 0]), float(kpts[i, 1])
        x2, y2 = float(kpts[j, 0]), float(kpts[j, 1])
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)

    for idx in set(i for pair in connections for i in pair):
        if idx >= len(kpts) or kpts[idx, 2] < CONF_THRESHOLD:
            continue
        x, y = float(kpts[idx, 0]), float(kpts[idx, 1])
        draw.ellipse([x - point_r, y - point_r, x + point_r, y + point_r], fill=color)


def render_pose_overlay(img: Image.Image, pose: np.ndarray) -> Image.Image:
    """Draw COCO-WholeBody skeleton on the image."""
    overlay = img.copy()
    kpts = denormalize(pose, img.width, img.height)
    draw = ImageDraw.Draw(overlay)

    # Body
    head_conns = [(0, 1), (0, 2), (1, 3), (2, 4)]
    body_conns = [c for c in BODY_SKELETON if c not in head_conns]
    draw_skeleton(draw, kpts, head_conns, COLOR_HEAD, width=2, point_r=3)
    draw_skeleton(draw, kpts, body_conns, COLOR_BODY, width=3, point_r=4)

    # Feet
    draw_skeleton(draw, kpts, FEET_SKELETON, COLOR_FEET, width=2, point_r=2)

    # Face landmarks (just dots, no skeleton — too dense)
    for idx in range(23, 91):
        if idx < len(kpts) and kpts[idx, 2] > CONF_THRESHOLD:
            x, y = float(kpts[idx, 0]), float(kpts[idx, 1])
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=COLOR_FACE)

    # Hands
    for hand_start, color in [(91, COLOR_HAND_L), (112, COLOR_HAND_R)]:
        hand_conns = [(a + hand_start, b + hand_start) for a, b in HAND_SKELETON_OFFSETS]
        draw_skeleton(draw, kpts, hand_conns, color, width=1, point_r=2)

    return overlay


def render_caption_panel(img: Image.Image, caption: str) -> Image.Image:
    """Image with caption text panel below, sized to fit the full text."""
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    margin = 16
    spacing = 4
    wrap_width = max(60, int(img.width / 9.5))
    wrapped = textwrap.fill(caption, width=wrap_width)

    # Measure text height
    tmp = Image.new("RGB", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp)
    bbox = tmp_draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing)
    text_h = bbox[3] - bbox[1]
    panel_h = text_h + margin * 2

    result = Image.new("RGB", (img.width, img.height + panel_h), (30, 30, 30))
    result.paste(img, (0, 0))
    draw = ImageDraw.Draw(result)
    draw.multiline_text((margin, img.height + margin), wrapped, fill=(240, 240, 240),
                        font=font, spacing=spacing)
    return result


def render_dino_heatmap(img: Image.Image, cls_emb: np.ndarray, patch_emb: np.ndarray) -> Image.Image:
    """Overlay DINOv3 CLS-to-patch cosine similarity as a heatmap.

    Uses percentile-based normalization for better contrast and a stronger
    blend so the heatmap is clearly visible over the face.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Cosine similarity: CLS vs each patch
    cls_norm = cls_emb / (np.linalg.norm(cls_emb) + 1e-8)
    patch_norms = patch_emb / (np.linalg.norm(patch_emb, axis=1, keepdims=True) + 1e-8)
    similarity = patch_norms @ cls_norm  # (N_patches,)

    # Reshape to 2D grid — patches = (H/16) × (W/16)
    n_patches = len(similarity)
    grid_size = int(np.sqrt(n_patches))
    if grid_size * grid_size != n_patches:
        grid_size = int(np.ceil(np.sqrt(n_patches)))
        padded = np.zeros(grid_size * grid_size)
        padded[:n_patches] = similarity
        similarity = padded
    heatmap = similarity.reshape(grid_size, grid_size)

    # Percentile-based normalization for better contrast
    lo = np.percentile(heatmap, 2)
    hi = np.percentile(heatmap, 98)
    heatmap = np.clip((heatmap - lo) / (hi - lo + 1e-8), 0, 1)

    # Expand heatmap to image dimensions using block repetition so each
    # patch maps exactly to its 16×16 pixel region (no interpolation shift).
    patch_h = img.height // grid_size
    patch_w = img.width // grid_size
    colored = cm.inferno(heatmap)[:, :, :3]
    colored_uint8 = (colored * 255).astype(np.uint8)
    # Repeat each row/col by the patch size
    full = np.repeat(np.repeat(colored_uint8, patch_h, axis=0), patch_w, axis=1)
    # Handle rounding remainder (e.g. 1024 not perfectly divisible)
    if full.shape[0] != img.height or full.shape[1] != img.width:
        heatmap_img = Image.fromarray(full).resize((img.width, img.height), Image.NEAREST)
    else:
        heatmap_img = Image.fromarray(full)

    # Stronger blend so the heatmap reads clearly
    overlay = Image.blend(img.convert("RGB"), heatmap_img, alpha=0.6)

    # Add colorbar legend
    fig, ax = plt.subplots(figsize=(0.4, 4), dpi=100)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap="inferno"), cax=ax)
    ax.set_ylabel("CLS→patch similarity (normalized)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.05, right=0.55, top=0.95, bottom=0.05)
    fig.canvas.draw()

    bar_arr = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    bar_arr = bar_arr.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    bar_img = Image.fromarray(bar_arr[:, :, :3])
    plt.close(fig)

    result = Image.new("RGB", (overlay.width + bar_img.width + 8, overlay.height), (30, 30, 30))
    result.paste(overlay, (0, 0))
    bar_y = (overlay.height - bar_img.height) // 2
    result.paste(bar_img, (overlay.width + 4, bar_y))

    return result


def render_t5_mask_chart(img: Image.Image, t5_mask: np.ndarray, caption: str,
                         t5_hidden: np.ndarray | None = None) -> Image.Image:
    """Visualize T5 encoding: per-token hidden-state magnitudes and mask boundary.

    When *t5_hidden* is provided, shows the L2 norm of each token's hidden state
    as a bar chart — real tokens in colour (viridis), padding in dark grey.
    Falls back to a simple mask bar chart if hidden states are unavailable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    n_real = int(t5_mask.sum())
    n_pad = len(t5_mask) - n_real

    fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100), dpi=100)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    if t5_hidden is not None:
        norms = np.linalg.norm(t5_hidden.astype(np.float32), axis=1)

        # Colour real tokens by magnitude, padding in grey
        real_norms = norms[:n_real]
        lo, hi = real_norms.min(), real_norms.max()
        norm_scaled = (real_norms - lo) / (hi - lo + 1e-8)
        cmap = matplotlib.colormaps["viridis"]
        colors = [cmap(v) for v in norm_scaled] + ["#2a2a2a"] * n_pad

        ax.bar(range(len(norms)), norms, color=colors, width=1.0, edgecolor="none")
        ax.axvline(x=n_real - 0.5, color="#ff5555", linewidth=1.5, linestyle="--", alpha=0.8)
        ax.text(n_real + 2, norms[:n_real].max() * 0.95, "← padding",
                color="#ff5555", fontsize=9, verticalalignment="top")
        ax.set_ylabel("Token hidden-state ‖h‖₂", color="white", fontsize=10)
        ax.set_title(f"T5 Encoding — {n_real} real tokens, {n_pad} padding  (hidden-state magnitudes)",
                     color="white", fontsize=11, pad=10)
    else:
        colors = ["#4CAF50" if m else "#333333" for m in t5_mask]
        ax.bar(range(len(t5_mask)), t5_mask.astype(float), color=colors, width=1.0, edgecolor="none")
        ax.set_ylim(0, 1.3)
        ax.set_title(f"T5 Attention Mask — {n_real} real tokens, {n_pad} padding",
                     color="white", fontsize=11, pad=10)

    ax.set_xlim(0, len(t5_mask))
    ax.set_xlabel("Token position", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#555555")

    short_caption = caption[:120] + ("..." if len(caption) > 120 else "")
    ax.text(0.02, 0.97, f'"{short_caption}"', transform=ax.transAxes,
            fontsize=8, color="#aaaaaa", style="italic", verticalalignment="top",
            wrap=True)

    fig.tight_layout()
    fig.canvas.draw()

    arr = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    arr = arr.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    result = Image.fromarray(arr[:, :, :3])
    plt.close(fig)

    return result


def render_combined_panel(pose_img: Image.Image, caption_img: Image.Image,
                          dino_img: Image.Image, t5_img: Image.Image,
                          target_size: int = 1024) -> Image.Image:
    """Assemble 4 panels into a 2×2 grid with labels."""
    cell_w = target_size
    cell_h = target_size
    padding = 4
    label_h = 28

    def fit(img: Image.Image) -> Image.Image:
        """Resize to fit cell while preserving aspect ratio, center on black."""
        ratio = min(cell_w / img.width, cell_h / img.height)
        new_w = int(img.width * ratio)
        new_h = int(img.height * ratio)
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        cell = Image.new("RGB", (cell_w, cell_h), (30, 30, 30))
        cell.paste(resized, ((cell_w - new_w) // 2, (cell_h - new_h) // 2))
        return cell

    panels = [
        ("Pose Keypoints (COCO-WholeBody)", pose_img),
        ("Caption (Gemma 3 27B)", caption_img),
        ("DINOv3 Patch Attention", dino_img),
        ("T5 Attention Mask", t5_img),
    ]

    grid_w = cell_w * 2 + padding * 3
    grid_h = (cell_h + label_h) * 2 + padding * 3
    result = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))
    draw = ImageDraw.Draw(result)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for (col, row), (label, img) in zip(positions, panels):
        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + label_h + padding)
        draw.text((x + 8, y + 4), label, fill=(200, 200, 200), font=font)
        fitted = fit(img)
        result.paste(fitted, (x, y + label_h))

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True, help="Source image (PNG)")
    parser.add_argument("--stratum-dir", type=Path, required=True, help="Stratum image directory")
    parser.add_argument("--output", type=Path, required=True, help="Output combined panel PNG")
    parser.add_argument("--save-individual", action="store_true", help="Also save individual overlays")
    args = parser.parse_args()

    # Load source image
    img = Image.open(args.image).convert("RGB")
    sd = args.stratum_dir

    # Load artifacts
    pose = np.load(sd / "pose.npy")
    caption = (sd / "caption.txt").read_text(encoding="utf-8").strip()
    dinov3_cls = np.load(sd / "dinov3_cls.npy")
    dinov3_patches = np.load(sd / "dinov3_patches.npy")
    t5_mask = np.load(sd / "t5_mask.npy")
    t5_hidden_path = sd / "t5_hidden.npy"
    t5_hidden = np.load(t5_hidden_path) if t5_hidden_path.exists() else None

    # Render panels
    pose_img = render_pose_overlay(img, pose)
    caption_img = render_caption_panel(img, caption)
    dino_img = render_dino_heatmap(img, dinov3_cls, dinov3_patches)
    t5_img = render_t5_mask_chart(img, t5_mask, caption, t5_hidden)

    # Combined
    combined = render_combined_panel(pose_img, caption_img, dino_img, t5_img)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.save(args.output, quality=95)
    print(f"Saved combined panel: {args.output} ({combined.width}×{combined.height})")

    if args.save_individual:
        stem = args.output.stem
        parent = args.output.parent
        for name, panel in [("pose", pose_img), ("caption", caption_img),
                            ("dino", dino_img), ("t5", t5_img)]:
            path = parent / f"{stem}_{name}.png"
            panel.save(path, quality=95)
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
