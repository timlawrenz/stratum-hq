import sys
import os
from pathlib import Path
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from visualize_example import render_combined_panel

examples_dir = Path("../examples")
blog_dir = Path("../blog/images")
blog_dir.mkdir(parents=True, exist_ok=True)

# The individual panels available
panels_map = {
    "Pose": "pose_keypoints.png",
    "Caption": "caption.png",
    "DINOv3": "dinov3_patch_attention.png",
    "T5 Mask": "t5_attention_mask.png",
    "Segmentation": "body_part_segmentation.png",
    "Depth": "depth_estimation.png",
    "Normals": "surface_normals.png",
}

bases = ["00010", "00028", "01000"]

# We will create various combinations and layouts to give the user choices
variants = [
    ("all", ["Pose", "Caption", "DINOv3", "T5 Mask", "Segmentation", "Depth", "Normals"]),
    ("vision_only", ["DINOv3", "Segmentation", "Depth", "Normals"]),
    ("spatial_only", ["Pose", "Segmentation", "Depth", "Normals"]),
    ("language_focus", ["Caption", "T5 Mask", "DINOv3", "Pose"]),
    ("geometry", ["Segmentation", "Depth", "Normals"]),
]

for base in bases:
    for var_name, keys in variants:
        panels = []
        for k in keys:
            path = examples_dir / f"{base}_combined_{panels_map[k]}"
            if path.exists():
                panels.append((k, Image.open(path)))
        
        if panels:
            # We can monkeypatch n_cols in visualize_example if we wanted, 
            # but let's just use the updated render_combined_panel
            grid = render_combined_panel(panels, target_size=800)
            out_path = blog_dir / f"{base}_{var_name}.png"
            grid.save(out_path, quality=95)
            print(f"Generated {out_path}")

