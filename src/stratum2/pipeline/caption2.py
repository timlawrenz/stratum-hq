"""Generate rich captions from geometry determinations via Ollama."""

import json
from pathlib import Path

# The base prompt instructs the VLM on its role.
CAPTION2_PROMPT_TEMPLATE = """You are an expert descriptive captioner for a text-to-image dataset.
Your task is to write a single, rich, dense paragraph describing the provided image.

Below is a block of DETERMINATIONS extracted deterministically from the image's geometry.
These are ground truth. You must NEVER contradict them.

DETERMINATIONS:
{determinations_text}

Your job is to VERBALIZE the geometry and ADD what the determinations omit:
1. Subject & Pose: Translate the measured relations (e.g., facing, limb positions) into natural prose.
2. Semantics: Name the posture or activity if obvious (e.g., cartwheel, kneeling, ballet) consistent with the geometry.
3. Visuals: Describe mood, lighting quality, color palette, fabric, texture, skin details, and expression.
4. Background: Describe the setting and environment.

Write strictly objective prose. No conversational filler, no preambles like "This image shows". Start the description immediately.
"""


def build_prompt(determinations_json: dict) -> str:
    """Render the determinations dict into a bulleted text block for the prompt."""
    lines = []

    # 1. Subject & Extent
    subj = determinations_json.get("subject", {})
    ext = determinations_json.get("subject_extent", {})
    if subj.get("n_detections", 1) == 1:
        lines.append("- exactly one primary subject detected")
    else:
        lines.append(f"- detector anomaly: {subj.get('detector_anomaly')}")

    if "h_position" in ext:
        lines.append(f"- subject horizontal position: {ext['h_position']}")

    # 2. Body parts
    parts = determinations_json.get("body_parts_visible", [])
    if parts:
        part_names = [p["part"] for p in parts]
        lines.append(f"- visible body regions: {', '.join(part_names)}")

    # 3. Orientation
    ori = determinations_json.get("orientation", {})
    if "upright_deg" in ori:
        deg = ori["upright_deg"]
        lines.append(
            f"- torso upright angle: {deg} degrees (0=upright, 90=horizontal, 180=inverted)"
        )

    # 4. Relations
    rels = determinations_json.get("relations", [])
    if rels:
        lines.append("- geometric relations:")
        for r in rels:
            lines.append(f"  * {r}")

    # Fallback if nothing
    if not lines:
        lines.append("- (no geometric determinations available)")

    determinations_text = "\n".join(lines)
    return CAPTION2_PROMPT_TEMPLATE.format(determinations_text=determinations_text)


def process(
    image_path: Path,
    output_dir: Path,
    ollama_url: str = "http://192.168.86.137:11434/api/generate",
    ollama_model: str = "gemma3:27b",
    **kwargs,
) -> bool:
    """To be implemented in Phase 4"""
    pass
