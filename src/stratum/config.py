"""Configuration constants and defaults for stratum."""

from __future__ import annotations

# --- Model IDs ---
DINO_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
T5_MODEL_ID = "t5-large"

# --- Pose ---
NUM_POSE_KEYPOINTS = 133  # COCO-WholeBody: 17 body + 6 feet + 68 face + 42 hands

# --- Sapiens (segmentation, depth, surface normals) ---
SAPIENS_SEG_REPO = "facebook/sapiens-seg-1b-torchscript"
SAPIENS_SEG_FILENAME = "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"
SAPIENS_DEPTH_REPO = "facebook/sapiens-depth-1b-torchscript"
SAPIENS_DEPTH_FILENAME = "sapiens_1b_render_people_epoch_88_torchscript.pt2"
SAPIENS_NORMAL_REPO = "facebook/sapiens-normal-1b-torchscript"
SAPIENS_NORMAL_FILENAME = "sapiens_1b_normal_render_people_epoch_115_torchscript.pt2"
NUM_SEG_CLASSES = 28
SAPIENS_INPUT_HEIGHT = 1024
SAPIENS_INPUT_WIDTH = 768

# --- Aspect ratio buckets ---
# All ~1 megapixel, all dims divisible by 64.
DEFAULT_ASPECT_BUCKETS: list[tuple[int, int]] = [
    (1024, 1024),  # Square, ratio 1.0
    (832, 1216),   # Portrait, ratio ~0.68
    (1216, 832),   # Landscape, ratio ~1.46
    (768, 1280),   # Tall portrait, ratio 0.6
    (1280, 768),   # Wide landscape, ratio ~1.67
    (704, 1344),   # Very tall, ratio ~0.52
    (1344, 704),   # Very wide, ratio ~1.91
]

# --- Caption prompt ---
CAPTION_PROMPT = (
    "Generate a single, dense paragraph describing this image for a text-to-image "
    "training dataset. Write in a strictly dry, objective, and descriptive tone. "
    "Do not use flowery language, subjective interpretations, or lists. "
    "Describe only what is visible: subject (including specific body build, muscle "
    "definition, skin texture, and visible anatomical landmarks), precise pose "
    "(mechanics of limb positioning, hand placement), clothing/accessories, lighting, "
    "background, composition/framing, and camera angle. "
    "Do not guess measurements (height, weight) or internal anatomy not visible. "
    "Do not include any conversational filler, preambles (like 'The image shows...'), "
    "or meta-commentary. Start the description immediately."
)

# --- Artifact filenames ---
METADATA_FILE = "metadata.json"
CAPTION_FILE = "caption.txt"
DINOV3_CLS_FILE = "dinov3_cls.npy"
DINOV3_PATCHES_FILE = "dinov3_patches.npy"
T5_HIDDEN_FILE = "t5_hidden.npy"
T5_MASK_FILE = "t5_mask.npy"
PIXEL_FILE = "pixel.npy"
POSE_FILE = "pose.npy"
SEG_FILE = "seg.npy"
DEPTH_FILE = "depth.npy"
NORMAL_FILE = "normal.npy"

# --- Supported image extensions ---
IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif",
})
