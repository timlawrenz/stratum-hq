This dataset is a derivative of the **Flickr-Faces-HQ (FFHQ)** dataset by
Tero Karras, Samuli Laine, and Timo Aila (NVIDIA), released under
[Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

### What this dataset contains

This dataset provides **pre-computed embeddings, captions, and pose estimates**
derived from the 70,000 aligned face images in FFHQ. It does **not** contain
the original images. To use this dataset you must obtain the original FFHQ
images separately from [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset).

### Changes made to the source material

The following artifacts were generated from each FFHQ image using
[stratum-hq](https://github.com/timlawrenz/stratum-hq):

| Artifact | Description | Model |
|----------|-------------|-------|
| `dinov3_cls.npy` | Global image embedding (1024-d, float16) | DINOv3 ViT-L/16 |
| `dinov3_patches.npy` | Per-patch embeddings (N×1024, float16) | DINOv3 ViT-L/16 |
| `pose.npy` | 133 COCO-WholeBody keypoints (133×3, float16) | DWPose |
| `caption.txt` | Natural-language image description | Gemma 3 27B via Ollama |
| `t5_hidden.npy` | Text encoder hidden states (512×1024, float16) | T5-Large |
| `t5_mask.npy` | T5 attention mask (512, uint8) | T5-Large |
| `seg.npy` | 28-class body-part segmentation (H×W, uint8) | Sapiens-1B |
| `depth.npy` | Relative depth, foreground-masked (H×W, float16) | Sapiens-1B |
| `normal.npy` | Surface normals (H×W×3, float16) | Sapiens-1B |

No original pixel data is distributed in this dataset.

### Caption generation

Captions were generated with
[**Gemma 3 27B**](https://huggingface.co/google/gemma-3-27b-it) served locally
via [Ollama](https://ollama.com/) (`gemma3:27b`). Each image was captioned with
the following system prompt:

> Generate a single, dense paragraph describing this image for a text-to-image
> training dataset. Write in a strictly dry, objective, and descriptive tone.
> Do not use flowery language, subjective interpretations, or lists.
> Describe only what is visible: subject (including specific body build, muscle
> definition, skin texture, and visible anatomical landmarks), precise pose
> (mechanics of limb positioning, hand placement), clothing/accessories, lighting,
> background, composition/framing, and camera angle.
> Do not guess measurements (height, weight) or internal anatomy not visible.
> Do not include any conversational filler, preambles (like 'The image shows...'),
> or meta-commentary. Start the description immediately.

### Example Overlays

The images below illustrate each data layer by overlaying it on a sample face.
These visualizations were generated with
[`scripts/visualize_example.py`](https://github.com/timlawrenz/stratum-hq/blob/main/scripts/visualize_example.py).

| Layer | Overlay |
|-------|---------|
| **Pose** (COCO-WholeBody skeleton) | ![pose overlay](https://raw.githubusercontent.com/timlawrenz/stratum-hq/main/examples/00028_combined_pose_keypoints.png) |
| **Caption** (Gemma 3 27B) | ![caption overlay](https://raw.githubusercontent.com/timlawrenz/stratum-hq/main/examples/00028_combined_caption.png) |
| **DINOv3** (CLS→patch attention) | ![dino heatmap](https://raw.githubusercontent.com/timlawrenz/stratum-hq/main/examples/00028_combined_dinov3_patch_attention.png) |
| **T5** (token attention mask) | ![t5 mask chart](https://raw.githubusercontent.com/timlawrenz/stratum-hq/main/examples/00028_combined_t5_attention_mask.png) |
| **Segmentation** (Sapiens body parts) | ![segmentation overlay](https://raw.githubusercontent.com/timlawrenz/stratum-hq/main/examples/00028_combined_body_part_segmentation.png) |
| **Depth** (Sapiens depth estimation) | ![depth heatmap](https://raw.githubusercontent.com/timlawrenz/stratum-hq/main/examples/00028_combined_depth_estimation.png) |
| **Surface Normals** (Sapiens) | ![normal map](https://raw.githubusercontent.com/timlawrenz/stratum-hq/main/examples/00028_combined_surface_normals.png) |

**Combined panel** (all four layers on three diverse FFHQ subjects):

| | | |
|---|---|---|
| ![combined 00028](https://raw.githubusercontent.com/timlawrenz/stratum-hq/main/examples/00028_combined.png) | ![combined 01000](https://raw.githubusercontent.com/timlawrenz/stratum-hq/main/examples/01000_combined.png) | ![combined 00010](https://raw.githubusercontent.com/timlawrenz/stratum-hq/main/examples/00010_combined.png) |

### Per-image licensing

The individual FFHQ images were published on Flickr under one of the following licenses:

- [Creative Commons BY 2.0](https://creativecommons.org/licenses/by/2.0/)
- [Creative Commons BY-NC 2.0](https://creativecommons.org/licenses/by-nc/2.0/)
- [Public Domain Mark 1.0](https://creativecommons.org/publicdomain/mark/1.0/)
- [Public Domain CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
- [U.S. Government Works](http://www.usa.gov/copyright.shtml)

The license and original author of each image are recorded in NVIDIA's official
metadata file **`ffhq-dataset-v2.json`** (255 MB), available from the
[FFHQ dataset repository](https://github.com/NVlabs/ffhq-dataset).

### Citation

If you use this dataset, please cite the original FFHQ paper:

```bibtex
@inproceedings{karras2019style,
  title     = {A Style-Based Generator Architecture for Generative Adversarial Networks},
  author    = {Karras, Tero and Laine, Samuli and Aila, Timo},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and
               Pattern Recognition (CVPR)},
  year      = {2019}
}
```
