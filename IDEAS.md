# IDEAS

What can be built on top of stratum-hq's pre-computed multimodal signals?
Generated from a brainstorming session. Organized by effort and ambition.

---

## The Data

Every image processed through stratum-hq produces:

| Signal | Shape | Description |
|--------|-------|-------------|
| `caption.txt` | text | Dense, objective paragraph (Gemma 3 27B) |
| `dinov3_cls.npy` | (1024,) float16 | Global style/composition embedding |
| `dinov3_patches.npy` | (N, 1024) float16 | Spatial patch tokens |
| `t5_hidden.npy` | (512, 1024) float16 | T5-Large hidden states |
| `t5_mask.npy` | (512,) uint8 | T5 attention mask |
| `pose.npy` | (133, 3) float16 | DWPose whole-body keypoints |
| `seg.npy` | (H, W) uint8 | 28-class body-part segmentation |
| `depth.npy` | (H, W) float16 | Foreground-masked relative depth |
| `normal.npy` | (H, W, 3) float16 | Surface normals (L2-normalized) |
| `pixel.npy` | (3, H, W) float16 | Bucketed RGB crop (opt-in) |

Everything is pre-computed, disk-based, and aligned to the same bucketed image.

---

## Models to Improve

### FLUME — FLUX fine-tune with native T5 conditioning
stratum already has T5-Large hidden states (512×1024). FLUX uses T5-XXL natively.
The gap is dimension mismatch, not architecture mismatch. A small projection layer
(1024→4096) lets you feed stratum's pre-computed T5 embeddings directly into a
frozen FLUX model for fine-tuning. Skip the T5 forward pass entirely — same trick
as prx-tg but applied to a production-grade model. 70k FFHQ stratum samples + FLUX
= a face model with zero text-encoder cost during training.

### SDXL with Disentangled Cross-Attention
The DEADiff paper (arXiv 2403.06951) showed cross-attention layers can be split so
different heads attend to style vs content. With stratum, DINO CLS (style) and
caption text (content) are already separated. Train SDXL cross-attention adapters
where half the heads bind to CLIP text (content) and half bind to projected DINO CLS
(style). At inference: one prompt, one reference image, no CFG juggling.

### Sapiens-to-SDXL Distillation
stratum's segmentation, depth, and normals all come from Sapiens-1B — a teacher
model's understanding of human anatomy. Train an SDXL-based student to internalize
that understanding: add auxiliary heads that predict segmentation/depth/normals from
the UNet's intermediate features. The model learns what a human body *is*, not just
what it looks like. Result: anatomically coherent generations even for unusual poses,
fewer extra limbs, correct joint articulation.

### Multi-Modal Conditioning Adapters for SDXL
Three new conditioning streams, each injected differently into a frozen SDXL UNet:
- **Style (DINO CLS)**: project 1024→1280, add to timestep embedding. Influences
  every block's modulation parameters. Global property, minimal params.
- **Pose (DWPose)**: 133 joint tokens with per-joint-type embeddings (body, feet,
  face, hands) + MLP projection of (x, y, confidence). Cross-attention in every
  transformer block, parallel to text cross-attention.
- **Layout (DINO patches)**: variable-length sequence, project 1024→768.
  Cross-attention in every block. Higher dropout during training (strongest signal).

Training dropout: style 15%, pose 10%, patches 40%, text 10%. Joint training with
shared dropout means the model learns to resolve conflicts between conditioning
signals — something three separately trained adapters (ControlNet + IP-Adapter + ...)
cannot do.

### Control-DINO for Image Generation
The Control-DINO paper (arXiv 2604.01761) uses DINOv3 features as ControlNet
conditioning for image-to-video. The same approach works for image generation:
use stratum's `dinov3_patches.npy` as the ControlNet conditioning signal. The
patches encode spatial layout — where things are relative to each other — in a
way that's richer than Canny edges or depth maps. Train a DINOv3 ControlNet for
SDXL or FLUX using stratum's pre-computed patches.

---

## LoRAs to Train

### Pose-Disentangled LoRA
Cluster the dataset by pose similarity (cosine distance on flattened keypoint
vectors from `pose.npy`). Train one LoRA per pose cluster. At inference: pick the
LoRA matching your desired pose, prompt normally. The LoRA encodes "how this model
poses" as a learned bias in the UNet weights, not as explicit conditioning.
Orthogonal to ControlNet — you can stack them.

### Anatomical Correctness LoRA
Use the Sapiens 28-class segmentation to create a loss that penalizes anatomically
impossible configurations. If the UNet produces a latent that, when decoded, shows
a hand where a foot should be, the segmentation head flags it. Train a LoRA that
minimizes this penalty. The LoRA becomes an "anatomy corrector" you can apply to
any SDXL generation.

### Depth-Aware Portrait LoRA
stratum depth maps encode foreground/background separation at the pixel level.
Train a LoRA where the loss function weights foreground pixels higher than
background. The model learns to allocate its capacity to the subject. Result:
sharper faces, blurrier backgrounds — the model learns the portrait photography
prior from data rather than from architecture.

### Style-Transfer LoRA from DINO CLS Clusters
Cluster images by DINO CLS similarity. Each cluster represents a coherent style
(color palette, lighting, composition). Train one LoRA per cluster. At inference:
pick the LoRA that matches your desired aesthetic. No reference image needed at
generation time — the style is baked into the weights.

---

## Plugins and Tools

### ComfyUI "Stratum Conditioner" Node
A single node that takes: a text prompt, an optional style reference image, an
optional pose skeleton (from OpenPose or DWPose), and an optional segmentation map.
Internally, it computes DINOv3 from the reference, encodes the pose, and feeds all
three into a multi-modal model. Makes research models usable by the ComfyUI
community immediately.

### Stratum Dataset Explorer / Curator
Navigate a stratum dataset by semantic dimensions:
- Pose similarity: vector DB on keypoints. "Show me all images with arms crossed."
- Style similarity: vector DB on DINO CLS. "Show me all images with warm lighting."
- Caption search: full-text search on `caption.txt`
- Body part presence: query by segmentation class presence
Useful for dataset curation AND for finding the perfect conditioning references
for generation.

### Pose-to-Prompt Reverse Mapper
Given a DWPose skeleton, what prompts produce this pose? Train a small model (or
use the existing caption data) to map pose → likely captions. At inference:
draw a stick figure, get a suggested prompt, generate. Bridges the gap between
visual posing and text prompting.

### Stratum-to-Model Adapter Framework
A library where you configure "I want to train model X using signals A, B, C from
stratum" and it wires up:
- The data loader (StratumDataset with correct signal subset)
- The conditioning projections (linear/MLP layers)
- The CFG dropout schedule (per-signal probabilities)
- The loss function (flow matching, MSE, hybrid)
Everything else — the specific LoRAs, the specific models, the specific plugins —
are instantiations of this framework.

---

## Research Directions

### Synthetic Data Engine
stratum produces (image, caption, pose, seg, depth, normals) tuples. This is
ground truth for training other models. Generate synthetic training data for:
- Human parsing models (segmentation)
- Monocular depth estimation (depth)
- Surface normal prediction (normals)
- Pose estimation (keypoints)
stratum becomes a data engine — the enrichment pipeline runs once, and every
downstream task benefits.

### Style-Consistent Multi-Image Generation
Take two images from stratum with different poses but similar DINO CLS embeddings
(same style). Can you generate a third image with a novel pose that preserves the
style? This is few-shot personalization without test-time fine-tuning. The DINO
CLS space already clusters by style — you're learning to sample from that cluster.

### Token-Level Text-to-Region Alignment
stratum has captions AND segmentation maps. Learn which words in the caption
correspond to which body parts in the segmentation. "left hand resting on hip" →
the model learns that "left hand" tokens attend to the left-hand segmentation
region. This enables text-based regional editing without manual masks. "Make the
left hand a fist" — the model knows where the left hand is from text alone.

### Pose-Conditioned Depth Estimation
stratum has both `pose.npy` and `depth.npy`. Predict depth from pose alone. If
successful, you can generate a plausible depth map for any pose skeleton, then use
that depth map as conditioning for image generation. The chain becomes:
pose → depth → image. Depth is the intermediate representation that bridges
skeletal pose and photorealistic rendering — essentially a learned graphics
pipeline.

### Cross-Dataset Style Transfer Benchmark
FFHQ (faces) vs photo-models (full body). Same stratum format, different domains.
Use DINO CLS from one domain + pose from the other → generate. Does the model
learn domain-invariant representations? This is a clean testbed for studying
domain gap in generative models — you control exactly which signals come from
which domain.

### Privacy-Preserving Dataset Release Study
stratum-ffhq doesn't include original images. But it includes DINOv3 patches —
a spatial grid of 1024-dim embeddings that captures enough structural information
to reconstruct approximations of faces. How much can you reconstruct from DINO
patches alone? What about from DINO CLS + pose + seg? This is a concrete study
in the privacy implications of feature-sharing. stratum gives you the tool to run
this experiment.

---

## The Eidolon Connection

### Project Eidolon v2: The Person Vector
[Project Eidolon](https://github.com/timlawrenz/eidolon) attempted to describe a
person (face, albedo, body) in a single vector using FLAME 3D morphable model +
ResNet-50 encoder. stratum provides everything eidolon needed but pre-computed
and at scale:

- DWPose 133 keypoints (includes 68 face landmarks = eidolon's original signal)
- DINOv3 CLS (already a person descriptor, just not geometrically interpretable)
- Body-part segmentation, depth, normals (spatial understanding beyond landmarks)

The fusion architecture:

```
pixel.npy ────┐
dinov3_cls ───┤ (teacher)
pose.npy ─────┤
seg.npy ──────┼──► Transformer Encoder ──► Person Vector
depth.npy ────┤                             = [FLAME(300) + SMPL(85)
normal.npy ───┘                               + albedo(512) + appearance(1024)]
```

Three advances over the original:
1. Face-only → full-body (SMPL-X instead of just FLAME)
2. DINOv3 as teacher loss (person vector aligns with pretrained visual space)
3. The person vector becomes a unified conditioning token for any diffusion model

### Eidolon PoC
The minimal proof of concept:
1. Train a tiny ConvNext/patch encoder: `pixel.npy → [dino_cls, pose_kpts]`
2. Concatenate: `person_vec = [dino_cls | flatten(pose)]` = 1423-dim
3. Feed person vector as conditioning into a diffusion model (e.g., prx-tg's NanoDiT)
4. Demonstrate interpolation: lerp between two person vectors → coherent mid-person
5. Demonstrate cross-signal prediction: pose → style, style → pose

---

## The Thread

stratum's core insight: pre-computed multimodal embeddings are a *lingua franca*.
Once you have (text, style, pose, anatomy, depth, normals) for every image:

- **Train** any model using any subset of these signals
- **Compose** signals across domains (FFHQ style + photo-model pose)
- **Evaluate** models on fine-grained dimensions (style preservation, pose accuracy,
  anatomical coherence)
- **Generate** training data for downstream tasks (the pipeline IS the data engine)

The most leveraged thing to build: the **stratum-to-model adapter framework**.
Everything else — LoRAs, models, plugins, research papers — are instantiations.