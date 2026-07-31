---
title: "Introducing Stratum-FFHQ: A Multi-Modal Enriched Face Dataset"
thumbnail: "thumbnail_1200x648.png"
authors:
- user: timlawrenz
---

# Introducing Stratum-FFHQ: A Multi-Modal Enriched Face Dataset

High-quality datasets are the lifeblood of modern generative AI, but raw pixels alone are no longer enough. To train advanced diffusion models, ControlNets, and multi-modal systems, you need rich, aligned contextual data for every single image.

Today, we are excited to release **Stratum-FFHQ**—an enriched, dataset-agnostic pipeline transformation of the renowned Flickr-Faces-HQ (FFHQ) dataset. 

Instead of just providing high-resolution RGB images, Stratum-FFHQ delivers a complete multi-modal artifact payload for every image, including dense captions, DINOv3 semantic embeddings, T5 text encodings, and Sapiens-derived spatial maps (depth, normals, and segmentation). 

## What's inside the dataset?

Stratum-FFHQ was processed using [Stratum-HQ](https://github.com/timlawrenz/stratum-hq), a dataset-agnostic image enrichment pipeline. Every image in the original FFHQ dataset maps to a rich set of aligned artifacts.

| Artifact Modality | Format | Description |
|----------|-------|-------------|
| **Text & Captions** | Parquet | Dense, objective descriptions of the image content alongside metadata. |
| **`t5_hidden`** | Tarred `.npy` | T5-Large text encoder hidden states mapped from the caption (`(512, 1024)`), alongside `t5_mask.npy`. |
| **`dinov3_cls`** | Tarred `.npy` | DINOv3-ViT-L/16 CLS token capturing the global style and composition. |
| **`dinov3_patches`**| Tarred `.npy` | DINOv3 spatial patch tokens for fine-grained semantic features. |
| **`pose`** | Tarred `.npy` | 133 DWPose keypoints, capturing facial landmarks and upper body posture (`[x, y, confidence]`). |
| **`seg`** | Tarred `.npy` | 28-class body-part and facial segmentation via Sapiens-1B. |
| **`depth`** | Tarred `.npy` | Sapiens relative depth estimation, foreground-masked and normalised to [0, 1]. |
| **`normal`** | Tarred `.npy` | Sapiens per-pixel surface normals (XYZ), L2-normalised. |

## Why use Stratum-FFHQ?

By pre-computing these expensive multi-modal features, Stratum-FFHQ dramatically accelerates the model development lifecycle and lowers the barrier to entry for independent researchers:

1. **Democratizing Training on Consumer Hardware**: If you are training Diffusion Transformers (DiTs) at home, keeping massive vision and text encoders in VRAM is a severe bottleneck. By ingesting pre-extracted DINOv3 patch embeddings and T5-Large hidden states directly from Stratum-FFHQ, you can avoid expensive forward passes and train highly capable text-to-image models entirely on a single consumer GPU.
2. **Train ControlNets Immediately**: With perfectly aligned depth maps, surface normals, and pose keypoints, you can bypass the heavy lifting of running Sapiens and DWPose on your own training cluster.
3. **Semantic Search & Analysis**: The DINOv3 CLS tokens can be used out-of-the-box for zero-shot clustering, similarity search, and semantic curation of the dataset.

## Optimized for High-Throughput Training

A major challenge with multi-modal datasets is finding a structure that supports high-speed GPU training without overwhelming file systems or forcing researchers to download terabytes of data they don't need. 

**We architected Stratum-FFHQ specifically for high-performance streaming.**

Instead of giant monolithic files or millions of tiny `.npy` arrays, Stratum-FFHQ uses **perfectly aligned, modality-isolated WebDataset shards**. Every modality (depth, pose, DINOv3, T5, etc.) is packaged into synchronized 1,000-item tarballs (e.g., `00000-00999.tar`). Meanwhile, the lightweight captions and metadata are stored in highly queryable Parquet files.

This allows you to lazily stream *only the specific modalities your model needs* directly into PyTorch without downloading the whole dataset. 

```python
import webdataset as wds

# 1. Generate aligned URLs for the modalities you want to train on
urls_depth = [f"https://huggingface.co/datasets/timlawrenz/stratum-ffhq/resolve/main/depth/{i:02d}000-{i:02d}999.tar" for i in range(70)]
urls_pose = [f"https://huggingface.co/datasets/timlawrenz/stratum-ffhq/resolve/main/pose/{i:02d}000-{i:02d}999.tar" for i in range(70)]

# 2. Initialize streaming pipelines and auto-decode the .npy files
ds_depth = wds.WebDataset(urls_depth, shardshuffle=False).decode().to_tuple("__key__", "npy")
ds_pose = wds.WebDataset(urls_pose, shardshuffle=False).decode().to_tuple("__key__", "npy")

# 3. Because the shards are perfectly aligned, we can seamlessly zip the streams over the network!
for (key_d, depth_map), (key_p, pose_data) in zip(ds_depth, ds_pose):
    print(f"Loaded depth map: {depth_map.shape} | Pose data: {pose_data.shape}")
    break
```

Alternatively, you can download specific chunks or modalities locally using the `huggingface_hub` CLI:

```bash
# Download only the depth maps and pose data for local development
hf download timlawrenz/stratum-ffhq --include "depth/*" "pose/*" --repo-type dataset
```

## In the Spirit of FFHQ

When the original Flickr-Faces-HQ (FFHQ) dataset was released, it set a gold standard for high-quality, openly accessible data that propelled the entire generative AI community forward. With Stratum-FFHQ, we want to continue that legacy. 

We believe that the next generation of AI breakthroughs shouldn't be locked behind corporate compute clusters or proprietary APIs. By freely sharing these computationally expensive, heavily processed multi-modal embeddings, we hope to lower the barrier to entry and keep high-quality research data open, accessible, and free for everyone.

The `stratum-hq` enrichment pipeline is entirely dataset-agnostic, meaning this same level of multi-modal enrichment can be applied to any domain. We invite the community to explore the dataset and share the incredible models and insights you build with it!
