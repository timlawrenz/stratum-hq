**Title: 🚀 Stratum-FFHQ: Modernizing a Generative AI Landmark**

In 2019, NVIDIA's release of the FFHQ dataset alongside StyleGAN catalyzed an entire generation of computer vision research. Projects spanning face restoration, GAN inversion, and semantic editing have all relied on the pristine quality of FFHQ. 

To give back to the open-source AI community and push open-weights research forward, we are thrilled to announce **stratum-ffhq**—a massive, free conditioning enrichment layer for the 70,000 images in the FFHQ dataset. 

**Why Stratum-FFHQ?**
As the community moves toward Diffusion Transformers and highly conditional generation, models need dense annotations. `stratum-ffhq` saves researchers thousands of GPU compute hours by providing state-of-the-art, pre-computed conditioning modalities out of the box:
*   **Vision Features:** DINOv3 ViT-L/16 (global `cls` and dense `patches`).
*   **Textual Conditioning:** Dense, objective descriptions (Gemma 3 27B) + `t5-large` hidden states.
*   **Geometric Controls:** Depth maps, surface normals, and 28-class body segmentation (Sapiens-1B).
*   **Pose Mechanics:** Dense facial/body keypoints (DWPose).

**⚖️ Crucial Licensing & Usage Notice**
To completely avoid licensing issues and respect NVIDIA's original dataset, **this repository does not contain any original pixel data**. We distribute *only* the extracted embeddings, tensors, and text arrays. 

To utilize this data:
1. Download the original images from the official FFHQ dataset repo.
2. Align the `image_id` keys in our parquet/npy files with the original images.
3. Ensure your use case complies with the original NVIDIA license (CC BY-NC-SA 4.0).