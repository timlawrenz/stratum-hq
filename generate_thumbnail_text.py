import matplotlib.pyplot as plt
import numpy as np
import webdataset as wds
import matplotlib.gridspec as gridspec
import textwrap

url_depth = "https://huggingface.co/datasets/timlawrenz/stratum-ffhq/resolve/main/depth/00000-00999.tar"
url_normal = "https://huggingface.co/datasets/timlawrenz/stratum-ffhq/resolve/main/normal/00000-00999.tar"
url_seg = "https://huggingface.co/datasets/timlawrenz/stratum-ffhq/resolve/main/seg/00000-00999.tar"
url_pose = "https://huggingface.co/datasets/timlawrenz/stratum-ffhq/resolve/main/pose/00000-00999.tar"

ds_depth = wds.WebDataset(url_depth).decode().to_tuple("__key__", "npy")
ds_normal = wds.WebDataset(url_normal).decode().to_tuple("__key__", "npy")
ds_seg = wds.WebDataset(url_seg).decode().to_tuple("__key__", "npy")
ds_pose = wds.WebDataset(url_pose).decode().to_tuple("__key__", "npy")

key_d, depth = next(iter(ds_depth))
key_n, normal = next(iter(ds_normal))
key_s, seg = next(iter(ds_seg))
key_p, pose = next(iter(ds_pose))

depth = depth.astype(np.float32)
normal = normal.astype(np.float32)
seg = seg.astype(np.float32)
pose = pose.astype(np.float32)

fig = plt.figure(figsize=(12.0, 6.48), dpi=100, facecolor='#111111')
gs = gridspec.GridSpec(2, 4, height_ratios=[5, 1.2], width_ratios=[1, 1, 1, 1], wspace=0.1, hspace=0.1)

ax1 = plt.subplot(gs[0, 0])
ax1.imshow(depth, cmap='inferno')
ax1.axis('off')
ax1.set_title("Depth Map", color='white', pad=15, fontsize=15)

ax2 = plt.subplot(gs[0, 1])
norm_vis = (normal + 1.0) / 2.0
ax2.imshow(norm_vis)
ax2.axis('off')
ax2.set_title("Surface Normals", color='white', pad=15, fontsize=15)

ax3 = plt.subplot(gs[0, 2])
ax3.imshow(seg, cmap='tab20')
ax3.axis('off')
ax3.set_title("Segmentation", color='white', pad=15, fontsize=15)

ax4 = plt.subplot(gs[0, 3], facecolor='#111111')
xs = pose[:, 0]
ys = pose[:, 1]
confs = pose[:, 2]

valid = confs > 0.3
ax4.scatter(xs[valid], ys[valid], c=confs[valid], cmap='viridis', s=20)
ax4.set_xlim(-1, 1)
ax4.set_ylim(1, -1)
ax4.set_aspect('equal', adjustable='box')
ax4.axis('off')
ax4.set_title("DWPose Keypoints", color='white', pad=15, fontsize=15)

caption_full = "Caption: A close-up, frontal portrait of an infant with light brown skin and dark hair. The infant has a rounded face, visible cheek fullness, and dark eyes. The nose is small with rounded nostrils. Lips are closed and neutral in expression. The infant is wearing a short-sleeved, lime green shirt with horizontal white and yellow stripes across the chest. The background consists of a blurred, bright blue surface, possibly fabric or a cushioned support. Lighting is diffuse and even, illuminating the face without strong shadows. The composition is tightly framed on the infant's face and upper chest, with the head occupying the majority of the image. The camera angle is at eye level with the subject."

ax_text = plt.subplot(gs[1, :], facecolor='#1a1a1a')
ax_text.axis('off')

wrapped_text = "\n".join(textwrap.wrap(caption_full, width=120))

ax_text.text(0.5, 0.5, wrapped_text, 
             color='#dddddd', fontsize=13, ha='center', va='center', style='italic', 
             bbox=dict(facecolor='#1a1a1a', edgecolor='#333333', boxstyle='round,pad=1.5'))

plt.suptitle("Stratum-FFHQ: Multi-Modal Enrichment", color='white', fontsize=22, y=0.95, fontweight='bold')

# INCREASED bottom margin from 0.05 to 0.08 to prevent cutoff
plt.subplots_adjust(left=0.04, right=0.96, top=0.85, bottom=0.08)

plt.savefig("/home/tim/source/activity/stratum-hq/thumbnail_final_clean_shifted.png", facecolor='#111111')
print("Saved thumbnail_final_clean_shifted.png")
