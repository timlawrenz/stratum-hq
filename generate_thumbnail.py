import matplotlib.pyplot as plt
import numpy as np
import webdataset as wds
import matplotlib.gridspec as gridspec

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

# Change: 1200x648px is recommended (12.0 x 6.48 at 100 dpi)
fig = plt.figure(figsize=(12.0, 6.48), dpi=100, facecolor='#111111')
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.1)

ax1 = plt.subplot(gs[0])
ax1.imshow(depth, cmap='inferno')
ax1.axis('off')
ax1.set_title("Depth Map", color='white', pad=20, fontsize=15)

ax2 = plt.subplot(gs[1])
norm_vis = (normal + 1.0) / 2.0
ax2.imshow(norm_vis)
ax2.axis('off')
ax2.set_title("Surface Normals", color='white', pad=20, fontsize=15)

ax3 = plt.subplot(gs[2])
ax3.imshow(seg, cmap='tab20')
ax3.axis('off')
ax3.set_title("Segmentation", color='white', pad=20, fontsize=15)

ax4 = plt.subplot(gs[3], facecolor='#111111')
xs = pose[:, 0]
ys = pose[:, 1]
confs = pose[:, 2]

valid = confs > 0.3
ax4.scatter(xs[valid], ys[valid], c=confs[valid], cmap='viridis', s=20)
ax4.set_xlim(-1, 1)
ax4.set_ylim(1, -1)
ax4.set_aspect('equal', adjustable='box')
ax4.axis('off')
ax4.set_title("DWPose Keypoints", color='white', pad=20, fontsize=15)

plt.suptitle("Stratum-FFHQ: Multi-Modal Enrichment", color='white', fontsize=22, y=0.92, fontweight='bold')
plt.subplots_adjust(left=0.05, right=0.95, top=0.8)

# The bbox_inches='tight' often strips exact dimensions, so we omit it to strictly enforce 1200x648
plt.savefig("/home/tim/source/activity/stratum-hq/thumbnail_1200x648.png", facecolor='#111111')
print("Saved thumbnail_1200x648.png")
