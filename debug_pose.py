import numpy as np
import webdataset as wds

url_pose = "https://huggingface.co/datasets/timlawrenz/stratum-ffhq/resolve/main/pose/00000-00999.tar"
ds_pose = wds.WebDataset(url_pose).decode().to_tuple("__key__", "npy")

key_p, pose = next(iter(ds_pose))
pose = pose.astype(np.float32)

xs = pose[:, 0]
ys = pose[:, 1]
confs = pose[:, 2]
valid = confs > 0.3

print("Valid points:", np.sum(valid))
print("X range:", xs[valid].min(), "to", xs[valid].max())
print("Y range:", ys[valid].min(), "to", ys[valid].max())
