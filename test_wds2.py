import webdataset as wds

url_depth = "https://huggingface.co/datasets/timlawrenz/stratum-ffhq/resolve/main/depth/{00000..00069}000-0{00000..00069}999.tar"
url_pose = "https://huggingface.co/datasets/timlawrenz/stratum-ffhq/resolve/main/pose/{00000..00069}000-0{00000..00069}999.tar"

try:
    ds_depth = wds.WebDataset(url_depth).to_tuple("__key__", "depth.npy")
    ds_pose = wds.WebDataset(url_pose).to_tuple("__key__", "pose.npy")

    for (key_d, depth), (key_p, pose) in zip(ds_depth, ds_pose):
        print(f"Matched! Key Depth: {key_d}, Key Pose: {key_p}")
        break
except Exception as e:
    print("Error:", e)
