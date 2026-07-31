import webdataset as wds

urls_depth = [f"https://huggingface.co/datasets/timlawrenz/stratum-ffhq/resolve/main/depth/{i:02d}000-{i:02d}999.tar" for i in range(70)]
urls_pose = [f"https://huggingface.co/datasets/timlawrenz/stratum-ffhq/resolve/main/pose/{i:02d}000-{i:02d}999.tar" for i in range(70)]

try:
    ds_depth = wds.WebDataset(urls_depth, shardshuffle=False).decode().to_tuple("__key__", "npy")
    ds_pose = wds.WebDataset(urls_pose, shardshuffle=False).decode().to_tuple("__key__", "npy")

    for (key_d, depth), (key_p, pose) in zip(ds_depth, ds_pose):
        print(f"Matched! Key: {key_d} - Depth shape: {depth.shape}, Pose shape: {pose.shape}")
        break
except Exception as e:
    print("Error:", repr(e))
