import webdataset as wds
import numpy as np

url_depth = "https://huggingface.co/datasets/timlawrenz/stratum-ffhq/resolve/main/depth/{00000..00069}000-0{00000..00069}999.tar"
url_pose = "https://huggingface.co/datasets/timlawrenz/stratum-ffhq/resolve/main/pose/{00000..00069}000-0{00000..00069}999.tar"

try:
    dataset = wds.DataPipeline(
        wds.ResampledShards(url_depth),
        wds.tarfile_to_samples(),
        wds.ResampledShards(url_pose),
        wds.tarfile_to_samples(),
    ).zip()

    for item in dataset:
        print("Success!", type(item))
        break
except Exception as e:
    print("Error:", e)
