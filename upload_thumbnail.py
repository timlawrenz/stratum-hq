import os
from huggingface_hub import HfApi

with open("/home/tim/.hermes/master.env", "r") as f:
    for line in f:
        if line.startswith("HF_TOKEN="):
            token = line.strip().split("=", 1)[1].strip("'\"")
            break

api = HfApi(token=token)
api.upload_file(
    path_or_fileobj="/home/tim/source/activity/stratum-hq/thumbnail_with_text.png",
    path_in_repo="thumbnail_1200x648.png",
    repo_id="timlawrenz/stratum-ffhq",
    repo_type="dataset",
    commit_message="Update thumbnail with caption text"
)
