import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
from rich.progress import track
from torch.utils.data import DataLoader

from .. import config
from ..datasets import VITDataset

torch.set_float32_matmul_precision("high")

dataloader_params = {
    "batch_size": None,
    "pin_memory": True,
    "num_workers": 1,
}

dino_vit_S14 = ("facebookresearch/dinov2", "dinov2_vits14")
dino_vit_B14 = ("facebookresearch/dinov2", "dinov2_vitb14")
dino_vit_L14 = ("facebookresearch/dinov2", "dinov2_vitl14")
dino_vit_g14 = ("facebookresearch/dinov2", "dinov2_vitg14")


@torch.inference_mode()
def dino_features(x, model, batch_size=128):
    x["input"] = x["input"].cuda()
    w, h = np.array(x["input"].shape[2:]) // 14
    all_features = []

    for i in range(0, len(x["input"]), batch_size):
        vec = x["input"][i : i + batch_size]
        features = model.forward_features(vec)["x_norm_patchtokens"]
        features = features.reshape(features.shape[0], w, h, -1)
        features = features.permute([3, 0, 1, 2]).contiguous()
        features = features.to("cpu").half().numpy()
        all_features.append(features)

    return np.concatenate(all_features, axis=1)


def save_data(features, x):
    dst_dir = os.path.join(config.DINO_TOMO_DIR, x["sample"])
    os.makedirs(dst_dir, exist_ok=True)
    tomo_path = os.path.join(dst_dir, x["tomo_name"])

    with h5py.File(tomo_path, "w") as fh:
        fh.create_dataset("data", data=x["data"])
        fh.create_dataset("mito", data=x["mito"])
        fh.create_dataset("granule", data=x["granule"])
        fh.create_dataset("dino_features", data=features)


if __name__ == "__main__":
    sample = sys.argv[1]
    split_file = os.path.join(config.EXP_DIR, "splits.csv")
    record_df = pd.read_csv(split_file)
    record_df = record_df[record_df["sample"] == sample]

    records = [row._asdict() for row in record_df.itertuples()]
    dataloader = DataLoader(VITDataset(records), **dataloader_params)

    torch.hub.set_dir(config.DINO_DIR)
    model = torch.hub.load(*dino_vit_g14).cuda()
    model.eval()

    for x in track(dataloader, description="[green]Computing features"):
        features = dino_features(x, model)
        save_data(features, x)
