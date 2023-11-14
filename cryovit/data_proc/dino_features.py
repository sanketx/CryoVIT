import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from rich.progress import track
from torch import nn
from torch.utils.data import DataLoader

from cryovit.config import Sample
from cryovit.datasets import VITDataset


torch.set_float32_matmul_precision("high")
dino_model = ("facebookresearch/dinov2", "dinov2_vitg14")

dataloader_params = {
    "batch_size": None,
    "pin_memory": True,
    "num_workers": 0,
}


@torch.inference_mode()
def dino_features(
    data: torch.Tensor,
    model: nn.Module,
    batch_size: int,
) -> NDArray[np.float16]:
    data = data.cuda()
    w, h = np.array(data.shape[2:]) // 14
    all_features = []

    for i in range(0, len(data), batch_size):
        vec = data[i : i + batch_size]
        features = model.forward_features(vec)["x_norm_patchtokens"]
        features = features.reshape(features.shape[0], w, h, -1)
        features = features.permute([3, 0, 1, 2]).contiguous()
        features = features.to("cpu").half().numpy()
        all_features.append(features)

    return np.concatenate(all_features, axis=1)


def save_data(
    features: NDArray[np.float16],
    tomo_name: str,
    src_dir: Path,
    dst_dir: Path,
) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(src_dir / tomo_name, dst_dir)

    with h5py.File(dst_dir / tomo_name, "r+") as fh:
        fh.create_dataset("dino_features", data=features)


def process_sample(
    dino_dir: Path,
    data_dir: Path,
    feature_dir: Path,
    batch_size: int,
    sample: Sample,
    **kwargs,
) -> None:
    src_dir = data_dir / "tomograms" / sample.name
    dst_dir = feature_dir / sample.name

    record_file = data_dir / "csv" / f"{sample.name}.csv"
    records = pd.read_csv(record_file)["tomo_name"]

    dataset = VITDataset(records, root=src_dir)
    dataloader = DataLoader(dataset, **dataloader_params)

    torch.hub.set_dir(dino_dir)
    model = torch.hub.load(*dino_model, verbose=False).cuda()
    model.eval()

    for i, x in track(
        enumerate(dataloader),
        description="[green]Computing features",
        total=len(dataloader),
    ):
        features = dino_features(x, model, batch_size)
        save_data(features, records[i], src_dir, dst_dir)
