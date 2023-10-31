import os
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .. import config
from ..data_modules import MultiSampleDataModule
from ..datasets import DinoDataset
from ..models import BasicUpsample
from ..models.losses import DiceLoss
from ..models.metrics import DiceMetric

torch.set_float32_matmul_precision("high")
exp_dir = os.path.join(config.EXP_DIR, "dino")

dataloader_params = {
    "batch_size": None,
    "pin_memory": True,
    "num_workers": 1,
    "prefetch_factor": 1,
    "persistent_workers": False,
}

datamodule_params = {
    "train_samples": ["WT"],
    "test_samples": ["dN17_BACHD"],
    "split_id": 0,
    "split_type": "split_10",
    "split_file": os.path.join(config.EXP_DIR, "splits.csv"),
    "dataset_class": defaultdict(lambda: DinoDataset),
    "dataset_params": {},
    "dataloader_params": dataloader_params,
}

model_params = {
    "losses": [(DiceLoss, {})],
    "metrics": [(DiceMetric, {"threshold": 0.5})],
    "lr": 1e-4,
}


if __name__ == "__main__":
    datamodule = MultiSampleDataModule(**datamodule_params)
    ckpt_path = os.path.join(exp_dir, "checkpoints", "dino-v15.ckpt")

    model = (
        BasicUpsample.load_from_checkpoint(ckpt_path, **model_params)
        .half()
        .to("cuda:0")
    )
    model.eval()

    for x in tqdm(datamodule.test_dataloader()):
        if (
            "BACHD-DN17_Grid1_Square1_Tomo00002__bin4_healthy_mitochondria"
            not in x["tomo_name"]
        ):
            continue

        features = x["dino_features"].unsqueeze(0).to("cuda:0")

        with torch.no_grad():
            for layer in model.layers[:-1]:
                features = layer(features)

        # features = torch.nn.functional.interpolate(features, scale_factor=(1, 2, 2))
        features = features.squeeze().permute([1, 2, 3, 0]).cpu()

        with h5py.File(os.path.join("features", x["tomo_name"]), "w") as fh:
            fh.create_dataset("data", data=x["data"].squeeze().numpy())
            fh.create_dataset("F8", data=features.numpy())

        break

    print(features.shape)
