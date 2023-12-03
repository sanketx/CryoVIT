import os
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import Dataset

from .. import config


class TomoDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        input_key: str,
        label_key: str,
        data_root: Path,
        train: bool = False,
        aux_keys: List[str] = [],
    ) -> None:
        self.records = records
        self.input_key = input_key
        self.label_key = label_key
        self.aux_keys = aux_keys
        self.data_root = data_root
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx) -> torch.Tensor:
        if idx >= len(self):
            raise IndexError

        record = self.records.iloc[idx]
        record = self._load_tomogram(record)

        if self.train:
            self._random_crop(record)

        return record

    def _load_tomogram(self, record: str) -> NDArray[np.uint8]:
        tomo_path = self.data_root / record["sample"] / record["tomo_name"]
        data = {"sample": record["sample"], "tomo_name": record["tomo_name"]}

        with h5py.File(tomo_path) as fh:
            data["input"] = fh[self.input_key][()]
            data["label"] = fh[self.label_key][()]
            data |= {key: fh[key][()] for key in self.aux_keys}

        return data

    def _random_crop(self, record):
        max_depth = 128
        side = 32 if self.input_key == "dino_features" else 512
        d, h, w = record["input"].shape[-3:]
        x, y, z = min(d, max_depth), side, side

        if (d, h, w) == (x, y, z):
            return  # nothing to be done

        delta_d = d - x + 1
        delta_h = h - y + 1
        delta_w = w - z + 1

        di = np.random.choice(delta_d) if delta_d > 0 else 0
        hi = np.random.choice(delta_h) if delta_h > 0 else 0
        wi = np.random.choice(delta_w) if delta_w > 0 else 0

        record["input"] = record["input"][..., di : di + x, hi : hi + y, wi : wi + z]

        if self.input_key == "dino_features":
            hi, wi, y, z = 16 * np.array([hi, wi, y, z])

        record["label"] = record["label"][di : di + x, hi : hi + y, wi : wi + z]


if __name__ == "__main__":
    split_file = os.path.join(config.EXP_DIR, "splits.csv")

    records = pd.read_csv(split_file)
    records = records[records["sample"] == "Q53_KD"]

    dataset = TomoDataset(
        records,
        input_key="dino_features",
        label_key="mito",
        aux_keys=["data"],
        data_root=Path(
            "/sdf/home/s/sanketg/projects/CryoVIT/cryovit_dataset/dino_features"
        ),
        train=True,
    )

    print(dataset[0].keys())

    for x in dataset:
        print(x["input"].shape, x["label"].shape)
