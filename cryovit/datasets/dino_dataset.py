import os

import h5py
import numpy as np
import pandas as pd
import torch
from rich.progress import track
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .. import config


class DinoDataset(Dataset):
    def __init__(self, records, train=False, **params):
        self.records = records
        self.train = train
        self.num_records = len(records)
        print(f"NUM RECORDS: {self.num_records}")

    def __len__(self):
        # if self.train:
        #     return 4 * self.num_records

        return self.num_records

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        record = self.records[idx % self.num_records].copy()
        record = self._load_tomogram(record)

        if self.train:
            self._random_crop(record)

        return record

    def _load_ai_label(self, record):
        tomo_path = os.path.join(
            config.AI_TOMO_DIR, record["sample"], record["tomo_name"]
        )

        with h5py.File(tomo_path) as fh:
            mito = fh["pred"][()]
            record["mito"] = np.where(mito > 0, 1, 0).astype(np.float32)

    def _load_tomogram(self, record):
        tomo_path = os.path.join(
            config.DINO_TOMO_DIR, record["sample"], record["tomo_name"]
        )

        with h5py.File(tomo_path) as fh:
            record["data"] = fh["data"][()]
            record["dino_features"] = fh["dino_features"][()]
            record["mito"] = fh["mito"][()].astype(np.float32)

        # if self.train:
        #     self._load_ai_label(record)

        return {
            k: record[k]
            for k in ["data", "mito", "dino_features", "tomo_name", "sample"]
        }

    def _random_crop(self, record):
        # side = np.random.choice([16, 24, 32])
        d, h, w = record["dino_features"].shape[1:]
        x, y, z = np.array([min(d, 128), 32, 32])

        if (d, h, w) == (x, y, z):
            return  # nothing to be done

        delta_d = d - x + 1
        delta_h = h - y + 1
        delta_w = w - z + 1

        di = np.random.choice(delta_d) if delta_d > 0 else 0
        hi = np.random.choice(delta_h) if delta_h > 0 else 0
        wi = np.random.choice(delta_w) if delta_w > 0 else 0

        record["dino_features"] = record["dino_features"][
            :, di : di + x, hi : hi + y, wi : wi + z
        ].copy()

        hi, wi, y, z = 16 * np.array([hi, wi, y, z])
        record["mito"] = record["mito"][di : di + x, hi : hi + y, wi : wi + z].copy()
        # TODO: Repeat for granule


if __name__ == "__main__":
    split_file = os.path.join(config.EXP_DIR, "splits.csv")

    record_df = pd.read_csv(split_file)
    record_df = record_df[record_df["sample"] == "WT"]

    records = [row._asdict() for row in record_df.itertuples()]
    dataset = DinoDataset(records, train=True)

    dataloader_params = {
        "batch_size": None,
        "pin_memory": False,
        "num_workers": 0,
        # "prefetch_factor": 1,
        "persistent_workers": False,
    }

    # dataloader = DataLoader(dataset, **dataloader_params)

    for i, x in enumerate(track(dataset)):
        print(i, x["data"].shape, x["dino_features"].shape)
