import os

import h5py
import numpy as np
import pandas as pd
import torch
from rich.progress import track
from torch.utils.data import DataLoader, Dataset

from .. import config


class DinoDataset(Dataset):
    def __init__(self, records, train=False, **params):
        self.records = records
        self.train = train
        self.num_records = len(records)
        print(f"NUM RECORDS: {self.num_records}")

    def __len__(self):
        if self.train:
            return 4 * self.num_records

        return self.num_records

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        record = self.records[idx % self.num_records].copy()
        record = self._load_tomogram(record)

        # if self.train:
        #     self._random_crop(record)

        return record

    def _load_tomogram(self, record):
        tomo_path = os.path.join(
            config.DINO_TOMO_DIR, record["sample"], record["tomo_name"]
        )

        with h5py.File(tomo_path) as fh:
            record["data"] = fh["data"][()]
            record["mito"] = fh["mito"][()].astype(np.float32)
            record["dino_features"] = fh["dino_features"][()]

        return {k: record[k] for k in ["data", "mito", "dino_features", "tomo_name"]}

    def _random_crop(self, record):
        side = np.random.choice([16, 24, 32])
        d, h, w = record["dino_features"].shape[1:]
        x, y, z = np.array([128, side, side])

        if (d, h, w) == (x, y, z):
            return  # nothing to be done

        delta_d = d - x + 1
        delta_h = h - y + 1
        delta_w = w - z + 1

        di = np.random.choice(delta_d)
        hi = np.random.choice(delta_h)
        wi = np.random.choice(delta_w)

        record["dino_features"] = record["dino_features"][
            :, di : di + x, hi : hi + y, wi : wi + z
        ].copy()

        hi, wi, y, z = 16 * np.array([hi, wi, y, z])
        record["mito"] = record["mito"][di : di + x, hi : hi + y, wi : wi + z].copy()
        # TODO: Repeat for granule


if __name__ == "__main__":
    split_file = os.path.join(config.EXP_DIR, "splits.csv")

    record_df = pd.read_csv(split_file)
    record_df = record_df[record_df["sample"] == "BACHD"]

    records = [row._asdict() for row in record_df.itertuples()]
    dataset = DinoDataset(records, train=True)

    dataloader_params = {
        "batch_size": None,
        "pin_memory": True,
        "num_workers": 8,
        "prefetch_factor": 4,
        "persistent_workers": False,
    }

    dataloader = DataLoader(dataset, **dataloader_params)

    for i, x in enumerate(track(dataloader)):
        for key in x:
            if key != "tomo_name":
                print(i, key, x[key].shape)
