import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .. import config


class VITDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self.num_records = len(records)

        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

        self.transform = transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
        )

    def __len__(self):
        return self.num_records

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        record = self.records[idx].copy()
        self._load_tomogram(record)
        self._transform(record)

        return record

    def _load_tomogram(self, record):
        tomo_path = os.path.join(
            config.TRAIN_TOMO_DIR, record["sample"], record["tomo_name"]
        )

        with h5py.File(tomo_path) as fh:
            record["data"] = fh["data"][()]
            record["mito"] = fh["mito"][()]
            record["granule"] = fh["granule"][()]

    def _transform(self, record):
        scale = (14 / 16, 14 / 16)
        _, h, w = record["data"].shape
        assert h % 16 == 0 and w % 16 == 0, f"Invalid height: {h} or width: {w}"

        data = np.expand_dims(record["data"], axis=1)
        data = np.repeat(data, 3, axis=1)
        data = torch.from_numpy(data).float()

        data = self.transform(data / 255.0)
        data = torch.nn.functional.interpolate(data, scale_factor=scale, mode="bicubic")
        record["input"] = data
