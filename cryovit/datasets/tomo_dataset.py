"""Dataset class for loading DINOv2 features for CryoVIT models."""

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class TomoDataset(Dataset):
    """A dataset class for handling and preprocessing tomographic data for CryoVIT models."""

    def __init__(
        self,
        records: pd.DataFrame,
        input_key: str,
        label_key: str,
        data_root: Path,
        train: bool = False,
        aux_keys: List[str] = [],
    ) -> None:
        """Creates a new TomoDataset object.

        Args:
            records (pd.DataFrame): A DataFrame containing records of tomograms.
            input_key (str): The key in the HDF5 file to access input features.
            label_key (str): The key in the HDF5 file to access labels.
            data_root (Path): The root directory where the tomograms are stored.
            train (bool): Flag to determine if the dataset is for training (enables transformations).
            aux_keys (List[str]): Additional keys for auxiliary data to load from the HDF5 files.
        """
        self.records = records
        self.input_key = input_key
        self.label_key = label_key
        self.aux_keys = aux_keys
        self.data_root = data_root
        self.train = train

    def __len__(self) -> int:
        """Returns the total number of tomograms in the dataset."""
        return len(self.records)

    def __getitem__(self, idx) -> Dict[str, Any]:
        """Retrieves a single item from the dataset.

        Args:
            idx (int): The index of the item.

        Returns:
            record (Dict[str, Any]): A dictionary containing the loaded data and labels.

        Raises:
            IndexError: If index is out of the range of the dataset.
        """
        if idx >= len(self):
            raise IndexError

        record = self.records.iloc[idx]
        record = self._load_tomogram(record)

        if self.train:
            self._random_crop(record)

        return record

    def _load_tomogram(self, record: str) -> Dict[str, Any]:
        """Loads a single tomogram based on the record information.

        Args:
            record (pd.Series): A series containing the sample and tomogram names.

        Returns:
            data (Dict[str, Any]): A dictionary with input data, label, and any auxiliary data.
        """
        tomo_path = self.data_root / record["sample"] / record["tomo_name"]
        data = {"sample": record["sample"], "tomo_name": record["tomo_name"]}

        with h5py.File(tomo_path) as fh:
            data["input"] = fh[self.input_key][()]
            data["label"] = fh[self.label_key][()]
            data |= {key: fh[key][()] for key in self.aux_keys}

        return data

    def _random_crop(self, record) -> None:
        """Applies a random crop to the input data in the record dictionary.

        Args:
            record (Dict[str, Any]): The record dictionary containing 'input' and 'label' data.
        """
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
