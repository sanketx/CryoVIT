import os
from pathlib import Path
from typing import Callable
from typing import Dict

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from cryovit.datasets import TomoDataset


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        split_file: Path,
        dataloader_fn: Callable,
        dataset_params: Dict = {},
    ):
        super().__init__()
        self.dataset_params = dataset_params
        self.dataloader_fn = dataloader_fn
        self.load_splits(split_file)

    def load_splits(self, split_file: Path):
        if not split_file.exists():
            raise RuntimeError(f"split file {split_file} not found")

        self.record_df = pd.read_csv(split_file)

    def train_dataloader(self):
        dataset = TomoDataset(
            records=self.train_df(),
            train=True,
            **self.dataset_params,
        )

        return self.dataloader_fn(dataset, shuffle=True)

    def val_dataloader(self):
        dataset = TomoDataset(
            records=self.val_df(),
            train=False,
            **self.dataset_params,
        )
        return self.dataloader_fn(dataset, shuffle=False)

    def test_dataloader(self):
        dataset = TomoDataset(
            records=self.test_df(),
            train=False,
            **self.dataset_params,
        )
        return self.dataloader_fn(dataset, shuffle=False)

    def predict_dataloader(self):
        return self.test_dataloader()

    def train_df(self) -> pd.DataFrame:
        raise NotImplementedError

    def val_df(self) -> pd.DataFrame:
        raise NotImplementedError

    def test_df(self) -> pd.DataFrame:
        raise NotImplementedError
