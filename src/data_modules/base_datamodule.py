import os

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class BaseDataModule(LightningDataModule):
    def __init__(self, split_file, dataset_class, dataset_params, dataloader_params):
        super().__init__()
        self.dataset_class = dataset_class
        self.dataset_params = dataset_params
        self.dataloader_params = dataloader_params
        self.load_splits(split_file)

    def load_splits(self, split_file):
        if not os.path.exists(split_file):
            raise RuntimeError(f"split file {split_file} not found")

        self.record_df = pd.read_csv(split_file)

    def train_dataloader(self):
        records = [row._asdict() for row in self.train_df().itertuples()]
        dataset = self.dataset_class["train"](
            records=records, train=True, **self.dataset_params
        )
        return DataLoader(dataset, shuffle=False, **self.dataloader_params)

    def val_dataloader(self):
        records = [row._asdict() for row in self.val_df().itertuples()]
        dataset = self.dataset_class["val"](records=records, **self.dataset_params)
        return DataLoader(dataset, **self.dataloader_params)

    def test_dataloader(self):
        records = [row._asdict() for row in self.test_df().itertuples()]
        dataset = self.dataset_class["test"](records=records, **self.dataset_params)
        return DataLoader(dataset, **self.dataloader_params)

    def predict_dataloader(self):
        records = [row._asdict() for row in self.predict_df().itertuples()]
        dataset = self.dataset_class["predict"](records=records, **self.dataset_params)
        return DataLoader(dataset, **self.dataloader_params)
