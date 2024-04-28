"""Module defining base data loading functionality for CryoVIT experiments."""

from pathlib import Path
from typing import Callable
from typing import Dict

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from cryovit.datasets import TomoDataset


class BaseDataModule(LightningDataModule):
    """Base module defining common functions for creating data loaders."""

    def __init__(
        self,
        split_file: Path,
        dataloader_fn: Callable,
        dataset_params: Dict = {},
    ) -> None:
        """Initializes the BaseDataModule with dataset parameters, a dataloader function, and a path to the split file.

        Args:
            split_file (Path): The path to the CSV file containing data splits.
            dataloader_fn (Callable): Function to create a DataLoader from a dataset.
            dataset_params (Dict, optional): Dictionary of parameters to pass to the dataset class..
        """
        super().__init__()
        self.dataset_params = dataset_params
        self.dataloader_fn = dataloader_fn
        self._load_splits(split_file)

    def _load_splits(self, split_file: Path) -> None:
        if not split_file.exists():
            raise RuntimeError(f"split file {split_file} not found")

        self.record_df = pd.read_csv(split_file)

    def train_dataloader(self) -> DataLoader:
        """Creates DataLoader for training data.

        Returns:
            DataLoader: A DataLoader instance for training data.
        """
        dataset = TomoDataset(
            records=self.train_df(),
            train=True,
            **self.dataset_params,
        )

        return self.dataloader_fn(dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Creates DataLoader for validation data.

        Returns:
            DataLoader: A DataLoader instance for validation data.
        """
        dataset = TomoDataset(
            records=self.val_df(),
            train=False,
            **self.dataset_params,
        )
        return self.dataloader_fn(dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Creates DataLoader for testing data.

        Returns:
            DataLoader: A DataLoader instance for testing data.
        """
        dataset = TomoDataset(
            records=self.test_df(),
            train=False,
            **self.dataset_params,
        )
        return self.dataloader_fn(dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        """Creates DataLoader for prediction data.

        Returns:
            DataLoader: A DataLoader instance for prediction data.
        """
        return self.test_dataloader()

    def train_df(self) -> pd.DataFrame:
        """Abstract method to generate train splits."""
        raise NotImplementedError

    def val_df(self) -> pd.DataFrame:
        """Abstract method to generate validation splits."""
        raise NotImplementedError

    def test_df(self) -> pd.DataFrame:
        """Abstract method to generate test splits."""
        raise NotImplementedError
