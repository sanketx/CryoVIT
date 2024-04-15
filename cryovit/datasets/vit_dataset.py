"""Dataset class for loading tomograms for DINOv2 models."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import Dataset
from torchvision.transforms import Normalize


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class VITDataset(Dataset):
    """Dataset class for Vision Transformer models, loading and processing tomograms."""

    def __init__(self, records: pd.Series, root: Path) -> None:
        """Initializes a dataset object to load tomograms, applying normalization and resizing for DINOv2 models.

        Args:
            records (pd.Series): A series containing paths to tomogram files.
            root (Path): Root directory where tomogram files are stored.
        """
        self.records = records
        self.root = root
        self.transform = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    def __len__(self) -> int:
        """Returns the number of tomograms in the dataset."""
        return len(self.records)

    def __getitem__(self, idx) -> torch.Tensor:
        """Retrieves a preprocessed tomogram tensor from the dataset by index.

        Args:
            idx (int): Index of the tomogram to retrieve.

        Returns:
            torch.Tensor: A tensor representing the normalized and resized tomogram.

        Raises:
            IndexError: If the index is out of the dataset's bounds.
        """
        if idx >= len(self):
            raise IndexError

        record = self.records[idx]
        data = self._load_tomogram(record)
        return self._transform(data)

    def _load_tomogram(self, record: str) -> NDArray[np.uint8]:
        """Loads a tomogram from disk.

        Args:
            record (str): The file path to the tomogram relative to the root directory.

        Returns:
            NDArray[np.uint8]: The loaded tomogram as a numpy array.
        """
        tomo_path = self.root / record

        with h5py.File(tomo_path) as fh:
            return fh["data"][()]

    def _transform(self, data: NDArray[np.uint8]) -> torch.Tensor:
        """Applies normalization and resizing transformations to the tomogram.

        Args:
            data (NDArray[np.uint8]): The loaded tomogram data as a numpy array.

        Returns:
            torch.Tensor: The transformed data as a PyTorch tensor.
        """
        scale = (14 / 16, 14 / 16)
        _, h, w = data.shape
        assert h % 16 == 0 and w % 16 == 0, f"Invalid height: {h} or width: {w}"

        data = np.expand_dims(data, axis=1)
        data = np.repeat(data, 3, axis=1)

        data = torch.from_numpy(data).float()
        data = self.transform(data / 255.0)
        return F.interpolate(data, scale_factor=scale, mode="bicubic")
