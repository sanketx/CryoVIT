"""Dataset class for loading tomograms for CryoViT experiments."""

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from torch import tensor
from torch.utils.data import Dataset

from cryovit.types import TomogramData


class TomoDataset(Dataset):
    """A dataset class for handling and preprocessing tomographic data for CryoVIT models."""

    def __init__(
        self,
        records: pd.DataFrame,
        input_key: str,
        label_key: str,
        split_key: str,
        data_root: Path,
        aux_keys: list[str] | None = None,
        train: bool = False,
    ) -> None:
        """Initializes a dataset object to load tomograms for model training, applying optional training crops.

        Args:
            records (pd.DataFrame): A DataFrame containing records of tomograms.
            input_key (str): The key in the HDF5 file to access input features.
            label_key (str): The key in the HDF5 file to access labels.
            split_key (str): The key in the DataFrame to access the split identifier.
            data_root (Path): The root directory where the tomograms are stored.
            aux_keys (Optional[List[str]]): Optional additional keys for auxiliary data to load from the HDF5 files.
            train (bool): Flag to determine if the dataset is for training (enables transformations).
        """

        if aux_keys is None:
            aux_keys = []
        self.records = records
        self.input_key = input_key
        self.label_key = label_key
        self.split_key = split_key
        self.aux_keys = aux_keys
        self.data_root = (
            data_root if isinstance(data_root, Path) else Path(data_root)
        )
        self.train = train

    def __len__(self) -> int:
        """Returns the total number of tomograms in the dataset."""

        return len(self.records)

    def __getitem__(self, idx: int) -> TomogramData:  # type: ignore
        """Retrieves a single item from the dataset.

        Args:
            idx (int): The index of the item.

        Returns:
            TomogramData: A dataclass containing the loaded data, labels, and metadata.

        Raises:
            IndexError: If index is out of the range of the dataset.
        """

        if idx >= len(self):
            raise IndexError

        record = self.records.iloc[idx]
        assert isinstance(record, pd.Series)
        data = self._load_tomogram(record)

        if self.train:
            self._random_crop(data)

        return TomogramData(
            sample=record["sample"],
            tomo_name=record["tomo_name"],
            split_id=data.get("split_id", None),
            data=data["input"],
            label=data["label"],
            aux_data={key: data[key] for key in self.aux_keys if key in data},
        )  # type: ignore

    def _load_tomogram(self, record: pd.Series) -> dict[str, Any]:
        """Loads a single tomogram based on the record information.

        Args:
            record (pd.Series): A series containing the sample and tomogram names.

        Returns:
            data (dict[str, Any]): A dictionary with input data, label, and any auxiliary data.
        """

        tomo_path = self.data_root / record["sample"] / record["tomo_name"]

        # sample, tomo_name, split_id, input, label

        data_dict = {
            "sample": record["sample"],
            "tomo_name": record["tomo_name"],
        }
        if self.split_key in record.index:
            data_dict["split_id"] = record[self.split_key]

        with h5py.File(tomo_path) as fh:
            assert (
                self.input_key in fh
            ), f"Input key '{self.input_key}' not found in {tomo_path}."
            assert (
                self.label_key in fh["labels"]
            ), f"Label key '{self.label_key}' not found in {tomo_path}/labels."  # type: ignore
            data: np.ndarray = fh[self.input_key][()]  # type: ignore
            if data.dtype == np.uint8:
                data = data.astype(np.float32) / 255.0
            if len(data.shape) == 3:
                data = data[np.newaxis, ...]  # add channel dimension
            data_dict["input"] = data
            data_dict["label"] = fh["labels"][self.label_key][()]  # type: ignore

            # Load auxiliary data if specified
            aux_dict = {}
            for key in self.aux_keys:
                if (
                    key == "sam_features" and key in fh
                ):  # handle cached SAM features separately
                    feature_dict = {}
                    for feature_key in fh[key]:  # type: ignore
                        feature_list = []
                        for i in range(len(fh[key][feature_key])):  # type: ignore
                            feature_list.append(tensor(fh[key][feature_key][str(i)][()]).half())  # type: ignore
                        feature_dict[feature_key] = feature_list
                    if "vision_features" not in feature_dict:
                        feature_dict["vision_features"] = feature_dict[
                            "backbone_fpn"
                        ][-1]
                    aux_dict[key] = feature_dict
                elif key in fh:
                    aux_dict[key] = fh[key][()]  # type: ignore
            data_dict |= aux_dict  # type: ignore

        return data_dict

    def _random_crop(self, data: dict[str, Any]) -> None:
        """Applies a random crop to the input data in the record dictionary.

        Args:
            record (dict[str, Any]): The record dictionary containing 'input' and 'label' data.
        """

        max_depth = 128
        side = 32 if self.input_key == "dino_features" else 512
        d, h, w = data["input"].shape[-3:]
        x, y, z = min(d, max_depth), side, side

        if (d, h, w) == (x, y, z):
            return  # nothing to be done

        delta_d = d - x + 1
        delta_h = h - y + 1
        delta_w = w - z + 1

        di = np.random.choice(delta_d) if delta_d > 0 else 0
        hi = np.random.choice(delta_h) if delta_h > 0 else 0
        wi = np.random.choice(delta_w) if delta_w > 0 else 0

        data["input"] = data["input"][
            ..., di : di + x, hi : hi + y, wi : wi + z
        ]

        if self.input_key == "dino_features":
            hi, wi, y, z = 16 * np.array([hi, wi, y, z])

        data["label"] = data["label"][di : di + x, hi : hi + y, wi : wi + z]

        for key in self.aux_keys:
            data[key] = data[key][..., di : di + x, hi : hi + y, wi : wi + z]
