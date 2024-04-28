"""Implementation of the single sample data module."""

from typing import Optional

import pandas as pd

from cryovit.config import Sample
from cryovit.data_modules.base_datamodule import BaseDataModule


class SingleSampleDataModule(BaseDataModule):
    """Data module for CryoVIT experiments involving a single sample."""

    def __init__(self, sample: Sample, split_id: Optional[int], **kwargs) -> None:
        """Creates a data module for a single sample.

        Args:
            sample (Sample): Specific sample used in this dataset configuration.
            split_id (Optional[int]): Optional split ID to be excluded from training and used for eval.
        """
        super(SingleSampleDataModule, self).__init__(**kwargs)
        self.sample = sample.name
        self.split_id = split_id

    def train_df(self) -> pd.DataFrame:
        """Train tomograms: exclude those from the sample with the specified split_id.

        Returns:
            pd.DataFrame: A dataframe specifying the train tomograms.
        """
        return self.record_df[
            (self.record_df["split_id"] != self.split_id)
            & (self.record_df["sample"] == self.sample)
        ]

    def val_df(self) -> pd.DataFrame:
        """Validation tomograms: optionally validate on tomograms with the specified split_id.

        Returns:
            pd.DataFrame: A dataframe specifying the validation tomograms.
        """
        if self.split_id is None:  # validate on train set
            return self.train_df()

        return self.record_df[
            (self.record_df["split_id"] == self.split_id)
            & (self.record_df["sample"] == self.sample)
        ]

    def test_df(self) -> pd.DataFrame:
        """Test tomograms: test on tomograms with the specified split_id.

        Returns:
            pd.DataFrame: A dataframe specifying the test tomograms.
        """
        return self.val_df()
