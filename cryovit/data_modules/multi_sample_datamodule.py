"""Implementation of the multi sample data module."""

from typing import List
from typing import Optional

import pandas as pd

from cryovit.config import Sample
from cryovit.data_modules.base_datamodule import BaseDataModule


class MultiSampleDataModule(BaseDataModule):
    """Data module for CryoVIT experiments involving multiple samples."""

    def __init__(
        self,
        sample: List[Sample],
        split_id: Optional[int],
        test_samples: List[Sample],
        **kwargs,
    ) -> None:
        """Creates a data module for multiple samples.

        Args:
            sample (List[Sample]): List of samples used for training.
            split_id (Optional[int]): Optional split ID to be excluded from training and used for eval.
            test_samples (List[Sample]): List of samples used for testing.
        """
        super(MultiSampleDataModule, self).__init__(**kwargs)
        self.sample = [s.name for s in sample]
        test_samples = [s.name for s in test_samples]
        self.test_samples = test_samples = test_samples if test_samples else self.sample
        self.split_id = split_id

    def train_df(self) -> pd.DataFrame:
        """Train tomograms: exclude those with the specified split_id.

        Returns:
            pd.DataFrame: A dataframe specifying the train tomograms.
        """
        return self.record_df[
            (self.record_df["split_id"] != self.split_id)
            & (self.record_df["sample"].isin(self.sample))
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
            & (self.record_df["sample"].isin(self.sample))
        ]

    def test_df(self) -> pd.DataFrame:
        """Test tomograms: test on tomograms from the test samples.

        Returns:
            pd.DataFrame: A dataframe specifying the test tomograms.
        """
        return self.record_df[(self.record_df["sample"].isin(self.test_samples))]
