"""Implementation of the fractional leave one out data module."""

from typing import List

import pandas as pd

from cryovit.config import Sample
from cryovit.data_modules.base_datamodule import BaseDataModule


class FractionalSampleDataModule(BaseDataModule):
    """Data module for fractional leave-one-out CryoVIT experiments."""

    def __init__(
        self,
        sample: Sample,
        split_id: int,
        all_samples: List[Sample],
        **kwargs,
    ) -> None:
        """Train on a fraction of tomograms and leave out one sample for evaluation.

        Args:
            sample (Sample): Sample excluded from training (used for testing).
            split_id (int): Number of splits to be used for training (1-10).
            all_samples (Tuple[Sample]): Tuple of all samples, including the LOO sample.
        """
        super(FractionalSampleDataModule, self).__init__(**kwargs)
        self.sample = sample.name
        self.all_samples = [s.name for s in all_samples]
        self.split_id = list(range(split_id))

    def train_df(self) -> pd.DataFrame:
        """Train tomograms: include a subset of all splits, leaving out one sample.

        Returns:
            pd.DataFrame: A dataframe specifying the train tomograms.
        """
        return self.record_df[
            (self.record_df["split_id"].isin(self.split_id))
            & (self.record_df["sample"] != self.sample)
            & (self.record_df["sample"].isin(self.all_samples))
        ]

    def val_df(self) -> pd.DataFrame:
        """Validation tomograms: validate on the train tomograms. Not really useful.

        Returns:
            pd.DataFrame: A dataframe specifying the validation tomograms.
        """
        return self.train_df()  # validate on train set

    def test_df(self) -> pd.DataFrame:
        """Test tomograms: test on tomograms from the held out sample.

        Returns:
            pd.DataFrame: A dataframe specifying the test tomograms.
        """
        return self.record_df[self.record_df["sample"] == self.sample]
