"""Implementation of the leave one out data module."""

from typing import List
from typing import Optional

import pandas as pd

from cryovit.config import Sample
from cryovit.data_modules.base_datamodule import BaseDataModule


class LOOSampleDataModule(BaseDataModule):
    """Data module for CryoVIT experiments leaving out one sample."""

    def __init__(
        self,
        sample: Sample,
        split_id: Optional[int],
        all_samples: List[Sample],
        **kwargs,
    ) -> None:
        """Creates a data module which leaves out one sample for evaluation.

        Args:
            sample (Sample): Sample excluded from training (used for testing).
            split_id (Optional[int]): Optional split ID for validation.
            all_samples (Tuple[Sample]): Tuple of all samples, including the LOO sample.
        """
        super(LOOSampleDataModule, self).__init__(**kwargs)
        self.sample = sample.name
        self.all_samples = [s.name for s in all_samples]
        self.split_id = split_id

    def train_df(self) -> pd.DataFrame:
        """Train tomograms: exclude those with the specified split_id and sample.

        Returns:
            pd.DataFrame: A dataframe specifying the train tomograms.
        """
        return self.record_df[
            (self.record_df["split_id"] != self.split_id)
            & (self.record_df["sample"] != self.sample)
            & (self.record_df["sample"].isin(self.all_samples))
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
            & (self.record_df["sample"] != self.sample)
            & (self.record_df["sample"].isin(self.all_samples))
        ]

    def test_df(self) -> pd.DataFrame:
        """Test tomograms: test on tomograms from the held out sample.

        Returns:
            pd.DataFrame: A dataframe specifying the test tomograms.
        """
        return self.record_df[self.record_df["sample"] == self.sample]
