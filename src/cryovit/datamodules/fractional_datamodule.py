"""Implementation of the fractional  data module."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from cryovit.datamodules.base_datamodule import BaseDataModule


class FractionalDataModule(BaseDataModule):
    """Data module for fractional CryoVIT experiments."""

    def __init__(
        self,
        sample: list[str],
        split_id: int | None,
        split_key: str | None,
        test_sample: int | None = None,
        **kwargs,
    ) -> None:
        """Train on a fraction of tomograms and leave out one sample for evaluation.

        Args:
            sample (list[str]): The samples to train and test on.
            split_id (Optional[int]): The number of splits used for training. If None, defaults to all splits.
            split_key (str): The key used to select splits using split_id.
            test_sample (Optional[int]): The split to exclude from training and use for testing.
        """

        super().__init__(**kwargs)
        # Validity checks
        assert (
            test_sample is not None
        ), "Fractional sample `test_sample` cannot be None."
        assert isinstance(
            test_sample, int
        ), f"Fractional sample 'test_sample' should be an integer. Got {test_sample} instead."

        # we only want to generate splits for the relevant samples
        self.record_df = self.record_df[self.record_df["sample"].isin(sample)]

        # Add new splits for fractional training
        num_samples = self.record_df.shape[0]
        n_splits = 11  # Using 10-fold + 1 for LOO
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        X = np.array([[0] for _ in range(num_samples)])
        splits = [-1 for _ in range(num_samples)]
        for f, (_, test) in enumerate(kf.split(X)):
            for idx in test:
                splits[idx] = f
        self.record_df[split_key] = splits

        self.sample = sample
        self.split_id = split_id
        self.split_key = split_key
        self.test_id = test_sample

    def train_df(self) -> pd.DataFrame:
        """Train tomograms: include a subset of all splits, leaving out one split.

        Returns:
            pd.DataFrame: A dataframe specifying the train tomograms.
        """

        assert self.record_df is not None
        all_splits = sorted(
            set(self.record_df[self.split_key].unique()) - {self.test_id}
        )
        assert (
            len(all_splits) == 10
        ), "There should be 10 splits for fractional training."
        if self.split_id is not None:
            training_splits = all_splits[: self.split_id]
        else:
            training_splits = all_splits

        return self.record_df[
            (self.record_df[self.split_key].isin(training_splits))
        ][["sample", "tomo_name"]]

    def val_df(self) -> pd.DataFrame:
        """Validation tomograms: validate on tomograms from the held out sample.

        Returns:
            pd.DataFrame: A dataframe specifying the validation tomograms.
        """

        assert self.record_df is not None
        return self.record_df[(self.record_df[self.split_key] == self.test_id)]

    def test_df(self) -> pd.DataFrame:
        """Test tomograms: test on tomograms from the held out sample.

        Returns:
            pd.DataFrame: A dataframe specifying the test tomograms.
        """

        assert self.record_df is not None
        if self.split_id is not None:
            keys = ["sample", "tomo_name", self.split_key]
        else:
            keys = ["sample", "tomo_name"]

        df = self.val_df()[keys]
        # replace split_id with split fraction if it exists
        if self.split_key in df.columns:
            df["split_id"] = self.split_id
        return df

    def predict_df(self) -> pd.DataFrame:
        """Predict tomograms: predict on the specified samples.

        Returns:
            pd.DataFrame: A dataframe specifying the predict tomograms.
        """

        assert self.record_df is not None
        return self.record_df[["sample", "tomo_name"]]
