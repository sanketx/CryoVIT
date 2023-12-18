from typing import List
from typing import Optional

from cryovit.config import Sample
from cryovit.data_modules.base_datamodule import BaseDataModule


class MultiSampleDataModule(BaseDataModule):
    def __init__(
        self,
        sample: List[Sample],
        split_id: Optional[int],
        test_samples: List[Sample],
        **kwargs,
    ):
        super(MultiSampleDataModule, self).__init__(**kwargs)
        self.sample = [s.name for s in sample]
        test_samples = [s.name for s in test_samples]
        self.test_samples = test_samples = test_samples if test_samples else self.sample
        self.split_id = split_id

    def train_df(self):
        return self.record_df[
            (self.record_df["split_id"] != self.split_id)
            & (self.record_df["sample"].isin(self.sample))
        ]

    def val_df(self):
        if self.split_id is None:  # validate on train set
            return self.train_df()

        return self.record_df[
            (self.record_df["split_id"] == self.split_id)
            & (self.record_df["sample"].isin(self.sample))
        ]

    def test_df(self):
        return self.record_df[(self.record_df["sample"].isin(self.test_samples))]
