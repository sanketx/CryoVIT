from typing import List
from typing import Optional

from cryovit.config import Sample
from cryovit.data_modules.base_datamodule import BaseDataModule


class LOOSampleDataModule(BaseDataModule):
    def __init__(
        self,
        sample: Sample,
        split_id: Optional[int],
        all_samples: List[Sample],
        **kwargs,
    ):
        super(LOOSampleDataModule, self).__init__(**kwargs)
        self.sample = sample.name
        self.all_samples = [s.name for s in all_samples]
        self.split_id = split_id

    def train_df(self):
        return self.record_df[
            (self.record_df["split_id"] != self.split_id)
            & (self.record_df["sample"] != self.sample)
            & (self.record_df["sample"].isin(self.all_samples))
        ]

    def val_df(self):
        if self.split_id is None:  # validate on train set
            return self.train_df()

        return self.record_df[
            (self.record_df["split_id"] == self.split_id)
            & (self.record_df["sample"] != self.sample)
            & (self.record_df["sample"].isin(self.all_samples))
        ]

    def test_df(self):
        return self.record_df[self.record_df["sample"] == self.sample]
