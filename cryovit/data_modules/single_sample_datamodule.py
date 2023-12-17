from typing import Optional

from cryovit.config import Sample
from cryovit.data_modules.base_datamodule import BaseDataModule


class SingleSampleDataModule(BaseDataModule):
    def __init__(self, sample: Sample, split_id: Optional[int], **kwargs):
        super(SingleSampleDataModule, self).__init__(**kwargs)
        self.sample = sample.name
        self.split_id = split_id

    def train_df(self):
        return self.record_df[
            (self.record_df["split_id"] != self.split_id)
            & (self.record_df["sample"] == self.sample)
        ]

    def val_df(self):
        if self.split_id is None:  # validate on train set
            return self.train_df()

        return self.record_df[
            (self.record_df["split_id"] == self.split_id)
            & (self.record_df["sample"] == self.sample)
        ]

    def test_df(self):
        return self.val_df()
