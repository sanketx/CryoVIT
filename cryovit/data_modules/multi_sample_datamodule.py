from cryovit.data_modules.base_datamodule import BaseDataModule


class MultiSampleDataModule(BaseDataModule):
    def __init__(self, samples, split_id, **kwargs):
        super(MultiSampleDataModule, self).__init__(**kwargs)
        self.samples = samples
        self.split_id = split_id

    def train_df(self):
        return self.record_df[
            (self.record_df["split_id"] != self.split_id)
            & (self.record_df["sample"].isin(self.samples))
        ]

    def val_df(self):
        return self.record_df[
            (self.record_df["split_id"] == self.split_id)
            & (self.record_df["sample"].isin(self.samples))
        ]

    def test_df(self):
        return self.record_df[(self.record_df["sample"].isin(self.samples))]
