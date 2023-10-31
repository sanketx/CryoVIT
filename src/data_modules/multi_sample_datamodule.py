from .base_datamodule import BaseDataModule


class MultiSampleDataModule(BaseDataModule):
    def __init__(self, train_samples, test_samples, split_id, split_type, **kwargs):
        super(MultiSampleDataModule, self).__init__(**kwargs)
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.split_id = split_id
        self.split_type = split_type

    def train_df(self):
        return self.record_df[
            (self.record_df[self.split_type] != self.split_id) &
            (self.record_df["sample"].isin(self.train_samples))
        ]

    def val_df(self):
        return self.record_df[
            (self.record_df[self.split_type] == self.split_id) &
            (self.record_df["sample"].isin(self.train_samples))
        ]

    def test_df(self):
        return self.record_df[
            (self.record_df["sample"].isin(self.test_samples))
        ]

    def predict_df(self):
        return self.record_df[
            (self.record_df["sample"].isin(self.train_samples)) |
            (self.record_df["sample"].isin(self.test_samples))
        ]
