import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from torch import nn


class BaseModule(LightningModule):
    def __init__(self, losses, metrics, lr=1e-4):
        super(BaseModule, self).__init__()
        self.lr = lr
        self.compile(losses, metrics)

    def forward(self):
        raise NotImplementedError

    def compile(self, losses, metrics):
        self.loss_fns = {
            "TRAIN": [fn(**params) for fn, params in losses],
            "VAL": [fn(**params) for fn, params in losses],
            "TEST": [fn(**params) for fn, params in losses],
        }
        self.metric_fns = nn.ModuleDict(
            {
                "TRAIN": nn.ModuleList([fn(**params) for fn, params in metrics]),
                "VAL": nn.ModuleList([fn(**params) for fn, params in metrics]),
                "TEST": nn.ModuleList([fn(**params) for fn, params in metrics]),
            }
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-3)

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms, on_step=True)

    def masked_predict(self, batch):
        data = batch["data"]
        mito_true = batch["mito"]

        mito_pred = self(data)
        mito_weight = (mito_true > -1.0).detach()

        mito_pred = torch.masked_select(mito_pred, mito_weight).view(-1, 1)
        mito_true = torch.masked_select(mito_true, mito_weight).view(-1, 1)

        return mito_pred, mito_true

    def step(self, batch, prefix):
        y_pred, y_true = self.masked_predict(batch)

        loss_fns = self.loss_fns[prefix]  # train, val, or test
        losses = [fn(y_pred, y_true) for fn in loss_fns]

        for loss_val, fn in zip(losses, loss_fns):
            self.log(
                f"{prefix}_{type(fn).__name__}",
                loss_val,
                prog_bar=True,
                on_epoch=False,
                on_step=True,
                sync_dist=True,
            )

        for metric_fn in self.metric_fns[prefix]:
            metric_fn(y_pred.detach(), y_true)

            self.log(
                f"{prefix}_{type(metric_fn).__name__}",
                metric_fn,
                prog_bar=True,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )

        return sum(losses)

    def training_step(self, batch, batch_idx):
        return self.step(batch, "TRAIN")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "VAL")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "TEST")

    def predict_step(self, batch, batch_idx):
        return {
            "data": batch["data"],
            "mito_pred": self(batch["data"]),
            "mito_true": batch["mito"],
            "tomo_name": batch["tomo_name"],
        }
