from copy import deepcopy

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from torch import nn


# add type params to functions


class BaseModel(LightningModule):
    def __init__(self, lr, weight_decay, losses, metrics):
        super(BaseModel, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay

        self.loss_fns = {
            "TRAIN": [deepcopy(fn) for fn in losses],
            "VAL": [deepcopy(fn) for fn in losses],
            "TEST": [deepcopy(fn) for fn in losses],
        }
        self.metric_fns = nn.ModuleDict(
            {
                "TRAIN": nn.ModuleList([deepcopy(fn) for fn in metrics]),
                "VAL": nn.ModuleList([deepcopy(fn) for fn in metrics]),
                "TEST": nn.ModuleList([deepcopy(fn) for fn in metrics]),
            }
        )

    def forward(self):
        raise NotImplementedError

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms, on_step=True)

    def _masked_predict(self, batch):
        data = batch["input"]
        y_true = batch["label"]

        y_pred_full = self(data)
        mask = (y_true > -1.0).detach()

        y_pred = torch.masked_select(y_pred_full, mask).view(-1, 1)
        y_true = torch.masked_select(y_true, mask).view(-1, 1)

        return y_pred, y_true, y_pred_full

    def _log_stats(self, losses, prefix):
        def log_stat(name, value, on_epoch, on_step):
            self.log(
                f"{prefix}_{name}",
                value,
                prog_bar=True,
                on_epoch=on_epoch,
                on_step=on_step,
                batch_size=1,
            )

        on_step = prefix == "TRAIN"

        # Log losses
        for loss_val, fn in zip(losses, self.loss_fns[prefix]):
            log_stat(type(fn).__name__, loss_val, not on_step, on_step)

        # Log metrics
        for metric_fn in self.metric_fns[prefix]:
            log_stat(type(metric_fn).__name__, metric_fn, True, False)

    def step(self, batch, prefix):
        y_pred, y_true, _ = self._masked_predict(batch)

        loss_fns = self.loss_fns[prefix]  # train, val, or test
        losses = [fn(y_pred, y_true) for fn in loss_fns]

        for metric_fn in self.metric_fns[prefix]:
            metric_fn(y_pred, y_true)

        self._log_stats(losses, prefix)
        return sum(losses)

    def training_step(self, batch, batch_idx):
        return self.step(batch, "TRAIN")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "VAL")

    def test_step(self, batch, batch_idx):
        y_pred, y_true, y_pred_full = self._masked_predict(batch)
        y_pred_full = (
            255
            * (y_pred_full - y_pred_full.min())
            / (y_pred_full.max() - y_pred_full.min())
        ).to(torch.uint8)

        results = {
            "sample": batch["sample"],
            "tomo_name": batch["tomo_name"],
            "data": batch["data"].cpu().numpy(),
            "label": batch["label"].cpu().numpy(),
            "preds": y_pred_full.cpu().numpy(),
        }

        for loss_fn in self.loss_fns["TEST"]:
            loss = loss_fn(y_pred, y_true)
            results[f"TEST_{type(loss_fn).__name__}"] = loss.item()

        for metric_fn in self.metric_fns["TEST"]:
            score = metric_fn(y_pred, y_true)
            results[f"TEST_{type(metric_fn).__name__}"] = score.item()
            metric_fn.reset()

        return results
