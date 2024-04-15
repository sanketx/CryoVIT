"""Base Model class for 3D Tomogram Segmentation."""

from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from torch import Tensor
from torch import nn
from torch.optim import Optimizer
from torchmetrics import Metric


class BaseModel(LightningModule):
    """Base model with configurable loss functions and metrics."""

    def __init__(
        self,
        lr: float,
        weight_decay: float,
        losses: List[nn.Module],
        metrics: List[Metric],
    ) -> None:
        """Initializes the BaseModel with specified learning rate, weight decay, loss functions, and metrics.

        Args:
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay factor for AdamW optimizer.
            losses (List[nn.Module]): List of loss function instances for training, validation, and testing.
            metrics (List[Metric]): List of metric function instances for training, validation, and testing.
        """
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
        """Should be implemented in subclass."""
        raise NotImplementedError("The forward method must be implemented by subclass.")

    def configure_optimizers(self) -> Optimizer:
        """Configures the optimizer with the initialization parameters."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        """Logs gradient norms just before the optimizer updates weights."""
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms, on_step=True)

    def _masked_predict(
        self, batch: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Performs prediction while applying a mask to the inputs and labels based on the label value."""
        data = batch["input"]
        y_true = batch["label"]

        y_pred_full = self(data)
        mask = (y_true > -1.0).detach()

        y_pred = torch.masked_select(y_pred_full, mask).view(-1, 1)
        y_true = torch.masked_select(y_true, mask).view(-1, 1)

        return y_pred, y_true, y_pred_full

    def _log_stats(self, losses: List[Tensor], prefix: str) -> None:
        """Logs computed loss and metric statistics for each training or validation step."""

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

    def step(self, batch: Dict[str, Tensor], prefix: str) -> float:
        """Processes a single batch of data, computes the loss and updates metrics."""
        y_pred, y_true, _ = self._masked_predict(batch)

        loss_fns = self.loss_fns[prefix]  # train, val, or test
        losses = [fn(y_pred, y_true) for fn in loss_fns]

        for metric_fn in self.metric_fns[prefix]:
            metric_fn(y_pred, y_true)

        self._log_stats(losses, prefix)
        return sum(losses)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> float:
        """Processes one batch during training."""
        return self.step(batch, "TRAIN")

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> float:
        """Processes one batch during validation."""
        return self.step(batch, "VAL")

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Any]:
        """Processes one batch during testing, captures predictions, and computes losses and metrics.

        Args:
            batch (Dict[str, Tensor]): The batch of data being processed.
            batch_idx (int): Index of the batch.

        Returns:
            Dict[str, Any]: A dictionary containing test results and metrics for this batch.
        """
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
