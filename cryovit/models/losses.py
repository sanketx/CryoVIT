"""Loss functions for training CryoVIT segmentation models."""

import torch
from torch import Tensor
from torch import nn
from torchvision.ops import sigmoid_focal_loss


class DiceLoss(nn.Module):
    """Dice Loss for imbalanced foreground / background segmentation tasks."""

    def __init__(self) -> None:
        """Initializes the DiceLoss instance."""
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Computes the Dice loss between predictions and true values.

        Args:
            y_pred (Tensor): Predicted probabilities, expected to be logits from the model.
            y_true (Tensor): Ground truth labels.

        Returns:
            Tensor: Computed Dice loss value.
        """
        y_pred = torch.sigmoid(y_pred)
        intersection = torch.sum(y_true * y_pred)
        denom = torch.sum(y_true) + torch.sum(y_pred)
        dice_loss = 1 - (2 * intersection) / (denom + 1e-3)

        return dice_loss


class FocalLoss(nn.Module):
    """Focal loss to address class imbalance by focusing more on hard-to-classify instances."""

    def __init__(self, gamma=2) -> None:
        """Initializes the FocalLoss instance.

        Args:
            gamma (float, optional): Focusing parameter. Default is 2.
        """
        super().__init__()
        self.gamma = gamma

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Computes the Focal loss between predictions and true values.

        Args:
            y_pred (Tensor): Predicted probabilities, expected to be logits from the model.
            y_true (Tensor): Ground truth labels.

        Returns:
            Tensor: Computed Focal loss value.
        """
        weight = (y_true.numel() - y_true.sum()) / y_true.numel()
        return sigmoid_focal_loss(
            y_pred,
            y_true,
            alpha=weight,
            gamma=self.gamma,
            reduction="mean",
        )
