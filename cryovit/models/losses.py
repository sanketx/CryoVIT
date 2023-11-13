import torch
from torch import nn
from torchvision.ops import sigmoid_focal_loss


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        intersection = torch.sum(y_true * y_pred)
        denom = torch.sum(y_true) + torch.sum(y_pred)
        dice_loss = 1 - (2 * intersection) / (denom + 1e-3)

        return dice_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        weight = (y_true.numel() - y_true.sum()) / y_true.numel()
        return sigmoid_focal_loss(
            y_pred,
            y_true,
            alpha=weight,
            gamma=self.gamma,
            reduction="mean",
        )
