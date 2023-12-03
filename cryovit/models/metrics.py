import torch
from torchmetrics import Metric


class DiceMetric(Metric):
    higher_is_better = True

    def __init__(self, threshold):
        super().__init__()
        self.thresh = threshold
        self.add_state("dice_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_pred = torch.where(y_pred < self.thresh, 0.0, 1.0)

        intersection = torch.sum(y_true * y_pred)
        denom = torch.sum(y_true) + torch.sum(y_pred)

        self.dice_score += 2 * intersection / (denom + 1e-3)
        self.total += 1

    def compute(self):
        return self.dice_score / self.total
