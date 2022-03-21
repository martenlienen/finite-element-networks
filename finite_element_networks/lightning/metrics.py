import numpy as np
import torch
from torchmetrics import Metric, MetricCollection, MetricTracker
from torchtyping import TensorType


class MeanAbsoluteError(Metric):
    def __init__(self):
        super().__init__()

        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: TensorType["time", "batch", "node", "feature"],
        target: TensorType["time", "batch", "node", "feature"],
    ):
        self.error += (preds - target).abs().sum()
        self.total += np.prod(preds.shape[:-1])

    def compute(self):
        if int(self.total) == 0:
            return 0.0
        else:
            return self.error / self.total


def main_metrics():
    metrics = {"mae": MeanAbsoluteError()}
    return MetricCollection(metrics)
