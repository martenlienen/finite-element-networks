from dataclasses import dataclass

import numpy as np


@dataclass
class MeanStdAccumulator:
    """Accumulate the feature statistics of a dataset in a single pass."""

    total: int = 0
    s1: np.ndarray = 0.0
    s2: np.ndarray = 0.0

    @staticmethod
    def sum(accs: list["MeanStdAccumulator"]):
        return MeanStdAccumulator(
            total=sum(acc.total for acc in accs),
            s1=sum(acc.s1 for acc in accs),
            s2=sum(acc.s2 for acc in accs),
        )

    def add(self, u: np.ndarray):
        """Add more data to the stats."""

        u = u.astype(np.float64).reshape((-1, u.shape[-1]))
        self.total += u.shape[0]
        self.s1 += u.sum(axis=0)
        self.s2 += (u ** 2).sum(axis=0)

    def mean_and_std(self):
        mean = (self.s1 / self.total).astype(np.float32)
        std = (np.sqrt(self.s2 / self.total - mean ** 2)).astype(np.float32)
        return mean, std
