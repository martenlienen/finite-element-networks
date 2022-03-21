from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn

from ..data import STBatch
from ..fen import FEN, FENQuery
from .metrics import main_metrics


class SequenceRegressionTask(pl.LightningModule):
    def __init__(
        self, model: FEN, *, loss: Optional[nn.Module] = None, standardize: bool = True
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("model",))

        self.model = model
        self.loss = loss or nn.L1Loss()
        self.standardize = standardize

        metrics = main_metrics()
        self.metrics = nn.ModuleDict(
            {
                # The underscores are here because ModuleDict is stupid and does not allow
                # you to have a 'train' key because it also has a `train` method
                "train_": metrics.clone(prefix="train/"),
                "val_": metrics.clone(prefix="val/"),
                "test_": metrics.clone(prefix="test/"),
            }
        )

    def forward(self, batch: STBatch):
        query = FENQuery.from_batch(batch, standardize=self.standardize)
        u_hat = self.model(query)
        if self.standardize:
            u_hat = batch.target_standardizer.undo(u_hat)

        return u_hat

    def training_step(self, batch: STBatch, batch_idx: int):
        u_hat = self(batch)
        loss = self.loss(u_hat, batch.target_u)

        self.log("train/loss", loss, batch_size=batch.batch_size)
        self.log_model_metrics("train", u_hat, batch)
        return {"loss": loss}

    def validation_step(self, batch: STBatch, batch_idx: int):
        u_hat = self(batch)
        self.log_model_metrics("val", u_hat, batch)
        return {}

    def test_step(self, batch: STBatch, batch_idx: int):
        u_hat = self(batch)
        self.log_model_metrics("test", u_hat, batch)
        return {}

    def log_model_metrics(self, stage: str, u_hat, batch):
        self.log_model_stats(stage, u_hat, batch)
        with torch.no_grad():
            values = self.metrics[stage + "_"](u_hat, batch.target_u)
            self.log_dict(values, batch_size=batch.batch_size)

    def log_model_stats(self, stage: str, u_hat, batch):
        nfe = float(self.model.stats.mean_nfe)
        self.log(f"{stage}/nfe", nfe, batch_size=batch.batch_size, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())
