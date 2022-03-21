import tempfile
from dataclasses import replace

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.utilities import move_data_to_device

from ..data import STBatch
from ..fen import FEN
from ..plots import (
    animate_data_and_predictions,
    animate_disentanglement,
    animate_flow_fields,
)


def wandb_image_from_buffer(buffer: bytes, format: str) -> wandb.Image:
    with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as f:
        f.write(buffer)
        f.close()
        return wandb.Image(f.name)


def create_plots(batch: STBatch, u_hat: torch.Tensor, model, interval):
    animation = animate_data_and_predictions(batch, u_hat, interval=interval)
    plots = {"animation": wandb_image_from_buffer(animation, "webp")}

    if isinstance(model, FEN):
        animation = animate_disentanglement(batch, u_hat, model, interval=interval)
        plots["disentanglement"] = wandb_image_from_buffer(animation, "webp")

        has_transport_terms = len(model.dynamics.transport_terms) > 0
        if has_transport_terms:
            animation = animate_flow_fields(
                batch, u_hat, model, interval=interval, normalize=False
            )
            plots["flow-fields"] = wandb_image_from_buffer(animation, "webp")

            animation = animate_flow_fields(
                batch, u_hat, model, interval=interval, normalize=True
            )
            plots["normalized-flow-fields"] = wandb_image_from_buffer(animation, "webp")

    return plots


class PlotsCallback(pl.Callback):
    def __init__(self, stage: str = "val", fps: float = 2.5):
        super().__init__()

        self.stage = stage
        self.interval = int(1000 / fps)

        self.ready = True

    def on_sanity_check_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.ready = False

    def on_sanity_check_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.ready = True

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        if not self.ready:
            return

        batch = move_data_to_device(self._get_batch(trainer), pl_module.device)
        try:
            with torch.no_grad():
                pl_module.eval()
                u_hat = pl_module(batch)
        except AssertionError as e:
            if len(e.args) >= 1 and "underflow in dt" in e.args[0]:
                log.warning(f"Could not plot: {e}")
                return
            else:
                raise

        if pl_module.standardize:
            u_hat = batch.target_standardizer.do(u_hat)
            batch = replace(batch, u=batch.standardizer.do(batch.u))

        media = {"epoch": trainer.current_epoch}
        for key, plot in create_plots(
            batch, u_hat, pl_module.model, interval=self.interval
        ).items():
            media[f"{self.stage}/{key}"] = plot

        trainer.logger.log_metrics(media, step=trainer.global_step)

    def _get_batch(self, trainer: pl.Trainer) -> STBatch:
        if self.stage == "val":
            return trainer.datamodule.get_interesting_batch()
        elif self.stage == "train":
            return next(iter(trainer.datamodule.train_dataloader()))
        elif self.stage == "test":
            return next(iter(trainer.datamodule.test_dataloader()))
        else:
            assert False
