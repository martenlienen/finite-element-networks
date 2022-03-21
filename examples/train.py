#!/usr/bin/env python

import argparse
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch

import finite_element_networks as fen
from finite_element_networks import (
    FEN,
    MLP,
    FENDomainInfo,
    FENDynamics,
    FreeFormTerm,
    ODESolver,
    TransportTerm,
)
from finite_element_networks.lightning import (
    BlackSeaDataModule,
    CylinderFlowDataModule,
    MultipleShootingCallback,
    ScalarFlowDataModule,
    SequenceRegressionTask,
)

try:
    import wandb

    from finite_element_networks.lightning.wandb import PlotsCallback

    wandb_available = True
except:
    wandb_available = False

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints", default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument(
        "dataset",
        choices=["black-sea", "scalar-flow", "cylinder-flow"],
        help="Dataset name",
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoints
    dataset_name = args.dataset

    project_root = Path(fen.__file__).resolve().parent.parent
    data_root = project_root / "data" / dataset_name
    if dataset_name == "black-sea":
        dm_class = BlackSeaDataModule
        stationary, autonomous = False, False
        time_dim = 2
        n_features = 3
    elif dataset_name == "scalar-flow":
        dm_class = ScalarFlowDataModule
        stationary, autonomous = False, True
        time_dim = 1
        n_features = 4
    elif dataset_name == "cylinder-flow":
        dm_class = CylinderFlowDataModule
        stationary, autonomous = True, True
        time_dim = 1
        n_features = 3
    else:
        raise RuntimeError(f"Unknown dataset {dataset_name}")
    dm = dm_class(
        data_root,
        FENDomainInfo.from_domain,
        num_workers=2,
        pin_memory=True,
        train_target_steps=10,
        eval_target_steps=10,
        batch_size=1,
    )

    dynamics = FENDynamics(
        [
            FreeFormTerm(
                FreeFormTerm.build_coefficient_mlp(
                    n_features=n_features,
                    time_dim=time_dim,
                    space_dim=2,
                    hidden_dim=96,
                    n_layers=4,
                    non_linearity=torch.nn.Tanh,
                    stationary=stationary,
                    autonomous=autonomous,
                ),
                stationary=stationary,
                autonomous=autonomous,
                zero_init=True,
            ),
            TransportTerm(
                TransportTerm.build_flow_field_mlp(
                    n_features=n_features,
                    time_dim=time_dim,
                    space_dim=2,
                    hidden_dim=96,
                    n_layers=4,
                    non_linearity=torch.nn.Tanh,
                    stationary=stationary,
                    autonomous=autonomous,
                ),
                stationary=stationary,
                autonomous=autonomous,
                zero_init=True,
            ),
        ]
    )
    model = FEN(dynamics, ODESolver("dopri5", atol=1e-6, rtol=1e-6, adjoint=False))
    task = SequenceRegressionTask(model, standardize=True)

    logger = pl.loggers.WandbLogger(project="ref-impl") if wandb_available else None
    callbacks = [MultipleShootingCallback(initial_steps=3, increase=1)]
    if wandb_available:
        callbacks.append(pl.callbacks.ModelCheckpoint(monitor="val/mae", mode="min"))
        callbacks.append(PlotsCallback())
    else:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=checkpoint_dir, monitor="val/mae", mode="min"
            )
        )
    gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(max_epochs=20, callbacks=callbacks, gpus=gpus, logger=logger)
    trainer.fit(task, dm)
    trainer.test(task, dm)


if __name__ == "__main__":
    main()
