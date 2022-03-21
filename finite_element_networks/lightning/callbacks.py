import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import lightning_getattr, lightning_setattr


class MultipleShootingCallback(pl.Callback):
    """This callback increases the length of the training sequences each epoch.

    This technique is well known in the SciML community and documented in their tutorials
    [1] as a way to avoid falling into local minima when training ODE based models. We can
    also see this as an instance of multiple shooting [2, 3] in the data space, where the
    penalty function enforcing the equality constraints at the splitting points is equal
    to the loss function.

    Note that the number of target steps will never increase over the initial number of
    target steps configured in the data module.

    [1] https://diffeqflux.sciml.ai/dev/examples/local_minima/
    [2] https://diffeqflux.sciml.ai/dev/examples/multiple_shooting/
    [3] Evren Mert Turan, Johannes Jäschke, "Multiple shooting for training neural
        differential equations on time series", https://arxiv.org/abs/2109.06786

    Attributes
    ----------
    initial_steps
        Number of target steps in the first epoch
    increase
        The target steps increase by this much in each following epoch
    target_steps_attr
        Name of the data module attribute that should be modified
    """

    def __init__(
        self,
        *,
        initial_steps: int = 3,
        increase: int = 1,
        target_steps_attr: str = "train_target_steps",
    ):
        super().__init__()

        self.initial_steps = initial_steps
        self.increase = increase
        self.target_steps_attr = target_steps_attr
        self.initial_target_steps = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.initial_target_steps = lightning_getattr(pl_module, self.target_steps_attr)

        # Set the initial steps in this hook because the trainer selects the train
        # dataloader internally before train_epoch_start is called.
        lightning_setattr(pl_module, self.target_steps_attr, self.initial_steps)
        trainer.reset_train_dataloader(pl_module)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.log(
            self.target_steps_attr,
            float(lightning_getattr(pl_module, self.target_steps_attr)),
            on_step=False,
            on_epoch=True,
        )

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Trainer loads the data loader before the train_epoch_start hook is called, so we
        # set the target steps already at the end of the previous epoch
        prev_target_steps = lightning_getattr(pl_module, self.target_steps_attr)
        target_steps = prev_target_steps + self.increase
        if self.initial_target_steps is not None:
            target_steps = min(target_steps, self.initial_target_steps)
        if target_steps != prev_target_steps:
            lightning_setattr(pl_module, self.target_steps_attr, target_steps)
            trainer.reset_train_dataloader(pl_module)
