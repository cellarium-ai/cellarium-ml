# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import pyro
import pytorch_lightning as pl
import torch
import math


class PyroTrainingPlan(pl.LightningModule):
    """
    Lightning module task to train Pyro scvi-tools modules.

    .. note:: This is a stripped down version of the :class:`~scvi.train.LowLevelPyroTrainingPlan`.
        https://github.com/scverse/scvi-tools/blob/bf2121975bdfc31bfb1f6feb6446c331188b47dd/scvi/train/_trainingplans.py#L745

    Args:
        pyro_module: A Pyro module. This object should have callable `model` and `guide` attributes or methods.
        loss_fn: A Pyro loss. Should be a subclass of :class:`~pyro.infer.ELBO`.
            If `None`, defaults to :class:`~pyro.infer.Trace_ELBO`.
        optim: A Pytorch optimizer class, e.g., :class:`~torch.optim.Adam`. If `None`,
            defaults to :class:`torch.optim.Adam`.
        optim_kwargs: Keyword arguments for optimiser. If `None`, defaults to `dict(lr=1e-3)`.
    """

    def __init__(
        self,
        pyro_module: pyro.nn.PyroModule,
        loss_fn: Optional[pyro.infer.ELBO] = None,
        optim: Optional[torch.optim.Optimizer] = None,
        optim_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.module = pyro_module

        optim_kwargs = optim_kwargs if isinstance(optim_kwargs, dict) else dict()
        if "lr" not in optim_kwargs.keys():
            optim_kwargs.update({"lr": 1e-3})
        self.optim_kwargs = optim_kwargs

        self.loss_fn = (
            pyro.infer.Trace_ELBO().differentiable_loss if loss_fn is None else loss_fn
        )
        self.optim = torch.optim.Adam if optim is None else optim
        # self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        """Training step for Pyro training."""
        args, kwargs = self.module._get_fn_args_from_batch(batch)
        loss = self.loss_fn(self.module.model, self.module.guide, *args, **kwargs)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        self.module.log(self)
        return loss

    def configure_optimizers(self):
        """Configure optimizers for the model."""
        optimizer = self.optim(self.module.parameters(), **self.optim_kwargs)
        # return optimizer
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.01)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000, eta_min=1e-3)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50_000, T_mult=1)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.optim_kwargs["lr"], total_steps=15000)
        # scheduler = OneCycleLR(optimizer, max_lr=self.optim_kwargs["lr"], total_steps=16500, pct_start=0.1, anneal_strategy="linear")
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, anneal_fn)
        # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-2, total_iters=10000)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_train_epoch_start(self):
        """
        Calls the ``set_epoch`` method on the iterable dataset of the given dataloader.

        If the dataset is ``IterableDataset`` and has ``set_epoch`` method defined, then
        ``set_epoch`` must be called at the beginning of every epoch to ensure shuffling
        applies a new ordering. This has no effect if shuffling is off.
        """
        dataloaders = self.trainer.fit_loop._combined_loader.flattened
        for dataloader in dataloaders:
            dataset = dataloader.dataset
            set_epoch = getattr(dataset, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(self.current_epoch)


class DummyTrainingPlan(pl.LightningModule):
    """
    Lightning module task that does not perform any actual optimization (no gradient updates).
    It can be used for cases where only the forward pass is required (e.g., for calculating
    sufficient statistics, EM algorithms, etc.).
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._dummy_param = torch.nn.Parameter(torch.tensor(0.0))

    def training_step(self, batch, batch_idx):
        args, kwargs = self.module._get_fn_args_from_batch(batch)
        self.module(*args, **kwargs)

    def configure_optimizers(self):
        return torch.optim.SGD([self._dummy_param], lr=1.0)

    def on_train_epoch_start(self):
        """
        Calls the ``set_epoch`` method on the iterable dataset of the given dataloader.

        If the dataset is ``IterableDataset`` and has ``set_epoch`` method defined, then
        ``set_epoch`` must be called at the beginning of every epoch to ensure shuffling
        applies a new ordering. This has no effect if shuffling is off.
        """
        dataloaders = self.trainer.fit_loop._combined_loader.flattened
        for dataloader in dataloaders:
            dataset = dataloader.dataset
            set_epoch = getattr(dataset, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(self.current_epoch)


class OneCycleLR(torch.optim.lr_scheduler.OneCycleLR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anneal_strategy = kwargs.get("anneal_strategy", "cos")

    def state_dict(self):
        state = super().state_dict()
        # We are dropping the `_scale_fn_ref` attribute because it is a `weakref.WeakMethod` and can't be pickled
        state.pop("anneal_func")
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # Validate anneal_strategy
        anneal_strategy = state_dict["anneal_strategy"]
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError("anneal_strategy must by one of 'cos' or 'linear', instead got {}".format(anneal_strategy))
        elif anneal_strategy == 'cos':
            self.anneal_func = self._annealing_cos
        elif anneal_strategy == 'linear':
            self.anneal_func = self._annealing_linear

def anneal_fn(step):
    f = 1
    lr_max = 1
    lr_min = 0
    T = 50000
    e, step = divmod(step, T)
    return lr_min + f**e * 0.5 * (lr_max - lr_min) * (1 + math.cos(step / T * math.pi))
