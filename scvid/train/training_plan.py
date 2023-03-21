# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import pyro
import pytorch_lightning as pl
import torch


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
        loss_fn: pyro.infer.elbo.ELBOModule,
        optim: Optional[torch.optim.Optimizer] = None,
        optim_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.loss_fn = loss_fn

        optim_kwargs = optim_kwargs if isinstance(optim_kwargs, dict) else dict()
        if "lr" not in optim_kwargs.keys():
            optim_kwargs.update({"lr": 1e-3})
        self.optim_kwargs = optim_kwargs

        self.optim = torch.optim.Adam if optim is None else optim

    def training_step(self, batch, batch_idx):
        """Training step for Pyro training."""
        args, kwargs = self.loss_fn._get_fn_args_from_batch(batch)
        loss = self.loss_fn(*args, **kwargs)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure optimizers for the model."""
        return self.optim(self.loss_fn.parameters(), **self.optim_kwargs)


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
