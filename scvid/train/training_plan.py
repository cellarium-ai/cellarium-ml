# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from importlib import import_module
from typing import Any

import lightning.pytorch as pl
import torch

from scvid.module import BaseModule


class TrainingPlan(pl.LightningModule):
    """
    Lightning module task to train scvi-distributed modules.

    Args:
        module: A scvid module to train.
        optim_fn: A Pytorch optimizer class, e.g., :class:`~torch.optim.Adam`. If `None`,
            defaults to :class:`torch.optim.Adam`.
        optim_kwargs: Keyword arguments for optimiser. If `None`, defaults to `dict(lr=1e-3)`.
        scheduler_fn: A Pytorch lr scheduler class, e.g., :class:`~torch.optim.lr_scheduler.CosineAnnealingLR`.
        scheduler_kwargs: Keyword arguments for lr scheduler.
    """

    def __init__(
        self,
        module: BaseModule,
        optim_fn: type[torch.optim.Optimizer] | str | None = None,
        optim_kwargs: dict | None = None,
        scheduler_fn: type[torch.optim.lr_scheduler.LRScheduler] | str | None = None,
        scheduler_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.module = module

        # set up optimizer and scheduler
        if isinstance(optim_fn, str):
            class_module, class_name = optim_fn.rsplit(".", 1)
            optim_module = import_module(class_module)
            self.optim_fn = getattr(optim_module, class_name)
        elif optim_fn is None:
            self.optim_fn = torch.optim.Adam
        else:
            self.optim_fn = optim_fn

        if isinstance(scheduler_fn, str):
            class_module, class_name = scheduler_fn.rsplit(".", 1)
            scheduler_module = import_module(class_module)
            self.scheduler_fn = getattr(scheduler_module, class_name)
        else:
            self.scheduler_fn = scheduler_fn

        optim_kwargs = {} if optim_kwargs is None else optim_kwargs
        if "lr" not in optim_kwargs:
            optim_kwargs["lr"] = 1e-3
        self.optim_kwargs = optim_kwargs
        self.scheduler_fn = scheduler_fn
        self.scheduler_kwargs = scheduler_kwargs

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor | None:
        args, kwargs = self.module._get_fn_args_from_batch(batch)
        loss = self.module(*args, **kwargs)
        if loss is not None:
            # Logging to TensorBoard by default
            self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers for the model."""
        optim_config = {}
        optim_config["optimizer"] = self.optim_fn(
            self.module.parameters(), **self.optim_kwargs
        )
        #  optim_config["optimizer"] = self.optim_fn(
        #      [
        #          {"params": self.module.W_kg_unconstrained, "lr": self.optim_kwargs["lr"]},
        #          {
        #              "params": self.module.sigma_unconstrained,
        #              "lr": self.optim_kwargs["lr"] / self.module.W_kg.numel(),
        #          },
        #      ],
        #      **self.optim_kwargs
        #  )
        if self.scheduler_fn is not None:
            assert self.scheduler_kwargs is not None
            scheduler = self.scheduler_fn(
                optim_config["optimizer"], **self.scheduler_kwargs
            )
            optim_config["lr_scheduler"] = {"scheduler": scheduler, "interval": "step"}
        return optim_config

    def on_train_epoch_start(self) -> None:
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


# make a picklable version of OneCycleLR
class OneCycleLR(torch.optim.lr_scheduler.OneCycleLR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anneal_strategy = kwargs.get("anneal_strategy", "cos")

    # remove anneal_func from state dict
    def state_dict(self):
        state_dict = super().state_dict()
        state_dict.pop("anneal_func")
        return state_dict

    # add anneal_func back to state dict
    def load_state_dict(self, state_dict):
        state_dict["anneal_func"] = self.anneal_func
        if state_dict["anneal_strategy"] == "cos":
            state_dict["anneal_func"] = self._annealing_cos
        if state_dict["anneal_strategy"] == "linear":
            state_dict["anneal_func"] = self._annealing_linear
        super().load_state_dict(state_dict)
