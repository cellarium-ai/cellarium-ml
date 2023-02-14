from typing import Optional

import pyro
import pytorch_lightning as pl
import torch


class PyroTrainingPlan(pl.LightningModule):
    """
    Lightning module task to train Pyro scvi-tools modules.

    Args:
        pyro_module
            An instance of :class:`~scvi.module.base.PyroBaseModuleClass`. This object
            should have callable `model` and `guide` attributes or methods.
        loss_fn
            A Pyro loss. Should be a subclass of :class:`~pyro.infer.ELBO`.
            If `None`, defaults to :class:`~pyro.infer.Trace_ELBO`.
        optim
            A Pytorch optimizer class, e.g., :class:`~torch.optim.Adam`. If `None`,
            defaults to :class:`torch.optim.Adam`.
        optim_kwargs
            Keyword arguments for optimiser. If `None`, defaults to `dict(lr=1e-3)`.
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        scale_elbo
            Scale ELBO using :class:`~pyro.poutine.scale`. Potentially useful for avoiding
            numerical inaccuracy when working with very large ELBO.
    """

    def __init__(
        self,
        pyro_module,
        loss_fn: Optional[pyro.infer.ELBO] = None,
        optim: Optional[torch.optim.Adam] = None,
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

    def training_step(self, batch, batch_idx):
        """Training step for Pyro training."""
        args, kwargs = self.module._get_fn_args_from_batch(batch)
        loss = self.loss_fn(self.module.model, self.module.guide, *args, **kwargs)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure optimizers for the model."""
        return self.optim(self.module.parameters(), **self.optim_kwargs)
