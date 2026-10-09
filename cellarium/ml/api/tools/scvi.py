# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule
from cellarium.ml.api._datamodule_context import temporary_batch_keys, temporary_val_split
from cellarium.ml.api.cellariumdata import CellariumData, fit_and_register
from cellarium.ml.api.utils import LossHistory, PreciseProgressBar
from cellarium.ml.models import SingleCellVariationalInference
from cellarium.ml.transforms import Densify, Filter
from cellarium.ml.utilities.data import AnnDataField, categories_to_codes

DEFAULT_ENCODER = {
    "hidden_layers": [
        {
            "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
            "init_args": {"out_features": 128, "label_to_bias_hidden_layers": []},
        }
    ],
    "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
}

DEFAULT_DECODER = {
    "hidden_layers": [
        {
            "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
            "init_args": {"out_features": 128, "label_to_bias_hidden_layers": []},
        }
    ],
    "final_layer": {
        "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
        "init_args": {"label_to_bias_hidden_layers": []},
    },
    "final_additive_bias": False,
}


def scvi(
    cdata: CellariumData,
    batch_key: str,
    n_latent: int = 10,
    max_epochs: int = 200,
    val_size: float = 0.1,
    early_stopping_patience: int = 10,
    scvi_kwargs: dict | None = None,
    optim_kwargs: dict | None = None,
    key_added: str = "scvi",
    log_every_n_steps: int = 10,
    val_check_interval: int = 500,
    accelerator: Literal["cpu", "mps", "cuda", "auto"] = "auto",
) -> None:
    """
    Train an scVI model on the data in the datamodule, using ``batch_key`` for batch correction.

    Args:
        cdata: :class:`CellariumData` instance containing the data.
        batch_key: The ``obs`` column to use as the batch covariate for scVI's batch correction.
            Injected into ``cdata.datamodule.batch_keys`` as ``"batch_index_n"`` if not already present.
        n_latent: Dimension of the scVI latent space.
        max_epochs: Maximum number of training epochs (a fallback; training usually stops earlier
            via early stopping on the validation ELBO).
        val_size: Fraction of cells held out for validation (used for early stopping). Restored to
            its original value once training completes.
        early_stopping_patience: Number of validation checks (see ``val_check_interval``) with no improvement in
            validation loss (ELBO) before stopping early.
        scvi_kwargs: Additional keyword arguments passed to :class:`SingleCellVariationalInference`,
            overriding the defaults (e.g. ``encoder``, ``decoder``, ``dispersion``, ``gene_likelihood``).
            For advanced use; most users should only need ``n_latent``.
        optim_kwargs: Additional keyword arguments passed to the optimizer (default: ``{"lr": 1e-3}``).
        key_added: The key under which the trained module and its history are stored in ``cdata.trained_modules``,
            replacing any module already there under that key.
        log_every_n_steps: How often (in training steps) per-step metrics such as ``train_loss`` are recorded in the
            training history. Metrics logged once per validation check, such as ``val_loss``, are recorded then.
        val_check_interval: Run validation every this many training steps, counted globally across epochs (so checks
            need not fall on epoch boundaries), or once per epoch if an epoch has fewer steps, whichever is more
            frequent. Each check evaluates all of the ``val_size`` held-out cells, so for very large datasets
            consider also lowering ``val_size``.
        accelerator: The accelerator to use for training the module, in ["cpu", "mps", "cuda", "auto"].

    Returns:
        None

    Note:
        A :class:`CellariumModule` instance containing the trained scVI model is stored, along with the
        training history (``train_loss`` every ``log_every_n_steps`` steps, ``val_loss`` every validation check), as
        ``cdata.trained_modules[key_added]``. If training is interrupted (e.g. ``KeyboardInterrupt``), the partially
        trained module is stored the same way, marked ``complete=False``.

    Note:
        Temporarily mutates ``cdata.datamodule`` (injecting a ``"batch_index_n"`` batch key,
        restricting ``obs_columns_to_validate`` to ``batch_key``, and setting aside ``val_size``
        for validation) for the duration of training, and restores the original state afterward.
    """
    if not batch_key:
        raise ValueError("batch_key must be provided; scvi() always performs batch correction.")

    if optim_kwargs is None:
        optim_kwargs = {"lr": 1e-3}

    datamodule: CellariumAnnDataDataModule = cdata.datamodule

    n_batch = datamodule.obs_key_nunique(batch_key)

    if cdata.hvg is not None:
        filter = Filter(filter_list=cdata.hvg.index[cdata.hvg].tolist(), ordering=True)
        cpu_transforms = [filter]
        var_names_g = cdata.hvg.index[cdata.hvg].tolist()
    else:
        cpu_transforms = []
        var_names_g = datamodule.var_names_g

    # the data the module needs to run, recorded with it so that e.g. `cdata.using("scvi")` can provide it again
    batch_keys = {"batch_index_n": AnnDataField(attr="obs", key=batch_key, convert_fn=categories_to_codes)}

    with (
        temporary_batch_keys(datamodule, batch_keys),
        temporary_val_split(datamodule, val_size=val_size),
    ):
        scvi_kwargs = dict(scvi_kwargs or {})
        # The model allows only one of the two KL warmup arguments (and defaults to 400 warmup epochs), so default to
        # a 1000 step warmup only if the user sets neither, and otherwise turn off the one they did not set.
        if "kl_warmup_steps" not in scvi_kwargs and "kl_warmup_epochs" not in scvi_kwargs:
            scvi_kwargs["kl_warmup_steps"] = 1000
        scvi_kwargs.setdefault("kl_warmup_steps", None)
        scvi_kwargs.setdefault("kl_warmup_epochs", None)
        model_kwargs = {
            "encoder": DEFAULT_ENCODER,
            "decoder": DEFAULT_DECODER,
            **scvi_kwargs,
        }
        module = CellariumModule(
            cpu_transforms=cpu_transforms,
            transforms=[Densify()],
            model=SingleCellVariationalInference(
                var_names_g=var_names_g,
                n_batch=n_batch,
                n_latent=n_latent,
                **model_kwargs,
            ),
            optim_fn=torch.optim.AdamW,
            optim_kwargs=optim_kwargs,
        )

        loss_history = LossHistory()
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices="auto",
            max_epochs=max_epochs,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=early_stopping_patience),
                PreciseProgressBar(),
            ],
            logger=loss_history,
            log_every_n_steps=log_every_n_steps,
            # count val_check_interval in global training steps, ignoring epoch boundaries
            check_val_every_n_epoch=None,
            val_check_interval=val_check_interval,
            enable_checkpointing=False,
        )

        # An interval longer than an epoch would validate less than once per epoch, so cap it at the epoch length
        # (which makes the checks fall at the end of each epoch). Done after construction because the number of
        # replicas is only known to the trainer.
        datamodule.setup("fit")
        n_replicas = trainer.num_devices * trainer.num_nodes
        n_batches_per_epoch = max(1, len(datamodule.train_dataset) // n_replicas)
        trainer.val_check_interval = min(val_check_interval, n_batches_per_epoch)

        fit_and_register(
            cdata,
            trainer,
            module,
            key_added,
            config={"batch_key": batch_key, "n_latent": n_latent, "val_size": val_size},
            loss_history=loss_history,
            batch_keys=batch_keys,
        )

    return
