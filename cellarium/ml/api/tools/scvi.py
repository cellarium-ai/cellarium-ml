# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from collections.abc import Sequence
from typing import Literal

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping
from torch import nn

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule
from cellarium.ml.api._datamodule_context import (
    batch_index_field,
    check_batch_key,
    temporary_batch_keys,
    temporary_val_split,
)
from cellarium.ml.api.cellariumdata import CellariumData, fit_and_register
from cellarium.ml.api.utils import LossHistory, PreciseProgressBar, RestoreBestTrainableWeights
from cellarium.ml.models import ScArches, SingleCellVariationalInference
from cellarium.ml.transforms import Densify, Filter
from cellarium.ml.utilities.data import AnnDataField

DEFAULT_ENCODER = {
    "hidden_layers": [
        {
            "class_path": "cellarium.ml.models.scvi.LinearWithBatch",
            "init_args": {"out_features": 128, "label_to_bias_hidden_layers": []},
        }
    ],
    "final_layer": {"class_path": "torch.nn.Linear", "init_args": {}},
}

# The final layer of the decoder takes the batch: its bias (a linear map of the batch one-hot) is a per-gene, per-batch
# offset of the logits of the normalized expression, i.e. a per-gene, per-batch scaling of expression that is the same
# for every cell state. This is the batch effect of gene-specific capture efficiency, and it is what lets scArches
# calibrate a new batch's genes without changing what a point of the latent space means.
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


def _select_genes(cdata: CellariumData) -> tuple[list[Filter], list[str]]:
    """The ``cpu_transforms`` that keep the highly variable genes of ``cdata`` (if it has any), and the gene names."""
    if cdata.hvg is not None:
        var_names_g = cdata.hvg.index[cdata.hvg].tolist()
        return [Filter(filter_list=var_names_g, ordering=True)], var_names_g
    print("Warning: running on all genes since no highly variable genes are in the .hvg slot")
    return [], cdata.datamodule.var_names_g


def _batch_categories(cdata: CellariumData, batch_key: str) -> list[str]:
    """
    The categories of the (categorical) obs column ``batch_key``, in the order of their codes, which are the batch
    indices of the cells (see :func:`~cellarium.ml.utilities.data.batch_categories_to_codes`). Recorded in the
    ``config`` of a trained module, so that the batch of each index is known wherever the module is used.
    """
    return [str(category) for category in cdata.datamodule.dadc.obs_categories(batch_key)]


def _model_kwargs(scvi_kwargs: dict | None) -> dict:
    """
    The keyword arguments of an scVI (family) model: the default encoder, decoder and KL warmup,
    then those of the user.
    """
    scvi_kwargs = dict(scvi_kwargs or {})
    # The model allows only one of the two KL warmup arguments (and defaults to 400 warmup epochs), so default to
    # a 1000 step warmup only if the user sets neither, and otherwise turn off the one they did not set.
    if "kl_warmup_steps" not in scvi_kwargs and "kl_warmup_epochs" not in scvi_kwargs:
        scvi_kwargs["kl_warmup_steps"] = 1000
    scvi_kwargs.setdefault("kl_warmup_steps", None)
    scvi_kwargs.setdefault("kl_warmup_epochs", None)
    return {
        "encoder": DEFAULT_ENCODER,
        "decoder": DEFAULT_DECODER,
        **scvi_kwargs,
    }


def _with_decoder_final_additive_bias(scvi_kwargs: dict | None) -> dict:
    """
    ``scvi_kwargs`` with ``final_additive_bias`` turned on in the decoder: that of ``scvi_kwargs`` if it has one, and
    otherwise the default decoder (a copy in either case, so neither is changed).
    """
    scvi_kwargs = dict(scvi_kwargs or {})
    decoder = copy.deepcopy(scvi_kwargs.get("decoder", DEFAULT_DECODER))
    decoder["final_additive_bias"] = True
    scvi_kwargs["decoder"] = decoder
    return scvi_kwargs


def _train(
    cdata: CellariumData,
    model: SingleCellVariationalInference | ScArches,
    cpu_transforms: Sequence[nn.Module],
    train_batch_keys: dict[str, AnnDataField],
    recorded_batch_keys: dict[str, AnnDataField],
    key_added: str,
    config: dict,
    max_epochs: int,
    val_size: float,
    early_stopping_patience: int,
    optim_kwargs: dict | None,
    log_every_n_steps: int,
    val_check_interval: int,
    accelerator: Literal["cpu", "mps", "cuda", "auto"],
    transforms: Sequence[nn.Module] | None = None,
    optim_fn: type[torch.optim.Optimizer] = torch.optim.AdamW,
    restore_best_weights: bool = False,
) -> None:
    """
    Train ``model`` (scVI or a model that extends or wraps it) on the data in ``cdata.datamodule``, with early stopping
    on the validation loss, and store the module in ``cdata.trained_modules[key_added]``.

    ``train_batch_keys`` are the data fields that the datamodule provides during training, and ``recorded_batch_keys``
    are the ones stored with the module, which is all that is needed for it to run (``cdata.using``) afterwards.
    ``transforms`` are those that run on the accelerator, by default ``[Densify()]``. With ``restore_best_weights``,
    the trainable parameters are put back to those of the best validation when training ends (see
    :class:`~cellarium.ml.api.utils.RestoreBestTrainableWeights`), instead of being those of the last step.
    """
    if optim_kwargs is None:
        optim_kwargs = {"lr": 1e-3}

    datamodule: CellariumAnnDataDataModule = cdata.datamodule

    with (
        temporary_batch_keys(datamodule, train_batch_keys),
        temporary_val_split(datamodule, val_size=val_size),
    ):
        module = CellariumModule(
            cpu_transforms=cpu_transforms,
            transforms=[Densify()] if transforms is None else list(transforms),
            model=model,
            optim_fn=optim_fn,
            optim_kwargs=optim_kwargs,
        )

        loss_history = LossHistory()
        callbacks: list[pl.Callback] = [
            EarlyStopping(monitor="val_loss", mode="min", patience=early_stopping_patience),
            PreciseProgressBar(),
        ]
        if restore_best_weights:
            callbacks.append(RestoreBestTrainableWeights(monitor="val_loss"))
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices="auto",
            max_epochs=max_epochs,
            callbacks=callbacks,
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
            config=config,
            loss_history=loss_history,
            batch_keys=recorded_batch_keys,
        )


def scvi(
    cdata: CellariumData,
    batch_key: str,
    n_latent: int = 10,
    decoder_final_additive_bias: bool = False,
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
        decoder_final_additive_bias: If True, the decoder adds a per-gene, per-batch number of background counts to
            the mean counts of each cell (independent of its library size and state), like ambient RNA or the
            spillover of segmentation. It is in addition to the per-gene, per-batch scaling of expression of the final
            layer of the decoder, and :func:`~cellarium.ml.api.tools.scarches` trains both for new batches. It turns on
            ``final_additive_bias`` in the decoder of ``scvi_kwargs`` if that has one, and otherwise in the default
            decoder. If False (default), the decoder is left as it is.
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
    check_batch_key(cdata.datamodule, batch_key)
    if decoder_final_additive_bias:
        scvi_kwargs = _with_decoder_final_additive_bias(scvi_kwargs)

    n_batch = cdata.datamodule.obs_key_nunique(batch_key)
    cpu_transforms, var_names_g = _select_genes(cdata)

    # the data the module needs to run, recorded with it so that e.g. `cdata.using("scvi")` can provide it again
    batch_keys = {"batch_index_n": batch_index_field(batch_key)}

    model = SingleCellVariationalInference(
        var_names_g=var_names_g,
        n_batch=n_batch,
        n_latent=n_latent,
        **_model_kwargs(scvi_kwargs),
    )
    _train(
        cdata,
        model,
        cpu_transforms,
        train_batch_keys=batch_keys,
        recorded_batch_keys=batch_keys,
        key_added=key_added,
        config={
            "batch_key": batch_key,
            "n_latent": n_latent,
            "val_size": val_size,
            "batch_categories": _batch_categories(cdata, batch_key),
            "decoder_final_additive_bias": model.decoder.final_additive_bias_layer is not None,
        },
        max_epochs=max_epochs,
        val_size=val_size,
        early_stopping_patience=early_stopping_patience,
        optim_kwargs=optim_kwargs,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        accelerator=accelerator,
    )
