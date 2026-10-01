# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule
from cellarium.ml.api._datamodule_context import temporary_batch_keys, temporary_val_split
from cellarium.ml.api.data_analysis import CellariumData
from cellarium.ml.models import SingleCellVariationalInference
from cellarium.ml.transforms import Densify
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
    early_stopping_patience: int = 20,
    scvi_kwargs: dict | None = None,
    accelerator: Literal["cpu", "mps", "cuda", "auto"] = "auto",
) -> CellariumModule:
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
        early_stopping_patience: Number of epochs with no improvement in validation loss (ELBO)
            before stopping early.
        scvi_kwargs: Additional keyword arguments passed to :class:`SingleCellVariationalInference`,
            overriding the defaults (e.g. ``encoder``, ``decoder``, ``dispersion``, ``gene_likelihood``).
            For advanced use; most users should only need ``n_latent``.
        accelerator: The accelerator to use for training the module, in ["cpu", "mps", "cuda", "auto"].

    Returns:
        A :class:`CellariumModule` instance containing the trained scVI model.

    Note:
        Temporarily mutates ``cdata.datamodule`` (injecting a ``"batch_index_n"`` batch key,
        restricting ``obs_columns_to_validate`` to ``batch_key``, and setting aside ``val_size``
        for validation) for the duration of training, and restores the original state afterward.
    """
    if not batch_key:
        raise ValueError("batch_key must be provided; scvi() always performs batch correction.")

    datamodule: CellariumAnnDataDataModule = cdata.datamodule

    n_batch = datamodule.obs_key_nunique(batch_key)

    with (
        temporary_batch_keys(
            datamodule, {"batch_index_n": AnnDataField(attr="obs", key=batch_key, convert_fn=categories_to_codes)}
        ),
        temporary_val_split(datamodule, val_size=val_size),
    ):
        model_kwargs = {
            "encoder": DEFAULT_ENCODER,
            "decoder": DEFAULT_DECODER,
            **(scvi_kwargs or {}),
        }
        module = CellariumModule(
            transforms=[Densify()],
            model=SingleCellVariationalInference(
                var_names_g=datamodule.var_names_g,
                n_batch=n_batch,
                n_latent=n_latent,
                **model_kwargs,
            ),
        )

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices="auto",
            max_epochs=max_epochs,
            callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=early_stopping_patience)],
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(module, datamodule)

    return module
