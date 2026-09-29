# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import anndata
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule
from cellarium.ml.api.data_analysis import CellariumData
from cellarium.ml.cli import compute_var_names_g
from cellarium.ml.models import StreamingPlaidGeometricSketch
from cellarium.ml.transforms import Log1p, NormalizeTotal


def geometric_sketch(
    cdata: CellariumData,
    target_n_cells: int = 1_000_000,
    embedding_module: CellariumModule | None = None,
    return_new_adata: bool = True,
) -> dict[str, anndata.AnnData | CellariumModule | pd.Series]:
    """
    Train a plaid geometric sketching model on the data in the datamodule,
    given a trained embedding model such as PCA or scVI.

    Args:
        cdata: :class:`CellariumData` instance containing the data.
        target_n_cells: The target number of cells to select using geometric sketching. This is very
            approximate.
        embedding_module: A trained :class:`CellariumModule` containing an embedding model such as PCA or scVI.
            If not provided, the embedding will be a random matrix projection, after NormalizeTotal and Log1p.
        return_new_adata: Whether to return a new AnnData object with the selected geometric sketch cells.

    Returns:
        Dict with keys:
            "obs_names_in_sketch": Pandas series boolean mask with index as all obs_names.
            "adata": The new AnnData object containing only the selected geometric sketch cells
                (if `return_new_adata` is True).
            "module": The trained StreamingPlaidGeometricSketch module (if `return_new_adata` is False).
        Note: updates cdata.datamodule.obs['in_sketch'] with a boolean mask for selected geometric sketch cells.
    """
    datamodule: CellariumAnnDataDataModule = cdata.datamodule
    if "obs_names_n" not in datamodule.batch_keys:
        raise ValueError("batch_keys in the datamodule needs to contain key 'obs_names_n' for geometric_sketch.")

    if embedding_module is None:
        embedding_module = CellariumModule(
            transforms=[
                NormalizeTotal(),
                Log1p(),
                torch.nn.Linear(in_features=datamodule.dadc.shape[1], out_features=128),
            ],
        )

    var_names_g = compute_var_names_g(
        cpu_transforms=embedding_module.cpu_transforms,  # type: ignore[arg-type]
        transforms=embedding_module.transforms + [embedding_module.model],  # type: ignore[arg-type]
        data=datamodule,
    )
    print("Computed variable names for geometric sketching:", var_names_g)

    target_bucket_ncells = 5
    min_cells_per_bucket_qc_threshold = 2

    module = CellariumModule(
        transforms=[embedding_module],
        model=StreamingPlaidGeometricSketch(
            var_names_g=var_names_g,
            target_voxels=target_n_cells // target_bucket_ncells,
            min_cells_per_bucket=min_cells_per_bucket_qc_threshold,
            max_cells_per_bucket=target_bucket_ncells,
            projector=None,
            store_cell_data=return_new_adata,
        ),
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(module, datamodule)

    reservoir = module.model.get_reservoir(return_cell_data=return_new_adata, max_cells=target_n_cells)
    sketch_obs_names = reservoir["obs_names"]

    datamodule_shuffle = datamodule.shuffle
    datamodule.shuffle = False
    predict_loader = datamodule.train_dataloader()
    obs_names_list = []
    embedding_list = []
    for batch in predict_loader:
        if "obs_names_n" not in batch:
            raise ValueError("batch_keys in the datamodule needs to contain 'obs_names_n' for geometric_sketch.")
        obs_names_list.append(batch["obs_names_n"])
        if return_new_adata:
            embedding_list.append(embedding_module(batch)["x_ng"])
    obs_names = np.concatenate(obs_names_list)
    if return_new_adata:
        embedding = torch.cat(embedding_list, dim=0).detach().cpu().numpy()
    datamodule.shuffle = datamodule_shuffle

    ordered_sketch_mask = np.isin(obs_names, sketch_obs_names)
    sketch_series = pd.Series(ordered_sketch_mask, index=obs_names)
    datamodule.dadc._obs["in_sketch"] = ordered_sketch_mask

    adata = None
    if return_new_adata:
        adata = anndata.AnnData(
            X=reservoir["x_ng"],
            obs=pd.DataFrame(index=reservoir["obs_names"]),
            var=datamodule.dadc.adatas[0].var.loc[var_names_g],
            obsm={"X_embedding": embedding},
        )

    return {
        "obs_names_in_sketch": sketch_series,
        "adata": adata,
        "module": module,
    }
