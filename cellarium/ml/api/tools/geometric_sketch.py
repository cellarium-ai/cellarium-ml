# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import anndata
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils._pytree import tree_map
from tqdm import tqdm

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule, CellariumPipeline
from cellarium.ml.api.cellariumdata import CellariumData
from cellarium.ml.models import IncrementalPCA, StreamingPlaidGeometricSketch
from cellarium.ml.models.model import CellariumModel, PredictMixin, TransformPrediction
from cellarium.ml.transforms import Densify, Filter, Log1p, NormalizeTotal
from cellarium.ml.utilities.data import AnnDataField, collate_fn, sparse_tensor_to_scipy_csr
from cellarium.ml.utilities.testing import assert_arrays_equal, assert_columns_and_array_lengths_equal


class RandomMatrixProjection(CellariumModel, PredictMixin):
    """
    Embeds ``x_ng`` via a fixed, untrained random matrix projection (i.e. a linear layer that is
    never trained). Used as :func:`geometric_sketch`'s default embedding when no
    ``embedding_module`` is provided, so that geometric sketching has some (arbitrary) notion of
    cell-to-cell distance to work with even without a trained embedding model like PCA or scVI.

    Args:
        var_names_g:
            The variable names schema for the input data validation.
        out_features:
            The dimensionality of the output random projection.
    """

    def __init__(self, var_names_g: np.ndarray, out_features: int = 128) -> None:
        super().__init__()
        self.var_names_g = var_names_g
        self.n_vars = len(var_names_g)
        self.out_features = out_features
        self.linear = torch.nn.Linear(in_features=self.n_vars, out_features=out_features, bias=False)
        self.linear.requires_grad_(False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.linear.reset_parameters()

    def forward(self, x_ng: torch.Tensor, var_names_g: np.ndarray) -> dict[str, torch.Tensor | None]:
        # no-op: this model is never trained, so there is nothing to accumulate across batches
        return {}

    def predict(self, x_ng: torch.Tensor, var_names_g: np.ndarray) -> TransformPrediction:
        """
        Randomly project the input data ``x_ng`` into a lower-dimensional space.

        Args:
            x_ng:
                Gene counts matrix.
            var_names_g:
                The list of the variable names in the input data.

        Returns:
            A dictionary with the following keys:

            - ``x_ng``: (misnomer) Random projection of the input data.
            - ``var_names_g``: The list of variable names for the output data.
        """
        assert_columns_and_array_lengths_equal("x_ng", x_ng, "var_names_g", var_names_g)
        assert_arrays_equal("var_names_g", var_names_g, "var_names_g", self.var_names_g)

        z_nk = self.linear(x_ng)
        var_names_k = np.array([f"random_dim{i}" for i in range(self.out_features)])
        return {"x_ng": z_nk, "var_names_g": var_names_k}


def compute_output_var_names_g(module: CellariumModule, datamodule: CellariumAnnDataDataModule) -> np.ndarray:
    # Run the embedding pipeline's `predict` (rather than `forward`, which is the training
    # step and doesn't update `var_names_g`) so that models like PCA report their actual
    # output var names (e.g. "PC1", "PC2", ...) instead of the input gene names.
    adata = datamodule.dadc[0]
    batch = tree_map(lambda field: field(adata), datamodule.batch_keys)
    collated = collate_fn([batch])
    pipeline = CellariumPipeline(list(module.cpu_transforms or []) + list(module.transforms) + [module.model])
    var_names_g = pipeline.predict(collated)["var_names_g"]
    assert isinstance(var_names_g, np.ndarray)
    return var_names_g


def geometric_sketch(
    cdata: CellariumData,
    target_n_cells: int = 1_000_000,
    embedding_module: CellariumModule | None = None,
    n_pcs: int | None = None,
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
        n_pcs: Number of principal components to use for the embedding if the embedding module is PCA.
            If None, all components are used. Raises ValueError if module is not PCA.
        return_new_adata: Whether to return a new AnnData object with the selected geometric sketch cells.

    Returns:
        Dict with keys:
            "obs_names_in_sketch": Pandas series boolean mask with index as all obs_names.
            "adata": The new AnnData object containing only the selected geometric sketch cells
                (if `return_new_adata` is True).
            "module": The trained StreamingPlaidGeometricSketch module (if `return_new_adata` is False).
        Note: stores the same boolean mask (a pandas series indexed by obs_names) in
        ``cdata.obs_computed['in_sketch']``.
    """
    datamodule: CellariumAnnDataDataModule = cdata.datamodule
    if "obs_names_n" not in datamodule.batch_keys:
        raise ValueError("batch_keys in the datamodule needs to contain key 'obs_names_n' for geometric_sketch.")

    if n_pcs is not None:
        if embedding_module is None:
            raise ValueError("n_pcs can only be specified if an embedding_module is provided.")
        else:
            if not isinstance(embedding_module.model, IncrementalPCA):
                raise ValueError("n_pcs can only be specified if the embedding_module's model is IncrementalPCA.")

    if embedding_module is None:
        embedding_var_names_g = datamodule.var_names_g if cdata.hvg is None else np.asarray(cdata.hvg.index[cdata.hvg])
        embedding_module = CellariumModule(
            cpu_transforms=(
                [] if cdata.hvg is None else [Filter(filter_list=cdata.hvg.index[cdata.hvg], ordering=True)]
            ),
            transforms=[
                Densify(),
                NormalizeTotal(),
                Log1p(),
            ],
            model=RandomMatrixProjection(var_names_g=embedding_var_names_g, out_features=128),
        )
        # this module is used directly below (not via `trainer.fit`), so it must be configured manually
        embedding_module.configure_model()
    var_names_g = compute_output_var_names_g(embedding_module, datamodule)

    target_bucket_ncells = 5
    min_cells_per_bucket_qc_threshold = 1

    module = CellariumModule(
        transforms=[embedding_module],
        model=StreamingPlaidGeometricSketch(
            var_names_g=var_names_g,
            target_voxels=target_n_cells // target_bucket_ncells,
            min_cells_per_bucket=min_cells_per_bucket_qc_threshold,
            max_cells_per_bucket=target_bucket_ncells,
            projector=None,
            limit_input_to_top_pcs=n_pcs,
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

    sketch_index = pd.Index(sketch_obs_names)

    datamodule_shuffle = datamodule.shuffle
    datamodule.shuffle = False
    predict_loader = datamodule.train_dataloader()
    obs_names_list = []
    raw_x_ng_list = []
    raw_obs_names_list = []
    for batch in tqdm(predict_loader, desc="Collecting raw data for sketch cells"):
        if "obs_names_n" not in batch:
            raise ValueError("batch_keys in the datamodule needs to contain 'obs_names_n' for geometric_sketch.")
        batch_obs_names = batch["obs_names_n"]
        obs_names_list.append(batch_obs_names)
        if return_new_adata:
            mask = sketch_index.get_indexer(batch_obs_names) >= 0
            if mask.any():
                raw_x_ng_list.append(sparse_tensor_to_scipy_csr(batch["x_ng"])[mask])
                raw_obs_names_list.append(batch_obs_names[mask])
    obs_names = np.concatenate(obs_names_list)
    datamodule.shuffle = datamodule_shuffle

    ordered_sketch_mask = sketch_index.get_indexer(obs_names) >= 0
    sketch_series = pd.Series(ordered_sketch_mask, index=obs_names)
    cdata.obs_computed["in_sketch"] = sketch_series

    adata = None
    if return_new_adata:
        # Rows of raw_x_ng/raw_obs_names are kept in the order they were collected (dataset
        # streaming order), not reservoir order: obs and X just need to agree with each other,
        # not with `sketch_obs_names`. Only the (small) embedding needs to be reindexed to match.
        raw_obs_names = np.concatenate(raw_obs_names_list)
        coverage = pd.Index(raw_obs_names).get_indexer(sketch_obs_names)
        if (coverage == -1).any():
            missing = np.asarray(sketch_obs_names)[coverage == -1]
            raise ValueError(f"Could not find raw data for sketch obs_names, e.g. {missing[:5].tolist()}")
        raw_x_ng = sp.vstack(raw_x_ng_list, format="csr")

        embedding_pos = sketch_index.get_indexer(raw_obs_names)
        embedding = reservoir["x_ng"].to_dense().cpu().numpy()[embedding_pos]

        var = datamodule.dadc.var
        ad_field = datamodule.batch_keys["var_names_g"]
        assert isinstance(ad_field, AnnDataField)
        var_col = ad_field.key
        var = var.set_index(var_col).copy() if var_col is not None else var.copy()
        if cdata.hvg is None:
            var["highly_variable"] = False
        else:
            var["highly_variable"] = cdata.hvg.reindex(var.index).fillna(False).astype(bool)

        adata = anndata.AnnData(
            X=raw_x_ng,
            obs=pd.DataFrame(index=raw_obs_names),
            var=var,
            obsm={"X_embedding": embedding},
        )

    return {
        "obs_names_in_sketch": sketch_series,
        "adata": adata,
        "module": module,
    }
