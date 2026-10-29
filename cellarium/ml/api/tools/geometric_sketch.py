# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


import warnings
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from typing import Any, Literal

import anndata
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from lightning.fabric.utilities.apply_func import move_data_to_device
from torch.utils._pytree import tree_map

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule, CellariumPipeline
from cellarium.ml.api._cell_ids import CELL_ID_COLUMN
from cellarium.ml.api._obsm_storage import ObsmWriter
from cellarium.ml.api.cellariumdata import CellariumData, TrainedModule, _device_of
from cellarium.ml.api.utils import resolve_device
from cellarium.ml.models import IncrementalPCA, StreamingPlaidGeometricSketch
from cellarium.ml.models.model import CellariumModel, PredictMixin, TransformPrediction
from cellarium.ml.transforms import Densify, Filter, Log1p, NormalizeTotal
from cellarium.ml.utilities.core import call_func_with_batch
from cellarium.ml.utilities.data import collate_fn, sparse_tensor_to_scipy_csr
from cellarium.ml.utilities.testing import assert_arrays_equal, assert_columns_and_array_lengths_equal


class RandomMatrixProjection(CellariumModel, PredictMixin):
    """
    Embeds ``x_ng`` via a fixed, untrained random matrix projection (i.e. a linear layer that is
    never trained). Used as :func:`geometric_sketch`'s default embedding when no
    ``embedding_module_key`` is provided, so that geometric sketching has some (arbitrary) notion of
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
    # Use a slice of up to one batch, not a single cell (`dadc[0]`), since models with BatchNorm (e.g. scVI)
    # need more than one cell per batch. Slicing works the same for h5ad and deltacells collections.
    adata = datamodule.dadc[: min(datamodule.batch_size, datamodule.dadc.n_obs)]
    batch = tree_map(lambda field: field(adata), datamodule.batch_keys)
    collated = collate_fn([batch])
    pipeline = CellariumPipeline(list(module.cpu_transforms or []) + list(module.transforms) + [module.model])
    # inference only: eval mode so BatchNorm uses (and doesn't update) its running stats
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            var_names_g = pipeline.predict(collated)["var_names_g"]
    finally:
        module.train(was_training)
    assert isinstance(var_names_g, np.ndarray)
    return var_names_g


@contextmanager
def _batch_embedder(module: CellariumModule, device: torch.device) -> Iterator[Callable[[dict[str, Any]], np.ndarray]]:
    """
    A function that embeds a batch that :meth:`CellariumData.predict_batches` yields without a module (so, with the raw
    counts) with ``module``, a trained embedding model, on ``device``: the module's ``cpu_transforms``, then the rest of
    its transforms and its model's ``predict``. The batch is left as it was. The module is in ``eval`` mode, and on
    ``device``, for the duration of the ``with`` block.
    """
    module.configure_model()  # idempotent; makes its transforms and model accessible
    was_training = module.training
    transforms_device, model_device = _device_of(module.transforms), _device_of(module.model)
    module.eval()
    module.transforms.to(device)
    module.model.to(device)

    @torch.no_grad()
    def embed_batch(batch: dict[str, Any]) -> np.ndarray:
        batch = dict(batch)  # the transforms put their results in the dictionary: not in the caller's
        if len(module.cpu_transforms) > 0:
            batch = module.cpu_transforms(batch)
        batch = module.transforms(move_data_to_device(batch, device))
        prediction = call_func_with_batch(module.model.predict, batch)
        return torch.as_tensor(prediction["x_ng"]).detach().cpu().numpy()

    try:
        yield embed_batch
    finally:
        module.train(was_training)
        if transforms_device is not None:
            module.transforms.to(transforms_device)
        if model_device is not None:
            module.model.to(model_device)


def geometric_sketch(
    cdata: CellariumData,
    target_n_cells: int = 1_000_000,
    embedding_module_key: str | None = None,
    n_pcs: int | None = None,
    return_new_adata: bool = True,
    accelerator: Literal["cpu", "mps", "cuda", "auto"] = "auto",
    seed: int = 0,
    store_embedding: bool = True,
) -> dict[str, anndata.AnnData | CellariumModule | pd.Series]:
    """
    Train a plaid geometric sketching model on the data in the datamodule,
    given a trained embedding model such as PCA or scVI.

    Args:
        cdata: :class:`CellariumData` instance containing the data.
        target_n_cells: The number of cells to select using geometric sketching. The sketch is exactly this
            size, unless the data has fewer cells than this (after pruning), in which case all of them are
            selected. The sketch is built as a grid over the embedding that is coarsened whenever it exceeds
            twice this many occupied voxels, keeping two random cells per voxel, and is brought to exactly this
            size at the end, as in the geometric sketching paper (see
            :class:`~cellarium.ml.models.StreamingPlaidGeometricSketch`).
        embedding_module_key: The key in ``cdata.trained_modules`` of a trained embedding model such as PCA
            (``"pca"``) or scVI (``"scvi"``), populated by training it with the api functions. Any data the model
            needs to run (such as scVI's batch column) is provided for you, see :meth:`CellariumData.using`.
            If not provided, the embedding will be a random matrix projection, after NormalizeTotal and Log1p.
            Raises ValueError if there is no trained module under this key.
        n_pcs: Number of principal components to use for the embedding if the embedding module is PCA.
            If None, all components are used. Raises ValueError if module is not PCA.
        return_new_adata: Whether to return a new AnnData object with the selected geometric sketch cells.
        accelerator: The accelerator to use for the pass over the data, in ["cpu", "mps", "cuda", "auto"]. The
            sketch itself is kept in tensors on the accelerator (except for "mps", where it is kept on the CPU),
            which is what makes it fast for millions of cells.
        seed: The random seed of the sketch: of the reservoir sampling and the final selection of cells, of the random
            projection (if there is no ``embedding_module_key``) and of the order in which the cells are streamed
            (``datamodule.shuffle_seed`` is set to it for the pass). The same seed on the same data gives the same
            sketch, up to floating point differences between devices. The cells are shuffled, not read in order,
            because the grid is set up from the first minibatch, which should be representative of the data.
        store_embedding: Whether to also embed every cell of ``cdata`` with the ``embedding_module_key`` module, in the
            second pass over the data (the one that collects the counts of the sketch cells), and store the embedding
            as ``cdata.obsm[f"X_{embedding_module_key}"]``, as :func:`~cellarium.ml.api.tools.embed` does, so that
            later steps (such as :func:`~cellarium.ml.api.tools.propagate_labels`) need no further pass over the counts
            to have the embedding of all the cells. It costs one embedding of each cell on the accelerator, and
            disk (or memory) for the embedding: about ``4 * n_dimensions`` bytes per cell, in a memory-mapped file in
            the temporary directory if it does not fit in ``cdata.max_gb`` (see :func:`~cellarium.ml.api.tools.embed`).
            It is the whole embedding of the module, also if ``n_pcs`` limits the dimensions that the sketch is made in.
            Nothing is stored (and nothing is embedded) if there is no ``embedding_module_key`` (a random
            projection is not an embedding to keep), if ``obsm`` already has the embedding of this module from
            :func:`~cellarium.ml.api.tools.embed`, or, with a warning, if it has an entry of that name that was not
            made by it. Pass ``False`` to leave this out.

    Returns:
        Dict with keys:
            "obs_names_in_sketch": Pandas series boolean mask over all the cells, indexed by
                :attr:`~cellarium.ml.api.CellariumData.cell_ids` (the key's name is historical).
            "adata": The new AnnData object containing only the selected geometric sketch cells
                (if `return_new_adata` is True), with the embedding in ``obsm[f"X_{embedding_module_key}"]``
                (``obsm["X_embedding"]`` if there is no ``embedding_module_key``), and the cell IDs of the cells in
                ``obs["cellarium_cell_id"]``.
            "module": The trained StreamingPlaidGeometricSketch module (if `return_new_adata` is False).
        Note: stores the same boolean mask in ``cdata.obs_computed['in_sketch']``.
    """
    if embedding_module_key is None:
        return _geometric_sketch(cdata, target_n_cells, None, n_pcs, return_new_adata, accelerator, seed)

    with cdata.using(embedding_module_key) as trained:
        stored_key = f"X_{embedding_module_key}"
        store: tuple[str, TrainedModule] | None = None
        if store_embedding:
            state = cdata._cached_embedding_state(embedding_module_key)
            if stored_key in cdata.obsm and state is None:
                warnings.warn(
                    f"obsm['{stored_key}'] exists and was not made by cml.tl.embed from the module "
                    f"'{embedding_module_key}', so it is not replaced by the embedding of all the cells.",
                    stacklevel=3,
                )
            elif state != "current":  # (if it is current, the embedding of all the cells is there already)
                store = (stored_key, trained)
        return _geometric_sketch(
            cdata,
            target_n_cells,
            trained.module,
            n_pcs,
            return_new_adata,
            accelerator,
            seed,
            embedding_key=stored_key,
            store=store,
        )


def _geometric_sketch(
    cdata: CellariumData,
    target_n_cells: int,
    embedding_module: CellariumModule | None,
    n_pcs: int | None,
    return_new_adata: bool,
    accelerator: Literal["cpu", "mps", "cuda", "auto"],
    seed: int,
    embedding_key: str = "X_embedding",
    store: tuple[str, TrainedModule] | None = None,
) -> dict[str, anndata.AnnData | CellariumModule | pd.Series]:
    datamodule: CellariumAnnDataDataModule = cdata.datamodule
    if "obs_names_n" not in datamodule.batch_keys:
        raise ValueError("batch_keys in the datamodule needs to contain key 'obs_names_n' for geometric_sketch.")

    if n_pcs is not None:
        if embedding_module is None:
            raise ValueError("n_pcs can only be specified if an embedding_module_key is provided.")
        else:
            if not isinstance(embedding_module.model, IncrementalPCA):
                raise ValueError("n_pcs can only be specified if the embedding module's model is IncrementalPCA.")

    if embedding_module is None:
        embedding_var_names_g = datamodule.var_names_g if cdata.hvg is None else np.asarray(cdata.hvg.index[cdata.hvg])
        # the projection is drawn from the global torch RNG: seed it for this, and leave the RNG as it was
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
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

    # The embedding module's own cpu_transforms (for example the library size normalization of sparse counts
    # in front of PCA) are made those of the sketch module, so that the dataloader runs them on the CPU, just as
    # when the embedding module was trained. The rest of the embedding module runs on the accelerator.
    embedding_module.configure_model()  # idempotent; makes its transforms and model accessible
    embedding_on_accelerator = CellariumModule(
        transforms=list(embedding_module.transforms),
        model=embedding_module.model,
        is_initialized=True,  # it is trained: do not reset its parameters
    )
    embedding_on_accelerator.configure_model()

    # The grid is allowed twice as many occupied voxels as there are cells in the sketch, and each voxel keeps
    # two cells: a coarsening event leaves between target_voxels / 2 and target_voxels voxels (more than
    # target_n_cells), and the model brings the sketch to exactly target_n_cells at the end by coarsening
    # a copy of the grid a little further, as in the geometric sketching paper.
    module = CellariumModule(
        cpu_transforms=list(embedding_module.cpu_transforms),
        transforms=[embedding_on_accelerator],
        model=StreamingPlaidGeometricSketch(
            var_names_g=var_names_g,
            target_voxels=2 * target_n_cells,
            target_n_cells=target_n_cells,
            min_cells_per_bucket=1,
            max_cells_per_bucket=2,
            projector=None,
            limit_input_to_top_pcs=n_pcs,
            store_cell_data=return_new_adata,
            seed=seed,
        ),
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    cdata._warn_if_read_amplified("The geometric sketch")
    # the retained cells are drawn as the cells stream by, so the order of the stream is part of the seed
    datamodule_shuffle_seed = datamodule.shuffle_seed
    datamodule.shuffle_seed = seed
    try:
        trainer.fit(module, datamodule)
    finally:
        datamodule.shuffle_seed = datamodule_shuffle_seed
        # the trainer and the module refer to each other, which would keep the module (and the sketch it holds, on
        # the accelerator) alive until the cyclic garbage collector ran; the module no longer needs the trainer
        module.trainer = None

    reservoir = module.model.get_reservoir(return_cell_data=return_new_adata, max_cells=target_n_cells)
    sketch_obs_names = reservoir["obs_names"]

    sketch_index = pd.Index(sketch_obs_names)

    # One more pass, to find the sketch cells in the data (by their position, which `predict_batches` gives for every
    # cell) and to collect their counts. Only the names of one batch at a time are looked up in the sketch, and the
    # mask of the sketch is one boolean per cell: nothing is kept for every cell but that (except the embedding of
    # every cell, if it is to be stored, which goes to `obsm`: to a file if it does not fit in memory).
    in_sketch = np.zeros(cdata.n_obs, dtype=bool)
    raw_x_ng_list = []
    raw_obs_names_list = []
    raw_positions_list = []
    writer: ObsmWriter | None = None
    with ExitStack() as stack:
        embed_batch = None
        if store is not None:
            assert embedding_module is not None
            embed_batch = stack.enter_context(_batch_embedder(embedding_module, resolve_device(accelerator)))
        for batch in cdata.predict_batches(desc="Collecting raw data for sketch cells"):
            positions = np.asarray(batch["row_indices_n"])
            if embed_batch is not None:
                values = embed_batch(batch)
                if writer is None:
                    writer = stack.enter_context(
                        ObsmWriter(cdata.n_obs, values.shape[1], cdata._max_bytes(), dtype=np.float32)
                    )
                if len(positions) > 1 and np.all(np.diff(positions) == 1):  # a contiguous run of cells
                    writer.array[positions[0] : positions[-1] + 1] = values
                else:
                    writer.array[positions] = values
            batch_obs_names = np.asarray(batch["obs_names_n"])
            mask = sketch_index.get_indexer(batch_obs_names) >= 0
            if mask.any():
                in_sketch[positions[mask]] = True
                if return_new_adata:
                    raw_x_ng_list.append(sparse_tensor_to_scipy_csr(batch["x_ng"])[mask])
                    raw_obs_names_list.append(batch_obs_names[mask])
                    raw_positions_list.append(positions[mask])
    if store is not None:
        assert writer is not None, "predict_batches yielded no batches"
        cdata.obsm[store[0]] = writer.result
        cdata._embedding_sources[store[0]] = (store[1], writer.result)

    sketch_series = pd.Series(in_sketch, index=cdata.cell_ids)
    cdata.obs_computed["in_sketch"] = sketch_series

    adata = None
    if return_new_adata:
        # Rows of raw_x_ng/raw_obs_names are kept in the order they were collected, not reservoir order: obs and X
        # just need to agree with each other, not with `sketch_obs_names`. Only the (small) embedding needs to be
        # reindexed to match.
        raw_obs_names = np.concatenate(raw_obs_names_list)
        coverage = pd.Index(raw_obs_names).get_indexer(sketch_obs_names)
        if (coverage == -1).any():
            missing = np.asarray(sketch_obs_names)[coverage == -1]
            raise ValueError(f"Could not find raw data for sketch obs_names, e.g. {missing[:5].tolist()}")
        raw_x_ng = sp.vstack(raw_x_ng_list, format="csr")
        del raw_x_ng_list  # the counts are in raw_x_ng now: do not hold a second copy of them

        embedding_pos = sketch_index.get_indexer(raw_obs_names)
        embedding = reservoir["x_ng"].to_dense().cpu().numpy()[embedding_pos]

        var = cdata.var.copy()
        if cdata.hvg is None:
            var["highly_variable"] = False
        else:
            var["highly_variable"] = cdata.hvg.reindex(var.index).fillna(False).astype(bool)

        raw_positions = np.concatenate(raw_positions_list)
        adata = anndata.AnnData(
            X=raw_x_ng,
            obs=pd.DataFrame({CELL_ID_COLUMN: cdata.cell_ids.to_numpy()[raw_positions]}, index=raw_obs_names),
            var=var,
            obsm={embedding_key: embedding},
        )

    return {
        "obs_names_in_sketch": sketch_series,
        "adata": adata,
        "module": module,
    }
