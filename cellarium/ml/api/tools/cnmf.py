# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import gc
import os
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.fabric.utilities.apply_func import move_data_to_device
from tqdm.auto import tqdm

from cellarium.ml import CellariumModule, CellariumPipeline
from cellarium.ml.api import _cnmf_consensus as cc
from cellarium.ml.api import _cnmf_plots as plots
from cellarium.ml.api._data_transforms import OnePassRecipe, fit_or_reuse
from cellarium.ml.api._obsm_storage import ObsmWriter
from cellarium.ml.api.cellariumdata import CellariumData, _device_of, fit_and_register
from cellarium.ml.api.utils import LossHistory, PreciseProgressBar, resolve_device
from cellarium.ml.models import AmortizedOnlineNonNegativeMatrixFactorization, OnlineNonNegativeMatrixFactorization
from cellarium.ml.models.ols import StreamingOrdinaryLeastSquares
from cellarium.ml.transforms import Densify, DivideByScale, Filter, NormalizeTotal, ZScore

_FORMAT_VERSION = 1
_KOTLIAR_DENSITY_THRESHOLD = 0.5
_KOTLIAR_LOCAL_NEIGHBORHOOD_SIZE = 0.3

Accelerator = Literal["cpu", "mps", "cuda", "auto"]


class _SaveFinalCheckpoint(pl.Callback):
    """Save a checkpoint of the module to ``path`` when training completes."""

    def __init__(self, path: str):
        self.path = path

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        trainer.save_checkpoint(self.path)


def _is_nmf_model(model: object) -> bool:
    """Whether ``model`` is a :class:`NonNegativeMatrixFactorization` (by class name, which survives module reloads)."""
    return any(c.__name__ == "NonNegativeMatrixFactorization" for c in type(model).__mro__)


@contextmanager
def _on_device(module: torch.nn.Module, device: torch.device) -> Iterator[None]:
    """Move ``module`` to ``device`` for the ``with`` block, and back to where it was."""
    original = _device_of(module)
    module.to(device)
    try:
        yield
    finally:
        if original is not None:
            module.to(original)


def _shell_module(
    var_names_g: np.ndarray,
    factors_per_k: dict[int, torch.Tensor],
    cpu_transforms: Sequence[torch.nn.Module],
    transforms: Sequence[torch.nn.Module],
) -> CellariumModule:
    """
    A module that holds the factors of the replicate runs, for consensus and for inferring loadings. Its model is an
    online NMF with solver ``"fista"``, which does not store loadings, so it needs none of the (huge)
    ``loadings_{k}_rnk`` buffers. ``r=1`` is a placeholder: every ``D`` buffer is replaced right away.
    """
    k_values = sorted(factors_per_k)
    model = OnlineNonNegativeMatrixFactorization(
        var_names_g=list(var_names_g),
        k_values=k_values,
        r=1,
        solver="fista",
        init="uniform_random",
    )
    for k in k_values:
        # PyTorch routes setattr through _buffers when the name is a registered buffer, so this replaces the (1, k, g)
        # placeholder with the (r, k, g) factors
        setattr(model, f"D_{k}_rkg", factors_per_k[k])
    # is_initialized=True keeps configure_model() from calling reset_parameters(), which would overwrite the factors
    module = CellariumModule(
        cpu_transforms=list(cpu_transforms) or None,
        transforms=list(transforms) or None,
        model=model,
        is_initialized=True,
    )
    module.configure_model()
    return module


class NMFOutput:
    """
    A trained (consensus) NMF model bound to the cells :attr:`cdata`, for choosing the number of programs k and
    computing programs and loadings: it runs the consensus step (:meth:`compute_consensus_factors`), picks k
    (:meth:`default_k_selection_plot`, :meth:`maximal_stability_k_selection_plot`, :meth:`plot_clustermap`), and
    computes per-cell loadings (:meth:`compute_loadings`) and programs over all genes
    (:meth:`refit_consensus_factors_zscored_tpm_all_genes`). It holds the consensus that has been computed, so
    that these steps can build on each other.

    It is returned by :func:`~cellarium.ml.api.tools.cnmf`. Models that were trained from the command line, perhaps in
    chunks, are combined with :meth:`from_checkpoints`, and :meth:`save` and :meth:`load` store and restore an
    :class:`NMFOutput` (without its data).

    The methods that make a pass over the cells (:meth:`calculate_reconstruction_error`, :meth:`compute_loadings` and
    :meth:`refit_consensus_factors_zscored_tpm_all_genes`) read :attr:`cdata`, or the ``cdata`` that is passed to them
    (for example a view of a few cells), and apply the transforms of the module to the cells just as in training.

    Args:
        nmf_module: A :class:`~cellarium.ml.core.CellariumModule` whose model is a trained
            :class:`~cellarium.ml.models.nmf.NonNegativeMatrixFactorization`, along with the transforms that preprocess
            the cells for it.
        cdata: The cells. The model's genes must be among the genes of the data.

    Attributes:
        nmf_module: The module.
        cdata: The cells.
    """

    def __init__(self, nmf_module: CellariumModule, cdata: CellariumData):
        nmf_module.configure_model()  # idempotent; makes the transforms and the model accessible
        if not _is_nmf_model(nmf_module.model):
            raise ValueError(
                "NMFOutput requires a module with a NonNegativeMatrixFactorization model, "
                f"got {type(nmf_module.model).__name__}."
            )
        missing = sorted(set(nmf_module.model.var_names_g) - set(cdata.var.index))
        if missing:
            raise ValueError(f"{len(missing)} genes of the model are not in the data, for example {missing[:5]}.")
        self.nmf_module = nmf_module
        self.cdata = cdata
        self._consensus: dict[int, dict[str, Any]] = {}
        self._rec_error: dict[int, float] | None = None

    # ---- construction ----

    @classmethod
    def from_cdata(cls, cdata: CellariumData, key: str = "cnmf") -> "NMFOutput":
        """
        An :class:`NMFOutput` for the module trained by :func:`~cellarium.ml.api.tools.cnmf`, which is stored in
        ``cdata.trained_modules[key]``.
        """
        if key not in cdata.trained_modules:
            raise ValueError(
                f"No trained module '{key}' in cdata.trained_modules. Available: {list(cdata.trained_modules)}."
            )
        return cls(cdata.trained_modules[key].module, cdata)

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: list[str],
        cdata: CellariumData,
        map_location: str | torch.device = "cpu",
    ) -> "NMFOutput":
        """
        Construct an :class:`NMFOutput` from a list of checkpoint files (for example, of runs from the command line),
        loading one checkpoint at a time to avoid holding all model weights in memory simultaneously.

        Two checkpoint splitting strategies are supported (and may be freely mixed):

        - **Split-by-k**: each checkpoint was trained on a disjoint set of k values.  The
          ``D_{k}_rkg`` tensors are placed directly into the combined model.
        - **Split-by-r**: each checkpoint was trained on the same k values but with a different
          (possibly smaller) number of replicates.  The ``D_{k}_rkg`` tensors are concatenated
          along the replicate dimension (dim 0).

        .. note::
            All checkpoints are assumed to have been trained with identical transforms and the
            same set of HVGs (``var_names_g``).  Only ``var_names_g`` is validated for
            consistency across checkpoints; transform agreement is not checked. The transforms of the first
            checkpoint are used.

        Args:
            checkpoint_paths: Ordered list of paths to ``.ckpt`` files produced by training a
                :class:`~cellarium.ml.models.nmf.NonNegativeMatrixFactorization` model.
            cdata: The cells to use for consensus and downstream analyses.
            map_location: Device onto which checkpoint tensors are loaded.  Defaults to
                ``"cpu"`` to keep peak memory low regardless of GPU availability.

        Returns:
            An :class:`NMFOutput` whose model carries the union of all ``D_{k}_rkg`` factor tensors from every
            checkpoint.
        """
        if not checkpoint_paths:
            raise ValueError("checkpoint_paths must not be empty")

        # Multiple checkpoints may contribute tensors for the same k (split-by-r scenario); they will be concatenated
        # along dim 0.
        d_tensors_per_k: dict[int, list[torch.Tensor]] = {}
        reference_var_names_g: np.ndarray | None = None
        cpu_transforms: list[torch.nn.Module] = []
        transforms: list[torch.nn.Module] = []

        for i, path in enumerate(tqdm(checkpoint_paths, desc="Loading checkpoints", unit="checkpoint")):
            module = CellariumModule.load_from_checkpoint(path, map_location=map_location, strict=False)
            model = module.model
            if not _is_nmf_model(model):
                raise ValueError(
                    f"Checkpoint {path!r} model must be NonNegativeMatrixFactorization, got {type(model).__name__}"
                )

            if reference_var_names_g is None:
                reference_var_names_g = model.var_names_g.copy()
                cpu_transforms = list(module.cpu_transforms)
                transforms = list(module.transforms)
            elif not np.array_equal(model.var_names_g, reference_var_names_g):
                raise ValueError(
                    f"var_names_g mismatch between checkpoint 0 and checkpoint {i} ({path!r}). "
                    "All checkpoints must have been trained on the same set of HVGs."
                )

            for k in model.k_values:
                d_tensors_per_k.setdefault(k, []).append(model.factors_dict[k].cpu().clone())  # (r_i, k, g)

            del module
            gc.collect()

        assert reference_var_names_g is not None
        factors_per_k = {k: torch.cat(tensors, dim=0) for k, tensors in d_tensors_per_k.items()}  # split-by-r
        shell = _shell_module(reference_var_names_g, factors_per_k, cpu_transforms, transforms)
        return cls(shell, cdata)

    # ---- state ----

    @property
    def model(self) -> Any:
        """The :class:`~cellarium.ml.models.nmf.NonNegativeMatrixFactorization` model."""
        return self.nmf_module.model

    @property
    def k_values(self) -> list[int]:
        """The values of k that the model was trained for."""
        return list(self.model.k_values)

    @property
    def consensus(self) -> dict[int, dict[str, Any]]:
        """
        The consensus of each k that :meth:`compute_consensus_factors` has computed, keyed by k. Each is a dictionary
        with the L1-normalized factors ``"consensus_D_kg"``, the ``"stability"`` (silhouette score) and the parameters
        that produced it.
        """
        if len(self._consensus) > 0:
            return self._consensus
        raise RuntimeError("Compute a consensus using compute_consensus_factors() -- entails hyperparameter choices")

    @property
    def reconstruction_error(self) -> dict[int, float]:
        """The reconstruction error of each k with a consensus, from :meth:`calculate_reconstruction_error`."""
        if self._rec_error is not None:
            return self._rec_error
        raise RuntimeError("Compute reconstruction error using calculate_reconstruction_error()")

    def __repr__(self) -> str:
        r = self.model.factors_dict[self.k_values[0]].shape[0]
        return (
            f"NMFOutput(k_values={self.k_values}, n_replicates={r}, n_obs={self.cdata.n_obs}, "
            f"consensus for k={sorted(self._consensus)}, "
            f"reconstruction error {'computed' if self._rec_error is not None else 'not computed'})"
        )

    def _k_list(self, k_values: int | Sequence[int] | None) -> list[int]:
        ks = self.k_values if k_values is None else [k_values] if isinstance(k_values, int) else list(k_values)
        invalid = [k for k in ks if k not in self.k_values]
        if invalid:
            raise ValueError(f"Invalid k value for the trained model: {invalid}. Choose from {self.k_values}.")
        return ks

    def _require_consensus(self, k: int) -> None:
        if k not in self.consensus:
            raise KeyError(f"Missing consensus_factors key k={k}. Choose from {list(self.consensus.keys())}")

    # ---- consensus ----

    def compute_consensus_factors(
        self,
        k_values: int | Sequence[int] | None = None,
        density_threshold: float = _KOTLIAR_DENSITY_THRESHOLD,
        local_neighborhood_size: float = _KOTLIAR_LOCAL_NEIGHBORHOOD_SIZE,
    ) -> None:
        """
        Run the "consensus" step of consensus NMF by filtering outliers and clustering replicates,
        taking the median program in each cluster. The consensus is added to :attr:`consensus`, replacing that of the
        same k, and the reconstruction error is forgotten, as it needs updating.

        Args:
            k_values: The k values to compute the consensus of, by default all of them.
            density_threshold: The density threshold to use for filtering. Kotliar default is 0.5
            local_neighborhood_size: The fraction of replicate runs that are considered neighbors. Kotliar default is
                0.3

        Raises:
            ValueError: If a parameter is out of range, or the parameters leave too few replicate factors to cluster
                (see :meth:`plot_density_histograms`).
        """
        for k in self._k_list(k_values):
            try:
                self._consensus[k] = cc.consensus_factors(
                    self.model.factors_dict[k], density_threshold, local_neighborhood_size
                )
            except ValueError as error:
                raise ValueError(f"k={k}: {error}") from error
        self._rec_error = None

    def plot_density_histograms(
        self,
        k_values: int | Sequence[int] | None = None,
        local_neighborhood_size: float = _KOTLIAR_LOCAL_NEIGHBORHOOD_SIZE,
    ) -> None:
        """
        Plot a histogram of the local neighborhood distances for each k, for choosing the density threshold.

        Args:
            k_values: The k values to plot. If None, use all available k values.
            local_neighborhood_size: The fraction of replicate runs that are considered neighbors.
        """
        for k in self._k_list(k_values):
            mean_neighbor_distance_m, n_neighbors = cc.neighbor_distances_for_plot(
                self.model.factors_dict[k], local_neighborhood_size
            )
            plots.plot_density_histogram(mean_neighbor_distance_m, n_neighbors, k)

    def plot_clustermap(
        self,
        k: int | Sequence[int] | None,
        density_threshold: float | None = None,
        local_neighborhood_size: float | None = None,
    ) -> None:
        """
        Make a clustermap plot of replicate factors to see how they cluster, and the histogram of the local
        neighborhood distances with the density threshold.

        Args:
            k: The number of components for NMF, or ``None`` for all of them.
            density_threshold: The density threshold for filtering factors. If None, it will use
                the cached consensus for k. If a value is provided that differs from it, the
                consensus is recomputed with that density_threshold before plotting, and cached.
            local_neighborhood_size: With a new ``density_threshold``, the local neighborhood size to compute the
                consensus with, by default that of the cached consensus (or else 0.3).
        """
        for k_val in self._k_list(k):
            cached = self._consensus.get(k_val)
            if density_threshold is not None and (cached is None or cached["density_threshold"] != density_threshold):
                size = local_neighborhood_size
                if size is None:
                    size = _KOTLIAR_LOCAL_NEIGHBORHOOD_SIZE if cached is None else cached["local_neighborhood_size"]
                self.compute_consensus_factors([k_val], density_threshold, size)
            self._require_consensus(k_val)
            result = self._consensus[k_val]
            if result["all_neighbor_distances"] is None:
                raise ValueError(f"k={k_val} has a single replicate, so there is nothing to cluster.")

            # the (quadratic in r * k) matrix of distances is not kept: it is recomputed here
            D_rkg = self.model.factors_dict[k_val]
            filtered_distance_matrix_ff = cc.filtered_distance_matrix(
                D_rkg, result["density_threshold"], result["local_neighborhood_size"]
            )
            plots.plot_clustermap(filtered_distance_matrix_ff, k_val)
            mean_neighbor_distance_m = result["all_neighbor_distances"].mean(dim=1)
            plots.plot_density_histogram(
                mean_neighbor_distance_m, result["n_neighbors"], k_val, density_threshold=result["density_threshold"]
            )

    # ---- passes over the cells ----

    def _cells(self, cdata: CellariumData | None) -> CellariumData:
        return self.cdata if cdata is None else cdata

    def calculate_reconstruction_error(
        self, cdata: CellariumData | None = None, accelerator: Accelerator = "auto"
    ) -> dict[int, float]:
        """
        Calculate the reconstruction error of the consensus of each k, in a pass over the cells. The error is stored
        in :attr:`reconstruction_error`.

        Args:
            cdata: The cells to read, by default :attr:`cdata`.
            accelerator: The accelerator for the pass, in ["cpu", "mps", "cuda", "auto"].

        Returns:
            The reconstruction error of each k that has a consensus.
        """
        consensus = self.consensus
        cdata = self._cells(cdata)
        rec_error = {k: 0.0 for k in consensus}
        for batch in cdata.predict_batches(
            self.nmf_module, device=resolve_device(accelerator), desc="Reconstruction error"
        ):
            errors_keyed_by_k = self.model.reconstruction_error(
                x_ng=batch["x_ng"], var_names_g=batch["var_names_g"], consensus_factors=consensus
            )
            for k, error in errors_keyed_by_k.items():
                rec_error[k] += error

        self._rec_error = rec_error
        return rec_error

    @torch.no_grad()
    def compute_loadings(
        self,
        k: int,
        key_added: str = "X_cnmf_k{k}",
        cdata: CellariumData | None = None,
        normalize: bool = True,
        accelerator: Accelerator = "auto",
        max_gb: float | None = None,
    ) -> None:
        """
        Compute the per-cell loadings of the consensus programs of ``k``, in a pass over the cells, and store them as
        ``cdata.obsm[key_added]``, an array of shape ``(n_obs, k)`` in the order of the cells.

        The array is in memory if it fits in ``max_gb`` (by default the ``max_gb`` of ``cdata``) and otherwise in a
        read-only memory-mapped file in the temporary directory, which is deleted when the array is no longer used.
        A view of ``cdata`` copies the rows of its cells into memory if they fit, and otherwise reads them from the
        file as needed. If the pass is interrupted, nothing is stored.

        Args:
            k: The number of consensus NMF factors. Its consensus must have been computed.
            key_added: The key in ``obsm`` to store the loadings under, replacing any entry there. ``{k}`` is replaced
                by ``k``.
            cdata: The cells to compute loadings for, by default :attr:`cdata`. It is where they are stored.
            normalize: Whether to normalize the loadings of each cell to sum to 1.
            accelerator: The accelerator for the pass, in ["cpu", "mps", "cuda", "auto"].
            max_gb: The most memory (in GiB) for the loadings, by default the ``max_gb`` of ``cdata``.
        """
        self._require_consensus(k)
        cdata = self._cells(cdata)
        consensus = self.consensus
        with ObsmWriter(cdata.n_obs, k, cdata._max_bytes(max_gb)) as writer:
            for batch in cdata.predict_batches(
                self.nmf_module, device=resolve_device(accelerator), desc=f"Loadings for k={k}"
            ):
                alpha_nk = self.model.infer_loadings(
                    x_ng=batch["x_ng"],
                    var_names_g=batch["var_names_g"],
                    obs_names_n=batch.get("obs_names_n", None),
                    consensus_factors=consensus,
                    k=k,
                    normalize=normalize,
                )
                positions = batch["row_indices_n"].numpy()
                values = alpha_nk.cpu().numpy()
                if len(positions) > 1 and np.all(np.diff(positions) == 1):  # a contiguous run of cells
                    writer.array[positions[0] : positions[-1] + 1] = values
                else:
                    writer.array[positions] = values
        cdata.obsm[key_added.format(k=k)] = writer.result

    @torch.no_grad()
    def refit_consensus_factors_zscored_tpm_all_genes(
        self,
        k: int,
        normalize_loadings: bool = False,
        mean_g: torch.Tensor | None = None,
        std_g: torch.Tensor | None = None,
        var_names_g: np.ndarray | None = None,
        cdata: CellariumData | None = None,
        accelerator: Accelerator = "auto",
    ) -> dict[str, torch.Tensor | np.ndarray]:
        """
        Refit the consensus factors using all genes, after TPM normalization, and Z-scoring, in a pass over the cells.

        For each cell, the loadings of the consensus programs are inferred as in :meth:`compute_loadings` (from the
        genes and the preprocessing of the model), and then the TPM-normalized, Z-scored expression of all genes is
        regressed on them by streaming ordinary least squares.

        Args:
            k: The run identifier, i.e. the number of factors to use from self.consensus to refit.
            normalize_loadings: Whether to normalize the inferred loadings. False for Kotliar default.
            mean_g: The mean expression of each gene for Z-scoring.
                NOTE: should be TPM normalized stats. By default (if none of ``mean_g``, ``std_g`` and ``var_names_g``
                is given) they are computed over the cells in a pass, which is reused by later calls.
            std_g: The standard deviation of each gene for Z-scoring.
            var_names_g: The names of the genes for Z-scoring.
            cdata: The cells to read, by default :attr:`cdata`.
            accelerator: The accelerator for the pass, in ["cpu", "mps", "cuda", "auto"].

        Returns:
            A dictionary with the refitted factors ``"factors_kg"`` and the gene names ``"var_names_g"``.
        """
        self._require_consensus(k)
        cdata = self._cells(cdata)
        device = resolve_device(accelerator)
        given = [mean_g is not None, std_g is not None, var_names_g is not None]
        if not any(given):
            tpm_stats = fit_or_reuse(cdata, OnePassRecipe(target_count=1_000_000), accelerator).module.model
            mean_g, std_g, var_names_g = tpm_stats.mean_g, tpm_stats.std_g, tpm_stats.var_names_g
        elif not all(given):
            raise ValueError("mean_g, std_g, and var_names_g must all be provided for Z-scoring")
        assert mean_g is not None and std_g is not None and var_names_g is not None

        # Both views of a cell start from the raw counts: the preprocessing of the model (which may be anything, for
        # a model from the command line) and, to regress on, all genes at TPM, Z-scored. The raw batches are sparse
        # on the cpu, as the transforms that run on the cpu need them.
        tpm_pipeline = CellariumPipeline([NormalizeTotal(target_count=1_000_000)])
        zscore_pipeline = CellariumPipeline(
            [Densify(), ZScore(mean_g=mean_g, std_g=std_g, var_names_g=var_names_g, eps=1e-12)]
        ).to(device)
        cpu_transforms = self.nmf_module.cpu_transforms
        transforms = self.nmf_module.transforms

        ols_solver = None
        target_var_names_g = None
        with _on_device(transforms, device):
            for raw_batch in cdata.predict_batches(None, device=None, desc=f"Refitting k={k}"):
                # the batch used for inference (HVGs, same transforms as used for training)
                inference_batch = move_data_to_device(cpu_transforms(dict(raw_batch)), device)
                inference_batch = transforms(inference_batch)
                loadings_nk = self.model.infer_loadings(
                    x_ng=inference_batch["x_ng"],
                    var_names_g=inference_batch["var_names_g"],
                    obs_names_n=inference_batch.get("obs_names_n", None),
                    consensus_factors=self.consensus,
                    k=k,
                    normalize=normalize_loadings,
                )

                # the batch used for refitting (all genes, TPM normalized, Z-scored globally - different from training)
                target_batch = zscore_pipeline(move_data_to_device(tpm_pipeline(dict(raw_batch)), device))

                # incremental OLS fit update
                if ols_solver is None:
                    ols_solver = StreamingOrdinaryLeastSquares(
                        var_names_g=np.arange(k).astype(str),
                        n_targets=len(target_batch["var_names_g"]),
                        univariate=False,
                        ridge_penalty=0.0,
                    ).to(device)
                ols_solver.update(x_ng=loadings_nk, y_nk=target_batch["x_ng"])
                target_var_names_g = target_batch["var_names_g"]

        assert ols_solver is not None and target_var_names_g is not None, "predict_batches yielded no batches"
        factors_kg = ols_solver.solve()
        assert isinstance(factors_kg, torch.Tensor)
        return {"factors_kg": factors_kg.cpu(), "var_names_g": target_var_names_g}

    # ---- choosing k ----

    def default_k_selection(self, cdata: CellariumData | None = None, accelerator: Accelerator = "auto") -> None:
        """
        Compute the stability and reconstruction error as a function of k, using the Kotliar default hyperparameters
        to run consensus (a density threshold of 0.5 and a local neighborhood size of 0.3). This computes the
        consensus of every k, and so replaces those that were computed before, and then makes a pass over the cells.
        """
        self.compute_consensus_factors(
            k_values=None,
            density_threshold=_KOTLIAR_DENSITY_THRESHOLD,
            local_neighborhood_size=_KOTLIAR_LOCAL_NEIGHBORHOOD_SIZE,
        )
        self.calculate_reconstruction_error(cdata, accelerator)

    def default_k_selection_plot(self, cdata: CellariumData | None = None, accelerator: Accelerator = "auto") -> None:
        """
        Make a k-selection plot with stability and reconstruction error curves as a function of k.
        This uses the Kotliar default hyperparameters to run consensus (see :meth:`default_k_selection`).
        """
        self.default_k_selection(cdata, accelerator)
        self.k_selection_plot()

    def maximal_stability_k_selection(
        self,
        fast_or_exhaustive: Literal["fast", "exhaustive"] = "fast",
        density_threshold_limits: tuple[float, float] = (0.05, 0.3),
        local_neighborhood_size: float = _KOTLIAR_LOCAL_NEIGHBORHOOD_SIZE,
        max_tolerable_fraction_eliminated: float = 0.3,
        cdata: CellariumData | None = None,
        accelerator: Accelerator = "auto",
    ) -> None:
        """
        Compute the stability and reconstruction error as a function of k. This uses the Kotliar default
        hyperparameter for local_neighborhood_size, but then makes the determination of density_threshold into an
        optimization problem, which it solves separately for each k. Arguably, this is an even better way to
        ultimately choose k. The consensus of every k is replaced, and then there is a pass over the cells.

        Args:
            fast_or_exhaustive: Whether to use fast or exhaustive search for density thresholds.
            density_threshold_limits: The lower and upper bounds for the density threshold search.
            local_neighborhood_size: Determines the number of neighboring runs to average distances over.
            max_tolerable_fraction_eliminated: Do not consider a density_threshold which eliminates more than
                this fraction of NMF runs.
            cdata: The cells to read for the reconstruction error, by default :attr:`cdata`.
            accelerator: The accelerator for the pass, in ["cpu", "mps", "cuda", "auto"].
        """
        for k in tqdm(self.k_values, desc="Searching for best density thresholds", unit="k"):
            try:
                self._consensus[k] = cc.maximal_stability_consensus(
                    self.model.factors_dict[k],
                    local_neighborhood_size=local_neighborhood_size,
                    density_threshold_limits=density_threshold_limits,
                    max_tolerable_fraction_eliminated=max_tolerable_fraction_eliminated,
                    search=fast_or_exhaustive,
                )
            except ValueError as error:
                raise ValueError(f"k={k}: {error}") from error
        self._rec_error = None
        self.calculate_reconstruction_error(cdata, accelerator)

    def maximal_stability_k_selection_plot(
        self,
        fast_or_exhaustive: Literal["fast", "exhaustive"] = "fast",
        density_threshold_limits: tuple[float, float] = (0.05, 0.3),
        local_neighborhood_size: float = _KOTLIAR_LOCAL_NEIGHBORHOOD_SIZE,
        max_tolerable_fraction_eliminated: float = 0.3,
        cdata: CellariumData | None = None,
        accelerator: Accelerator = "auto",
    ) -> None:
        """
        Make a k-selection plot with stability and reconstruction error curves as a function of k, with the density
        threshold of each k chosen to maximize its stability (see :meth:`maximal_stability_k_selection`, which has the
        arguments).
        """
        self.maximal_stability_k_selection(
            fast_or_exhaustive=fast_or_exhaustive,
            density_threshold_limits=density_threshold_limits,
            local_neighborhood_size=local_neighborhood_size,
            max_tolerable_fraction_eliminated=max_tolerable_fraction_eliminated,
            cdata=cdata,
            accelerator=accelerator,
        )
        self.k_selection_plot()

    def k_selection_plot(self) -> None:
        """
        Make the k-selection plot with stability and reconstruction error curves as a function of k, for the k that have
        both a consensus and a reconstruction error.
        """
        consensus, rec_error = self.consensus, self.reconstruction_error
        ks = sorted(set(consensus) & set(rec_error))
        if not ks:
            raise RuntimeError("No k has both a consensus and a reconstruction error.")
        plots.plot_k_selection(
            stability={k: consensus[k]["stability"] for k in ks},
            reconstruction_error={k: rec_error[k] for k in ks},
        )

    # ---- save and load ----

    def save(self, path: str) -> None:
        """
        Save the factors of the replicate runs, the transforms, the consensus and the reconstruction error (but not
        :attr:`cdata`) to ``path``, for :meth:`load`.
        """
        torch.save(
            {
                "format_version": _FORMAT_VERSION,
                "var_names_g": self.model.var_names_g,
                "factors": {k: v.detach().cpu() for k, v in self.model.factors_dict.items()},
                "cpu_transforms": list(self.nmf_module.cpu_transforms),
                "transforms": list(self.nmf_module.transforms),
                "consensus": self._consensus,
                "rec_error": self._rec_error,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, cdata: CellariumData) -> "NMFOutput":
        """
        Load an :class:`NMFOutput` saved with :meth:`save`, bound to the cells ``cdata``.

        .. warning::
            The file holds pickled transforms, so load only files that you trust.
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format_version") != _FORMAT_VERSION:
            raise ValueError(
                f"{path!r} has format version {state.get('format_version')}, but this version reads {_FORMAT_VERSION}."
            )
        shell = _shell_module(state["var_names_g"], state["factors"], state["cpu_transforms"], state["transforms"])
        output = cls(shell, cdata)
        output._consensus = state["consensus"]
        output._rec_error = state["rec_error"]
        return output


def cnmf(
    cdata: CellariumData,
    k_values: int | Sequence[int],
    n_replicates: int = 100,
    latent_dim: int = 256,
    max_epochs: int = 20,
    amortized_kwargs: dict | None = None,
    key_added: str = "cnmf",
    checkpoint_dir: str | None = None,
    accelerator: Accelerator = "auto",
) -> NMFOutput:
    """
    Train an (amortized, online) consensus NMF model on the data in the datamodule, for each of ``k_values`` at once,
    and return an :class:`NMFOutput` for choosing k and computing programs and loadings.

    The raw counts are restricted to the highly variable genes (:attr:`CellariumData.hvg`), and each gene is divided
    by its standard deviation over the cells (computed in a pass over the data, unless an earlier call has already
    computed it), as in cNMF (Kotliar et al., 2019). ``n_replicates`` replicate runs are trained for each k and combined
    by :meth:`NMFOutput.compute_consensus_factors`.

    Args:
        cdata: :class:`CellariumData` instance containing the data. If ``cdata.hvg`` is not set (see
            :func:`~cellarium.ml.api.preprocessing.highly_variable_genes`), a warning is issued and all genes are used.
        k_values: The numbers of gene expression programs to infer.
        n_replicates: The number of replicate runs, for each k (for consensus).
        latent_dim: The dimension of the latent space of the encoder that predicts the loadings.
        max_epochs: Maximum number of training epochs (a fallback; training usually stops earlier, when the
            model's convergence criteria are met).
        amortized_kwargs: Additional keyword arguments passed to
            :class:`~cellarium.ml.models.AmortizedOnlineNonNegativeMatrixFactorization`, for example the stopping
            thresholds. For advanced use.
        key_added: The key under which the trained module and its history are stored in ``cdata.trained_modules``,
            replacing any module already there under that key.
        checkpoint_dir: A directory (made if needed) to save the trained model's checkpoint ``NMF.ckpt`` to when
            training completes, for combining with others using :meth:`NMFOutput.from_checkpoints`. By default no
            checkpoint is saved.
        accelerator: The accelerator to use for training the module, in ["cpu", "mps", "cuda", "auto"].

    Returns:
        An :class:`NMFOutput` for the trained model, bound to ``cdata``.

    Note:
        A :class:`CellariumModule` instance containing the trained model is stored, along with the training history,
        as ``cdata.trained_modules[key_added]``. If training is interrupted (e.g. ``KeyboardInterrupt``), the partially
        trained module is stored the same way, marked ``complete=False``. Training uses a single device.
    """
    ks = [k_values] if isinstance(k_values, int) else list(k_values)
    if not ks or any(not isinstance(k, (int, np.integer)) or k < 1 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError(f"k_values must be distinct positive integers, got {k_values}.")
    ks = sorted(int(k) for k in ks)

    # the statistics of the raw counts, for dividing each gene by its standard deviation
    recipe = OnePassRecipe(normalize_total=False)
    onepass_model = fit_or_reuse(cdata, recipe, accelerator).module.model
    cpu_transforms: list[torch.nn.Module] = []
    if cdata.hvg is None:
        warnings.warn("cdata.hvg is not set, so cNMF uses all genes. See highly_variable_genes().", stacklevel=2)
        var_names_g = cdata.datamodule.var_names_g
    else:
        gene_filter = Filter(filter_list=cdata.hvg.index[cdata.hvg].tolist(), ordering=True)
        cpu_transforms.append(gene_filter)
        var_names_g = gene_filter.filter_list

    datamodule = cdata.datamodule
    module = CellariumModule(
        cpu_transforms=cpu_transforms or None,
        transforms=[
            Densify(),
            # the statistics are those of all genes: DivideByScale picks the genes it is given
            DivideByScale(scale_g=onepass_model.std_g, var_names_g=onepass_model.var_names_g, eps=1e-4),
        ],
        model=AmortizedOnlineNonNegativeMatrixFactorization(
            var_names_g=var_names_g,
            k_values=ks,
            r=n_replicates,
            latent_dim=latent_dim,
            total_n_cells=datamodule.n_train,
            batch_size=datamodule.batch_size,
            **(amortized_kwargs or {}),
        ),
    )

    config = {
        "k_values": ks,
        "n_replicates": n_replicates,
        "latent_dim": latent_dim,
        "max_epochs": max_epochs,
        "std_statistics": recipe.fingerprint(),
    }
    loss_history = LossHistory()
    callbacks: list[pl.Callback] = [PreciseProgressBar()]
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        callbacks.append(_SaveFinalCheckpoint(os.path.join(checkpoint_dir, "NMF.ckpt")))
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=loss_history,
        log_every_n_steps=1,
        enable_checkpointing=False,
    )
    trained = fit_and_register(cdata, trainer, module, key_added, config=config, loss_history=loss_history)

    return NMFOutput(trained.module, cdata)
