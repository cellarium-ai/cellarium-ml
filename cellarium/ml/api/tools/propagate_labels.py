# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from cellarium.ml.api.cellariumdata import CellariumData, _per_cell_series
from cellarium.ml.api.tools.embed import ensure_embedding
from cellarium.ml.api.utils import release_memory_after, resolve_device

_EPSILON = 1e-6  # added to distances before they are inverted into weights
# with method="auto", the nearest neighbors are exact if (cells to query) x (labeled cells) is at most this, for the
# device that does the work: the exact search costs one distance for each pair (about 1.5e9 pairs per second on a
# laptop CPU, which is also about what mps does; the cuda value is an estimate), while the approximate one first
# builds an index of the labeled cells
_MAX_EXACT_PAIRS = {"cpu": 5e11, "mps": 5e11, "cuda": 5e12}
_REFERENCE_TILE = 8192  # labeled cells (rows) compared with the queries at a time in the exact search
_QUERY_TILE = 1024  # cells queried at a time in the exact search
_MAX_DEVICE_BYTES = 4 * 2**30  # the most of the labeled cells that the exact search keeps on the accelerator


@contextlib.contextmanager
def _threads(n_jobs: int | None) -> Iterator[None]:
    """Use ``n_jobs`` threads (of numba, which pynndescent runs on, and of torch) within the block."""
    if n_jobs is None:
        yield
        return
    try:
        import numba
    except ImportError:  # the exact search does not need it
        numba = None
    torch_threads = torch.get_num_threads()
    numba_threads = numba.get_num_threads() if numba is not None else None
    torch.set_num_threads(n_jobs)
    if numba is not None:
        numba.set_num_threads(min(n_jobs, numba.config.NUMBA_NUM_THREADS))
    try:
        yield
    finally:
        torch.set_num_threads(torch_threads)
        if numba is not None:
            numba.set_num_threads(numba_threads)


class _ExactIndex:
    """
    The exact nearest neighbors of points among some others, by brute force: the squared distances to tiles of the
    others are made as matrix products (on the CPU or the accelerator) and the ``k`` smallest are kept. It answers
    ``query`` as a pynndescent index does.
    """

    def __init__(self, x: np.ndarray, device: torch.device) -> None:
        self.device = device
        # translating all the points leaves the distances as they are, and keeps the squared norms in the products
        # (|q|^2 - 2 q.r + |r|^2) from being large numbers that cancel
        self.center = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        reference = torch.from_numpy(x - self.center)
        if device.type != "cpu" and reference.numel() * reference.element_size() <= _MAX_DEVICE_BYTES:
            reference = reference.to(device)
        self.reference = reference
        self.squared_norms = (reference * reference).sum(dim=1)

    def _closest(self, queries: torch.Tensor, k: int) -> torch.Tensor:
        """The positions of the ``k`` nearest points of each of the queries (on the accelerator)."""
        best_scores = queries.new_empty((len(queries), 0))
        best_indices = torch.empty((len(queries), 0), dtype=torch.int64, device=self.device)
        for start in range(0, len(self.reference), _REFERENCE_TILE):
            stop = start + _REFERENCE_TILE
            block = self.reference[start:stop].to(self.device)
            # |r|^2 - 2 q.r: the squared distance without the |q|^2 that is the same for all the points
            scores = torch.addmm(self.squared_norms[start:stop].to(self.device), queries, block.T, alpha=-2)
            scores, indices = scores.topk(min(k, scores.shape[1]), dim=1, largest=False)
            scores = torch.cat([best_scores, scores], dim=1)
            indices = torch.cat([best_indices, indices + start], dim=1)
            best_scores, order = scores.topk(min(k, scores.shape[1]), dim=1, largest=False)
            best_indices = indices.gather(1, order)
        return best_indices

    def query(self, x: np.ndarray, k: int, epsilon: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """The ``k`` nearest points of each row of ``x``: their positions and distances (``epsilon`` is not used)."""
        k = min(k, len(self.reference))
        queries = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32) - self.center).to(self.device)
        indices = np.empty((len(queries), k), dtype=np.int64)
        distances = np.empty((len(queries), k), dtype=np.float32)
        for start in range(0, len(queries), _QUERY_TILE):
            tile = queries[start : start + _QUERY_TILE]
            closest = self._closest(tile, k)
            # the distances from the differences, which do not suffer from the cancellation of the products
            neighbors = self.reference[closest.to(self.reference.device)].to(self.device)
            tile_distances = (tile[:, None, :] - neighbors).norm(dim=2)
            tile_distances, order = tile_distances.sort(dim=1)
            indices[start : start + len(tile)] = closest.gather(1, order).cpu().numpy()
            distances[start : start + len(tile)] = tile_distances.cpu().numpy()
        return indices, distances


def _approximate_index(x: np.ndarray, k: int, seed: int) -> Any:
    """
    A pynndescent index of the points ``x`` that answers queries with all the threads of the process: one copy of the
    index in memory, however many threads search it.
    """
    from pynndescent import NNDescent

    index = NNDescent(
        x,
        metric="euclidean",
        n_neighbors=min(max(k + 1, 30), len(x) - 1),
        random_state=seed,
        n_jobs=-1,
        low_memory=True,
        parallel_batch_queries=True,
        verbose=False,
    )
    index.prepare()
    return index


def _neighbors(
    index: Any, embedding: Any, cells: np.ndarray, k: int, epsilon: float, chunk_size: int
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The ``k`` nearest labeled cells of the ``cells`` (positions, in increasing order) in chunks of ``chunk_size``:
    for each chunk, its cells, and their neighbors and distances."""
    for start in range(0, len(cells), chunk_size):
        positions = cells[start : start + chunk_size]
        x = np.ascontiguousarray(embedding[positions], dtype=np.float32)
        indices, distances = index.query(x, k=k, epsilon=epsilon)
        yield positions, indices, distances


@dataclass(frozen=True)
class PropagateLabelsResult:
    """
    What :func:`propagate_labels` found.

    Attributes:
        key_added: The key in ``obs_computed`` of the labels of all the cells (``f"{key_added}_confidence"`` and
            ``f"{key_added}_distance"`` hold the confidence and the distance to the nearest labeled cell).
        rep: The key in ``obsm`` of the embedding that the labels were propagated in.
        method: How the nearest labeled cells were found, ``"exact"`` or ``"approximate"`` (see
            :func:`propagate_labels`).
        n_reference: The number of cells that had a label.
        n_propagated: The number of cells that were given a label.
        agreement: The fraction of the ``n_checked`` held-out labeled cells whose label the others gave back: how well
            the embedding can reproduce the labels (see :func:`propagate_labels`).
        n_checked: The number of held-out labeled cells that the agreement is of.
        per_label: A table with one row for each label (the index): ``n_reference`` (labeled cells), ``n_checked``
            (held-out cells), ``agreement`` (the fraction of those that were given back their label, ``NaN`` if there
            were none) and ``most_confused_with`` (the label that the held-out cells that were not given back
            their label were most often given, or ``None``).
    """

    key_added: str
    rep: str
    method: str
    n_reference: int
    n_propagated: int
    agreement: float
    n_checked: int
    per_label: pd.DataFrame = field(repr=False)


def _vote(codes: np.ndarray, indices: np.ndarray, distances: np.ndarray, n_labels: int, weighted: bool) -> tuple:
    """The label (as a code) that wins the vote of the neighbors of each cell, and the fraction of the vote it won."""
    n_cells = len(indices)
    weights = 1.0 / (distances + _EPSILON) if weighted else np.ones(distances.shape)
    flat = (np.arange(n_cells)[:, None] * n_labels + codes[indices]).ravel()
    scores = np.bincount(flat, weights=weights.ravel(), minlength=n_cells * n_labels).reshape(n_cells, n_labels)
    winner = scores.argmax(axis=1)
    return winner, scores[np.arange(n_cells), winner] / scores.sum(axis=1)


def _held_out_agreement(
    index: Any,
    x_reference: np.ndarray,
    codes: np.ndarray,
    k: int,
    epsilon: float,
    n_labels: int,
    weighted: bool,
    n_check: int,
    seed: int,
) -> tuple[float, int, pd.DataFrame]:
    """
    How well the labeled cells give each other their labels: for ``n_check`` of them (chosen at random), the label
    that the ``k`` nearest of the *other* labeled cells vote for, compared to their own.
    """
    n_reference = len(codes)
    checked = np.sort(np.random.default_rng(seed).choice(n_reference, size=min(n_check, n_reference), replace=False))
    indices, distances = index.query(x_reference[checked], k=k + 1, epsilon=epsilon)
    # leave the cell itself out: it is its own nearest neighbor (or, if the search missed that, drop the farthest one)
    is_self = indices == checked[:, None]
    drop = np.where(is_self.any(axis=1), is_self.argmax(axis=1), k)
    keep = np.ones(indices.shape, dtype=bool)
    keep[np.arange(len(checked)), drop] = False
    indices, distances = indices[keep].reshape(len(checked), k), distances[keep].reshape(len(checked), k)
    predicted, _ = _vote(codes, indices, distances, n_labels, weighted)

    truth = codes[checked]
    n_checked = np.bincount(truth, minlength=n_labels)
    correct = np.bincount(truth[predicted == truth], minlength=n_labels)
    confusion = np.bincount(truth * n_labels + predicted, minlength=n_labels**2).reshape(n_labels, n_labels)
    np.fill_diagonal(confusion, 0)
    confused_with = np.where(confusion.max(axis=1) > 0, confusion.argmax(axis=1), -1)
    table = pd.DataFrame(
        {
            "n_reference": np.bincount(codes, minlength=n_labels),
            "n_checked": n_checked,
            "agreement": np.where(n_checked > 0, correct / np.maximum(n_checked, 1), np.nan),
            "most_confused_with": confused_with,
        }
    )
    return float((truth == predicted).mean()), len(checked), table


@release_memory_after
def propagate_labels(
    cdata: CellariumData,
    labels_key: str,
    rep: str,
    key_added: str | None = None,
    k: int = 15,
    weights: Literal["distance", "uniform"] = "distance",
    method: Literal["auto", "exact", "approximate"] = "auto",
    n_check: int = 20_000,
    min_agreement: float = 0.7,
    chunk_size: int = 50_000,
    n_jobs: int | None = None,
    epsilon: float = 0.1,
    accelerator: Literal["cpu", "mps", "cuda", "auto"] = "auto",
    max_gb: float | None = None,
    seed: int = 0,
) -> PropagateLabelsResult:
    """
    Give a label to every cell of ``cdata`` from the labels of some of them, by a vote of its ``k`` nearest labeled
    cells in an embedding. This is how labels found on a subsample of the cells, for example the clusters that scanpy
    found in the AnnData of a geometric sketch (see :meth:`CellariumData.update_labels_from_anndata`), are brought to
    all the cells of a dataset that is too large to cluster.

    The labeled cells (the *reference*) are those with a value in ``obs_computed[labels_key]``; they keep their
    labels, and the others are given one. The pass over the other cells is one query per cell, in chunks, reading the
    embedding of ``rep`` from ``obsm`` (which can be on disk). The counts are not read. The nearest labeled cells are
    found with one of two methods (``method``):

    * ``"exact"``: by brute force, as matrix products of the cells with the labeled cells, on ``accelerator`` (the CPU
      threads, or an accelerator). Nothing is built first and nothing is approximated, so this is best when the cells
      to label are few compared to the labeled ones (30 000 cells among 4 million, say),
    * ``"approximate"``: with a `pynndescent <https://pynndescent.readthedocs.io>`_ index (part of
      ``cellarium-ml[scanpy]``) of the labeled cells, which takes a while to build and some memory (a few GB for
      millions of cells), and then answers queries very fast, with all the CPU threads. This is best when there are
      many cells to label, as when propagating the labels of 100 000 cells to 90 million.

    ``"auto"`` chooses the exact search if the number of distances it takes (the cells to label and to check, times the
    labeled cells) is small enough for the device (about 5e11 for the CPU, about a few minutes), and the approximate
    one otherwise.

    Three things are stored in ``obs_computed``, each a :class:`pandas.Series` indexed by
    :attr:`CellariumData.cell_ids`, for all the cells:

    * ``key_added``: the labels (a categorical with the categories of ``labels_key``),
    * ``f"{key_added}_confidence"``: the fraction of the vote (of the weights, with ``weights="distance"``) that the
      label won; ``1.0`` for the labeled cells. Slice on it to leave out the doubtful cells,
    * ``f"{key_added}_distance"``: the distance to the nearest labeled cell, in the embedding; ``0.0`` for the labeled
      cells. A cell far from every labeled cell is in a part of the data that the reference does not cover.

    ``rep`` should be the embedding that the labels were found in. Cluster the AnnData with ``use_rep`` set to the
    embedding that :meth:`CellariumData.to_anndata` gives it (``X_pca``, ``X_scvi``), and give the key of the same
    embedding of all the cells here. The labels are checked against it: a random sample of ``n_check`` labeled cells is
    held out one at a time, and the others vote for its label. The fraction of them that are given their own label
    (``agreement``, and the same for each label, in ``per_label``) says how well the embedding reproduces the labels. A
    warning is raised if it is below ``min_agreement`` or if any label is reproduced for fewer than half of its cells,
    which is what happens when the labels were found in a different space (for example after batch correction, or in
    a PCA that scanpy computed itself). The labels are given anyway. Even in the right space, the labels of a fine
    clustering that splits a continuous population are not fully reproduced, so the value that is worrying depends on
    the data; look at ``per_label``.

    A sketch has a flatter density than the data (it keeps rare cells), so near the boundary of a large population and
    a rare one the vote favors the rare one more than the data would.

    Args:
        cdata: The cells to label: a :class:`CellariumData` or a view. Its ``obs_computed[labels_key]`` and the
            embedding ``rep`` are of its cells.
        labels_key: The key in ``cdata.obs_computed`` of the labels of some of the cells (the others are missing).
        rep: The embedding to find the nearest labeled cells in: a key of ``cdata.obsm`` (an embedding of all the
            cells, as :func:`~cellarium.ml.api.tools.embed` makes), or the key of a module in
            ``cdata.trained_modules`` (such as ``"pca"``), whose embedding of all the cells, ``obsm["X_pca"]``, is
            made with :func:`~cellarium.ml.api.tools.embed` unless it is there from that module already.
        key_added: The key in ``obs_computed`` for the labels, by default ``f"{labels_key}_propagated"``.
        k: The number of nearest labeled cells that vote.
        weights: ``"distance"`` to weight each vote by the inverse of the distance (plus a small constant), or
            ``"uniform"``.
        method: ``"exact"`` or ``"approximate"`` nearest neighbors, or ``"auto"`` to choose by the size of the search
            (see above).
        n_check: The most labeled cells to hold out to check the labels against the embedding (all of them if there
            are fewer). The check is a query for each of them, so it costs as much as labeling that many cells.
        min_agreement: The agreement below which a warning is raised.
        chunk_size: The number of cells to query at a time (this bounds the memory used for the votes).
        n_jobs: The number of CPU threads to use (to build the index and answer its queries, and for the exact search
            on the CPU); by default, all of them. The threads share one copy of the index, so more of them cost no
            more memory.
        epsilon: The accuracy of the approximate search, as in pynndescent: a larger value searches more, and is slower
            and closer to the exact neighbors (``0.0`` is about twice as fast as the default). It is not used by the
            exact search.
        accelerator: The accelerator for the exact search, and to make the embedding with if ``rep`` names a module
            that has not embedded the cells yet, in ["cpu", "mps", "cuda", "auto"].
        max_gb: The most memory (in GiB) for that embedding, by default the ``max_gb`` of ``cdata``.
        seed: The seed of the nearest neighbor index and of the choice of the held-out cells.

    Returns:
        A :class:`PropagateLabelsResult`: the agreement of the check, overall and for each label.

    Raises:
        KeyError: If ``cdata.obs_computed`` has no ``labels_key``, or ``rep`` is neither in ``obsm`` nor a trained
            module.
        ValueError: If there are not more than ``k`` labeled cells, or all the cells are labeled already, or
            ``method`` or ``n_jobs`` is not valid.
        ImportError: If the approximate search is used and pynndescent is not installed.
    """
    if labels_key not in cdata.obs_computed:
        raise KeyError(f"No '{labels_key}' in obs_computed. Available: {list(cdata.obs_computed)}")
    if weights not in ("distance", "uniform"):
        raise ValueError(f"weights must be 'distance' or 'uniform', got {weights!r}.")
    if method not in ("auto", "exact", "approximate"):
        raise ValueError(f"method must be 'auto', 'exact' or 'approximate', got {method!r}.")
    if n_jobs is not None and n_jobs < 1:
        raise ValueError(f"n_jobs must be at least 1 (or None for all the threads), got {n_jobs}.")
    key_added = f"{labels_key}_propagated" if key_added is None else key_added

    n = cdata.n_obs
    labels = _per_cell_series(cdata.obs_computed[labels_key], n, f"obs_computed['{labels_key}']").astype("category")
    labeled = labels.notna().to_numpy()
    n_reference = int(labeled.sum())
    if n_reference <= k:
        raise ValueError(f"There are {n_reference} labeled cells, but at least k + 1 = {k + 1} are needed.")
    if n_reference == n:
        raise ValueError(f"Every cell has a label in obs_computed['{labels_key}'], so there is nothing to propagate.")
    categories = labels.cat.categories
    n_labels = len(categories)
    codes_all = labels.cat.codes.to_numpy().astype(np.int64)  # -1 for the cells without a label

    device = resolve_device(accelerator)
    targets = np.flatnonzero(~labeled)
    if method == "auto":
        n_queries = len(targets) + min(n_check, n_reference)
        method = "exact" if n_queries * n_reference <= _MAX_EXACT_PAIRS[device.type] else "approximate"
    if method == "approximate":
        try:
            import pynndescent  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "The approximate nearest neighbor search needs pynndescent: pip install pynndescent (or "
                "cellarium-ml[scanpy]). The exact search, with method='exact', does not need it."
            ) from error

    rep_key = ensure_embedding(cdata, rep, accelerator=accelerator, max_gb=max_gb)
    embedding = cdata.obsm[rep_key]
    if len(embedding) != n or np.ndim(embedding) != 2:
        raise ValueError(f"obsm['{rep_key}'] must have one row for each of the {n} cells, got shape {embedding.shape}.")

    reference = np.flatnonzero(labeled)
    codes = codes_all[reference]
    x_reference = np.ascontiguousarray(embedding[reference], dtype=np.float32)
    weighted = weights == "distance"
    predicted = codes_all.copy()
    confidence = np.ones(n, dtype=np.float32)
    nearest = np.zeros(n, dtype=np.float32)
    n_chunks = -(-len(targets) // chunk_size)

    with _threads(n_jobs):
        index = _ExactIndex(x_reference, device) if method == "exact" else _approximate_index(x_reference, k, seed)
        agreement, n_checked, per_label = _held_out_agreement(
            index, x_reference, codes, k, epsilon, n_labels, weighted, n_check, seed
        )
        for cells, indices, distances in tqdm(
            _neighbors(index, embedding, targets, k, epsilon, chunk_size),
            total=n_chunks,
            desc="Propagating labels",
            unit="chunk",
        ):
            winner, share = _vote(codes, indices, distances, n_labels, weighted)
            predicted[cells] = winner
            confidence[cells] = share
            nearest[cells] = distances.min(axis=1)
    cdata.obs_computed[key_added] = pd.Series(pd.Categorical.from_codes(predicted, categories=categories))
    cdata.obs_computed[f"{key_added}_confidence"] = pd.Series(confidence)
    cdata.obs_computed[f"{key_added}_distance"] = pd.Series(nearest)

    per_label = per_label.set_axis(categories)
    per_label["most_confused_with"] = [
        categories[code] if code >= 0 else None for code in per_label["most_confused_with"].to_numpy()
    ]
    per_label = per_label[per_label["n_reference"] > 0]
    unreliable = per_label.index[(per_label["agreement"] < 0.5).to_numpy()].tolist()
    if agreement < min_agreement or unreliable:
        reasons = []
        if agreement < min_agreement:
            reasons.append(f"only {agreement:.0%} of {n_checked} held-out labeled cells were given their own label")
        if unreliable:
            reasons.append(f"fewer than half of the cells of {unreliable[:5]} were given their own label")
        warnings.warn(
            f"The embedding '{rep_key}' does not reproduce the labels in obs_computed['{labels_key}']: "
            + " and ".join(reasons)
            + ". Were the labels found in a different space (for example another PCA, or after batch correction)? "
            "The labels were propagated anyway; see per_label of the result.",
            stacklevel=2,
        )
    return PropagateLabelsResult(
        key_added=key_added,
        rep=rep_key,
        method=method,
        n_reference=n_reference,
        n_propagated=len(targets),
        agreement=agreement,
        n_checked=n_checked,
        per_label=per_label,
    )
