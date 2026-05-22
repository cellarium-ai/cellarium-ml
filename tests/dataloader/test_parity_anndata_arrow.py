# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Parity test: verify that an AnnData dataloader and an Arrow dataloader built from the
same source data (with the same NormalizeTotal → Log1p transforms applied) yield
identical cells across a full epoch for every batch_size / num_workers combination.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import torch
from anndata import AnnData
from torch.utils.data import DataLoader

from cellarium.ml.data import (
    DistributedAnnDataCollection,
    DistributedArrowDataCollection,
    IterableDistributedAnnDataCollectionDataset,
)
from cellarium.ml.models.data_preformatter import DataPreformatter
from cellarium.ml.transforms import Log1p, NormalizeTotal
from cellarium.ml.utilities.data import (
    AnnDataField,
    categories_to_codes,
    collate_fn,
    get_categories,
)

# Required for multi-worker DataLoaders on macOS / Linux with limited open-file descriptors.
torch.multiprocessing.set_sharing_strategy("file_system")

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

N_CELL = 10
N_GENE = 5
LIMITS = [2, 5, 10]
CELL_TYPE_CATS = np.array(["B", "NK", "T"])

# ---------------------------------------------------------------------------
# Module-level helper — must be picklable for num_workers > 0
# ---------------------------------------------------------------------------


def _to_float32(x: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
    """Convert sparse or dense X to float32 numpy array."""
    if isinstance(x, scipy.sparse.spmatrix):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def h5ads_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write 3 h5ad shards (limits=[2, 5, 10]) to a temp directory."""
    tmp_path = tmp_path_factory.mktemp("h5ads")

    rng = np.random.default_rng(2024)
    # Use positive integers so NormalizeTotal is well-defined (eps guards against zeros)
    X = rng.integers(1, 50, size=(N_CELL, N_GENE)).astype(np.int32)
    rng_ct = rng.integers(0, len(CELL_TYPE_CATS), N_CELL)
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(
                CELL_TYPE_CATS[rng_ct],
                categories=list(CELL_TYPE_CATS),
            ),
        },
        index=[f"ref_cell{i:03d}" for i in range(N_CELL)],
    )
    var = pd.DataFrame(index=[f"gene{i:03d}" for i in range(N_GENE)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = AnnData(X=X, obs=obs, var=var)
        for i, (start, end) in enumerate(zip([0] + LIMITS, LIMITS)):
            sliced = adata[start:end].copy()
            # Preserve global category list in every shard
            sliced.obs["cell_type"] = sliced.obs["cell_type"].cat.set_categories(list(CELL_TYPE_CATS))
            sliced.write(str(tmp_path / f"adata.{i:03d}.h5ad"))

    return tmp_path


@pytest.fixture(scope="module")
def dadc_anndata(h5ads_dir: Path) -> DistributedAnnDataCollection:
    """DistributedAnnDataCollection over the 3 test shards."""
    filenames = str(h5ads_dir / "adata.{000..002}.h5ad")
    return DistributedAnnDataCollection(
        filenames,
        limits=LIMITS,
        max_cache_size=3,
        cache_size_strictly_enforced=False,
    )


@pytest.fixture(scope="module")
def dadc_arrow(
    dadc_anndata: DistributedAnnDataCollection,
    tmp_path_factory: pytest.TempPathFactory,
) -> DistributedArrowDataCollection:
    """
    Write a single Arrow shard from the pre-transformed AnnData cells, then return
    a DistributedArrowDataCollection pointing at that shard.

    The NormalizeTotal → Log1p transform pipeline is applied here (same as in the
    test), so the Arrow shard stores the final transformed float values.
    """
    arrow_dir = tmp_path_factory.mktemp("arrows")

    # Load all cells from the AnnData collection
    adata_all = dadc_anndata[list(range(N_CELL))]

    # Build float32 expression matrix and apply transforms
    x_tensor = torch.tensor(_to_float32(adata_all.X))
    normalize = NormalizeTotal(target_count=10_000)
    log1p = Log1p()
    x_tensor = normalize(x_ng=x_tensor)["x_ng"]
    x_tensor = log1p(x_ng=x_tensor)["x_ng"]
    x_ng = x_tensor.numpy()  # shape (N_CELL, N_GENE), float32

    # Extract obs fields
    obs_names = np.asarray(adata_all.obs_names)
    cell_type_codes = categories_to_codes(adata_all.obs["cell_type"])  # int32
    cell_type_cats = get_categories(adata_all.obs["cell_type"])  # string array
    var_names = np.asarray(adata_all.var_names)

    # Write Arrow shard — classification rules:
    #   var_names_g        → schema metadata (ends with _g, not _ng)
    #   cell_type_categories → schema metadata (ends with _categories)
    #   x_ng               → per-row FixedSizeBinary float16
    #   obs_names_n        → per-row large_utf8
    #   cell_type_n        → per-row int32
    dp = DataPreformatter(output_dir=str(arrow_dir))
    dp._write_arrow(
        {
            "x_ng": x_ng,
            "obs_names_n": obs_names,
            "cell_type_n": cell_type_codes,
            "var_names_g": var_names,
            "cell_type_categories": cell_type_cats,
        }
    )

    arrow_file = str(arrow_dir / "rank00_shard000000.arrow")
    return DistributedArrowDataCollection(
        [arrow_file],
        shard_size=N_CELL,
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_no_shuffle_parity(
    dadc_anndata: DistributedAnnDataCollection,
    dadc_arrow: DistributedArrowDataCollection,
    batch_size: int,
    num_workers: int,
) -> None:
    """
    AnnData and Arrow dataloaders must yield the same cells (same data values,
    same obs_names, same category codes) across a full epoch.

    Comparison is done after sorting by obs_names to be robust to any difference
    in batch-delivery ordering across workers.
    """
    normalize = NormalizeTotal(target_count=10_000)
    log1p = Log1p()

    # --- AnnData dataset ---
    adata_ds = IterableDistributedAnnDataCollectionDataset(
        dadc_anndata,
        batch_keys={
            "x_ng": AnnDataField(attr="X", convert_fn=_to_float32),
            "obs_names_n": AnnDataField(attr="obs_names"),
            "var_names_g": AnnDataField(attr="var_names"),
            "cell_type_n": AnnDataField(attr="obs", key="cell_type", convert_fn=categories_to_codes),
            "cell_type_categories": AnnDataField(attr="obs", key="cell_type", convert_fn=get_categories),
        },
        batch_size=batch_size,
        shuffle=False,
        iteration_strategy="same_order",
    )

    # --- Arrow dataset ---
    arrow_ds = IterableDistributedAnnDataCollectionDataset(
        dadc_arrow,
        batch_keys=None,
        batch_size=batch_size,
        shuffle=False,
        iteration_strategy="same_order",
    )

    adata_loader = DataLoader(adata_ds, num_workers=num_workers, collate_fn=collate_fn)
    arrow_loader = DataLoader(arrow_ds, num_workers=num_workers, collate_fn=collate_fn)

    # --- Collect all batches ---
    adata_x_parts: list[np.ndarray] = []
    adata_obs_parts: list[np.ndarray] = []
    adata_ct_parts: list[np.ndarray] = []
    adata_var_names: np.ndarray | None = None
    adata_ct_cats: np.ndarray | None = None

    for batch in adata_loader:
        # Apply NormalizeTotal → Log1p (both transforms take/return x_ng as torch.Tensor)
        x_ng = normalize(x_ng=batch["x_ng"])["x_ng"]
        x_ng = log1p(x_ng=x_ng)["x_ng"]
        adata_x_parts.append(x_ng.numpy())
        adata_obs_parts.append(np.asarray(batch["obs_names_n"]))
        adata_ct_parts.append(batch["cell_type_n"].numpy())
        if adata_var_names is None:
            adata_var_names = np.asarray(batch["var_names_g"])
            adata_ct_cats = np.asarray(batch["cell_type_categories"])

    arrow_x_parts: list[np.ndarray] = []
    arrow_obs_parts: list[np.ndarray] = []
    arrow_ct_parts: list[np.ndarray] = []
    arrow_var_names: np.ndarray | None = None
    arrow_ct_cats: np.ndarray | None = None

    for batch in arrow_loader:
        # x_ng comes back as float16 tensor; cast to float32 for comparison
        arrow_x_parts.append(batch["x_ng"].float().numpy())
        arrow_obs_parts.append(np.asarray(batch["obs_names_n"]))
        arrow_ct_parts.append(batch["cell_type_n"].numpy())
        if arrow_var_names is None:
            arrow_var_names = np.asarray(batch["var_names_g"])
            arrow_ct_cats = np.asarray(batch["cell_type_categories"])

    # --- Concatenate and sort by obs_names for robust ordering comparison ---
    adata_x = np.concatenate(adata_x_parts, axis=0)
    adata_obs = np.concatenate(adata_obs_parts, axis=0)
    adata_ct = np.concatenate(adata_ct_parts, axis=0)

    arrow_x = np.concatenate(arrow_x_parts, axis=0)
    arrow_obs = np.concatenate(arrow_obs_parts, axis=0)
    arrow_ct = np.concatenate(arrow_ct_parts, axis=0)

    adata_sort = np.argsort(adata_obs)
    arrow_sort = np.argsort(arrow_obs)

    # --- Assertions ---

    # Both loaders must cover all N_CELL cells exactly once
    assert len(adata_obs) == N_CELL, f"AnnData: expected {N_CELL} cells, got {len(adata_obs)}"
    assert len(arrow_obs) == N_CELL, f"Arrow:   expected {N_CELL} cells, got {len(arrow_obs)}"

    # Same cell ordering after sort
    np.testing.assert_array_equal(
        adata_obs[adata_sort],
        arrow_obs[arrow_sort],
        err_msg="obs_names_n mismatch between AnnData and Arrow loaders",
    )

    # Expression values agree within float16 round-trip tolerance.
    # After NormalizeTotal(1e4) + Log1p, values are in [0, log(10001)] ≈ [0, 9.2].
    # Float16 absolute error at that magnitude ≈ 9.2 × 2⁻¹⁰ ≈ 0.009, so atol=1e-2.
    np.testing.assert_allclose(
        adata_x[adata_sort],
        arrow_x[arrow_sort],
        atol=1e-2,
        err_msg="x_ng mismatch between AnnData and Arrow loaders",
    )

    # Category codes must match exactly
    np.testing.assert_array_equal(
        adata_ct[adata_sort],
        arrow_ct[arrow_sort],
        err_msg="cell_type_n codes mismatch between AnnData and Arrow loaders",
    )

    # Constant fields must match exactly
    assert adata_var_names is not None
    assert arrow_var_names is not None
    assert adata_ct_cats is not None
    assert arrow_ct_cats is not None
    np.testing.assert_array_equal(
        adata_var_names,
        arrow_var_names,
        err_msg="var_names_g mismatch",
    )
    np.testing.assert_array_equal(
        adata_ct_cats,
        arrow_ct_cats,
        err_msg="cell_type_categories mismatch",
    )
