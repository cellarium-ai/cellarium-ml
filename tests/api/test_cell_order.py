# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import math
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellarium.ml.api import _cell_order
from cellarium.ml.api._cell_order import grouped_obs_columns, tile_mixing
from cellarium.ml.api.cellariumdata import CellariumData


def test_tile_mixing_is_about_one_in_random_order_and_near_zero_when_grouped():
    rng = np.random.default_rng(0)
    grouped = np.repeat(np.arange(20), 500)  # 20 values, one after another
    diversity, mixing = tile_mixing(rng.permutation(grouped), tile_size=100)
    assert diversity == pytest.approx(1 - 1 / 20)
    assert mixing == pytest.approx(1.0, abs=0.02)  # a tile's (unbiased) diversity has that of all the cells as mean
    _, mixing = tile_mixing(grouped, tile_size=100)
    assert mixing == 0.0  # every tile has one value


def test_tile_mixing_counts_missing_values_as_a_value_and_needs_some_diversity():
    # tiles of 5 and 5: two different cells of a tile differ with probability 25 / 45, and two cells of all with 1/2
    diversity, mixing = tile_mixing(np.array([-1, 0] * 50), tile_size=10)
    assert (diversity, mixing) == (pytest.approx(0.5), pytest.approx((25 / 45) / 0.5))
    diversity, mixing = tile_mixing(np.zeros(100, dtype=np.int64), tile_size=10)
    assert diversity == 0.0 and math.isnan(mixing)
    _, mixing = tile_mixing(np.array([0, 1] * 50), tile_size=1)  # a tile of one cell has no diversity
    assert math.isnan(mixing)


def test_tile_mixing_is_the_same_whatever_way_it_is_counted(monkeypatch):
    codes = np.random.default_rng(1).integers(-1, 30, size=1_003)  # a last tile that is not full
    expected = tile_mixing(codes, tile_size=10)
    monkeypatch.setattr(_cell_order, "_MAX_DENSE_COUNTS", 0)  # count by sorting instead of with a dense table
    monkeypatch.setattr(_cell_order, "_CHUNK_CELLS", 64)  # and a few tiles at a time
    assert tile_mixing(codes, tile_size=10) == pytest.approx(expected)


@pytest.fixture
def per_experiment(tmp_path) -> CellariumData:
    """Two files of 20 cells, one experiment each (as per-experiment files are), with a column that is well mixed and
    one that has nearly one value."""
    paths = []
    for i in range(2):
        j = np.arange(20)
        obs = pd.DataFrame(
            {
                "experiment": pd.Categorical([f"exp{i}"] * 20, categories=["exp0", "exp1"]),
                "mixed": pd.Categorical(np.where(j % 2, "a", "b"), categories=["a", "b"]),
                "nearly_constant": pd.Categorical(["x"] * 19 + ["y"], categories=["x", "y"]),
                "n_counts": np.arange(20),
            },
            index=[f"exp{i}_cell{k}" for k in j],
        )
        x = sp.csr_matrix(np.random.default_rng(i).poisson(2.0, size=(20, 5)).astype(np.float32))
        path = tmp_path / f"exp{i}.h5ad"
        ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=[f"g{k}" for k in range(5)])).write_h5ad(path)
        paths.append(str(path))
    return CellariumData(h5ad_paths=paths)


def test_to_deltacells_warns_before_writing_cells_that_are_grouped(per_experiment, tmp_path, deltacells_kwargs):
    pytest.importorskip("deltacells._core")
    assert per_experiment.obs.categorical_columns == ["experiment", "mixed", "nearly_constant"]
    with pytest.warns(UserWarning, match=r"grouped by some obs columns.*: experiment \(0%\)\. Training") as record:
        new = per_experiment.to_deltacells(
            str(tmp_path / "grouped"), tile_size=4, level=3, deltacells_kwargs=deltacells_kwargs
        )
    message = str(next(w.message for w in record if "grouped" in str(w.message)))
    assert "mixed" not in message and "nearly_constant" not in message

    # the same for the obs of a deltacells dataset and of a view (in which "mixed" has one value: not checked)
    assert grouped_obs_columns(new.obs, tile_size=4) == [("experiment", 0.0)]
    view = per_experiment[np.arange(0, 40, 2)]
    assert view.obs.categorical_columns == ["experiment", "mixed", "nearly_constant"]
    # 10 cells of each experiment: one tile of 4 straddles them, 2 and 2 (diversity 8 / 12), so (8 / 12 * 4 / 20) / 0.5
    assert grouped_obs_columns(view.obs, tile_size=4) == [("experiment", pytest.approx(4 / 15))]


def test_to_deltacells_in_one_tile_does_not_warn(per_experiment, tmp_path, deltacells_kwargs):
    pytest.importorskip("deltacells._core")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # a single tile: the tiles are all of the cells, so there is nothing to compare
        per_experiment.to_deltacells(
            str(tmp_path / "one_tile"), tile_size=40, level=3, deltacells_kwargs=deltacells_kwargs
        )
