# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellarium.ml.api._datamodule_context import temporary_batch_keys
from cellarium.ml.api._view_collections import ViewCollection, unwrap_collection
from cellarium.ml.api.cellariumdata import CellariumData, CellariumDataView, TrainedModule
from cellarium.ml.api.tools import pca
from cellarium.ml.data import DistributedAnnDataCollection
from cellarium.ml.utilities.data import AnnDataField, categories_to_codes

# the `cdata` fixture (tests/api/conftest.py): 40 cells in two h5ad files of 20, as h5ad files or as a deltacells
# dataset in tiles of 15, with obs columns `cell_type` and `n_counts`


def _expected_x(cdata: CellariumData, positions: np.ndarray) -> np.ndarray:
    """The dense counts of the cells ``positions`` of ``cdata``, read from its own collection."""
    return np.asarray(cdata.datamodule.dadc[positions].X.todense())


def _epoch(cdata: CellariumData) -> dict:
    """Everything the (training) dataloader of ``cdata`` yields in one epoch."""
    names, rows, sizes = [], [], []
    for batch in cdata.datamodule.train_dataloader():
        names.extend(batch["obs_names_n"])
        rows.append(batch["x_ng"].to_dense().numpy())
        sizes.append(len(batch["obs_names_n"]))
    return {"names": names, "x": np.concatenate(rows), "sizes": sizes}


# ---- ViewCollection ----


def test_view_collection_limits_follow_the_shards_of_the_source(cdata):
    source = unwrap_collection(cdata.datamodule.dadc)
    first = source.limits[0]  # 20 for h5ad, 15 for deltacells
    view = ViewCollection(source, np.array([3, 5, first + 1]))
    assert view.n_obs == 3 and len(view) == 3
    assert view.limits == [2, 3]  # the first two cells are in the first shard
    # a shard without view cells is dropped
    view = ViewCollection(source, np.array([first + 1, first + 2]))
    assert view.limits == [2]


def test_view_collection_reads_the_cells_of_the_source_in_the_requested_order(cdata):
    source = unwrap_collection(cdata.datamodule.dadc)
    positions = np.array([1, 4, 16, 22, 30, 39])
    view = ViewCollection(source, positions)
    request = [5, 0, 3, 3, 1]
    got = np.asarray(view[request].X.todense())
    assert np.array_equal(got, np.asarray(source[positions[request]].X.todense()))
    assert np.array_equal(np.asarray(view[2:4].X.todense()), np.asarray(source[positions[2:4]].X.todense()))
    assert np.array_equal(np.asarray(view[-1].X.todense()), np.asarray(source[[39]].X.todense()))
    mask = np.array([True, False, False, True, False, True])
    assert np.array_equal(np.asarray(view[mask].X.todense()), np.asarray(source[positions[mask]].X.todense()))
    with pytest.raises(IndexError):
        view[6]


def test_view_collection_delegates_what_does_not_depend_on_the_cells(cdata):
    source = unwrap_collection(cdata.datamodule.dadc)
    view = ViewCollection(source, np.array([0, 39]))
    assert view.n_vars == source.n_vars
    assert view.var.equals(source.var)
    assert list(view.var_names) == list(source.var_names)
    assert view.supports_prefetch == source.supports_prefetch
    assert view.obs_key_nunique("cell_type") == source.obs_key_nunique("cell_type")
    assert list(view.obs_categories("cell_type")) == list(source.obs_categories("cell_type"))
    assert view.reference_adata is not None


def test_view_collection_of_a_view_collection_is_flat(cdata):
    source = unwrap_collection(cdata.datamodule.dadc)
    inner = ViewCollection(source, np.arange(0, 40, 2))
    outer = ViewCollection(inner, np.array([1, 5, 7]))
    assert outer.source is source
    assert outer.positions.tolist() == [2, 10, 14]
    assert unwrap_collection(outer) is source


def test_view_collection_rejects_bad_positions(cdata):
    source = unwrap_collection(cdata.datamodule.dadc)
    with pytest.raises(ValueError, match="at least one cell"):
        ViewCollection(source, np.array([], dtype=np.int64))
    with pytest.raises(ValueError, match="strictly increasing"):
        ViewCollection(source, np.array([3, 3]))
    with pytest.raises(ValueError, match="strictly increasing"):
        ViewCollection(source, np.array([3, 2]))
    with pytest.raises(IndexError):
        ViewCollection(source, np.array([0, 40]))


# ---- selecting cells ----


def test_selection_forms(cdata):
    mask = np.zeros(40, dtype=bool)
    mask[[3, 10, 11]] = True
    for selection, expected in [
        (mask, [3, 10, 11]),
        (np.array([11, 3, 10, 3]), [3, 10, 11]),  # sorted and without repeats
        ([5, 2], [2, 5]),
        (slice(10, 14), [10, 11, 12, 13]),
        (slice(None, None, 15), [0, 15, 30]),
        (np.array([-1, 0]), [0, 39]),
        (7, [7]),
        (np.int64(-1), [39]),
    ]:
        view = cdata[selection]
        assert isinstance(view, CellariumDataView)
        assert view.root_indices.tolist() == expected
        assert view.n_obs == len(expected)
        assert view.parent is cdata and view.root is cdata


def test_series_mask_must_have_the_index_of_obs(cdata):
    names = cdata.obs.index
    mask = pd.Series(np.arange(40) % 4 == 0, index=names)
    assert cdata[mask].root_indices.tolist() == list(range(0, 40, 4))
    with pytest.raises(ValueError, match="index"):
        cdata[pd.Series(np.arange(40) % 4 == 0, index=names[::-1])]
    with pytest.raises(ValueError, match="index"):
        cdata[pd.Series(np.arange(10) % 4 == 0, index=names[:10])]
    with pytest.raises(TypeError, match="boolean"):
        cdata[pd.Series(np.arange(40), index=names)]


def test_bad_selections_raise(cdata):
    with pytest.raises(ValueError, match="empty"):
        cdata[np.zeros(40, dtype=bool)]
    with pytest.raises(ValueError, match="empty"):
        cdata[[]]
    with pytest.raises(ValueError, match="empty"):
        cdata[5:5]
    with pytest.raises(IndexError):
        cdata[40]
    with pytest.raises(IndexError):
        cdata[np.ones(39, dtype=bool)]
    with pytest.raises(TypeError, match="boolean mask"):
        cdata[["file0_cell0"]]


def test_cdata_is_not_iterable(cdata):
    with pytest.raises(TypeError):
        iter(cdata)


def test_view_of_a_view_composes_with_the_root(cdata):
    first = cdata[np.arange(0, 40, 2)]
    second = first[[1, 5, 7]]
    assert isinstance(second, CellariumDataView)
    assert second.parent is first and second.root is cdata
    assert second.root_indices.tolist() == [2, 10, 14]
    assert second.n_obs == 3
    # the cells are the right ones
    assert np.array_equal(_expected_x(second, np.arange(3)), _expected_x(cdata, np.array([2, 10, 14])))
    assert second.obs.index.tolist() == cdata.obs.index[[2, 10, 14]].tolist()
    # the data is read from the root's collection directly
    assert unwrap_collection(second.datamodule.dadc) is unwrap_collection(cdata.datamodule.dadc)


# ---- obs ----


def test_view_obs_is_the_obs_of_the_selected_cells(cdata):
    positions = np.array([2, 3, 17, 20, 38])
    view = cdata[positions]
    expected = cdata.obs.to_frame().iloc[positions]
    assert len(view.obs) == 5
    assert view.obs.columns == cdata.obs.columns
    assert view.obs.index.tolist() == expected.index.tolist()
    assert view.obs["n_counts"].tolist() == expected["n_counts"].tolist()
    assert view.obs["cell_type"].astype(str).tolist() == expected["cell_type"].astype(str).tolist()
    frame = view.obs[["cell_type", "n_counts"]]
    assert frame.index.tolist() == expected.index.tolist()
    assert view.obs.to_frame()["n_counts"].tolist() == expected["n_counts"].tolist()
    chunks = list(view.obs.iter_batches(batch_size=2, columns=["n_counts"]))
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert pd.concat(chunks)["n_counts"].tolist() == expected["n_counts"].tolist()


def test_view_obs_loc_selects_by_name_within_the_view(cdata):
    positions = np.array([2, 3, 17, 20, 38])
    view = cdata[positions]
    names = cdata.obs.index
    wanted = [names[20], names[2]]
    got = view.obs.loc[wanted, ["n_counts"]]
    assert got.index.tolist() == wanted
    assert got["n_counts"].tolist() == cdata.obs.to_frame().loc[wanted, "n_counts"].tolist()
    with pytest.raises(KeyError, match="not found"):
        view.obs.loc[[names[0]]]  # a cell that is not in the view


def test_view_of_view_obs_reads_the_root_obs(cdata):
    second = cdata[np.arange(0, 40, 2)][[1, 5, 7]]
    full = cdata.obs.to_frame()
    assert second.obs["n_counts"].tolist() == full["n_counts"].iloc[[2, 10, 14]].tolist()


# ---- the datamodule of a view ----


def test_an_epoch_over_a_view_yields_each_cell_of_the_view_once(cdata):
    positions = np.array([0, 1, 5, 6, 7, 16, 19, 20, 21, 22, 30, 33, 39])
    view = cdata[positions]
    view.datamodule.batch_size = 4
    view.datamodule.setup("fit")
    epoch = _epoch(view)
    assert sorted(epoch["names"]) == sorted(cdata.obs.index[positions])
    assert epoch["sizes"] == [4, 4, 4, 1]
    # the counts that came with each cell are those of the cell
    by_name = dict(zip(cdata.obs.index[positions], _expected_x(cdata, positions)))
    assert all(np.array_equal(row, by_name[name]) for name, row in zip(epoch["names"], epoch["x"]))


def test_the_datamodule_of_a_view_has_the_settings_of_the_parent(cdata):
    cdata.datamodule.batch_size = 7
    parent = cdata.datamodule
    view = cdata[np.arange(0, 40, 3)]
    dm = view.datamodule
    assert dm is not parent
    assert dm.batch_size == 7
    assert dm.shuffle == parent.shuffle and dm.num_workers == parent.num_workers
    assert dm.iteration_strategy == parent.iteration_strategy
    assert set(dm.batch_keys) == set(parent.batch_keys) and dm.batch_keys is not parent.batch_keys
    assert len(dm.dadc) == view.n_obs == 14
    assert dm.n_train == 14  # train_size=1.0 of the view, not of the parent
    assert len(parent.dadc) == 40


def test_a_view_trains_a_model_on_its_own_cells(cdata):
    view = cdata[np.arange(0, 40, 2)]
    with pytest.warns(UserWarning, match="all genes"):
        pca(view, n_components=3, accelerator="cpu")
    assert "pca" in view.trained_modules
    assert view.trained_modules["pca"].module.model.n_components == 3
    # the statistics were fit to the cells of the view, and the parent has nothing
    assert len(view._fit_cache) > 0
    assert len(cdata.trained_modules) == 0 and len(cdata._fit_cache) == 0


def test_views_use_extra_batch_keys_of_trained_modules(make_h5ad_files):
    cdata = CellariumData(h5ad_paths=make_h5ad_files(n_files=2, cells_per_file=20, n_genes=30))
    view = cdata[np.arange(0, 40, 2)]
    h5ad_dadc = unwrap_collection(view.datamodule.dadc)
    original = list(h5ad_dadc.obs_columns_to_validate)
    extra = {"n_counts_n": AnnDataField(attr="obs", key="n_counts")}
    with temporary_batch_keys(view.datamodule, extra):
        # the h5ad files that the view reads are validated for the new column, and nothing else
        assert h5ad_dadc.obs_columns_to_validate == ["n_counts"]
        assert "n_counts_n" in view.datamodule.batch_keys
    assert h5ad_dadc.obs_columns_to_validate == original
    assert "n_counts_n" not in view.datamodule.batch_keys


# ---- h5ad: a batch of scattered cells spans more files than the cache holds ----


@pytest.fixture
def many_files(tmp_path):
    """Six h5ad files of eight cells, and their counts as one dense matrix."""
    rng = np.random.default_rng(0)
    paths, blocks = [], []
    for i in range(6):
        X = sp.csr_matrix(rng.poisson(3.0, size=(8, 5)).astype(np.float32))
        obs = pd.DataFrame(
            {"cell_type": pd.Categorical(rng.choice(["a", "b"], 8), categories=["a", "b"])},
            index=[f"f{i}_c{j}" for j in range(8)],
        )
        obs.index.name = "barcode"
        path = tmp_path / f"f{i}.h5ad"
        ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{k}" for k in range(5)])).write_h5ad(path)
        paths.append(str(path))
        blocks.append(X)
    return paths, sp.vstack(blocks).toarray()


def test_a_request_spanning_more_files_than_the_cache_is_read_in_groups(many_files):
    paths, dense = many_files
    cdata = CellariumData(h5ad_paths=paths)
    dadc = cdata.datamodule.dadc
    assert isinstance(dadc, DistributedAnnDataCollection)
    assert dadc.max_cache_size == 2 and dadc.cache_size_strictly_enforced
    positions = np.array([40, 1, 9, 17, 25, 33, 47, 2, 10])  # all six files, unsorted
    with pytest.raises(ValueError, match="Expected the number of anndata files"):
        dadc[positions]  # indexing stays strict
    batch = dadc.read(positions)
    assert np.array_equal(np.asarray(batch.X.todense()), dense[positions])
    assert batch.obs_names.tolist() == [f"f{p // 8}_c{p % 8}" for p in positions]
    assert len(dadc.cache) <= 2


def test_a_view_over_many_files_trains_on_exactly_its_cells(many_files):
    paths, dense = many_files
    cdata = CellariumData(h5ad_paths=paths)
    positions = np.arange(0, 48, 3)
    view = cdata[positions]
    view.datamodule.batch_size = 6
    view.datamodule.setup("fit")
    epoch = _epoch(view)
    assert sorted(epoch["names"]) == sorted(f"f{p // 8}_c{p % 8}" for p in positions)
    assert epoch["sizes"] == [6, 6, 4]
    assert np.array_equal(epoch["x"].sum(axis=0), dense[positions].sum(axis=0))


def test_h5ad_batches_have_the_categories_of_the_schema_in_the_schema_order(h5ad_cdata):
    # the category *order* decides the codes of `categories_to_codes`, so it must not depend on the cells asked for
    dadc = h5ad_cdata.datamodule.dadc
    expected = list(dadc.schema.attr_values["obs"].dtypes["cell_type"].categories)
    assert expected != sorted(expected)  # the test data has its categories out of alphabetical order
    for cells in ([0, 1], [25, 3, 21, 4], [39, 0, 20, 1, 21, 2, 22, 3], list(range(40)), [30, 2, 21, 1, 22, 0]):
        for batch in (dadc[cells], dadc.read(cells)):
            assert list(batch.obs["cell_type"].cat.categories) == expected
    view = h5ad_cdata[np.arange(0, 40, 3)]
    field = AnnDataField(attr="obs", key="cell_type", convert_fn=categories_to_codes)
    codes = np.concatenate([field(view.datamodule.dadc[np.arange(i, min(i + 4, 14))]) for i in range(0, 14, 4)])
    codes_from_obs = pd.Categorical(view.obs["cell_type"], categories=expected).codes
    assert np.array_equal(codes, codes_from_obs)


# ---- state of a view ----


def test_a_view_has_its_own_state_with_snapshots_of_obsm_and_obs_computed(cdata):
    names = cdata.obs.index
    cdata.obsm["emb"] = np.arange(80, dtype=np.float32).reshape(40, 2)
    cdata.obs_computed["flag"] = pd.Series(np.arange(40) % 2 == 0, index=names)
    cdata.obs_computed["score"] = np.arange(40) * 1.5
    cdata.hvg = np.arange(len(cdata.var)) % 2 == 0
    positions = np.array([3, 8, 21, 39])
    view = cdata[positions]
    assert np.array_equal(view.obsm["emb"], cdata.obsm["emb"][positions])
    assert view.obs_computed["flag"].index.tolist() == names[positions].tolist()
    assert view.obs_computed["flag"].tolist() == cdata.obs_computed["flag"].iloc[positions].tolist()
    assert np.array_equal(view.obs_computed["score"], np.arange(40)[positions] * 1.5)
    # everything else starts empty
    assert view.hvg is None and len(view.trained_modules) == 0 and len(view._fit_cache) == 0
    # and is independent of the parent
    view.obsm["emb"][0, 0] = -1
    view.obsm["other"] = np.zeros((4, 1))
    assert cdata.obsm["emb"][3, 0] == 6.0 and "other" not in cdata.obsm
    with pytest.raises(ValueError, match="n_obs"):
        view.obsm["bad"] = np.zeros((40, 1))
    cdata.obsm["later"] = np.zeros((40, 1))
    assert "later" not in view.obsm  # a snapshot, not a link


def test_the_repr_of_a_view_says_what_it_is(cdata):
    text = repr(cdata[:10])
    assert text.startswith("CellariumDataView(shape [10, 30]")
    assert "view of 40 cells" in text


def test_parent_modules_are_only_available_through_explicit_assignment(cdata):
    cdata.trained_modules["mod"] = TrainedModule(module=None)  # type: ignore[arg-type]
    view = cdata[:10]
    with pytest.raises(ValueError, match="No trained module 'mod'"):
        with view.using("mod"):
            pass
    view.trained_modules["mod"] = cdata.trained_modules["mod"]
    with view.using("mod") as trained:
        assert trained is cdata.trained_modules["mod"]


# ---- update_from ----


def _label_view(cdata, positions, labels, key="subtype"):
    """A view of ``positions`` with the Series ``labels`` (indexed by obs names) as ``obs_computed[key]``."""
    view = cdata[positions]
    view.obs_computed[key] = pd.Series(labels, index=view.obs.index)
    return view


def test_update_from_copies_the_values_to_their_cells_and_leaves_the_rest_missing(cdata):
    positions = np.array([2, 5, 30])
    view = _label_view(cdata, positions, ["x", "y", "x"])
    cdata.obs_computed.update_from(view, "subtype")
    result = cdata.obs_computed["subtype"]
    assert isinstance(result, pd.Series) and len(result) == 40
    assert result.index.tolist() == cdata.obs.index.tolist()
    assert str(result.dtype) == "string"
    assert result.iloc[positions].tolist() == ["x", "y", "x"]
    assert result.drop(result.index[positions]).isna().all()


@pytest.mark.parametrize(
    "values, expected_dtype",
    [
        (np.array([True, False, True]), "boolean"),
        (np.array([1, 2, 3]), "Int64"),
        (np.array([1.5, 2.5, np.nan]), "float64"),
        (pd.Categorical(["a", "b", "a"], categories=["a", "b", "c"]), "category"),
        (np.array(["u", "v", "w"], dtype=object), "string"),
    ],
)
def test_update_from_keeps_the_kind_of_value_in_a_dtype_with_missing_values(cdata, values, expected_dtype):
    positions = np.array([2, 5, 30])
    view = cdata[positions]
    view.obs_computed["v"] = values  # an array or Categorical: matched to the cells by position
    cdata.obs_computed.update_from(view, "v")
    result = cdata.obs_computed["v"]
    assert str(result.dtype) == expected_dtype
    assert result.drop(result.index[positions]).isna().all()
    assert result.iloc[positions].tolist()[:2] == pd.Series(values).tolist()[:2]
    if expected_dtype == "category":
        assert result.cat.categories.tolist() == ["a", "b", "c"]


def test_several_views_fill_in_one_entry(cdata):
    first = _label_view(cdata, np.array([0, 1, 2]), ["t1", "t2", "t1"])
    second = _label_view(cdata, np.array([10, 11]), ["b1", "b2"])
    cdata.obs_computed.update_from(first, "subtype")
    cdata.obs_computed.update_from(second, "subtype")
    result = cdata.obs_computed["subtype"]
    assert result.iloc[[0, 1, 2, 10, 11]].tolist() == ["t1", "t2", "t1", "b1", "b2"]
    assert result.notna().sum() == 5


def test_a_later_update_wins_silently_unless_overwrite_is_false(cdata):
    first = _label_view(cdata, np.array([0, 1, 2]), ["a", "a", "a"])
    cdata.obs_computed.update_from(first, "subtype")
    second = _label_view(cdata, np.array([2, 3]), ["b", "b"])
    with pytest.raises(ValueError, match="1 cells already have a value"):
        cdata.obs_computed.update_from(second, "subtype", overwrite=False)
    assert cdata.obs_computed["subtype"].iloc[[2, 3]].isna().tolist() == [False, True]  # nothing changed
    cdata.obs_computed.update_from(second, "subtype")  # overwrite=True is the default
    assert cdata.obs_computed["subtype"].iloc[[0, 1, 2, 3]].tolist() == ["a", "a", "b", "b"]
    # without a conflict, overwrite=False is fine
    third = _label_view(cdata, np.array([20]), ["c"])
    cdata.obs_computed.update_from(third, "subtype", overwrite=False)
    assert cdata.obs_computed["subtype"].iloc[20] == "c"


def test_missing_values_of_the_view_do_not_replace_existing_ones(cdata):
    cdata.obs_computed.update_from(_label_view(cdata, np.array([0]), ["a"]), "subtype")
    view = _label_view(cdata, np.array([0, 1]), [pd.NA, "z"])
    cdata.obs_computed.update_from(view, "subtype", overwrite=False)  # the NA at cell 0 replaces nothing
    assert cdata.obs_computed["subtype"].iloc[[0, 1]].tolist() == ["a", "z"]
    view = _label_view(cdata, np.array([0, 1]), ["q", pd.NA], key="subtype")
    cdata.obs_computed.update_from(view, "subtype")
    assert cdata.obs_computed["subtype"].iloc[[0, 1]].tolist() == ["q", "z"]


def test_new_labels_extend_the_categories_of_a_categorical_entry(cdata):
    first = cdata[[0, 1]]
    first.obs_computed["c"] = pd.Series(pd.Categorical(["a", "b"]), index=first.obs.index)
    cdata.obs_computed.update_from(first, "c")
    second = cdata[[5, 6]]
    second.obs_computed["c"] = pd.Series(pd.Categorical(["b", "z"]), index=second.obs.index)
    cdata.obs_computed.update_from(second, "c")
    result = cdata.obs_computed["c"]
    assert str(result.dtype) == "category"
    assert result.cat.categories.tolist() == ["a", "b", "z"]
    assert result.iloc[[0, 1, 5, 6]].tolist() == ["a", "b", "b", "z"]


def test_update_from_can_store_under_another_key(cdata):
    view = _label_view(cdata, np.array([4, 7]), ["p", "q"])
    cdata.obs_computed.update_from(view, "subtype", as_key="t_cell_subtype")
    assert "subtype" not in cdata.obs_computed
    assert cdata.obs_computed["t_cell_subtype"].iloc[[4, 7]].tolist() == ["p", "q"]


def test_update_from_works_through_nested_views(cdata):
    middle = cdata[np.arange(0, 40, 2)]
    inner = middle[[1, 3]]  # root cells 2 and 6
    inner.obs_computed["s"] = pd.Series(["u", "v"], index=inner.obs.index)
    middle.obs_computed.update_from(inner, "s")
    assert middle.obs_computed["s"].iloc[[1, 3]].tolist() == ["u", "v"]
    assert middle.obs_computed["s"].notna().sum() == 2
    cdata.obs_computed.update_from(middle, "s")
    result = cdata.obs_computed["s"]
    assert result.iloc[[2, 6]].tolist() == ["u", "v"] and result.notna().sum() == 2
    # straight from the inner view to the root, too
    cdata.obs_computed.update_from(inner, "s", as_key="direct")
    assert cdata.obs_computed["direct"].equals(result)


def test_update_from_rejects_data_it_cannot_match(cdata, make_h5ad_files):
    view = _label_view(cdata, np.array([1, 2]), ["a", "b"])
    other = CellariumData(h5ad_paths=make_h5ad_files(n_files=2, cells_per_file=20, n_genes=30, seed=1))
    with pytest.raises(ValueError, match="same root"):
        other.obs_computed.update_from(view, "subtype")
    # a view whose cells are not among those of the target
    narrow = cdata[np.arange(10, 20)]
    with pytest.raises(ValueError, match="not among"):
        narrow.obs_computed.update_from(view, "subtype")
    with pytest.raises(KeyError, match="nothing"):
        cdata.obs_computed.update_from(view, "nothing")
    assert len(cdata.obs_computed) == 0


def test_update_from_checks_what_it_is_given(cdata):
    view = cdata[[1, 2]]
    view.obs_computed["wrong_index"] = pd.Series(["a", "b"], index=["x", "y"])
    with pytest.raises(ValueError, match="index"):
        cdata.obs_computed.update_from(view, "wrong_index")
    view.obs_computed["two_d"] = np.zeros((2, 2))
    with pytest.raises(ValueError, match="one entry per cell"):
        cdata.obs_computed.update_from(view, "two_d")
    view.obs_computed["text"] = pd.Series(["a", "b"], index=view.obs.index)
    cdata.obs_computed["number"] = np.arange(40, dtype=float)
    with pytest.raises(ValueError, match="do not fit"):
        cdata.obs_computed.update_from(view, "text", as_key="number")
    assert cdata.obs_computed["number"].tolist() == list(range(40))  # unchanged


def test_update_from_the_data_itself_is_a_copy_under_another_key(cdata):
    cdata.obs_computed["a"] = pd.Series(np.arange(40.0), index=cdata.obs.index)
    cdata.obs_computed.update_from(cdata, "a", as_key="b")
    assert cdata.obs_computed["b"].tolist() == cdata.obs_computed["a"].tolist()
