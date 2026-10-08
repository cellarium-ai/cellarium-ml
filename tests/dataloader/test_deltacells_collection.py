# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
import pickle
from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import torch

# These tests start DataLoader worker processes and Lightning trainers; running them in parallel under pytest-xdist
# can exhaust the memory of a laptop. Under xdist (workers have PYTEST_XDIST_WORKER set) the module is skipped:
# run it with plain `pytest` (about 4 minutes, one process at a time).
if os.environ.get("PYTEST_XDIST_WORKER") is not None:
    pytest.skip("memory heavy: run without pytest-xdist (-n)", allow_module_level=True)

# `deltacells._core` is the compiled extension; a bare `deltacells` can also be an unrelated folder on sys.path
pytest.importorskip("deltacells._core")
pytest.importorskip("pyarrow")

from deltacells.obs import ObsSchema  # noqa: E402
from deltacells.writer import DatasetWriter  # noqa: E402

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule  # noqa: E402
from cellarium.ml.cli import (  # noqa: E402
    compute_batch_index_n_categories,
    compute_n_cats_per_cov,
    compute_n_obs,
    compute_n_vars,
    compute_y_categories,
)
from cellarium.ml.data import (  # noqa: E402
    DeltaCellsBatch,
    DistributedCollection,
    DistributedDeltaCellsCollection,
    IterableDistributedAnnDataCollectionDataset,
)
from cellarium.ml.utilities.data import (  # noqa: E402
    AnnDataField,
    categories_to_codes,
    collate_fn,
    densify,
    to_torch_sparse_csr,
)
from tests.common import BoringModel  # noqa: E402

torch.multiprocessing.set_sharing_strategy("file_system")


@pytest.fixture()
def obs() -> pd.DataFrame:
    n_cell = 20
    obs = pd.DataFrame(
        data={
            "batch": np.concatenate([np.zeros(3), np.ones(n_cell)]).astype(int)[:n_cell],
            "assay": np.array(["10x", "dropseq"] * (n_cell // 2 + 1))[:n_cell],
        }
    )
    obs["batch"] = obs["batch"].astype("category")
    obs["assay"] = obs["assay"].astype("category")
    return obs


def build_dataset(path: Path, limits: list[int], obs: pd.DataFrame, **writer_kwargs) -> None:
    n_cell, tile = limits[-1], limits[0]
    X = scipy.sparse.csr_matrix(np.arange(n_cell, dtype=np.float32).reshape(n_cell, 1))
    frame = obs.iloc[:n_cell].reset_index(drop=True).copy()
    frame["obs_names"] = [f"cell{i}" for i in range(n_cell)]
    var = pd.DataFrame({"gene_name": ["geneA"]}, index=pd.Index(["gene0"], name="var_id"))
    with DatasetWriter(
        str(path), n_genes=1, tile_size=tile, var_names=["gene0"], var=var, obs_schema=ObsSchema.infer([frame]),
        level=3, n_chunks=1, **writer_kwargs,
    ) as writer:  # fmt: skip
        for lo in range(0, n_cell, tile):
            writer.add_tile(X[lo : lo + tile], obs=frame.iloc[lo : lo + tile])


@pytest.fixture(params=[[3, 6, 9, 12], [4, 8, 11]])  # 4 even tiles, and 3 tiles with a shorter last one
def dadc(tmp_path: Path, obs: pd.DataFrame, request: pytest.FixtureRequest) -> DistributedDeltaCellsCollection:
    build_dataset(tmp_path / "ds", request.param, obs)
    return DistributedDeltaCellsCollection(str(tmp_path / "ds"), cache_dir=str(tmp_path / "cache"))


# ------------------------------------------------------------------------------------------------ the collection


def test_collection_interface(dadc: DistributedDeltaCellsCollection):
    assert isinstance(dadc, DistributedCollection) and dadc.supports_prefetch
    assert dadc.n_obs == len(dadc) == dadc.limits[-1] and dadc.n_vars == 1
    assert dadc.limits in ([3, 6, 9, 12], [4, 8, 11])
    assert list(dadc.var_names) == ["gene0"] and list(dadc.var["gene_name"]) == ["geneA"]
    assert dadc.obs_columns_available == ["batch", "assay", "obs_names"]
    assert "DistributedDeltaCellsCollection" in repr(dadc)
    clone = pickle.loads(pickle.dumps(dadc))
    assert clone.limits == dadc.limits and (clone[[1, 2]].X != dadc[[1, 2]].X).nnz == 0


def test_a_dataset_without_var_names_is_rejected(tmp_path: Path):
    with DatasetWriter(str(tmp_path / "ds"), n_genes=2, tile_size=2, level=3) as w:
        w.add_tile(scipy.sparse.csr_matrix(np.ones((2, 2), dtype=np.float32)))
    with pytest.raises(ValueError, match="var_names"):
        DistributedDeltaCellsCollection(str(tmp_path / "ds"))


def test_batch_is_anndata_like(dadc: DistributedDeltaCellsCollection, obs: pd.DataFrame):
    n = dadc.n_obs
    idx = [n - 1, 0, 2, 2]
    batch = dadc[idx]
    assert isinstance(batch, DeltaCellsBatch) and batch.n_obs == len(batch) == 4 and batch.shape == (4, 1)
    np.testing.assert_array_equal(batch.X.toarray().ravel(), np.array(idx, dtype=np.float32))
    assert batch.X.dtype == np.float32 and batch.X is batch.X  # decoded once
    assert list(batch.var_names) == ["gene0"] and list(batch.var.index) == ["gene0"]
    assert list(batch.obs_names) == [f"cell{i}" for i in idx]
    for key in ("batch", "assay"):
        series = batch.obs[key]
        assert isinstance(series.dtype, pd.CategoricalDtype)
        # the dataset's global categories, as strings
        assert list(series.cat.categories) == [str(c) for c in obs[key].cat.categories]
        assert series.tolist() == [str(v) for v in obs[key].iloc[idx]]
    df = batch.obs[["assay", "batch"]]
    assert list(df.columns) == ["assay", "batch"] and "assay" in batch.obs and "nope" not in batch.obs
    # other ways of indexing
    assert dadc[3].X.toarray().item() == 3.0 and dadc[-1].X.toarray().item() == n - 1
    assert dadc[1:5:2].X.toarray().ravel().tolist() == [1.0, 3.0]
    mask = np.zeros(n, dtype=bool)
    mask[[0, 4]] = True
    assert dadc[mask].X.toarray().ravel().tolist() == [0.0, 4.0]
    with pytest.raises(AttributeError, match="provides only the AnnData attributes"):
        batch.obsm  # noqa: B018


def test_attributes_are_read_lazily(dadc: DistributedDeltaCellsCollection):
    batch = dadc[[1, 5]]
    batch.obs["batch"]
    batch.var_names
    assert dadc.dataset.stats["tiles_decoded"] == 0 and dadc.dataset.stats["tiles_fetched"] == 0, (
        "no tile is read for metadata"
    )
    batch.X
    assert dadc.dataset.stats["tiles_decoded"] >= 1


def test_obs_reads_do_not_warn(dadc: DistributedDeltaCellsCollection, recwarn):
    dadc[[0, 1]].obs["assay"]
    assert not [w for w in recwarn.list if "local cache" in str(w.message)]
    assert "assay" in dadc.dataset.obs.localized_columns()


def test_a_dataset_without_obs_raises_a_clear_error(tmp_path: Path):
    with DatasetWriter(str(tmp_path / "ds"), n_genes=1, tile_size=3, var_names=["g"], level=3) as w:
        w.add_tile(scipy.sparse.csr_matrix(np.ones((3, 1), dtype=np.float32)))
    dadc = DistributedDeltaCellsCollection(str(tmp_path / "ds"))
    assert dadc.obs_columns_available == [] and dadc[[0]].X.toarray().item() == 1.0
    with pytest.raises(ValueError, match="no obs"):
        dadc[[0]].obs["x"]


# --------------------------------------------------------------------------------------- fields and the dataset


def test_iterable_dataset_anndatafields(dadc: DistributedDeltaCellsCollection, obs: pd.DataFrame):
    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc,
        batch_keys={
            "x_ng": AnnDataField("X", convert_fn=densify),
            "var_names_g": AnnDataField("var_names"),
            "batch_n": AnnDataField("obs", key="batch", convert_fn=categories_to_codes),
            "assay_n": AnnDataField("obs", key="assay", convert_fn=categories_to_codes),
            "batch_assay_n2": AnnDataField("obs", key=["batch", "assay"], convert_fn=categories_to_codes),
        },
        batch_size=1,
        shuffle=False,
        test_mode=True,
    )
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn)
    for i, batch in enumerate(data_loader):
        assert batch["x_ng"].shape == (1, 1) and batch["x_ng"].dtype == torch.float32
        assert batch["x_ng"].item() == float(i)
        assert batch["var_names_g"].tolist() == ["gene0"]
        assert batch["batch_n"] == torch.tensor([obs["batch"].cat.codes[i]])
        assert batch["assay_n"] == torch.tensor([obs["assay"].cat.codes[i]])
        torch.testing.assert_close(
            torch.cat([batch["batch_n"], batch["assay_n"]]).unsqueeze(0), batch["batch_assay_n2"]
        )
    assert i == dadc.n_obs - 1


def test_sparse_x_through_the_worker_and_collate(dadc: DistributedDeltaCellsCollection):
    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc,
        batch_keys={"x_ng": AnnDataField("X", convert_fn=to_torch_sparse_csr)},  # type: ignore[arg-type]
        batch_size=2,
        shuffle=False,
    )
    data_loader = torch.utils.data.DataLoader(dataset, num_workers=1, collate_fn=collate_fn)
    values = []
    for batch in data_loader:
        assert batch["x_ng"].layout == torch.sparse_csr and batch["x_ng"].dtype == torch.float32
        values.extend(batch["x_ng"].to_dense().ravel().tolist())
    assert sorted(values) == [float(i) for i in range(dadc.n_obs)]


# A deliberately small selection of cases (worker processes are expensive: each one starts a Python interpreter):
# (iteration_strategy, shuffle, num_workers, batch_size, drop_incomplete_batch, start_idx, end_idx)
ITERABLE_DATASET_CASES = [
    ("same_order", False, 0, 1, False, 0, None),
    ("same_order", True, 0, 3, False, 2, 10),
    ("same_order", True, 0, 2, True, 0, 10),
    ("same_order", False, 1, 2, False, 2, None),
    ("same_order", True, 2, 3, True, 0, None),
    ("cache_efficient", False, 0, 3, True, 2, None),
    ("cache_efficient", True, 0, 1, False, 0, 10),
    ("cache_efficient", True, 0, 2, False, 2, None),
    ("cache_efficient", False, 1, 3, False, 0, 10),
    ("cache_efficient", True, 2, 2, True, 2, 10),
]


@pytest.mark.parametrize(
    "iteration_strategy, shuffle, num_workers, batch_size, drop_incomplete_batch, start_idx, end_idx",
    ITERABLE_DATASET_CASES,
)
def test_iterable_dataset(
    dadc: DistributedDeltaCellsCollection,
    iteration_strategy: Literal["same_order", "cache_efficient"],
    shuffle: bool,
    num_workers: int,
    batch_size: int,
    drop_incomplete_batch: bool,
    start_idx: int | None,
    end_idx: int | None,
):
    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc,
        iteration_strategy=iteration_strategy,
        batch_keys={"x_ng": AnnDataField("X", convert_fn=densify)},
        batch_size=batch_size,
        shuffle=shuffle,
        drop_incomplete_batch=drop_incomplete_batch,
        start_idx=start_idx,
        end_idx=end_idx,
        test_mode=True,
    )
    data_loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, collate_fn=collate_fn)

    all_batches = list(data_loader)
    miss_counts = list(int(batch["miss_count"]) for batch in all_batches for _ in batch["x_ng"])
    actual_idx = list(int(i) for batch in all_batches for i in batch["x_ng"])

    worker_ids = list(int(batch["worker_id"]) for batch in all_batches for _ in batch["x_ng"])
    tiles = np.searchsorted([0] + dadc.limits, actual_idx, side="right")
    for worker in set(worker_ids):
        miss_count = max(c for c, w in zip(miss_counts, worker_ids) if w == worker)
        assert miss_count == len(set([o for o, w in zip(tiles, worker_ids) if w == worker])), (
            "each tile is fetched once"
        )

    n_obs = dataset.end_idx - dataset.start_idx
    expected_idx = list(range(dataset.start_idx, dataset.end_idx))
    expected_len = n_obs
    if drop_incomplete_batch and n_obs % batch_size != 0:
        expected_len = n_obs // batch_size * batch_size
    assert expected_len == len(actual_idx)

    # assert entire dataset is sampled
    if not shuffle and iteration_strategy == "same_order":
        assert expected_idx[:expected_len] == actual_idx
    else:
        if drop_incomplete_batch and n_obs % batch_size != 0:
            assert len(set(expected_idx) - set(actual_idx)) < batch_size
        else:
            assert set(expected_idx) == set(actual_idx)


# ------------------------------------------------------------------------------------------------ prefetch look-ahead


class RecordingCollection(DistributedDeltaCellsCollection):
    """Records the order of reads and of prefetch announcements."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.events: list[tuple[str, tuple[int, ...]]] = []

    def _tiles(self, indices) -> tuple[int, ...]:
        return tuple(sorted(set(np.searchsorted(self.limits, np.asarray(indices), side="right").tolist())))

    def prefetch(self, indices):
        self.events.append(("prefetch", self._tiles(indices)))
        super().prefetch(indices)

    def __getitem__(self, index):
        self.events.append(("read", self._tiles(index)))
        return super().__getitem__(index)


@pytest.mark.parametrize("iteration_strategy", ["same_order", "cache_efficient"])
@pytest.mark.parametrize("lookahead", [1, 2, 3])
def test_the_dataset_announces_upcoming_tiles_before_they_are_read(
    tmp_path: Path, obs: pd.DataFrame, iteration_strategy: str, lookahead: int
):
    build_dataset(tmp_path / "ds", [3, 6, 9, 12], obs)
    dadc = RecordingCollection(str(tmp_path / "ds"), cache_dir=str(tmp_path / "cache"))
    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc,
        iteration_strategy=iteration_strategy,  # type: ignore[arg-type]
        batch_keys={"x_ng": AnnDataField("X", convert_fn=densify)},
        batch_size=2,
        shuffle=True,
        prefetch_lookahead=lookahead,
    )
    values = [float(v) for batch in torch.utils.data.DataLoader(dataset, collate_fn=collate_fn) for v in batch["x_ng"]]
    assert sorted(values) == [float(i) for i in range(12)]

    announced: set[int] = set()
    read: set[int] = set()
    n_announced_before_reads = 0
    for kind, tiles in dadc.events:
        if kind == "prefetch":
            announced.update(tiles)
        else:
            if read:  # after the first batch, every tile that is read was announced earlier (or already read)
                assert set(tiles) <= announced | read, f"tile(s) {set(tiles) - announced - read} were read unannounced"
            read.update(tiles)
            n_announced_before_reads += len(announced)
    assert announced, "prefetch hints were sent while iterating"
    assert read == {0, 1, 2, 3}
    # each batch is announced once, so the number of announcements is at most the number of batches
    assert sum(1 for kind, _ in dadc.events if kind == "prefetch") <= sum(
        1 for kind, _ in dadc.events if kind == "read"
    )


def test_h5ad_style_collections_get_no_prefetch_calls(tmp_path: Path, obs: pd.DataFrame):
    build_dataset(tmp_path / "ds", [3, 6, 9, 12], obs)

    class NoPrefetch(RecordingCollection):
        supports_prefetch = False

    dadc = NoPrefetch(str(tmp_path / "ds"), cache_dir=str(tmp_path / "cache"))
    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc, batch_keys={"x_ng": AnnDataField("X", convert_fn=densify)}, batch_size=2
    )
    assert len(list(torch.utils.data.DataLoader(dataset, collate_fn=collate_fn))) == 6
    assert all(kind == "read" for kind, _ in dadc.events)


def test_prefetch_lookahead_zero_disables_it(tmp_path: Path, obs: pd.DataFrame):
    build_dataset(tmp_path / "ds", [3, 6, 9, 12], obs)
    dadc = RecordingCollection(str(tmp_path / "ds"), cache_dir=str(tmp_path / "cache"))
    dataset = IterableDistributedAnnDataCollectionDataset(
        dadc, batch_keys={"x_ng": AnnDataField("X", convert_fn=densify)}, batch_size=2, prefetch_lookahead=0
    )
    list(torch.utils.data.DataLoader(dataset, collate_fn=collate_fn))
    assert all(kind == "read" for kind, _ in dadc.events)


# -------------------------------------------- prepare and the datamodule


def make_datamodule(dadc, **kwargs) -> CellariumAnnDataDataModule:
    batch_keys = kwargs.pop(
        "batch_keys",
        {
            "x_ng": AnnDataField("X", convert_fn=densify),
            "batch_n": AnnDataField("obs", key="batch", convert_fn=categories_to_codes),
        },
    )
    return CellariumAnnDataDataModule(dadc, batch_keys=batch_keys, **kwargs)


def test_prepare_data_localizes_exactly_the_columns_the_batch_keys_use(dadc: DistributedDeltaCellsCollection):
    store = dadc.dataset.obs
    assert store.localized_columns() == []
    make_datamodule(dadc, batch_size=2).prepare_data()
    assert store.localized_columns() == ["batch"]
    make_datamodule(
        dadc,
        batch_size=2,
        batch_keys={
            "x_ng": AnnDataField("X", convert_fn=densify),
            "ba": AnnDataField("obs", key=["assay", "batch"], convert_fn=categories_to_codes),
            "names": AnnDataField("obs_names", convert_fn=lambda x: np.asarray(x)),
        },
    ).prepare_data()
    assert sorted(store.localized_columns()) == ["assay", "batch", "obs_names"]


def test_prepare_data_with_the_whole_obs_and_extra_columns(tmp_path: Path, obs: pd.DataFrame):
    build_dataset(tmp_path / "ds", [3, 6, 9, 12], obs)
    everything = DistributedDeltaCellsCollection(str(tmp_path / "ds"), cache_dir=str(tmp_path / "c1"))
    everything.prepare({"o": AnnDataField("obs")})
    assert sorted(everything.dataset.obs.localized_columns()) == ["assay", "batch", "obs_names"]
    extra = DistributedDeltaCellsCollection(str(tmp_path / "ds"), cache_dir=str(tmp_path / "c2"), obs_columns=["assay"])
    extra.prepare({"x_ng": AnnDataField("X", convert_fn=densify)})  # no obs field: only the extra column
    assert extra.dataset.obs.localized_columns() == ["assay"]


def test_prepare_rejects_unsupported_attributes_and_unknown_columns(
    dadc: DistributedDeltaCellsCollection, tmp_path: Path
):
    with pytest.raises(ValueError, match="cannot provide 'obsm'"):
        dadc.prepare({"z": AnnDataField("obsm", key="X_pca")})
    with pytest.raises(KeyError, match="not in the dataset"):
        dadc.prepare({"y": AnnDataField("obs", key="nope")})
    with DatasetWriter(str(tmp_path / "noobs"), n_genes=1, tile_size=3, var_names=["g"], level=3) as w:
        w.add_tile(scipy.sparse.csr_matrix(np.ones((3, 1), dtype=np.float32)))
    no_obs = DistributedDeltaCellsCollection(str(tmp_path / "noobs"))
    no_obs.prepare({"x": AnnDataField("X", convert_fn=densify)})  # nothing needed, nothing to do
    with pytest.raises(ValueError, match="has no obs"):
        no_obs.prepare({"y": AnnDataField("obs", key="batch")})


def test_cli_link_helpers_work_with_the_collection(dadc: DistributedDeltaCellsCollection, obs: pd.DataFrame):
    dm = make_datamodule(
        dadc,
        batch_size=2,
        batch_keys={
            "x_ng": AnnDataField("X", convert_fn=densify),
            "y_categories": AnnDataField("obs", key="assay", convert_fn=lambda s: np.asarray(s.cat.categories)),
            "categorical_covariate_index_nd": AnnDataField(
                "obs", key=["batch", "assay"], convert_fn=categories_to_codes
            ),
            "batch_index_n": AnnDataField("obs", key=["batch", "assay"], convert_fn=categories_to_codes),
        },
    )
    assert compute_n_obs(dm) == dadc.n_obs and compute_n_vars(dm) == 1
    assert list(compute_y_categories(dm)) == list(obs["assay"].cat.categories)
    assert compute_n_cats_per_cov(dm) == [len(obs["batch"].cat.categories), len(obs["assay"].cat.categories)]
    assert compute_batch_index_n_categories(dm) == len(obs["batch"].cat.categories) * len(obs["assay"].cat.categories)


def test_train_val_split(dadc: DistributedDeltaCellsCollection):
    dm = make_datamodule(dadc, batch_size=2, train_size=0.5, shuffle=True, num_workers=1)  # validation: the rest
    dm.prepare_data()
    dm.setup("fit")
    train = [int(v) for batch in dm.train_dataloader() for v in batch["x_ng"]]
    val = [int(v) for batch in dm.val_dataloader() for v in batch["x_ng"]]
    assert sorted(train + val) == list(range(dadc.n_obs)) and not set(train) & set(val)
    assert len(train) == dm.n_train and len(val) == dm.n_val


# (iteration_strategy, shuffle, num_workers, batch_size, resume_step): few cases, trainers are heavy
@pytest.mark.parametrize(
    "iteration_strategy, shuffle, num_workers, batch_size, resume_step",
    [
        ("same_order", False, 0, 3, 1),
        ("same_order", True, 0, 1, 5),
        ("cache_efficient", True, 0, 3, 5),
        ("cache_efficient", False, 1, 3, 1),
    ],
)
def test_load_from_checkpoint(
    dadc: DistributedDeltaCellsCollection,
    iteration_strategy: Literal["same_order", "cache_efficient"],
    shuffle: bool,
    num_workers: int,
    batch_size: int,
    tmp_path: Path,
    resume_step: int,
):
    def datamodule() -> CellariumAnnDataDataModule:
        return CellariumAnnDataDataModule(
            dadc=dadc,
            batch_keys={"x_ng": AnnDataField("X", convert_fn=densify)},
            batch_size=batch_size,
            iteration_strategy=iteration_strategy,
            shuffle=shuffle,
            test_mode=True,
            num_workers=num_workers,
        )

    module1 = CellariumModule(model=BoringModel())
    trainer1 = pl.Trainer(
        accelerator="cpu",
        max_epochs=3,
        logger=False,
        callbacks=[pl.callbacks.ModelCheckpoint(every_n_train_steps=1, save_top_k=-1)],
        default_root_dir=tmp_path,
    )
    trainer1.fit(module1, datamodule())

    module2 = CellariumModule(model=BoringModel())
    trainer2 = pl.Trainer(accelerator="cpu", max_epochs=3, logger=False)
    try:
        ckpt_path = tmp_path / f"checkpoints/epoch=0-step={resume_step}.ckpt"
        trainer2.fit(module2, datamodule(), ckpt_path=ckpt_path)
    except FileNotFoundError:
        ckpt_path = tmp_path / f"checkpoints/epoch=1-step={resume_step}.ckpt"
        trainer2.fit(module2, datamodule(), ckpt_path=ckpt_path)

    iter_data1 = collate_fn(module1.model.iter_data)
    iter_data2 = collate_fn(module2.model.iter_data)
    torch.testing.assert_close(iter_data1["x_ng"], iter_data2["x_ng"])
