# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from cellarium.ml.callbacks.prediction_writer_arrow import PredictionWriterArrow
from cellarium.ml.data import (
    DistributedArrowDataCollection,
    IterableDistributedAnnDataCollectionDataset,
)

# ---------------------------------------------------------------------------
# Shared data dimensions
# ---------------------------------------------------------------------------

N_CELL = 10
N_GENE = 5
# Three shards: 2 cells, 3 cells, 5 cells
SHARD_LIMITS = [2, 5, 10]
VAR_NAMES = np.array([f"gene{i:03d}" for i in range(N_GENE)])
Y_CATEGORIES = np.array(["type_a", "type_b", "type_c"])


# ---------------------------------------------------------------------------
# Helper: write a single Arrow shard via PredictionWriterArrow._write_arrow
# ---------------------------------------------------------------------------


def _write_shard(dp: PredictionWriterArrow, x: np.ndarray, obs_names: np.ndarray, y: np.ndarray) -> None:
    """Write one shard for the given slice of cells."""
    dp._write_arrow(
        {
            "x_ng": x,
            "obs_names_n": obs_names,
            "y_n": y.astype(np.int32),
            "total_mrna_umis_n": x.sum(axis=1).astype(np.float32),
            "var_names_g": VAR_NAMES,
            "y_categories": Y_CATEGORIES,
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def source_arrays() -> dict:
    """Ground-truth numpy arrays for all N_CELL cells."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((N_CELL, N_GENE)).astype(np.float32)
    obs_names = np.array([f"cell{i:03d}" for i in range(N_CELL)], dtype=object)
    y = rng.integers(0, len(Y_CATEGORIES), N_CELL).astype(np.int32)
    return {"x": x, "obs_names": obs_names, "y": y}


@pytest.fixture(scope="module")
def arrows_path(tmp_path_factory: pytest.TempPathFactory, source_arrays: dict) -> Path:
    """Write three Arrow shards matching SHARD_LIMITS to a temp directory."""
    tmp_path = tmp_path_factory.mktemp("arrows")
    dp = PredictionWriterArrow(output_dir=str(tmp_path))
    starts = [0] + SHARD_LIMITS[:-1]
    for start, end in zip(starts, SHARD_LIMITS):
        _write_shard(
            dp,
            source_arrays["x"][start:end],
            source_arrays["obs_names"][start:end],
            source_arrays["y"][start:end],
        )
    return tmp_path


@pytest.fixture(params=[(i, j) for i in (1, 2, 3) for j in (True, False)])
def dat(arrows_path: Path, request: pytest.FixtureRequest) -> DistributedArrowDataCollection:
    """Parametrized DistributedArrowDataCollection over the three test shards."""
    max_cache_size, cache_size_strictly_enforced = request.param
    filenames = [str(arrows_path / f"rank00_shard{i:06d}.arrow") for i in range(3)]
    return DistributedArrowDataCollection(
        filenames,
        limits=SHARD_LIMITS,
        max_cache_size=max_cache_size,
        cache_size_strictly_enforced=cache_size_strictly_enforced,
    )


# ---------------------------------------------------------------------------
# PredictionWriterArrow: field classification and file writing
# ---------------------------------------------------------------------------


def test_preformatter_field_classification(tmp_path: Path) -> None:
    """Check that the right fields land in columns vs schema metadata."""
    import pyarrow as pa

    dp = PredictionWriterArrow(output_dir=str(tmp_path))
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, N_GENE)).astype(np.float32)
    obs_names = np.array([f"c{i}" for i in range(4)], dtype=object)
    y = np.array([0, 1, 0, 2], dtype=np.int32)

    dp._write_arrow(
        {
            "x_ng": x,
            "obs_names_n": obs_names,
            "y_n": y,
            "total_mrna_umis_n": x.sum(axis=1).astype(np.float32),
            "var_names_g": VAR_NAMES,
            "y_categories": Y_CATEGORIES,
        }
    )

    arrow_file = str(tmp_path / "rank00_shard000000.arrow")
    with pa.memory_map(arrow_file, "r") as src:
        reader = pa.ipc.open_file(src)
        batch = reader.get_batch(0)
        meta = reader.schema.metadata

    # --- Per-row columns ---
    assert "x_ng" in batch.schema.names
    assert pa.types.is_fixed_size_binary(batch.schema.field("x_ng").type)
    assert batch.schema.field("x_ng").type.byte_width == N_GENE * 2

    assert "obs_names_n" in batch.schema.names
    assert pa.types.is_large_string(batch.schema.field("obs_names_n").type)

    assert "y_n" in batch.schema.names
    assert pa.types.is_int32(batch.schema.field("y_n").type)

    assert "total_mrna_umis_n" in batch.schema.names
    assert pa.types.is_float32(batch.schema.field("total_mrna_umis_n").type)

    # --- Schema metadata ---
    assert "var_names_g" not in batch.schema.names
    assert b"var_names_g" in meta
    assert "y_categories" not in batch.schema.names
    assert b"y_categories" in meta

    # Check metadata contents
    assert meta[b"cellarium_arrow_version"] == b"1"
    assert int(meta[b"n_genes"]) == N_GENE
    assert meta[b"var_names_g"].decode().split("\n") == list(VAR_NAMES)
    assert json.loads(meta[b"y_categories"].decode()) == list(Y_CATEGORIES)


def test_preformatter_shard_counter(tmp_path: Path) -> None:
    """Counter increments and files are named correctly."""
    dp = PredictionWriterArrow(output_dir=str(tmp_path))
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, N_GENE)).astype(np.float32)
    obs = np.array(["a", "b"], dtype=object)
    y = np.array([0, 1], dtype=np.int32)

    for i in range(4):
        _write_shard(dp, x, obs, y)
        assert dp._shard_counter == i + 1
        assert os.path.exists(str(tmp_path / f"rank00_shard{i:06d}.arrow"))


def test_preformatter_tensor_input(tmp_path: Path) -> None:
    """PredictionWriterArrow accepts torch.Tensor inputs via _to_numpy."""
    dp = PredictionWriterArrow(output_dir=str(tmp_path))
    x_tensor = torch.randn(3, N_GENE)
    obs = np.array([f"c{i}" for i in range(3)], dtype=object)
    y = np.array([0, 0, 1], dtype=np.int32)

    dp._write_arrow(
        {
            "x_ng": x_tensor,
            "obs_names_n": obs,
            "y_n": y,
            "var_names_g": VAR_NAMES,
            "y_categories": Y_CATEGORIES,
        }
    )

    dadc = DistributedArrowDataCollection(
        [str(tmp_path / "rank00_shard000000.arrow")],
        shard_size=3,
    )
    batch = dadc[list(range(3))]
    # float16 round-trip; compare with float32-cast original
    np.testing.assert_allclose(
        batch["x_ng"].astype(np.float32),
        x_tensor.numpy().astype(np.float16).astype(np.float32),
        atol=1e-3,
    )


@pytest.mark.parametrize("compression", ["zstd", "lz4", None])
def test_preformatter_compression(tmp_path: Path, compression: str | None) -> None:
    """Compressed shards are smaller than uncompressed and round-trip values identically."""
    rng = np.random.default_rng(77)
    # Use sparse-ish integer counts (realistic UMI data) so compression has something to work with
    n_cells, n_genes = 50, 200
    x = rng.integers(0, 5, size=(n_cells, n_genes)).astype(np.float32)
    x[x < 3] = 0  # ~60 % sparsity
    obs = np.array([f"c{i:03d}" for i in range(n_cells)], dtype=object)
    y = np.zeros(n_cells, dtype=np.int32)
    var_names = np.array([f"gene{i:04d}" for i in range(n_genes)])

    compressed_dir = tmp_path / f"compressed_{compression}"
    uncompressed_dir = tmp_path / "uncompressed"

    dp_compressed = PredictionWriterArrow(output_dir=str(compressed_dir), compression=compression)
    dp_uncompressed = PredictionWriterArrow(output_dir=str(uncompressed_dir), compression=None)

    batch = {"x_ng": x, "obs_names_n": obs, "y_n": y, "var_names_g": var_names}
    dp_compressed._write_arrow(batch)
    dp_uncompressed._write_arrow(batch)

    compressed_file = compressed_dir / "rank00_shard000000.arrow"
    uncompressed_file = uncompressed_dir / "rank00_shard000000.arrow"
    assert compressed_file.exists()
    assert uncompressed_file.exists()

    if compression is not None:
        assert compressed_file.stat().st_size < uncompressed_file.stat().st_size, (
            f"{compression} compressed file should be smaller than uncompressed"
        )

    # Values must be identical regardless of compression
    dadc = DistributedArrowDataCollection([str(compressed_file)], shard_size=n_cells)
    result = dadc[list(range(n_cells))]
    x_ng = result["x_ng"]
    obs_names_n = result["obs_names_n"]
    assert isinstance(x_ng, np.ndarray)
    assert isinstance(obs_names_n, np.ndarray)
    np.testing.assert_allclose(
        x_ng.astype(np.float32),
        x.astype(np.float16).astype(np.float32),
        atol=1e-3,
    )
    np.testing.assert_array_equal(obs_names_n, obs)


def test_preformatter_async_writes(tmp_path: Path) -> None:
    """predict() offloads writes to the thread pool; on_predict_end() flushes and all files are readable."""
    rng = np.random.default_rng(99)
    n_cells, n_genes = 20, 10
    var_names = np.array([f"gene{i:04d}" for i in range(n_genes)])

    # Use 3 workers so we exercise the bounded-queue backpressure path with 12 batches.
    dp = PredictionWriterArrow(output_dir=str(tmp_path), compression=None, num_write_workers=3)

    n_batches = 12
    batches = []
    for i in range(n_batches):
        x = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
        obs = np.array([f"b{i}c{j:03d}" for j in range(n_cells)], dtype=object)
        y = np.zeros(n_cells, dtype=np.int32)
        batches.append({"x_ng": x, "obs_names_n": obs, "y_n": y, "var_names_g": var_names})
        # _submit_write is async — files may not exist yet
        dp._submit_write(batches[-1])

    assert dp._shard_counter == n_batches, "counter must be incremented synchronously in main thread"

    # Flush — blocks until all worker threads finish
    dp.flush()

    for i in range(n_batches):
        path = tmp_path / f"rank00_shard{i:06d}.arrow"
        assert path.exists(), f"shard {i} not written after flush"

    # Verify one shard round-trips correctly
    dadc = DistributedArrowDataCollection([str(tmp_path / "rank00_shard000000.arrow")], shard_size=n_cells)
    result = dadc[list(range(n_cells))]
    np.testing.assert_allclose(
        result["x_ng"].astype(np.float32),
        batches[0]["x_ng"].astype(np.float16).astype(np.float32),
        atol=1e-3,
    )


# ---------------------------------------------------------------------------
# DistributedArrowDataCollection: init
# ---------------------------------------------------------------------------


def test_init_empty_cache(dat: DistributedArrowDataCollection) -> None:
    """Cache starts empty — unlike AnnData, no shard is read on __init__."""
    assert len(dat.cache) == 0


def test_init_properties(dat: DistributedArrowDataCollection) -> None:
    assert dat.n_obs == N_CELL
    assert len(dat) == N_CELL
    assert dat.limits == SHARD_LIMITS
    # n_vars requires reading the schema footer of shard 0
    assert dat.n_vars == N_GENE


@pytest.mark.parametrize("num_shards", [1, 3, 5])
@pytest.mark.parametrize("last_shard_size", [1, 2, None])
def test_init_shard_size(tmp_path: Path, num_shards: int, last_shard_size: int | None) -> None:
    """limits are computed correctly from shard_size / last_shard_size."""
    shard_size = 3
    rng = np.random.default_rng(99)
    out_dir = tmp_path / f"shards_{num_shards}_{last_shard_size}"
    dp = PredictionWriterArrow(output_dir=str(out_dir))
    filenames = []
    for i in range(num_shards):
        n = last_shard_size if (last_shard_size is not None and i == num_shards - 1) else shard_size
        x = rng.standard_normal((n, N_GENE)).astype(np.float32)
        obs = np.array([f"c{j}" for j in range(n)], dtype=object)
        y = np.zeros(n, dtype=np.int32)
        dp._write_arrow({"x_ng": x, "obs_names_n": obs, "y_n": y, "var_names_g": VAR_NAMES})
        filenames.append(str(out_dir / f"rank00_shard{i:06d}.arrow"))

    dadc = DistributedArrowDataCollection(
        filenames,
        shard_size=shard_size,
        last_shard_size=last_shard_size,
    )

    expected = num_shards * shard_size
    if last_shard_size is not None:
        expected = expected - shard_size + last_shard_size
    assert len(dadc) == expected


# ---------------------------------------------------------------------------
# DistributedArrowDataCollection: schema metadata
# ---------------------------------------------------------------------------


def test_schema_metadata(dat: DistributedArrowDataCollection) -> None:
    meta = dat.get_schema_metadata()

    assert meta["n_obs"] == N_CELL
    assert meta["n_vars"] == N_GENE
    np.testing.assert_array_equal(meta["var_names_g"], VAR_NAMES)
    np.testing.assert_array_equal(meta["y_categories"], Y_CATEGORIES)


# ---------------------------------------------------------------------------
# DistributedArrowDataCollection: indexing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "row_select",
    [(slice(0, 2), 1), (slice(1, 4), 2), ([1, 2, 4, 4], 2), ([6, 1, 3], 3)],
    ids=["one shard", "two shards", "sorted two shards", "unsorted three shards"],
)
def test_indexing(
    dat: DistributedArrowDataCollection,
    source_arrays: dict,
    row_select: tuple,
) -> None:
    oidx, n_shards = row_select
    max_cache_size = dat.max_cache_size
    cache_size_strictly_enforced = dat.cache_size_strictly_enforced

    if cache_size_strictly_enforced and n_shards > max_cache_size:
        with pytest.raises(ValueError, match="Expected the number of Arrow shards"):
            dat[oidx]
        return

    result = dat[oidx]

    # Determine integer indices for comparison against source_arrays
    if isinstance(oidx, slice):
        idx_list = list(range(*oidx.indices(N_CELL)))
    else:
        idx_list = list(oidx)

    # x_ng: float16 round-trip tolerance
    np.testing.assert_allclose(
        result["x_ng"].astype(np.float32),
        source_arrays["x"][idx_list].astype(np.float16).astype(np.float32),
        atol=1e-3,
    )
    # obs_names_n: exact
    np.testing.assert_array_equal(result["obs_names_n"], source_arrays["obs_names"][idx_list])
    # y_n: exact integer codes
    np.testing.assert_array_equal(result["y_n"], source_arrays["y"][idx_list])
    # total_mrna_umis_n: derived from float16 x, so compare to recomputed value
    np.testing.assert_allclose(
        result["total_mrna_umis_n"],
        source_arrays["x"][idx_list].astype(np.float16).sum(axis=1).astype(np.float32),
        atol=1e-2,
    )
    # Metadata fields present in every response
    np.testing.assert_array_equal(result["var_names_g"], VAR_NAMES)
    np.testing.assert_array_equal(result["y_categories"], Y_CATEGORIES)


def test_indexing_single_int(dat: DistributedArrowDataCollection, source_arrays: dict) -> None:
    """Single integer index is converted to a list-of-one internally."""
    result = dat[0]
    assert result["x_ng"].shape == (1, N_GENE)
    np.testing.assert_array_equal(result["obs_names_n"], source_arrays["obs_names"][[0]])


# ---------------------------------------------------------------------------
# DistributedArrowDataCollection: pickle (worker handoff)
# ---------------------------------------------------------------------------


def test_pickle_empty_cache_after_unpickle(dat: DistributedArrowDataCollection) -> None:
    """__setstate__ rebuilds an empty LRU — no file I/O on worker init."""
    new_dat: DistributedArrowDataCollection = pickle.loads(pickle.dumps(dat))
    # Key difference from AnnData: cache is EMPTY after unpickling
    assert len(new_dat.cache) == 0


def test_pickle_reads_correctly_after_unpickle(dat: DistributedArrowDataCollection, source_arrays: dict) -> None:
    new_dat: DistributedArrowDataCollection = pickle.loads(pickle.dumps(dat))

    result = new_dat[list(range(2))]
    np.testing.assert_allclose(
        result["x_ng"].astype(np.float32),
        source_arrays["x"][:2].astype(np.float16).astype(np.float32),
        atol=1e-3,
    )
    np.testing.assert_array_equal(result["obs_names_n"], source_arrays["obs_names"][:2])


# ---------------------------------------------------------------------------
# IterableDistributedAnnDataCollectionDataset with batch_keys=None (Arrow mode)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "row_select",
    [(slice(0, 2), 1), (slice(1, 4), 2), ([1, 2, 4, 4], 2), ([6, 1, 3], 3)],
    ids=["one shard", "two shards", "sorted two shards", "unsorted three shards"],
)
def test_indexing_dataset(
    dat: DistributedArrowDataCollection,
    source_arrays: dict,
    row_select: tuple,
) -> None:
    """Dataset with batch_keys=None passes __getitem__ through to dadc directly."""
    oidx, n_shards = row_select
    max_cache_size = dat.max_cache_size
    cache_size_strictly_enforced = dat.cache_size_strictly_enforced

    dataset = IterableDistributedAnnDataCollectionDataset(dat, batch_keys=None)

    if cache_size_strictly_enforced and n_shards > max_cache_size:
        with pytest.raises(ValueError, match="Expected the number of Arrow shards"):
            dataset[oidx]
        return

    result = dataset[oidx]

    if isinstance(oidx, slice):
        idx_list = list(range(*oidx.indices(N_CELL)))
    else:
        idx_list = list(oidx)

    x_ng = result["x_ng"]
    obs_names_n = result["obs_names_n"]
    assert isinstance(x_ng, np.ndarray)
    assert isinstance(obs_names_n, np.ndarray)
    assert x_ng.shape == (len(idx_list), N_GENE)
    np.testing.assert_allclose(
        x_ng.astype(np.float32),
        source_arrays["x"][idx_list].astype(np.float16).astype(np.float32),
        atol=1e-3,
    )
    np.testing.assert_array_equal(obs_names_n, source_arrays["obs_names"][idx_list])


def test_pickle_dataset(dat: DistributedArrowDataCollection, source_arrays: dict) -> None:
    dataset = IterableDistributedAnnDataCollectionDataset(dat, batch_keys=None)
    new_dataset: IterableDistributedAnnDataCollectionDataset = pickle.loads(pickle.dumps(dataset))

    # Cache inside the dadc should still be empty after unpickling
    assert len(new_dataset.dadc.cache) == 0  # type: ignore[union-attr]

    result = new_dataset[list(range(2))]
    x_ng = result["x_ng"]
    obs_names_n = result["obs_names_n"]
    assert isinstance(x_ng, np.ndarray)
    assert isinstance(obs_names_n, np.ndarray)
    np.testing.assert_allclose(
        x_ng.astype(np.float32),
        source_arrays["x"][:2].astype(np.float16).astype(np.float32),
        atol=1e-3,
    )
    np.testing.assert_array_equal(obs_names_n, source_arrays["obs_names"][:2])
