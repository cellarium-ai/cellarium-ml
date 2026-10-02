# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import json
import multiprocessing
import os
import pickle
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from conftest import make_counts

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from deltacells import (  # noqa: E402
    CountingBackend,
    DatasetWriter,
    LocalBackend,
    Manifest,
    RangeFile,
    ThrottledBackend,
    open_dataset,
)
from deltacells.cli import main  # noqa: E402
from deltacells.obs import ObsSchema, resolve_cache_dir  # noqa: E402

N, TILE, PER_FILE = 350, 100, 2  # 4 tiles -> 2 obs shards


def make_obs(n, seed=0):
    rng = np.random.default_rng(seed)
    cell_type = rng.choice(np.array(["alpha", "beta", "gamma", None], dtype=object), n, p=[0.4, 0.3, 0.2, 0.1])
    return pd.DataFrame(
        {
            "cell_type": pd.Categorical(cell_type),  # with missing values
            "donor": pd.Categorical(rng.choice([f"d{i}" for i in range(7)], n)),
            "n_counts": rng.integers(0, 10**6, n).astype(np.int64),
            "age": np.where(rng.random(n) < 0.1, np.nan, rng.normal(50, 10, n)),
            "doublet": rng.random(n) < 0.2,
            "barcode": [f"BC{rng.integers(1 << 40):x}" for _ in range(n)],
            "sex": pd.array(rng.choice(["F", "M", None], n), dtype="string"),  # pandas string dtype with missing values
        }
    )


def make_var(n_genes):
    return pd.DataFrame(
        {
            "gene_name": pd.Categorical([f"name{i % 11}" for i in range(n_genes)]),
            "gene_id": pd.array([f"ENS{i:05d}" for i in range(n_genes)], dtype="string"),
            "gene_version": np.arange(n_genes, dtype=np.float64),
        },
        index=pd.Index([f"g{i}" for i in range(n_genes)], name="var_id"),
    )


def expected_column(obs, name, idx):
    """Ground truth for ``ObsStore.take``: categoricals as codes of the (sorted-by-appearance) vocabulary, the rest as is."""
    s = obs[name].iloc[idx]
    if isinstance(s.dtype, pd.CategoricalDtype):
        vocab = list(obs[name].cat.categories)
        return np.array([-1 if pd.isna(v) else vocab.index(v) for v in s])
    return s.to_numpy()


def assert_column_equal(got, want, name):
    if want.dtype == object or got.dtype == object:
        assert [None if pd.isna(v) else v for v in got] == [None if pd.isna(v) else v for v in want], name
    elif np.issubdtype(want.dtype, np.floating):
        np.testing.assert_allclose(got, want, equal_nan=True, err_msg=name)
    else:
        assert np.array_equal(got, want), name


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("obsds") / "ds")
    full = make_counts(N, 30, density=0.2, seed=100)
    obs = make_obs(N)
    var = make_var(30)
    order = np.random.default_rng(1).permutation(30)
    schema = ObsSchema.infer([obs])
    with DatasetWriter(
        root,
        n_genes=30,
        tile_size=TILE,
        gene_order=order,
        var=var,
        obs_schema=schema,
        obs_tiles_per_file=PER_FILE,
        level=3,
        n_chunks=2,
    ) as w:
        for lo in range(0, N, TILE):
            w.add_tile(full[lo : lo + TILE], obs=obs.iloc[lo : lo + TILE])
    return root, full, obs, var, order, schema


@pytest.fixture
def cache(tmp_path):
    return str(tmp_path / "cache")


# ---------------------------------------------------------------------------------------------- schema


def test_infer_kinds_vocabularies_and_json_roundtrip():
    obs = make_obs(50)
    schema = ObsSchema.infer([obs])
    kinds = {c.name: c.kind for c in schema.columns}
    assert kinds == {
        "cell_type": "category",
        "donor": "category",
        "n_counts": "numeric",
        "age": "numeric",
        "doublet": "bool",
        "barcode": "string",
        "sex": "string",
    }
    assert schema["cell_type"].categories == ["alpha", "beta", "gamma"] and schema["cell_type"].arrow_type == "int8"
    assert schema["n_counts"].arrow_type == "int64" and schema["age"].arrow_type == "double"
    again = ObsSchema.from_json(schema.to_json())
    assert [c.__dict__ for c in again.columns] == [c.__dict__ for c in schema.columns]
    assert schema.arrow_schema().names == schema.names


def test_vocabulary_is_the_union_across_frames_in_order_of_appearance():
    a = pd.DataFrame({"x": pd.Categorical(["b", "a", "b"], categories=["b", "a"])})
    b = pd.DataFrame({"x": pd.Categorical(["c", "a"], categories=["c", "a", "d"])})
    schema = ObsSchema.infer([a, b])
    assert schema["x"].categories == ["b", "a", "c", "d"]
    t = schema.encode(b)
    assert t.column("x").to_pylist() == [2, 1]  # codes follow the *global* vocabulary, not the frame's own categories


def test_high_cardinality_categoricals_become_strings_at_the_threshold():
    big = pd.DataFrame({"x": pd.Categorical([f"v{i}" for i in range(30)]), "y": pd.Categorical(["a", "b"] * 15)})
    assert ObsSchema.infer([big], max_categories=29)["x"].kind == "string"
    assert ObsSchema.infer([big], max_categories=30)["x"].kind == "category"
    assert ObsSchema.infer([big], max_categories=29)["y"].kind == "category"
    assert ObsSchema.infer([big])["x"].kind == "category"  # the default threshold is 20,000
    from deltacells.obs import DEFAULT_MAX_CATEGORIES

    assert DEFAULT_MAX_CATEGORIES == 20_000


def test_union_exceeding_the_threshold_across_frames_gives_strings():
    frames = [pd.DataFrame({"x": pd.Categorical([f"s{k}_{i}" for i in range(10)])}) for k in range(5)]
    assert ObsSchema.infer(frames, max_categories=40)["x"].kind == "string"
    assert ObsSchema.infer(frames, max_categories=50)["x"].kind == "category"


def test_code_dtype_grows_with_the_vocabulary():
    s = ObsSchema.infer([pd.DataFrame({"x": pd.Categorical([str(i) for i in range(200)])})])
    assert s["x"].arrow_type == "int16"
    s = ObsSchema.infer([pd.DataFrame({"x": pd.Categorical([str(i) for i in range(40_000)])})], max_categories=50_000)
    assert s["x"].arrow_type == "int32"


def test_infer_selects_and_excludes_columns():
    obs = make_obs(20)
    assert ObsSchema.infer([obs], exclude=["barcode", "sex"]).names == [
        "cell_type",
        "donor",
        "n_counts",
        "age",
        "doublet",
    ]
    assert ObsSchema.infer([obs], columns=["age", "donor"]).names == ["age", "donor"]


def test_numeric_types_are_merged_across_frames():
    a, b = pd.DataFrame({"x": np.array([1, 2], dtype=np.int16)}), pd.DataFrame({"x": np.array([1.5], dtype=np.float32)})
    assert ObsSchema.infer([a, b])["x"].pa_type() == pa.float32()  # (Arrow spells float32 "float")
    assert ObsSchema.infer([b, a])["x"].pa_type() == pa.float32()


def test_infer_errors():
    with pytest.raises(ValueError, match="no obs frames"):
        ObsSchema.infer([])
    with pytest.raises(ValueError, match="in some frames"):
        ObsSchema.infer([pd.DataFrame({"x": ["a"]}), pd.DataFrame({"x": [1.0]})])
    with pytest.raises(ValueError, match="different columns"):
        ObsSchema.infer([pd.DataFrame({"x": [1]}), pd.DataFrame({"y": [1]})])
    with pytest.raises(ValueError, match="unsupported dtype"):
        ObsSchema.infer([pd.DataFrame({"t": pd.to_datetime(["2020-01-01"])})])
    with pytest.raises(ValueError, match="lacks columns"):
        ObsSchema.infer([pd.DataFrame({"x": [1]}), pd.DataFrame({"y": [1]})], columns=["x"])


def test_encode_missing_values_unknown_values_and_column_mismatch():
    obs = make_obs(60)
    schema = ObsSchema.infer([obs])
    t = schema.encode(obs)
    assert t.num_rows == 60 and t.schema.equals(schema.arrow_schema())
    codes = np.array(t.column("cell_type").to_pylist())
    assert ((codes == -1) == obs["cell_type"].isna().to_numpy()).all()
    # a category column may be given as plain strings
    as_strings = obs.assign(cell_type=obs["cell_type"].astype(object))
    assert schema.encode(as_strings).column("cell_type").to_pylist() == t.column("cell_type").to_pylist()
    with pytest.raises(ValueError, match="not in the dataset's vocabulary"):
        schema.encode(obs.assign(donor=pd.Categorical(["unseen"] * 60)))
    with pytest.raises(ValueError, match="not in the dataset's vocabulary"):
        schema.encode(obs.assign(cell_type=["zzz"] * 60))
    with pytest.raises(ValueError, match="missing"):
        schema.encode(obs.drop(columns=["age"]))
    with pytest.raises(ValueError, match="unexpected"):
        schema.encode(obs.assign(extra=1))


def test_schema_constructor_validation():
    from deltacells.obs import ColumnSpec

    with pytest.raises(ValueError, match="duplicate"):
        ObsSchema([ColumnSpec("a", "numeric", "int64"), ColumnSpec("a", "numeric", "int64")])
    with pytest.raises(ValueError, match="unknown column kind"):
        ObsSchema([ColumnSpec("a", "weird", "int64")])
    with pytest.raises(ValueError, match="categories"):
        ObsSchema([ColumnSpec("a", "category", "int8")])
    with pytest.raises(KeyError, match="available"):
        ObsSchema([ColumnSpec("a", "numeric", "int64")])["b"]
    with pytest.raises(ValueError, match="not a deltacells obs schema"):
        ObsSchema.from_json(json.dumps({"format": "other"}))


# ---------------------------------------------------------------------------------------------- writer and layout


def test_shards_row_groups_manifest_and_statistics(built):
    root, *_ = built
    m = Manifest.from_json(Path(os.path.join(root, "manifest.json")).read_text())
    assert m.has_obs and m.obs_tiles_per_file == PER_FILE and m.n_obs_shards == 2 and m.has_var_table
    assert len(m.obs_shard_bytes) == len(m.obs_shard_sha256) == 2 and len(m.obs_fingerprint) == 24
    assert list(m.obs_shard_tiles(1)) == [2, 3]
    for s in range(2):
        path = os.path.join(root, m.obs_shard_name(s))
        assert os.path.getsize(path) == m.obs_shard_bytes[s]
        md = pq.read_metadata(path)
        assert [md.row_group(g).num_rows for g in range(md.num_row_groups)] == [
            m.tile_cells[t] for t in m.obs_shard_tiles(s)
        ]
        assert not md.row_group(0).column(0).is_stats_set  # statistics are off: they only inflate the footer
        assert md.row_group(0).column(0).compression == "ZSTD"
    assert os.path.exists(os.path.join(root, "obs", "schema.json"))


def test_obs_is_deterministic(tmp_path):
    obs, full = make_obs(120, seed=3), make_counts(120, 20, seed=3)
    schema = ObsSchema.infer([obs])
    fps = []
    for name in ("a", "b"):
        with DatasetWriter(str(tmp_path / name), n_genes=20, tile_size=60, level=3, obs_schema=schema) as w:
            w.add_tile(full[:60], obs=obs.iloc[:60])
            w.add_tile(full[60:], obs=obs.iloc[60:])
        fps.append(json.loads(Path(tmp_path / name / "manifest.json").read_text())["obs_fingerprint"])
    assert fps[0] == fps[1]


def test_writer_obs_argument_rules(tmp_path):
    obs, full = make_obs(40), make_counts(40, 10, seed=4)
    schema = ObsSchema.infer([obs])
    w = DatasetWriter(str(tmp_path / "a"), n_genes=10, tile_size=40, level=3, obs_schema=schema)
    with pytest.raises(ValueError, match="obs must be given"):
        w.add_tile(full)
    with pytest.raises(ValueError, match="40 cells"):
        w.add_tile(full, obs=obs.iloc[:30])
    with pytest.raises(ValueError, match="missing"):
        w.add_tile(full, obs=obs.drop(columns=["age"]))
    assert not os.path.exists(os.path.join(tmp_path, "a", "tiles", "tile_000000.dct")), (
        "nothing is written for a rejected tile"
    )
    w.add_tile(full, obs=obs)
    w.close()
    plain = DatasetWriter(str(tmp_path / "b"), n_genes=10, tile_size=40, level=3)
    with pytest.raises(ValueError, match="obs must be given"):
        plain.add_tile(full, obs=obs)
    with pytest.raises(ValueError, match="var must have"):
        DatasetWriter(str(tmp_path / "c"), n_genes=10, tile_size=40, var=make_var(9))


def test_writer_accepts_an_already_encoded_table(tmp_path):
    obs, full = make_obs(40), make_counts(40, 10, seed=5)
    schema = ObsSchema.infer([obs])
    with DatasetWriter(str(tmp_path / "d"), n_genes=10, tile_size=40, level=3, obs_schema=schema) as w:
        w.add_tile(full, obs=schema.encode(obs))
    ds = open_dataset(str(tmp_path / "d"), cache_dir=str(tmp_path / "c"), obs_columns=["age"])
    assert_column_equal(ds.obs.take([3, 7], "age")["age"], obs["age"].to_numpy()[[3, 7]], "age")


def test_verify_passes_and_detects_tampering(built, tmp_path):
    root, *_ = built
    open_dataset(root).obs.verify()
    bad = str(tmp_path / "bad")
    shutil.copytree(root, bad)
    path = os.path.join(bad, "obs", "obs_0001.parquet")
    data = bytearray(Path(path).read_bytes())
    data[len(data) // 2] ^= 0xFF
    Path(path).write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        open_dataset(bad).obs.verify()
    os.remove(path)
    with pytest.raises(FileNotFoundError):
        open_dataset(bad).obs.verify()


def test_manifest_obs_validation(built):
    root, *_ = built
    d = json.loads(Path(os.path.join(root, "manifest.json")).read_text())
    for mutate in (
        lambda d: d.update(obs_tiles_per_file=0),
        lambda d: d.update(obs_fingerprint=""),
        lambda d: d.update(obs_shard_bytes=[1]),
        lambda d: d.update(obs_shard_sha256=["x", "y", "z"]),
    ):
        bad = json.loads(json.dumps(d))
        mutate(bad)
        with pytest.raises(ValueError, match="obs fields"):
            Manifest.from_json(json.dumps(bad))
    assert Manifest.from_json(json.dumps(d)).has_obs


# ---------------------------------------------------------------------------------------------- reading


def test_take_matches_ground_truth_for_every_kind(built, cache):
    root, _, obs, *_ = built
    ds = open_dataset(root, cache_dir=cache)
    rng = np.random.default_rng(7)
    ds.obs.localize()
    for _ in range(10):
        idx = rng.integers(0, N, int(rng.integers(1, 120)))  # unsorted, repeated, across tiles and shards
        got = ds.obs.take(idx, ds.obs.columns)
        assert list(got) == ds.obs.columns
        for name in ds.obs.columns:
            assert_column_equal(got[name], expected_column(obs, name, idx), name)


def test_take_basics(built, cache):
    root, _, obs, *_ = built
    store = open_dataset(root, cache_dir=cache).obs
    store.localize(["age", "cell_type"])
    assert store.take([], ["age"])["age"].shape == (0,)
    assert_column_equal(store.take([5], "age")["age"], obs["age"].to_numpy()[[5]], "age")  # a single name is accepted
    with pytest.raises(KeyError, match="unknown obs columns"):
        store.take([1], ["nope"])
    for bad in ([-1], [N], [0, 10**6]):
        with pytest.raises(IndexError):
            store.take(bad, ["age"])
    assert store.kind("cell_type") == "category" and store.categories("cell_type") == ["alpha", "beta", "gamma"]
    with pytest.raises(ValueError, match="not categorical"):
        store.categories("age")
    assert "age" in store.localized_columns() and "donor" not in store.localized_columns()


def test_on_demand_fetch_warns_and_works(built, cache):
    root, _, obs, *_ = built
    store = open_dataset(root, cache_dir=cache).obs
    with pytest.warns(UserWarning, match="not in the local cache"):
        got = store.take([1, 2, 3], ["donor"])
    assert np.array_equal(got["donor"], expected_column(obs, "donor", [1, 2, 3]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a column that is already local must not warn
        store.take([1, 2, 3], ["donor"])


def test_to_pandas(built, cache):
    root, _, obs, *_ = built
    store = open_dataset(root, cache_dir=cache).obs
    df = store.to_pandas(["cell_type", "age", "barcode", "sex", "doublet"])
    assert len(df) == N and isinstance(df.index, pd.RangeIndex)
    assert isinstance(df["cell_type"].dtype, pd.CategoricalDtype) and list(df["cell_type"].cat.categories) == [
        "alpha",
        "beta",
        "gamma",
    ]
    assert df["cell_type"].isna().equals(obs["cell_type"].isna())
    assert (df["cell_type"].dropna().astype(str) == obs["cell_type"].dropna().astype(str)).all()
    assert_column_equal(df["age"].to_numpy(), obs["age"].to_numpy(), "age")
    assert list(df["barcode"]) == list(obs["barcode"]) and df["doublet"].tolist() == obs["doublet"].tolist()
    assert [None if pd.isna(v) else v for v in df["sex"]] == [None if pd.isna(v) else v for v in obs["sex"]]
    plain = store.to_pandas(["cell_type"], categorical=False)
    assert plain["cell_type"].dtype == object
    assert list(store.to_pandas().columns) == obs.columns.tolist()


def test_dataset_obs_columns_argument_and_absent_obs(built, cache, tmp_path):
    root, *_ = built
    ds = open_dataset(root, cache_dir=cache, obs_columns=["age", "donor"])
    assert sorted(ds.obs.localized_columns()) == ["age", "donor"]
    with pytest.raises(KeyError):
        open_dataset(root, cache_dir=cache, obs_columns=["nope"])
    with DatasetWriter(str(tmp_path / "plain"), n_genes=10, tile_size=20, level=3) as w:
        w.add_tile(make_counts(20, 10, seed=6))
    plain = open_dataset(str(tmp_path / "plain"))
    assert plain.obs is None and plain.var is None
    with pytest.raises(ValueError, match="no obs"):
        open_dataset(str(tmp_path / "plain"), obs_columns=["a"])


def test_pickling_a_dataset_with_obs(built, cache):
    root, _, obs, *_ = built
    ds = open_dataset(root, cache_dir=cache, obs_columns=["age"])
    ds.obs.take([1], ["age"])
    assert ds.obs._readers  # the memory map is open
    clone = pickle.loads(pickle.dumps(ds))
    assert clone.cache_dir == cache and not clone.obs._readers  # nothing mapped is pickled; reopened lazily
    assert_column_equal(clone.obs.take([5, 6], ["age"])["age"], obs["age"].to_numpy()[[5, 6]], "age")
    ds.close()
    assert not ds.obs._readers


def test_var_table_roundtrip(built):
    root, _, _, var, order, _ = built
    ds = open_dataset(root)
    got = ds.var
    pd.testing.assert_frame_equal(got, var.iloc[order], check_dtype=False, check_categorical=False)
    assert got.index.name == "var_id" and list(got.index) == [f"g{j}" for j in order]
    assert isinstance(got["gene_name"].dtype, pd.CategoricalDtype)
    assert ds.var is got  # cached


# ---------------------------------------------------------------------------------------------- localization


@pytest.fixture(scope="module")
def wide(tmp_path_factory):
    """12,000 cells in 6 tiles of 2,000 (2 obs shards of 3 tiles) with 30 incompressible float columns and a few small ones."""
    root = str(tmp_path_factory.mktemp("wide") / "ds")
    n, tile = 12_000, 2_000
    rng = np.random.default_rng(11)
    obs = pd.DataFrame({f"f{i:02d}": rng.normal(size=n) for i in range(30)})
    obs["cell_type"] = pd.Categorical(rng.choice(list("abcde"), n))
    obs["donor"] = pd.Categorical(rng.choice([f"d{i}" for i in range(50)], n))
    full = make_counts(n, 10, density=0.05, seed=11)
    schema = ObsSchema.infer([obs])
    with DatasetWriter(
        root, n_genes=10, tile_size=tile, level=3, n_chunks=1, obs_schema=schema, obs_tiles_per_file=3
    ) as w:
        for lo in range(0, n, tile):
            w.add_tile(full[lo : lo + tile], obs=obs.iloc[lo : lo + tile])
    return root, obs


def test_localizing_reads_only_the_requested_columns(wide, cache):
    root, obs = wide
    backend = CountingBackend(LocalBackend(root))
    ds = open_dataset(backend, cache_dir=cache)
    total_obs = sum(ds.manifest.obs_shard_bytes)
    store = ds.obs
    sizes = store.column_nbytes()
    assert sizes["cell_type"] < 0.01 * total_obs < sizes["f00"]
    store.columns  # noqa: B018  (reads the schema)
    backend.reset()
    assert store.localize(["cell_type"]) == ["cell_type"]
    assert backend.bytes_read < 0.15 * total_obs, (
        f"read {backend.bytes_read} of {total_obs} obs bytes for one small column"
    )
    assert backend.n_reads == 0, "obs shards are only ever read with ranged reads"
    first = backend.bytes_read
    backend.reset()
    assert store.localize(["f03"]) == ["f03"]  # adding a column: only its bytes (plus the footers) are read
    assert backend.bytes_read < 0.1 * total_obs + sizes["f03"] + first
    backend.reset()
    assert store.localize(["cell_type", "f03"]) == []
    assert backend.bytes_read == 0 and backend.n_range_reads == 0
    got = store.take(np.arange(0, 12_000, 7), ["cell_type", "f03"])
    assert np.array_equal(got["cell_type"], expected_column(obs, "cell_type", np.arange(0, 12_000, 7)))
    assert np.array_equal(got["f03"], obs["f03"].to_numpy()[::7])
    assert backend.bytes_read == 0, "lookups after localizing never touch the dataset"


def test_localize_is_additive_and_leaves_existing_files_alone(wide, cache):
    root, _ = wide
    store = open_dataset(root, cache_dir=cache).obs
    store.localize(["donor"])
    path = store._path("donor")
    stat = (path.stat().st_mtime_ns, path.stat().st_size)
    store.localize(["f01", "f02"])
    assert (path.stat().st_mtime_ns, path.stat().st_size) == stat
    assert sorted(store.localized_columns()) == ["donor", "f01", "f02"]
    assert not [p for p in store.cache_dir.iterdir() if ".tmp" in p.name]


def test_threads_do_not_change_the_result(wide, tmp_path):
    root, obs = wide
    results = []
    for threads in (1, 6):
        store = open_dataset(root, cache_dir=str(tmp_path / f"c{threads}")).obs
        store.localize(["f05", "donor"], threads=threads)
        results.append(store.take(np.arange(0, 12_000, 13), ["f05", "donor"]))
    assert np.array_equal(results[0]["f05"], results[1]["f05"]) and np.array_equal(
        results[0]["donor"], results[1]["donor"]
    )


def test_localizing_from_a_throttled_remote_is_correct(built, cache):
    root, _, obs, *_ = built
    remote = ThrottledBackend(LocalBackend(root), latency_s=0.01, bandwidth_bytes_per_s=50e6)
    store = open_dataset(remote, cache_dir=cache).obs
    store.localize(["age"], threads=4)
    assert_column_equal(store.take(np.arange(N), ["age"])["age"], obs["age"].to_numpy(), "age")


def test_cache_location_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv("DELTACELLS_CACHE", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert resolve_cache_dir() == tmp_path / "home" / ".cache" / "deltacells"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert resolve_cache_dir() == tmp_path / "xdg" / "deltacells"
    monkeypatch.setenv("DELTACELLS_CACHE", str(tmp_path / "env"))
    assert resolve_cache_dir() == tmp_path / "env"
    assert resolve_cache_dir(str(tmp_path / "arg")) == tmp_path / "arg"


def test_environment_variable_cache_is_used(built, tmp_path, monkeypatch):
    root, *_ = built
    monkeypatch.setenv("DELTACELLS_CACHE", str(tmp_path / "envcache"))
    store = open_dataset(root).obs
    store.localize(["age"])
    assert store._path("age").exists() and str(store._path("age")).startswith(str(tmp_path / "envcache"))


def test_caches_are_keyed_by_dataset_content(tmp_path, cache):
    obs, full = make_obs(40, seed=1), make_counts(40, 10, seed=1)
    other = obs.assign(age=obs["age"] + 1)
    roots = []
    for name, df in (("a", obs), ("b", other)):
        with DatasetWriter(
            str(tmp_path / name), n_genes=10, tile_size=40, level=3, obs_schema=ObsSchema.infer([df])
        ) as w:
            w.add_tile(full, obs=df)
        roots.append(str(tmp_path / name))
    a, b = (open_dataset(r, cache_dir=cache).obs for r in roots)
    assert a.cache_dir != b.cache_dir
    a.localize(["age"])
    b.localize(["age"])
    assert_column_equal(a.take([0, 1], "age")["age"], obs["age"].to_numpy()[:2], "age")
    assert_column_equal(b.take([0, 1], "age")["age"], other["age"].to_numpy()[:2], "age")
    # the same dataset opened again (even from a copy) shares the cache and fetches nothing
    shutil.copytree(roots[0], str(tmp_path / "a_copy"))
    again = CountingBackend(LocalBackend(str(tmp_path / "a_copy")))
    store = open_dataset(again, cache_dir=cache).obs
    store.columns  # noqa: B018
    again.reset()
    assert store.localize(["age"]) == [] and again.bytes_read == 0


def _localize_in_process(root, cache, columns):
    from deltacells import open_dataset as _open

    return _open(root, cache_dir=cache).obs.localize(columns)


def test_concurrent_processes_fetch_each_column_once(wide, cache):
    root, obs = wide
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(4, mp_context=ctx) as pool:
        futures = [pool.submit(_localize_in_process, root, cache, ["f07", "cell_type"]) for _ in range(4)]
        fetched = [f.result() for f in futures]
    assert sum(bool(f) for f in fetched) == 1, f"exactly one process should do the work, got {fetched}"
    store = open_dataset(root, cache_dir=cache).obs
    assert sorted(store.localized_columns()) == ["cell_type", "f07"]
    assert not [p for p in store.cache_dir.iterdir() if ".tmp" in p.name]
    assert np.array_equal(store.take([10, 11999], ["f07"])["f07"], obs["f07"].to_numpy()[[10, 11999]])


@pytest.mark.parametrize("damage", ["truncate", "empty", "garbage"])
def test_a_corrupt_cache_file_is_detected_and_rebuilt(built, cache, damage):
    root, _, obs, *_ = built
    store = open_dataset(root, cache_dir=cache).obs
    store.localize(["age"])
    path = store._path("age")
    data = path.read_bytes()
    path.write_bytes({"truncate": data[: len(data) // 2], "empty": b"", "garbage": b"not an arrow file" * 100}[damage])
    store.close()
    got = store.take([1, 2, 3], ["age"])  # detected on open, deleted, fetched again
    assert_column_equal(got["age"], obs["age"].to_numpy()[[1, 2, 3]], "age")


def test_a_batch_count_mismatch_in_the_cache_is_detected(built, cache, tmp_path):
    root, _, obs, *_ = built
    store = open_dataset(root, cache_dir=cache).obs
    store.localize(["age"])
    # replace the cached column by a valid Arrow file with the wrong number of batches
    path = store._path("age")
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, pa.schema([("age", pa.float64())])) as w:
        w.write_batch(pa.record_batch([pa.array([1.0, 2.0])], names=["age"]))
    store.close()
    assert_column_equal(store.take([4], ["age"])["age"], obs["age"].to_numpy()[[4]], "age")


# ---------------------------------------------------------------------------------------------- backends


def test_read_range_on_every_backend(tmp_path):
    (tmp_path / "x").write_bytes(bytes(range(256)) * 4)
    local = LocalBackend(str(tmp_path))
    assert local.read_range("x", 10, 5) == bytes(range(10, 15)) and local.size("x") == 1024
    assert local.read_range("x", 1020, 100) == bytes(range(252, 256)) and local.read_range("x", 5000, 10) == b""
    counting = CountingBackend(local)
    counting.read_range("x", 0, 100)
    counting.read("x")
    assert (
        counting.bytes_read == 100 + 1024
        and counting.n_range_reads == 1
        and counting.n_reads == 1
        and counting.log[0] == ("x", 0, 100)
    )
    counting.reset()
    assert counting.bytes_read == 0 and counting.log == []

    class Plain(LocalBackend):  # a backend without a native ranged read falls back to reading the whole object
        read_range = LocalBackend.__mro__[1].read_range
        size = LocalBackend.__mro__[1].size

    assert Plain(str(tmp_path)).read_range("x", 10, 5) == bytes(range(10, 15))
    assert Plain(str(tmp_path)).size("x") == 1024


def test_gcs_backend_ranged_reads_and_size_use_the_client_api():
    from deltacells import GCSBackend

    calls = []

    class Blob:
        size = 1234

        def __init__(self, name):
            self.name = name

        def download_as_bytes(self, start=None, end=None):
            calls.append((self.name, start, end))
            return b"z" * (end - start + 1)

        def reload(self):
            calls.append((self.name, "reload"))

    class Client:
        def bucket(self, name):
            return self

        def blob(self, name):
            return Blob(name)

    b = GCSBackend("bkt", "p", client=Client())
    assert b.read_range("tiles/t", 100, 50) == b"z" * 50
    assert calls[-1] == ("p/tiles/t", 100, 149), "GCS ranges are inclusive at the end"
    assert b.read_range("tiles/t", 0, 0) == b""
    assert b.size("tiles/t") == 1234 and calls[-1] == ("p/tiles/t", "reload")


def test_throttled_range_reads_are_delayed(tmp_path):
    import time

    (tmp_path / "x").write_bytes(b"0" * 100_000)
    b = ThrottledBackend(LocalBackend(str(tmp_path)), latency_s=0.1)
    t0 = time.perf_counter()
    assert len(b.read_range("x", 0, 10)) == 10
    assert time.perf_counter() - t0 >= 0.09 and b.size("x") == 100_000


def test_range_file_semantics(tmp_path):
    (tmp_path / "x").write_bytes(bytes(range(200)))
    counting = CountingBackend(LocalBackend(str(tmp_path)))
    f = RangeFile(counting, "x")
    assert f.readable() and f.seekable() and f.tell() == 0
    assert f.read(5) == bytes(range(5)) and f.tell() == 5
    f.seek(-10, os.SEEK_END)
    assert f.tell() == 190 and f.read() == bytes(range(190, 200)) and f.read(5) == b""
    f.seek(20)
    buf = bytearray(4)
    assert f.readinto(buf) == 4 and bytes(buf) == bytes(range(20, 24))
    f.seek(3, os.SEEK_CUR)
    assert f.tell() == 27 and f.seek(-100) == 0
    assert counting.n_reads == 0 and counting.bytes_read == 5 + 10 + 4


def test_parquet_footer_and_one_column_can_be_read_through_a_range_file(built):
    root, *_ = built
    counting = CountingBackend(LocalBackend(root))
    name = "obs/obs_0000.parquet"
    f = RangeFile(counting, name)
    md = pq.read_metadata(f)
    assert md.num_row_groups == 2 and md.num_columns == 7
    got = pq.ParquetFile(RangeFile(counting, name), metadata=md, pre_buffer=True).read_row_group(
        1, columns=["n_counts"]
    )
    assert got.num_rows == 100 and got.column_names == ["n_counts"]


# ---------------------------------------------------------------------------------------------- converter and CLI


anndata = pytest.importorskip("anndata")


def write_obs_shards(directory, sizes, full, obs, var=None):
    os.makedirs(directory, exist_ok=True)
    lo = 0
    var = (
        var
        if var is not None
        else pd.DataFrame(
            {"gene_symbol": [f"s{j}" for j in range(full.shape[1])]}, index=[f"gene{j}" for j in range(full.shape[1])]
        )
    )
    for i, n in enumerate(sizes):
        part = obs.iloc[lo : lo + n].copy()
        part.index = [f"cell{j}" for j in range(lo, lo + n)]
        for c in obs.columns:  # like real shards: each file's categoricals only know the values that occur in that file
            if isinstance(obs[c].dtype, pd.CategoricalDtype):
                part[c] = part[c].cat.remove_unused_categories()
        anndata.AnnData(X=full[lo : lo + n], obs=part, var=var).write_h5ad(f"{directory}/s_{i:03d}.h5ad")
        lo += n


def test_converter_stores_obs_with_global_vocabularies_and_var(tmp_path, cache):
    from deltacells.convert import convert_h5ad

    n = 470
    full = make_counts(n, 25, density=0.2, seed=120, max_value=30)
    obs = make_obs(n, seed=9).drop(columns=["sex"])  # (the h5ad writer cannot round-trip nullable strings)
    obs["barcode"] = pd.Categorical(obs["barcode"])  # a categorical with huge cardinality: must become a string column
    write_obs_shards(tmp_path / "in", [137, 200, 5, 128], full, obs)
    out = str(tmp_path / "out")
    m = convert_h5ad(
        str(tmp_path / "in" / "*.h5ad"),
        out,
        tile_size=100,
        max_categories=50,
        level=3,
        n_chunks=2,
        log=None,
        obs_tiles_per_file=3,
    )
    assert m.has_obs and m.n_obs_shards == 2 and m.has_var_table
    ds = open_dataset(out, cache_dir=cache)
    schema = ds.obs.schema
    assert (
        schema["barcode"].kind == "string"
        and schema["cell_type"].kind == "category"
        and schema["obs_names"].kind == "string"
    )
    assert schema["cell_type"].categories == ["alpha", "beta", "gamma"] and schema["donor"].kind == "category"
    idx = np.random.default_rng(0).integers(0, n, 200)
    ds.obs.localize()
    got = ds.obs.take(idx, ["cell_type", "donor", "age", "barcode", "obs_names", "doublet", "n_counts"])
    assert np.array_equal(got["cell_type"], expected_column(obs, "cell_type", idx))
    assert np.array_equal(got["donor"], expected_column(obs, "donor", idx))
    assert list(got["barcode"]) == list(obs["barcode"].astype(str).iloc[idx])
    assert list(got["obs_names"]) == [f"cell{j}" for j in idx]
    assert_column_equal(got["age"], obs["age"].to_numpy()[idx], "age")
    assert np.array_equal(got["n_counts"], obs["n_counts"].to_numpy()[idx])
    var = ds.var
    assert list(var.index) == [f"gene{j}" for j in ds.gene_order] and list(var["gene_symbol"]) == [
        f"s{j}" for j in ds.gene_order
    ]
    ds.obs.verify()


def test_converter_obs_options(tmp_path):
    from deltacells.convert import convert_h5ad

    n = 120
    full = make_counts(n, 15, density=0.3, seed=121)
    obs = make_obs(n, seed=10).drop(columns=["sex"])
    write_obs_shards(tmp_path / "in", [60, 60], full, obs)
    base = dict(tile_size=60, level=3, n_chunks=1, log=None)
    m = convert_h5ad(
        str(tmp_path / "in" / "*.h5ad"),
        str(tmp_path / "a"),
        obs_exclude=["barcode", "age"],
        obs_names=False,
        var=False,
        **base,
    )
    store = open_dataset(str(tmp_path / "a")).obs
    assert (
        "barcode" not in store.columns
        and "age" not in store.columns
        and "obs_names" not in store.columns
        and not m.has_var_table
    )
    m = convert_h5ad(str(tmp_path / "in" / "*.h5ad"), str(tmp_path / "b"), obs=False, **base)
    assert not m.has_obs and open_dataset(str(tmp_path / "b")).obs is None
    assert not os.path.exists(tmp_path / "b" / "obs")
    obs2 = obs.assign(obs_names="clash")
    write_obs_shards(tmp_path / "in2", [60, 60], full, obs2)
    with pytest.raises(ValueError, match="obs_names"):
        convert_h5ad(str(tmp_path / "in2" / "*.h5ad"), str(tmp_path / "c"), **base)
    convert_h5ad(str(tmp_path / "in2" / "*.h5ad"), str(tmp_path / "d"), obs_names=False, **base)


def test_cli_obs_commands(wide, tmp_path, capsys):
    root, obs = wide
    cache = str(tmp_path / "cache")
    assert main(["obs", "info", root, "--cache-dir", cache]) == 0
    text = capsys.readouterr().out
    assert "cell_type" in text and "f00" in text and "(all columns)" in text and "per 100M cells" in text
    assert main(["obs", "localize", root, "--columns", "cell_type,f02", "--cache-dir", cache, "--threads", "2"]) == 0
    assert "fetched ['cell_type', 'f02']" in capsys.readouterr().out
    assert main(["obs", "localize", root, "--columns", "cell_type", "--cache-dir", cache]) == 0
    assert "already local" in capsys.readouterr().out
    assert main(["obs", "localize", root, "--cache-dir", cache]) == 2  # neither --columns nor --all
    assert main(["obs", "localize", root, "--all", "--cache-dir", cache]) == 0
    assert len(open_dataset(root, cache_dir=cache).obs.localized_columns()) == 32
    assert main(["info", root]) == 0
    assert "obs:" in capsys.readouterr().out


def test_cli_without_obs(tmp_path, capsys):
    with DatasetWriter(str(tmp_path / "p"), n_genes=10, tile_size=20, level=3) as w:
        w.add_tile(make_counts(20, 10, seed=7))
    assert main(["obs", "info", str(tmp_path / "p")]) == 1
    assert main(["obs", "localize", str(tmp_path / "p"), "--all"]) == 1
    assert main(["info", str(tmp_path / "p")]) == 0
    assert "obs:            none" in capsys.readouterr().out


def test_cli_verify_covers_obs(built, tmp_path, capsys):
    root, *_ = built
    assert main(["verify", root]) == 0
    assert "verified obs" in capsys.readouterr().out
    bad = str(tmp_path / "bad")
    shutil.copytree(root, bad)
    p = os.path.join(bad, "obs", "obs_0000.parquet")
    data = bytearray(Path(p).read_bytes())
    data[100] ^= 0xFF
    Path(p).write_bytes(data)
    assert main(["verify", bad]) == 1
    assert "checksum" in capsys.readouterr().err
