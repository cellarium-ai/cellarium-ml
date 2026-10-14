# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Remote (``gs://``) inputs and outputs, tested with an in-memory fsspec filesystem standing in for the bucket."""

import os

import numpy as np
import pandas as pd
import pytest
from conftest import make_counts, same

fsspec = pytest.importorskip("fsspec")
anndata = pytest.importorskip("anndata")

from deltacells import DatasetWriter, open_dataset  # noqa: E402
from deltacells.convert import convert_h5ad, resolve_files  # noqa: E402
from deltacells.sink import FsspecSink, LocalSink, open_sink  # noqa: E402
from deltacells.staging import Stager, read_h5ad_meta  # noqa: E402


class FakeBucket:
    """An in-memory filesystem that records the order of the uploads and deletions."""

    def __init__(self, fail_puts: int = 0):
        self.fs = fsspec.filesystem("memory")
        self.events: list[tuple[str, str]] = []
        self.fail_puts = fail_puts  # the next uploads that fail, as a flaky network would

    def put_file(self, local, remote):
        if self.fail_puts > 0:
            self.fail_puts -= 1
            raise ConnectionError("flaky")
        self.events.append(("put", remote))
        self.fs.put_file(local, remote)

    def rm(self, path, *args, **kwargs):
        for p in [path] if isinstance(path, str) else path:
            self.events.append(("rm", p))
        return self.fs.rm(path, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.fs, name)


@pytest.fixture
def bucket(monkeypatch):
    monkeypatch.setattr("deltacells.sink.time.sleep", lambda s: None)  # do not wait between the retries of a test
    b = FakeBucket()
    yield b
    if b.fs.exists("bucket"):
        b.fs.rm("bucket", recursive=True)


def put_h5ad(bucket, tmp_path, n_obs=30, n_genes=12, name="in/a.h5ad", seed=0, with_big_obs=True):
    x = make_counts(n_obs, n_genes, density=0.3, seed=seed, max_value=20)
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(
                np.random.default_rng(seed).choice(["b", "a", "c"], n_obs), categories=["b", "a", "c"]
            ),
            "score": np.arange(n_obs, dtype=np.float64),
            "flag": np.arange(n_obs) % 2 == 0,
        },
        index=[f"c{seed}_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame({"symbol": [f"s{j}" for j in range(n_genes)]}, index=[f"g{j}" for j in range(n_genes)])
    local = str(tmp_path / f"{name.replace('/', '_')}")
    anndata.AnnData(X=x, obs=obs, var=var).write_h5ad(local)
    bucket.fs.put_file(local, f"bucket/{name}")
    return x, local, f"gs://bucket/{name}"


# ---------------------------------------------------------------------------------------------- the sink


def test_sink_recognises_the_files_of_a_dataset(bucket):
    for name in ("manifest.json", "tiles/tile_000012.dct", "obs/obs_0003.parquet", "obs/schema.json", "var.parquet"):
        bucket.fs.pipe(f"bucket/p/{name}", b"x")
    bucket.fs.pipe("bucket/p/tiles/tile_000013.dct.tmp123", b"x")  # a leftover of an interrupted write
    sink = FsspecSink("gs://bucket/p", bucket)
    assert len(sink.check(overwrite=True)) == 6
    bucket.fs.pipe("bucket/p/tiles/other.txt", b"x")
    with pytest.raises(FileExistsError, match="other.txt"):
        sink.check(overwrite=True)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        sink.check(overwrite=False)


def test_sink_deletes_the_manifest_first(bucket):
    for name in ("tiles/tile_000000.dct", "manifest.json", "var.parquet"):
        bucket.fs.pipe(f"bucket/p/{name}", b"x")
    FsspecSink("gs://bucket/p", bucket).prepare(overwrite=True)
    assert bucket.events[0] == ("rm", "bucket/p/manifest.json")
    assert not bucket.fs.exists("bucket/p/tiles")


def test_sink_refuses_an_output_that_contains_an_input(bucket, tmp_path):
    sink = FsspecSink("gs://bucket/data", bucket)
    with pytest.raises(ValueError, match="inside the output"):
        sink.check(True, ["gs://bucket/data/sub/a.h5ad"])
    sink.check(True, ["gs://bucket/database/a.h5ad", "gs://other/data/a.h5ad", str(tmp_path / "a.h5ad")])  # not inside
    local = LocalSink(str(tmp_path / "out"))
    with pytest.raises(ValueError, match="inside the output"):
        local.check(True, [str(tmp_path / "out" / ".." / "out" / "a.h5ad")])
    local.check(True, [str(tmp_path / "outer" / "a.h5ad"), "gs://bucket/out/a.h5ad"])  # not inside


def test_sink_retries_an_upload(bucket, tmp_path):
    (tmp_path / "f").write_text("data")
    sink = open_sink("gs://bucket/p", bucket)
    bucket.fail_puts = 2
    sink.put(str(tmp_path / "f"), "f")
    assert bucket.fs.cat("bucket/p/f") == b"data"
    bucket.fail_puts = 10
    with pytest.raises(OSError, match="failed after"):
        sink.put(str(tmp_path / "f"), "g")


# ---------------------------------------------------------------------------------------------- the writer


def write_tiles(writer, n_tiles=3, tile_size=10, n_genes=5):
    for i in range(n_tiles):
        writer.add_tile(make_counts(tile_size, n_genes, seed=i))


def test_writer_uploads_each_tile_and_the_manifest_last(bucket, tmp_path):
    with DatasetWriter(
        "gs://bucket/p",
        n_genes=5,
        tile_size=10,
        level=3,
        filesystem=bucket,
        scratch_dir=str(tmp_path),
        upload_threads=2,
    ) as w:
        scratch = w.root
        assert os.path.dirname(scratch) == str(tmp_path) and w.is_remote
        write_tiles(w)
    puts = [p for kind, p in bucket.events if kind == "put"]
    assert puts[-1] == "bucket/p/manifest.json" and len(puts) == 4
    assert os.listdir(tmp_path) == []  # the scratch directory is gone
    assert bucket.fs.exists("bucket/p/tiles/tile_000002.dct")


def test_writer_keeps_only_a_few_tiles_in_scratch(bucket, tmp_path):
    peak = 0
    with DatasetWriter(
        "gs://bucket/p",
        n_genes=5,
        tile_size=10,
        level=3,
        filesystem=bucket,
        scratch_dir=str(tmp_path),
        upload_threads=1,
    ) as w:
        for i in range(40):
            w.add_tile(make_counts(10, 5, seed=i))
            peak = max(peak, len(os.listdir(os.path.join(w.root, "tiles"))))
    assert peak <= 4 * 1 + 1  # the backlog is bounded, whatever the number of tiles


def test_an_aborted_writer_leaves_uploaded_files_but_no_manifest(bucket, tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        with DatasetWriter(
            "gs://bucket/p", n_genes=5, tile_size=10, level=3, filesystem=bucket, scratch_dir=str(tmp_path)
        ) as w:
            write_tiles(w, 2)
            raise RuntimeError("boom")
    assert os.listdir(tmp_path) == []
    assert not bucket.fs.exists("bucket/p/manifest.json")
    assert bucket.fs.exists("bucket/p/tiles/tile_000000.dct")  # nothing is deleted from a bucket
    DatasetWriter("gs://bucket/p", n_genes=5, tile_size=10, filesystem=bucket, overwrite=True).abort()  # replaces it
    assert not bucket.fs.exists("bucket/p/tiles/tile_000000.dct")


def test_a_failed_upload_is_an_error_and_cleans_the_scratch(bucket, tmp_path):
    bucket.fail_puts = 1000
    with pytest.raises(OSError, match="failed after"):
        with DatasetWriter(
            "gs://bucket/p", n_genes=5, tile_size=10, level=3, filesystem=bucket, scratch_dir=str(tmp_path)
        ) as w:
            write_tiles(w, 6)
    assert os.listdir(tmp_path) == []
    assert not bucket.fs.exists("bucket/p/manifest.json")


# ---------------------------------------------------------------------------------------------- reading remote h5ad files


def test_remote_metadata_equals_local_metadata(bucket, tmp_path):
    x, local, url = put_h5ad(bucket, tmp_path)
    remote, from_disk = read_h5ad_meta(url, bucket, var=True), read_h5ad_meta(local, var=True)
    backed = anndata.read_h5ad(local, backed="r")
    for meta in (remote, from_disk):
        assert meta.n_obs == 30 and list(meta.var_names) == [f"g{j}" for j in range(12)]
        pd.testing.assert_frame_equal(meta.obs, backed.obs)  # the same dtypes, categories and their order
        pd.testing.assert_frame_equal(meta.var, backed.var)
    assert remote.size == os.path.getsize(local)
    backed.file.close()


def test_a_missing_remote_file_is_not_found(bucket):
    with pytest.raises(FileNotFoundError):
        read_h5ad_meta("gs://bucket/in/nothing.h5ad", bucket)


def test_remote_paths_are_not_globbed():
    assert resolve_files(["gs://b/z_10.h5ad", "gs://b/z_2.h5ad*"], sort=False) == [
        "gs://b/z_10.h5ad",
        "gs://b/z_2.h5ad*",
    ]


def test_the_stager_deletes_a_file_after_its_last_use(bucket, tmp_path):
    urls = [put_h5ad(bucket, tmp_path, name=f"in/f{i}.h5ad", seed=i)[2] for i in range(3)]
    local = tmp_path / "plain.h5ad"
    local.write_text("x")
    files = [urls[0], str(local), urls[1], urls[2]]
    with Stager(files, [2, 1, 1, 0], scratch_dir=str(tmp_path), filesystem=bucket) as st:
        scratch = st.scratch
        assert scratch is not None and os.path.dirname(scratch) == str(tmp_path)
        st.prefetch_through(3)
        a = st.local(0)
        assert os.path.getsize(a) == bucket.fs.size("bucket/in/f0.h5ad") and st.local(1) == str(local)
        st.release(0)
        assert os.path.exists(a)  # one use is left
        st.release(0)
        assert not os.path.exists(a)
        with pytest.raises(RuntimeError, match="released"):
            st.local(0)
        st.release(1)
        assert local.exists()  # a local file is never touched
        b = st.local(2)
        st.release(2)
        assert not os.path.exists(b)
        assert sorted(os.listdir(scratch)) == []  # f3 has no uses: never downloaded
    assert not os.path.exists(scratch)


def test_the_stager_retries_a_download_and_checks_its_size(bucket, tmp_path, monkeypatch):
    monkeypatch.setattr("deltacells.sink.time.sleep", lambda s: None)
    url = put_h5ad(bucket, tmp_path)[2]
    calls = {"n": 0}

    class Flaky(FakeBucket):
        def get_file(self, remote, local, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("flaky")
            if calls["n"] == 2:
                with open(local, "wb") as f:
                    f.write(b"short")  # a truncated download
                return
            self.fs.get_file(remote, local, **kwargs)

    flaky = Flaky()
    with Stager(
        [url], [1], scratch_dir=str(tmp_path), sizes=[bucket.fs.size("bucket/in/a.h5ad")], filesystem=flaky
    ) as st:
        path = st.local(0)
        assert calls["n"] == 3 and os.path.getsize(path) == flaky.fs.size("bucket/in/a.h5ad")
        assert [f for f in os.listdir(os.path.dirname(path)) if f.endswith(".part")] == []


# ---------------------------------------------------------------------------------------------- the conversion


def test_conversion_of_remote_files_to_a_bucket_equals_the_local_conversion(bucket, tmp_path):
    inputs = [put_h5ad(bucket, tmp_path, n_obs=n, seed=i, name=f"in/f{i}.h5ad") for i, n in enumerate([13, 30, 7])]
    kw = dict(tile_size=8, level=3, threads=1, log=None, sort_files=False)
    local = convert_h5ad([p for _, p, _ in inputs], str(tmp_path / "local"), workers=1, **kw)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    for workers in (1, 3):
        out = f"gs://bucket/out{workers}"
        remote = convert_h5ad(
            [u for _, _, u in inputs], out, workers=workers, filesystem=bucket, staging_dir=str(scratch), **kw
        )
        assert remote.tile_bytes == local.tile_bytes and remote.tile_nnz == local.tile_nnz
        for i in range(local.n_tiles):
            name = f"tiles/tile_{i:06d}.dct"
            assert bucket.fs.cat(f"bucket/out{workers}/{name}") == (tmp_path / "local" / name).read_bytes()
        for name in ("obs/obs_0000.parquet", "var.parquet", "gene_order.npy", "var_names.txt"):
            assert bucket.fs.cat(f"bucket/out{workers}/{name}") == (tmp_path / "local" / name).read_bytes()
        puts = [p for kind, p in bucket.events if kind == "put" and p.startswith(f"bucket/out{workers}/")]
        assert puts[-1] == f"bucket/out{workers}/manifest.json"
        assert os.listdir(scratch) == []
    ds = open_dataset(str(tmp_path / "local"))
    assert same(ds[:].to_scipy(), np.vstack([x.toarray() for x, _, _ in inputs])[:, ds.gene_order])


def test_mismatched_var_names_in_remote_files_are_found_by_the_scan(bucket, tmp_path):
    put_h5ad(bucket, tmp_path, name="in/a.h5ad")
    put_h5ad(bucket, tmp_path, n_genes=11, name="in/b.h5ad")
    with pytest.raises(ValueError, match="differ|columns"):
        convert_h5ad(
            ["gs://bucket/in/a.h5ad", "gs://bucket/in/b.h5ad"], str(tmp_path / "o"), filesystem=bucket, log=None
        )
    assert not os.path.exists(tmp_path / "o")


def test_genes_are_named_by_their_symbol_when_the_var_names_are_ids(tmp_path):
    import scipy.sparse as sp

    x = sp.csr_matrix(np.array([[1.0, 70000.0], [2.0, 3.0]], dtype=np.float32))
    var = pd.DataFrame({"feature_name": ["ACTB", "MALAT1"]}, index=["101", "102"])
    path = str(tmp_path / "ids.h5ad")
    anndata.AnnData(x, obs=pd.DataFrame(index=["a", "b"]), var=var).write_h5ad(path)
    with pytest.raises(ValueError, match=r"102 \(MALAT1\)"):
        convert_h5ad([path], str(tmp_path / "o"), tile_size=2, level=3, log=None, sort_genes=False)
    messages: list[str] = []
    manifest = convert_h5ad([path], str(tmp_path / "o"), tile_size=2, level=3, log=messages.append, clip=True)
    assert list(manifest.metadata["clipped_counts"]["genes"]) == ["102 (MALAT1)"]
    assert any("102 (MALAT1)" in m for m in messages)


def test_a_failure_is_explained_and_leaves_the_inputs_alone(bucket, tmp_path):
    import scipy.sparse as sp

    x = sp.csr_matrix(np.array([[1.0, 2.0], [2.0, 70000.0]], dtype=np.float32))  # the second tile fails
    local = str(tmp_path / "big.h5ad")
    anndata.AnnData(x, obs=pd.DataFrame(index=["a", "b"]), var=pd.DataFrame(index=["g0", "g1"])).write_h5ad(local)
    bucket.fs.put_file(local, "bucket/in/big.h5ad")
    messages: list[str] = []
    with pytest.raises(ValueError, match="g1") as info:
        convert_h5ad(
            ["gs://bucket/in/big.h5ad"], "gs://bucket/out", tile_size=1, level=3, sort_genes=False, workers=1,
            filesystem=bucket, staging_dir=str(tmp_path), log=messages.append,
        )  # fmt: skip
    # the way to the incomplete dataset that was left is in the error (Python 3.11+ notes) or in the log (before that)
    hint = "".join(getattr(info.value, "__notes__", [])) + " ".join(messages)
    assert "overwrite=True" in hint and "gs://bucket/out" in hint
    assert bucket.fs.exists("bucket/out/tiles/tile_000000.dct") and not bucket.fs.exists("bucket/out/manifest.json")
    assert bucket.fs.exists("bucket/in/big.h5ad")
