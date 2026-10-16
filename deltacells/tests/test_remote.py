# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Remote (``gs://``) inputs and outputs, tested with an in-memory fsspec filesystem standing in for the bucket."""

import glob
import json
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


class FolderMarkers(FakeBucket):
    """Like gcsfs on a real bucket: ``find`` also returns the empty "folder" placeholder objects (names ending in ``/``) that the
    cloud console makes with "create folder", here for the output prefix itself and for a ``tiles/`` folder in it."""

    def __init__(self, root: str):
        super().__init__()
        self.root = root

    def exists(self, path, **kwargs):
        return path.strip("/") == self.root or self.fs.exists(path, **kwargs)

    def find(self, path, **kwargs):
        return [f"{self.root}/", f"{self.root}/tiles/", *self.fs.find(path, **kwargs)]


def test_folder_placeholders_are_not_files_that_block_an_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr("deltacells.sink.time.sleep", lambda s: None)
    b = FolderMarkers("bucket/p")
    try:
        sink = FsspecSink("gs://bucket/p", b)
        assert sink.existing() == [] and sink.check(overwrite=False) == [] and sink.check(overwrite=True) == []

        for name in ("manifest.json", "tiles/tile_000000.dct"):  # a dataset in it: only its files are listed
            b.fs.pipe(f"bucket/p/{name}", b"x")
        assert sink.existing() == ["manifest.json", "tiles/tile_000000.dct"]
        assert sink.check(overwrite=True) == ["manifest.json", "tiles/tile_000000.dct"]

        b.fs.pipe("bucket/p/notes.txt", b"precious")  # something else is still reported, and only that
        with pytest.raises(FileExistsError, match=r"\['notes.txt'\]"):
            sink.check(overwrite=True)
        b.fs.rm("bucket/p/notes.txt")

        # end to end: a whole conversion into the prefix that the console made, with --overwrite
        url = put_h5ad(b, tmp_path, n_obs=10, name="in/a.h5ad")[2]
        manifest = convert_h5ad(
            [url],
            "gs://bucket/p",
            tile_size=4,
            level=3,
            log=None,
            filesystem=b,
            staging_dir=str(tmp_path),
            overwrite=True,
        )
        assert manifest.n_cells == 10 and b.fs.exists("bucket/p/manifest.json")
        assert not b.fs.exists("bucket/p/tiles/tile_000099.dct")
    finally:
        if b.fs.exists("bucket"):
            b.fs.rm("bucket", recursive=True)


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


class Forbidden(FakeBucket):
    """A bucket that refuses uploads the way gcsfs reports a 403 (an ``OSError``, not a ``PermissionError``)."""

    def __init__(self):
        super().__init__()
        self.attempts = 0

    def put_file(self, local, remote):
        self.attempts += 1
        raise OSError("Forbidden: https://storage.googleapis.com/upload/b/x\nProvided scope(s) are not authorized")


def test_a_refused_upload_is_not_retried_and_says_what_to_check(bucket):
    forbidden = Forbidden()
    with pytest.raises(PermissionError, match=r"(?s)was refused.*gcloud auth application-default login"):
        FsspecSink("gs://bucket/p", forbidden).probe()
    assert forbidden.attempts == 1  # a refusal does not change by trying again


def test_the_probe_leaves_nothing_and_a_leftover_one_does_not_block_an_overwrite(bucket):
    sink = FsspecSink("gs://bucket/p", bucket)
    sink.probe()
    assert [kind for kind, _ in bucket.events] == ["put", "rm"] and sink.existing() == []

    class NoDelete(FakeBucket):
        def rm(self, path, *args, **kwargs):
            raise PermissionError("no delete")

    stuck = NoDelete()
    with pytest.raises(OSError, match=r"could not be deleted.*\.deltacells_write_test_"):
        FsspecSink("gs://bucket/q", stuck).probe()
    assert (
        len(sink.check(True)) == 0 and len(FsspecSink("gs://bucket/q", stuck).check(True)) == 1
    )  # it is a dataset file


def test_a_conversion_finds_out_that_it_cannot_write_before_it_scans(bucket, tmp_path):
    url = put_h5ad(bucket, tmp_path)[2]
    messages: list[str] = []
    with pytest.raises(PermissionError, match="application-default login"):
        convert_h5ad([url], "gs://bucket/out", filesystem=Forbidden(), log=messages.append)
    assert not [m for m in messages if "metadata" in m]  # it did not start reading the files
    assert not os.path.exists("bucket/out")


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
    puts = [p for kind, p in bucket.events if kind == "put" and ".deltacells_write_test" not in p]  # (not the probe's)
    assert puts[-1] == "bucket/p/manifest.json" and len(puts) == 4
    assert not [p for p in bucket.fs.find("bucket/p") if "write_test" in p]  # the probe deleted its file
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
        # exactly the files of the local dataset, and the same manifest (but for where the files came from)
        uploaded = {p.removeprefix(f"/bucket/out{workers}/") for p in bucket.fs.find(f"bucket/out{workers}")}
        assert uploaded == {
            os.path.relpath(p, tmp_path / "local")
            for p in glob.glob(f"{tmp_path}/local/**", recursive=True)
            if os.path.isfile(p)
        }
        remote_manifest = json.loads(bucket.fs.cat(f"bucket/out{workers}/manifest.json"))
        local_manifest = json.loads((tmp_path / "local" / "manifest.json").read_text())
        assert {k: v for k, v in remote_manifest.items() if k != "metadata"} == {
            k: v for k, v in local_manifest.items() if k != "metadata"
        }
    ds = open_dataset(str(tmp_path / "local"))
    assert same(ds[:].to_scipy(), np.vstack([x.toarray() for x, _, _ in inputs])[:, ds.gene_order])


def test_remote_inputs_are_staged_a_few_at_a_time(tmp_path):
    class Recording(FakeBucket):
        def __init__(self):
            super().__init__()
            self.staged: list[int] = []  # the files in the scratch directory at each download, that one included

        def get_file(self, remote, local, **kwargs):
            self.fs.get_file(remote, local, **kwargs)
            self.staged.append(
                len(os.listdir(os.path.dirname(local)))
            )  # the finished files and this one (a .part file)

    rec = Recording()
    try:
        urls = [put_h5ad(rec, tmp_path, n_obs=5, seed=i, name=f"in/f{i}.h5ad")[2] for i in range(12)]
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        convert_h5ad(
            urls, str(tmp_path / "out"), tile_size=4, level=3, log=None, filesystem=rec, staging_dir=str(scratch),
            workers=1, sort_genes_max_files=1, sort_files=False,
        )  # fmt: skip
        assert len(rec.staged) == 12 + 1  # each file once for its tiles, and one more for the gene order
        # the window is 4 * workers + max(2, workers) + 2 = 8 files (see convert_h5ad), never all 12 at once
        assert max(rec.staged) <= 8
        assert os.listdir(scratch) == []
    finally:
        rec.fs.rm("bucket", recursive=True)


def test_a_remote_output_is_replaced_only_if_it_is_a_dataset(bucket, tmp_path):
    urls = [put_h5ad(bucket, tmp_path, n_obs=10, seed=i, name=f"in/f{i}.h5ad")[2] for i in range(2)]
    kw = dict(tile_size=4, level=3, log=None, filesystem=bucket, staging_dir=str(tmp_path), workers=1, sort_files=False)
    convert_h5ad(urls, "gs://bucket/prefix", **kw)
    with pytest.raises(FileExistsError, match="overwrite"):  # a dataset is there
        convert_h5ad(urls[:1], "gs://bucket/prefix", **kw)

    bucket.fs.pipe("bucket/prefix/tiles/tile_000099.dct", b"stale")  # a leftover of a bigger dataset is replaced
    manifest = convert_h5ad(urls[:1], "gs://bucket/prefix", overwrite=True, **kw)
    assert manifest.n_cells == 10 and not bucket.fs.exists("bucket/prefix/tiles/tile_000099.dct")

    bucket.fs.pipe("bucket/prefix/notes.txt", b"precious")  # anything else is never deleted: nothing happens at all
    before = len(bucket.events)
    with pytest.raises(FileExistsError, match="notes.txt"):
        convert_h5ad(urls, "gs://bucket/prefix", overwrite=True, **kw)
    assert bucket.fs.cat("bucket/prefix/notes.txt") == b"precious" and len(bucket.events) == before
    assert json.loads(bucket.fs.cat("bucket/prefix/manifest.json"))["n_cells"] == 10


def test_a_conversion_never_deletes_its_inputs(bucket, tmp_path):
    x, local, url = put_h5ad(bucket, tmp_path, name="in/a.h5ad")
    with pytest.raises(ValueError, match="inside the output"):  # a local input in the output directory
        convert_h5ad([local], os.path.dirname(local), overwrite=True, log=None)
    with pytest.raises(ValueError, match="inside the output"):  # a remote input under the output prefix
        convert_h5ad([url], "gs://bucket/in", overwrite=True, log=None, filesystem=bucket)
    assert os.path.exists(local) and bucket.fs.exists("bucket/in/a.h5ad")
    assert not [e for e in bucket.events if e[0] == "rm"]


def test_mismatched_var_names_in_remote_files_are_found_by_the_scan(bucket, tmp_path):
    put_h5ad(bucket, tmp_path, name="in/a.h5ad")
    put_h5ad(bucket, tmp_path, n_genes=11, name="in/b.h5ad")
    with pytest.raises(ValueError, match="differ|columns"):
        convert_h5ad(
            ["gs://bucket/in/a.h5ad", "gs://bucket/in/b.h5ad"], str(tmp_path / "o"), filesystem=bucket, log=None
        )
    assert not os.path.exists(tmp_path / "o")


def test_a_failure_is_explained_and_leaves_the_inputs_alone(bucket, tmp_path):
    import scipy.sparse as sp

    n = 2000  # the second tile has more counts above 65535 than a tile accepts (1024), so it fails
    data = np.full((2 * n, 2), 2.0, dtype=np.float32)
    data[n:, 1] = 70000
    local = str(tmp_path / "big.h5ad")
    obs = pd.DataFrame(index=[f"c{i}" for i in range(2 * n)])
    anndata.AnnData(sp.csr_matrix(data), obs=obs, var=pd.DataFrame(index=["g0", "g1"])).write_h5ad(local)
    bucket.fs.put_file(local, "bucket/in/big.h5ad")
    messages: list[str] = []
    with pytest.raises(ValueError, match="tile 1 .*does not look like raw counts") as info:
        convert_h5ad(
            ["gs://bucket/in/big.h5ad"], "gs://bucket/out", tile_size=n, level=3, sort_genes=False, workers=1,
            filesystem=bucket, staging_dir=str(tmp_path), log=messages.append,
        )  # fmt: skip
    # the way to the incomplete dataset that was left is in the error (Python 3.11+ notes) or in the log (before that)
    hint = "".join(getattr(info.value, "__notes__", [])) + " ".join(messages)
    assert "overwrite=True" in hint and "gs://bucket/out" in hint
    assert bucket.fs.exists("bucket/out/tiles/tile_000000.dct") and not bucket.fs.exists("bucket/out/manifest.json")
    assert bucket.fs.exists("bucket/in/big.h5ad")
    assert os.listdir(tmp_path) == [os.path.basename(local)]  # nothing is left in the scratch directory

    # as the hint says: with the input fixed, running it again with overwrite=True replaces what was left
    data[n:, 1] = 2
    anndata.AnnData(sp.csr_matrix(data), obs=obs, var=pd.DataFrame(index=["g0", "g1"])).write_h5ad(local)
    bucket.fs.put_file(local, "bucket/in/big.h5ad")
    kw = dict(tile_size=n, level=3, sort_genes=False, workers=1, filesystem=bucket, staging_dir=str(tmp_path), log=None)
    with pytest.raises(FileExistsError, match="overwrite"):
        convert_h5ad(["gs://bucket/in/big.h5ad"], "gs://bucket/out", **kw)
    manifest = convert_h5ad(["gs://bucket/in/big.h5ad"], "gs://bucket/out", overwrite=True, **kw)
    assert manifest.n_cells == 2 * n and bucket.fs.exists("bucket/out/manifest.json")
