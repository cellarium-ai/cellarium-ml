# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import pickle
import time

import pytest

from deltacells import GCSBackend, LocalBackend, ThrottledBackend, open_backend


def test_local_backend(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.bin").write_bytes(b"hello")
    b = LocalBackend(str(tmp_path))
    assert bytes(b.read("sub/x.bin")) == b"hello" and b.exists("sub/x.bin") and not b.exists("nope")
    with pytest.raises(FileNotFoundError):
        b.read("nope")
    assert pickle.loads(pickle.dumps(b)).root == b.root


def test_open_backend_dispatch(tmp_path):
    assert isinstance(open_backend(str(tmp_path)), LocalBackend)
    assert isinstance(open_backend(f"file://{tmp_path}"), LocalBackend)
    g = open_backend("gs://my-bucket/some/prefix/")
    assert isinstance(g, GCSBackend) and g.describe() == "gs://my-bucket/some/prefix"
    assert open_backend(g) is g
    with pytest.raises(ValueError, match="scheme"):
        open_backend("s3://bucket/x")


class _FakeBlob:
    def __init__(self, store, name):
        self.store, self.name = store, name

    def download_as_bytes(self):
        return self.store[self.name]

    def exists(self):
        return self.name in self.store


class _FakeClient:
    def __init__(self, store):
        self.store = store

    def bucket(self, name):
        return self

    def blob(self, name):
        return _FakeBlob(self.store, name)


def test_gcs_backend_with_fake_client_and_pickling():
    store = {"pre/fix/manifest.json": b"{}", "pre/fix/tiles/t": b"abc"}
    b = GCSBackend("bkt", "pre/fix", client=_FakeClient(store))
    assert b.read("tiles/t") == b"abc" and b.exists("manifest.json") and not b.exists("zzz")
    state = b.__getstate__()
    assert state["_client"] is None and state["_bucket"] is None  # not pickled: workers create their own client


def test_throttled_backend(tmp_path):
    (tmp_path / "x").write_bytes(b"0" * 1_000_000)
    b = ThrottledBackend(LocalBackend(str(tmp_path)), latency_s=0.1, bandwidth_bytes_per_s=5_000_000)  # 0.1 + 0.2 s
    t0 = time.perf_counter()
    assert len(b.read("x")) == 1_000_000
    assert 0.28 <= time.perf_counter() - t0 < 0.6
    assert b.exists("x") and "throttled" in b.describe()
