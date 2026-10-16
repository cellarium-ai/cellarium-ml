# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Where tiles live. A backend reads whole named objects (a tile is one object, so one GET per tile)."""

from __future__ import annotations

import io
import os
import time
from abc import ABC, abstractmethod

import numpy as np


class Backend(ABC):
    """Read-only access to the objects of a dataset (``manifest.json``, ``tiles/tile_000000.dct``, ...)."""

    @abstractmethod
    def read(self, name: str):
        """Return the object's bytes as a bytes-like object (``bytes`` or a uint8 ``ndarray``)."""

    @abstractmethod
    def exists(self, name: str) -> bool: ...

    def size(self, name: str) -> int:
        """Size of the object in bytes."""
        return len(self.read(name))

    def read_range(self, name: str, offset: int, length: int) -> bytes:
        """``length`` bytes starting at ``offset`` (fewer if the object ends first). Backends should override this with a real
        ranged read; the default reads the whole object."""
        return bytes(memoryview(self.read(name))[offset : offset + length])

    def describe(self) -> str:
        return type(self).__name__


class LocalBackend(Backend):
    """Tiles in a local directory."""

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(os.path.expanduser(root))

    def read(self, name: str) -> np.ndarray:
        return np.fromfile(os.path.join(self.root, name), dtype=np.uint8)

    def exists(self, name: str) -> bool:
        return os.path.exists(os.path.join(self.root, name))

    def size(self, name: str) -> int:
        return os.path.getsize(os.path.join(self.root, name))

    def read_range(self, name: str, offset: int, length: int) -> bytes:
        with open(os.path.join(self.root, name), "rb") as f:
            f.seek(offset)
            return f.read(length)

    def describe(self) -> str:
        return f"local:{self.root}"


class GCSBackend(Backend):
    """Tiles under ``gs://bucket/prefix`` (requires ``google-cloud-storage``; credentials as for that library).

    The client is created lazily and not pickled, so a dataset using this backend can be sent to DataLoader workers.
    Note: not exercised by this package's test-suite (no network access there); the throttled-backend tests and benchmarks
    emulate a remote store.
    """

    def __init__(self, bucket: str, prefix: str = "", client=None) -> None:
        self.bucket_name, self.prefix = bucket, prefix.strip("/")
        self._client = client
        self._bucket = None

    def _blob(self, name: str):
        if self._bucket is None:
            if self._client is None:
                try:
                    from google.cloud import storage
                except ImportError as e:  # pragma: no cover
                    raise ImportError("GCSBackend needs google-cloud-storage (pip install google-cloud-storage)") from e
                self._client = storage.Client()
            self._bucket = self._client.bucket(self.bucket_name)
        return self._bucket.blob(f"{self.prefix}/{name}" if self.prefix else name)

    def read(self, name: str) -> bytes:
        return self._blob(name).download_as_bytes()

    def exists(self, name: str) -> bool:
        return self._blob(name).exists()

    def size(self, name: str) -> int:
        blob = self._blob(name)
        blob.reload()
        return int(blob.size)

    def read_range(self, name: str, offset: int, length: int) -> bytes:
        if length <= 0:
            return b""
        return self._blob(name).download_as_bytes(start=offset, end=offset + length - 1)  # `end` is inclusive

    def describe(self) -> str:
        return f"gs://{self.bucket_name}/{self.prefix}"

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_client"] = None
        state["_bucket"] = None
        return state


class ThrottledBackend(Backend):
    """Wrap a backend and delay each read to emulate a remote store: ``latency_s`` plus ``nbytes / bandwidth_bytes_per_s``.

    The delay is per read (per thread), so concurrent reads overlap the way independent connections do; bandwidth is therefore
    *per stream*, not aggregate. Meant for tests and benchmarks.
    """

    def __init__(self, inner: Backend, *, latency_s: float = 0.0, bandwidth_bytes_per_s: float | None = None) -> None:
        self.inner, self.latency_s, self.bandwidth = inner, latency_s, bandwidth_bytes_per_s

    def read(self, name: str):
        t0 = time.perf_counter()
        data = self.inner.read(name)
        delay = self.latency_s + (len(data) / self.bandwidth if self.bandwidth else 0.0)
        remaining = delay - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)
        return data

    def read_range(self, name: str, offset: int, length: int) -> bytes:
        t0 = time.perf_counter()
        data = self.inner.read_range(name, offset, length)
        delay = self.latency_s + (len(data) / self.bandwidth if self.bandwidth else 0.0)
        remaining = delay - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)
        return data

    def size(self, name: str) -> int:
        return self.inner.size(name)

    def exists(self, name: str) -> bool:
        return self.inner.exists(name)

    def describe(self) -> str:
        bw = f"{self.bandwidth / 1e6:.0f} MB/s" if self.bandwidth else "unlimited"
        return f"throttled({self.inner.describe()}, latency={self.latency_s * 1e3:.0f} ms, {bw})"


class CountingBackend(Backend):
    """Wrap a backend and count what is read (for tests and benchmarks): ``bytes_read``, ``n_reads`` (whole-object reads),
    ``n_range_reads`` and the ``log`` of ``(name, offset, length)`` ranged reads (``offset`` is ``None`` for whole reads)."""

    def __init__(self, inner: Backend) -> None:
        self.inner = inner
        self.reset()

    def reset(self) -> None:
        self.bytes_read = 0
        self.n_reads = 0
        self.n_range_reads = 0
        self.log: list[tuple[str, int | None, int]] = []

    def read(self, name: str):
        data = self.inner.read(name)
        self.n_reads += 1
        self.bytes_read += len(data)
        self.log.append((name, None, len(data)))
        return data

    def read_range(self, name: str, offset: int, length: int) -> bytes:
        data = self.inner.read_range(name, offset, length)
        self.n_range_reads += 1
        self.bytes_read += len(data)
        self.log.append((name, offset, len(data)))
        return data

    def size(self, name: str) -> int:
        return self.inner.size(name)

    def exists(self, name: str) -> bool:
        return self.inner.exists(name)

    def describe(self) -> str:
        return f"counting({self.inner.describe()})"


class RangeFile(io.RawIOBase):
    """A read-only seekable file object over one backend object, fetching exactly the ranges that are read (no read-ahead).

    Hand it to ``pyarrow.parquet.ParquetFile`` to read only the footer and the requested column chunks of a remote file.
    """

    def __init__(self, backend: Backend, name: str, size: int | None = None) -> None:
        self.backend, self.name = backend, name
        self._size = backend.size(name) if size is None else size
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        base = {os.SEEK_SET: 0, os.SEEK_CUR: self._pos, os.SEEK_END: self._size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._size - self._pos
        size = max(0, min(size, self._size - self._pos))
        data = self.backend.read_range(self.name, self._pos, size) if size else b""
        self._pos += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_backend(uri: str | Backend) -> Backend:
    """``Backend`` from a local path, ``file://`` or ``gs://bucket/prefix`` URI (or pass a Backend through)."""
    if isinstance(uri, Backend):
        return uri
    uri = str(uri)
    if uri.startswith("gs://"):
        bucket, _, prefix = uri[len("gs://") :].partition("/")
        return GCSBackend(bucket, prefix)
    if uri.startswith("file://"):
        uri = uri[len("file://") :]
    if "://" in uri:
        raise ValueError(f"unsupported URI scheme in {uri!r} (supported: local paths, file://, gs://)")
    return LocalBackend(uri)
