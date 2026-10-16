# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Where a dataset is written: a local directory, or a remote prefix such as ``gs://bucket/prefix``.

A :class:`Sink` hides the difference, so that the writer has one code path: it writes every file to a local working
directory and hands each finished file to the sink (a no-op for a local output, an upload for a remote one).
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from typing import Any

from deltacells.manifest import GENE_ORDER_NAME, MANIFEST_NAME, OBS_SCHEMA_NAME, VAR_NAMES_NAME, VAR_TABLE_NAME

# The files a dataset is made of (and the temporary names they are written under). Only these are ever deleted from an
# existing output, so that ``overwrite`` cannot destroy anything that is not a deltacells dataset.
_DATASET_FILE = re.compile(
    "^("
    + "|".join(re.escape(n) for n in (MANIFEST_NAME, GENE_ORDER_NAME, VAR_NAMES_NAME, VAR_TABLE_NAME, OBS_SCHEMA_NAME))
    + r"|tiles/tile_\d+\.dct|obs/obs_\d+\.parquet|\.deltacells_write_test_\d+)(\.tmp\d*)?$"
)

_UPLOAD_ATTEMPTS = 4
_PROBE_NAME = (
    ".deltacells_write_test_{}"  # the file that :meth:`FsspecSink.probe` makes and deletes (also a "dataset file")
)

CREDENTIALS_HINT = (
    "Check the credentials: for gs:// paths deltacells uses the Google Application Default Credentials, not the login of "
    "gcloud or gsutil. On a Compute Engine VM these are the VM's service account by default, and its access scopes may allow "
    "reading but not writing. Run `gcloud auth application-default login` to use your own account instead."
)


def is_remote(path: str) -> bool:
    """Whether ``path`` is a URL (``gs://...``) rather than a local path."""
    return "://" in path


def split_url(url: str) -> tuple[str, str]:
    """``("gs", "bucket/prefix")`` for ``gs://bucket/prefix``."""
    protocol, _, path = url.partition("://")
    return protocol, path


def get_filesystem(url: str, filesystem: Any = None) -> Any:
    """The fsspec filesystem for ``url`` (``filesystem`` if given, for testing). ``gs://`` needs ``gcsfs`` and uses the default credentials."""
    if filesystem is not None:
        return filesystem
    try:
        import fsspec
    except ImportError as e:  # pragma: no cover
        raise ImportError("remote paths need fsspec and gcsfs (pip install deltacells[gcs])") from e
    return fsspec.filesystem(split_url(url)[0])


def _refused(e: Exception) -> bool:
    """Whether ``e`` is the storage refusing the credentials (not authenticated or not authorized), which trying again does not
    change. gcsfs raises an ``OSError`` whose message starts with ``Forbidden`` for a 403, and an ``HttpError`` with a ``code``
    for the other statuses."""
    if isinstance(e, PermissionError):
        return True
    return getattr(e, "code", None) in (401, 403) or (
        isinstance(e, OSError) and str(e).startswith(("Forbidden", "Unauthorized"))
    )


def retry(fn, what: str, attempts: int = _UPLOAD_ATTEMPTS):
    """Call ``fn()``, trying again with a growing pause if it fails for a reason that may be temporary. A missing file and a
    refusal of the credentials are not temporary: they are raised at once (the refusal as a ``PermissionError``)."""
    for attempt in range(attempts):
        try:
            return fn()
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
            raise  # not temporary
        except Exception as e:
            if _refused(e):
                raise PermissionError(f"{what} was refused: {e}\n{CREDENTIALS_HINT}") from e
            if attempt == attempts - 1:
                raise OSError(f"{what} failed after {attempts} attempts: {e}") from e
            time.sleep(2**attempt)


class Sink:
    """The destination of a dataset. Names are relative to it, with ``/`` as the separator (``tiles/tile_000000.dct``)."""

    local: bool
    url: str

    def existing(self) -> list[str]:
        """The names of all the files that are already there."""
        raise NotImplementedError

    def delete(self, names: list[str]) -> None:
        """Delete these files (the manifest first, so that a half-deleted dataset is not taken for a complete one)."""
        raise NotImplementedError

    def put(self, local_path: str, name: str) -> None:
        """Store the local file as ``name`` (not used by a local sink, whose files are written in place)."""
        raise NotImplementedError

    def check(self, overwrite: bool, inputs: tuple[str, ...] | list[str] = ()) -> list[str]:
        """Check that a dataset can be written here and return the names of the files that ``overwrite`` would delete.

        Raises if an input lies inside the output, if the output has anything in it and ``overwrite`` is not given, or if
        it has anything in it that is not a deltacells file. Nothing is changed.
        """
        self._check_inputs(inputs)
        names = self.existing()
        if not names:
            return []
        if not overwrite:
            raise FileExistsError(
                f"{self.url} is not empty; pass overwrite=True to replace it (the dataset that is there, or the "
                "incomplete one left by a failed run, is deleted first)"
            )
        unknown = [n for n in names if not _DATASET_FILE.match(n)]
        if unknown:
            raise FileExistsError(
                f"{self.url} has files that are not part of a deltacells dataset (for example {unknown[:3]}), so it is "
                "not overwritten; delete them yourself or write to another place"
            )
        return names

    def probe(self) -> None:
        """Check that files can be written here, by writing (and deleting) a tiny one: raises if not. Nothing for a local
        output, whose files are written where they are by the writer."""

    def prepare(self, overwrite: bool, inputs: tuple[str, ...] | list[str] = ()) -> None:
        """:meth:`check`, then delete what is there."""
        names = self.check(overwrite, inputs)
        if names:
            self.delete(names)

    def _check_inputs(self, inputs: tuple[str, ...] | list[str]) -> None:
        for path in inputs:
            if self._contains(path):
                raise ValueError(
                    f"the input {path!r} is inside the output {self.url!r}; overwriting would delete it. "
                    "Write the dataset somewhere else"
                )

    def _contains(self, path: str) -> bool:
        raise NotImplementedError


class LocalSink(Sink):
    local = True

    def __init__(self, root: str) -> None:
        self.url = root
        self.root = os.path.abspath(os.path.expanduser(root))
        self.created_root = not os.path.exists(self.root)

    def _contains(self, path: str) -> bool:
        if is_remote(path):
            return False
        real, root = os.path.realpath(os.path.expanduser(path)), os.path.realpath(self.root)
        return real == root or real.startswith(root + os.sep)

    def existing(self) -> list[str]:
        if not os.path.exists(self.root):
            return []
        if not os.path.isdir(self.root):
            raise NotADirectoryError(f"{self.url} exists and is not a directory")
        return sorted(
            os.path.relpath(os.path.join(directory, f), self.root).replace(os.sep, "/")
            for directory, _, files in os.walk(self.root)
            for f in files
        )

    def delete(self, names: list[str]) -> None:
        for name in sorted(names, key=lambda n: n != MANIFEST_NAME):
            os.remove(os.path.join(self.root, name))
        for sub in ("tiles", "obs"):
            try:
                os.rmdir(os.path.join(self.root, sub))
            except OSError:
                pass  # not empty (holds something that is not ours) or not there

    def discard(self) -> None:
        """Remove what a failed run left: the dataset files in the directory, and the directory if this run created it."""
        self.delete([n for n in self.existing() if _DATASET_FILE.match(n)])
        if self.created_root:
            try:
                os.rmdir(self.root)
            except OSError:
                pass


class FsspecSink(Sink):
    local = False

    def __init__(self, url: str, filesystem: Any = None) -> None:
        self.url = url
        self.fs = get_filesystem(url, filesystem)
        self.root = self._norm(split_url(url)[1])
        if not self.root:
            raise ValueError(f"{url!r} has no bucket")

    def _norm(self, path: str) -> str:
        return self.fs._strip_protocol(path).strip("/")

    def _contains(self, path: str) -> bool:
        if not is_remote(path):
            return False
        other = self._norm(split_url(path)[1])
        return other == self.root or other.startswith(self.root + "/")

    def existing(self) -> list[str]:
        if not self.fs.exists(self.root):
            return []
        names = []
        for raw in self.fs.find(self.root):
            if raw.endswith("/"):  # a "folder" placeholder (the empty object that the cloud console makes), not a file
                continue
            path = self._norm(raw)
            names.append(path[len(self.root) + 1 :] if path.startswith(self.root + "/") else path)
        return sorted(names)

    def delete(self, names: list[str]) -> None:
        ordered = sorted(names, key=lambda n: n != MANIFEST_NAME)
        if ordered and ordered[0] == MANIFEST_NAME:
            self.fs.rm(f"{self.root}/{MANIFEST_NAME}")
            ordered = ordered[1:]
        if ordered:
            self.fs.rm([f"{self.root}/{n}" for n in ordered])

    def put(self, local_path: str, name: str) -> None:
        retry(lambda: self.fs.put_file(local_path, f"{self.root}/{name}"), f"uploading {name} to {self.url}")

    def probe(self) -> None:
        """Upload a tiny file and delete it again, so that a missing permission is found now and not after the (long) scan of
        the inputs. A refusal is a ``PermissionError`` that says what to check."""
        name = _PROBE_NAME.format(os.getpid())
        with tempfile.TemporaryDirectory() as scratch:
            local = os.path.join(scratch, "probe")
            with open(local, "wb") as f:
                f.write(b"deltacells write test\n")
            self.put(local, name)
        try:
            retry(lambda: self.fs.rm(f"{self.root}/{name}"), f"deleting {name} in {self.url}")
        except FileNotFoundError:
            pass  # already gone
        except OSError as e:
            raise OSError(
                f"{self.url} can be written to but the test file {name} that was just uploaded could not be deleted ({e}); "
                "delete it yourself (the files of a dataset could not be replaced either)"
            ) from e


def open_sink(output: str, filesystem: Any = None) -> Sink:
    """The sink for ``output``: a local directory, or a remote prefix if it is a URL."""
    return FsspecSink(output, filesystem) if is_remote(output) else LocalSink(output)
