# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import os
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Literal

from anndata import AnnData, read_h5ad
from google.auth.exceptions import DefaultCredentialsError
from google.cloud.storage import Client

url_schemes = ("http:", "https:", "ftp:")
backed_mode_type = Literal["r"] | bool | None
backed_mode_default: backed_mode_type = False

# Default persistent cache for GCS-downloaded h5ad files. Shards are written
# here on first access and reused on subsequent accesses or after a checkpoint
# restart on the same machine. Pass cache_dir=None to stream directly from GCS
# with no local disk usage.
GCS_CACHE_DIR: Path = Path.home() / ".cache" / "cellarium_gcs_cache"


def read_h5ad_gcs(
    filename: str,
    storage_client: Client | None = None,
    backed: backed_mode_type = backed_mode_default,
    cache_dir: Path | str | None = GCS_CACHE_DIR,
) -> AnnData:
    r"""
    Read ``.h5ad``-formatted hdf5 file from the Google Cloud Storage.

    Example::

        >>> adata = read_h5ad_gcs("gs://dsp-cellarium-cas-public/test-data/test_0.h5ad")

    Args:
        filename: Path to the data file in Cloud Storage.
        backed: See :func:`anndata.read_h5ad` for details on backed mode.
            ['r', True] will load in backed mode instead of fully loading into memory.
            [False, None] will use in-memory mode.
        cache_dir: Directory for caching downloaded files on local disk. On first
            access the shard is saved here; subsequent accesses read from disk instead
            of re-downloading. If a write fails (e.g. disk full), falls back to
            streaming from GCS. Set to ``None`` to always stream with no disk usage.
    """
    if not filename.startswith("gs:"):
        raise ValueError("The filename must start with 'gs:' protocol name.")
    # parse bucket and blob names from the filename
    filename = re.sub(r"^gs://?", "", filename)
    bucket_name, blob_name = filename.split("/", 1)

    if storage_client is None:
        try:
            storage_client = Client()
        except DefaultCredentialsError:
            storage_client = Client.create_anonymous_client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if cache_dir is not None:
        local_path = Path(cache_dir) / bucket_name / blob_name
        if local_path.exists():
            return read_h5ad(str(local_path), backed=backed)
        # Cache miss — download to a staging file then rename atomically so that
        # a concurrent worker or an interrupted run never sees a partial file.
        local_path.parent.mkdir(parents=True, exist_ok=True)
        staging = local_path.with_suffix(local_path.suffix + ".download")
        try:
            with open(staging, "wb") as f:
                blob.download_to_file(f)
                f.flush()
            staging.rename(local_path)
            return read_h5ad(str(local_path), backed=backed)
        except OSError:
            # Disk full or permission error — discard the staging file and fall
            # through to the no-cache path below.
            with contextlib.suppress(OSError):
                staging.unlink()

    # No cache (or cache write failed) — stream without leaving anything on disk.
    if backed not in [True, "r"]:
        with blob.open("rb") as f:
            return read_h5ad(f)

    # Backed mode without a persistent cache: download to an anonymous temp file.
    # Flushed and closed before h5py opens it to avoid truncation from an
    # unflushed write buffer. The unlink removes the directory entry immediately;
    # on Linux/macOS the inode persists until h5py closes its fd (when the
    # AnnData is GC'd or evicted from the LRU cache).
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp_file:
        temp_path = tmp_file.name
        blob.download_to_file(tmp_file)
        tmp_file.flush()

    try:
        return read_h5ad(temp_path, backed=backed)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_path)


def read_h5ad_url(filename: str, backed: backed_mode_type = backed_mode_default) -> AnnData:
    r"""
    Read ``.h5ad``-formatted hdf5 file from the URL.

    Example::

        >>> adata = read_h5ad_url(
        ...     "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad"
        ... )
        >>> adata = read_h5ad_url(
        ...     "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad",
        ...     backed='r'
        ... )

    Args:
        filename: URL of the data file.
        backed: See :func:`anndata.read_h5ad` for details on backed mode.
            ['r', True] will load in backed mode instead of fully loading into memory.
            [False, None] will use in-memory mode.
    """
    if not any(filename.startswith(scheme) for scheme in url_schemes):
        raise ValueError("The filename must start with 'http:', 'https:', or 'ftp:' protocol name.")

    if backed not in [True, "r"]:
        # Anonymous TemporaryFile: no path, no flush needed, deleted automatically.
        with urllib.request.urlopen(filename) as response:
            with tempfile.TemporaryFile() as tmp_file:
                shutil.copyfileobj(response, tmp_file)
                return read_h5ad(tmp_file)

    # Backed mode needs a real path; flush and close before h5py opens it.
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp_file:
        temp_path = tmp_file.name
        with urllib.request.urlopen(filename) as response:
            shutil.copyfileobj(response, tmp_file)
        tmp_file.flush()

    try:
        return read_h5ad(temp_path, backed=backed)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def read_h5ad_local(filename: str, backed: backed_mode_type = backed_mode_default) -> AnnData:
    r"""
    Read ``.h5ad``-formatted hdf5 file from the local disk.

    Args:
        filename: Path to the local data file.
        backed: See :func:`anndata.read_h5ad` for details on backed mode.
            ['r', True] will load in backed mode instead of fully loading into memory.
            [False, None] will use in-memory mode.
    """
    if not filename.startswith("file:"):
        raise ValueError("The filename must start with 'file:' protocol name.")
    filename = re.sub(r"^file://?", "", filename)
    return read_h5ad(filename, backed=backed)


def read_h5ad_file(
    filename: str,
    backed: backed_mode_type = backed_mode_default,
    cache_dir: Path | str | None = GCS_CACHE_DIR,
    **kwargs,
) -> AnnData:
    r"""
    Read ``.h5ad``-formatted hdf5 file from a filename.

    Args:
        filename: Path to the data file.
        backed: See :func:`anndata.read_h5ad` for details on backed mode.
            ['r', True] will load in backed mode instead of fully loading into memory.
            [False, None] will use in-memory mode.
        cache_dir: Directory for caching GCS-downloaded files on local disk.
            Only used when ``filename`` starts with ``gs://``. Set to ``None`` to
            stream GCS files directly into memory with no disk usage.
    """
    if filename.startswith("gs:"):
        return read_h5ad_gcs(filename, backed=backed, cache_dir=cache_dir, **kwargs)

    if filename.startswith("file:"):
        return read_h5ad_local(filename, backed=backed)

    if any(filename.startswith(scheme) for scheme in url_schemes):
        return read_h5ad_url(filename, backed=backed)

    return read_h5ad(filename, backed=backed)
