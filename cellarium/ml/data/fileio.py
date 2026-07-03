# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
import shutil
import tempfile
import urllib.request
from typing import Literal

from anndata import AnnData, read_h5ad
from google.auth.exceptions import DefaultCredentialsError
from google.cloud.storage import Client

url_schemes = ("http:", "https:", "ftp:")
backed_mode_type = Literal["r"] | bool | None
backed_mode_default: backed_mode_type = False


def read_h5ad_gcs(
    filename: str,
    storage_client: Client | None = None,
    backed: backed_mode_type = backed_mode_default,
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

    if backed not in [True, "r"]:
        # Stream directly into memory — no temp file, no disk I/O.
        with blob.open("rb") as f:
            return read_h5ad(f)

    # Backed mode requires h5py to have a real seekable file path. The file is
    # flushed and closed before h5py opens it to avoid Python's write buffer
    # leaving the last chunk off disk (which causes h5py to report a truncated
    # file). After os.unlink the directory entry is gone but the inode stays
    # allocated until h5py closes its fd (i.e. when the AnnData is GC'd).
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp_file:
        temp_path = tmp_file.name
        blob.download_to_file(tmp_file)
        tmp_file.flush()

    try:
        return read_h5ad(temp_path, backed=backed)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


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


def read_h5ad_file(filename: str, backed: backed_mode_type = backed_mode_default, **kwargs) -> AnnData:
    r"""
    Read ``.h5ad``-formatted hdf5 file from a filename.

    Args:
        filename: Path to the data file.
        backed: See :func:`anndata.read_h5ad` for details on backed mode.
            ['r', True] will load in backed mode instead of fully loading into memory.
            [False, None] will use in-memory mode.
    """
    if filename.startswith("gs:"):
        return read_h5ad_gcs(filename, **kwargs)

    if filename.startswith("file:"):
        return read_h5ad_local(filename, backed=backed)

    if any(filename.startswith(scheme) for scheme in url_schemes):
        return read_h5ad_url(filename)

    return read_h5ad(filename, backed=backed)
