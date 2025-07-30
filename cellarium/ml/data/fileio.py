# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
import shutil
import tempfile
import urllib.request
from typing import Literal

from anndata import AnnData, read_h5ad
from google.cloud.storage import Client

url_schemes = ("http:", "https:", "ftp:")
backed_mode_type = Literal["r"] | bool | None
backed_mode_default: backed_mode_type = "r"


def read_h5ad_gcs(
    filename: str,
    storage_client: Client | None = None,
    backed: Literal["r", "r+"] | bool | None = backed_mode_default,
) -> AnnData:
    r"""
    Read ``.h5ad``-formatted hdf5 file from the Google Cloud Storage.

    Example::

        >>> adata = read_h5ad_gcs("gs://dsp-cellarium-cas-public/test-data/test_0.h5ad")

    Args:
        filename: Path to the data file in Cloud Storage.
        backed: If 'r', load in backed mode instead of fully loading into memory.
               If 'r+', load in backed mode with write access (only X can be modified).
               If True, equivalent to 'r'. Default is None (load into memory).
    """
    if not filename.startswith("gs:"):
        raise ValueError("The filename must start with 'gs:' protocol name.")
    # parse bucket and blob names from the filename
    filename = re.sub(r"^gs://?", "", filename)
    bucket_name, blob_name = filename.split("/", 1)

    if storage_client is None:
        storage_client = Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # write to a named temporary file
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp_file:
        temp_path = tmp_file.name
        blob.download_to_file(tmp_file)
        try:
            return read_h5ad(temp_path, backed=backed)
        finally:
            try:
                os.unlink(temp_path)  # clean up the temp file
            except OSError:
                pass  # if there's an error during cleanup, continue


def read_h5ad_url(filename: str, backed: Literal["r", "r+"] | bool | None = backed_mode_default) -> AnnData:
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
         backed: If 'r', load in backed mode instead of fully loading into memory.
               If 'r+', load in backed mode with write access (only X can be modified).
               If True, equivalent to 'r'. Default is None (load into memory).
    """
    if not any(filename.startswith(scheme) for scheme in url_schemes):
        raise ValueError("The filename must start with 'http:', 'https:', or 'ftp:' protocol name.")

    # write to a named temporary file
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp_file:
        temp_path = tmp_file.name
        with urllib.request.urlopen(filename) as response:
            shutil.copyfileobj(response, tmp_file)
        try:
            return read_h5ad(temp_path, backed=backed)
        finally:
            try:
                os.unlink(temp_path)  # clean up the temp file
            except OSError:
                pass  # if there's an error during cleanup, continue


def read_h5ad_local(filename: str, backed: Literal["r", "r+"] | bool | None = backed_mode_default) -> AnnData:
    r"""
    Read ``.h5ad``-formatted hdf5 file from the local disk.

    Args:
        filename: Path to the local data file.
        backed: If 'r', load in backed mode instead of fully loading into memory.
               If 'r+', load in backed mode with write access (only X can be modified).
               If True, equivalent to 'r'. Default is None (load into memory).

    """
    if not filename.startswith("file:"):
        raise ValueError("The filename must start with 'file:' protocol name.")
    filename = re.sub(r"^file://?", "", filename)
    return read_h5ad(filename, backed=backed)


def read_h5ad_file(filename: str, backed: Literal["r", "r+"] | bool | None = backed_mode_default, **kwargs) -> AnnData:
    r"""
    Read ``.h5ad``-formatted hdf5 file from a filename.

    Args:
        filename: Path to the data file.
        backed: If 'r', load in backed mode instead of fully loading into memory.
               If 'r+', load in backed mode with write access (only X can be modified).
               If True, equivalent to 'r'. Default is None (load into memory).

    """
    if filename.startswith("gs:"):
        return read_h5ad_gcs(filename, **kwargs)

    if filename.startswith("file:"):
        return read_h5ad_local(filename)

    if any(filename.startswith(scheme) for scheme in url_schemes):
        return read_h5ad_url(filename)

    return read_h5ad(filename, backed=backed)
