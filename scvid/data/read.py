# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import re
from typing import Optional

from anndata import AnnData, read_h5ad
from google.cloud.storage import Client


def read_h5ad_gcs(filename: str, storage_client: Optional[Client] = None) -> AnnData:
    r"""
    Read `.h5ad`-formatted hdf5 file from the Google Cloud Storage.

    Example::

        >>> adata = read_h5ad_gcs("gs://dsp-cell-annotation-service/benchmark_v1/benchmark_v1.000.h5ad")

    Args:
        filename (str): Path to the data file in Cloud Storage.
    """
    assert filename.startswith("gs:")
    # parse bucket and blob names from the filename
    filename = re.sub(r"^gs://?", "", filename)
    bucket_name, blob_name = filename.split("/", 1)

    if storage_client is None:
        storage_client = Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    with blob.open("rb") as f:
        return read_h5ad(f)


def read_h5ad_local(filename: str) -> AnnData:
    r"""
    Read `.h5ad`-formatted hdf5 file from the local disk.

    Args:
        filename (str): Path to the local data file.
    """
    assert filename.startswith("file:")
    filename = re.sub(r"^file://?", "", filename)
    return read_h5ad(filename)


def read_h5ad_file(filename: str, **kwargs) -> AnnData:
    r"""
    Read `.h5ad`-formatted hdf5 file from a filename.

    Args:
        filename (str): Path to the data file.
    """
    if filename.startswith("gs:"):
        return read_h5ad_gcs(filename, **kwargs)

    if filename.startswith("file:"):
        return read_h5ad_local(filename)

    return read_h5ad(filename)
