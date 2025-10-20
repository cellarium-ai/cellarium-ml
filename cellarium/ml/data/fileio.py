# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import pickle
import re
import shutil
import tempfile
import urllib.request

from anndata import AnnData, read_h5ad
from google.cloud.storage import Client

url_schemes = ("http:", "https:", "ftp:")


def read_h5ad_gcs(filename: str, storage_client: Client | None = None) -> AnnData:
    r"""
    Read ``.h5ad``-formatted hdf5 file from the Google Cloud Storage.

    Example::

        >>> adata = read_h5ad_gcs("gs://dsp-cellarium-cas-public/test-data/test_0.h5ad")

    Args:
        filename: Path to the data file in Cloud Storage.
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

    with blob.open("rb") as f:
        return read_h5ad(f)


def read_h5ad_url(filename: str) -> AnnData:
    r"""
    Read ``.h5ad``-formatted hdf5 file from the URL.

    Example::

        >>> adata = read_h5ad_url(
        ...     "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad"
        ... )

    Args:
        filename: URL of the data file.
    """
    if not any(filename.startswith(scheme) for scheme in url_schemes):
        raise ValueError("The filename must start with 'http:', 'https:', or 'ftp:' protocol name.")
    with urllib.request.urlopen(filename) as response:
        with tempfile.TemporaryFile() as tmp_file:
            shutil.copyfileobj(response, tmp_file)
            return read_h5ad(tmp_file)


def read_h5ad_local(filename: str) -> AnnData:
    r"""
    Read ``.h5ad``-formatted hdf5 file from the local disk.

    Args:
        filename: Path to the local data file.
    """
    if not filename.startswith("file:"):
        raise ValueError("The filename must start with 'file:' protocol name.")
    filename = re.sub(r"^file://?", "", filename)
    return read_h5ad(filename)


def read_h5ad_file(filename: str, **kwargs) -> AnnData:
    r"""
    Read ``.h5ad``-formatted hdf5 file from a filename.

    Args:
        filename: Path to the data file.
    """
    if filename.startswith("gs:"):
        return read_h5ad_gcs(filename, **kwargs)

    if filename.startswith("file:"):
        return read_h5ad_local(filename)

    if any(filename.startswith(scheme) for scheme in url_schemes):
        return read_h5ad_url(filename)

    return read_h5ad(filename)


def read_pkl_from_gcs(filename: str, storage_client: Client | None = None):
    r"""
    Read ``.pkl``-formatted pickle file from the Google Cloud Storage.

    Example::

        >>> adata = read_pkl_from_gcs("gs://cellarium-file-system/curriculum/human_10x_ebd_lrexp_extract/models/shared_metadata/final_y_categories.pkl")

    Args:
        filename: Path to the data file in Cloud Storage.
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

    # Download the blob content as a byte string (in memory, no file is saved)
    pickle_data = blob.download_as_bytes()

    # Load the pickle data from the byte string directly in memory
    return pickle.loads(pickle_data)
