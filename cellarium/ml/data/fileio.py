# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import re
import shutil
import tempfile
import urllib.request

import h5py
from anndata import AnnData, read_h5ad
from google.cloud.storage import Blob, Client

url_schemes = ("http:", "https:", "ftp:")


def read_n_cells_h5ad_gcs(
    filename: str,
    storage_client: Client | None = None,
) -> int:
    r"""
    Read the number of cells from an ``.h5ad``-formatted hdf5 file from Google Cloud Storage.

    Example::

        >>> n_cells = read_n_cells_h5ad_gcs("gs://dsp-cellarium-cas-public/test-data/test_0.h5ad")

    Args:
        filename: Path to the data file in Cloud Storage.
        storage_client: (Optional) Google Cloud Storage client.
    """
    blob = _gcs_blob(filename=filename, storage_client=storage_client)

    with blob.open("rb") as f:
        with h5py.File(f, "r") as h5file:
            file_id = h5file.id  # h5py File ID
            obs_group = h5py.h5g.open(file_id, b"obs")
            key = b"index" if b"index" in obs_group else b"_index"
            index_dataset = h5py.h5d.open(obs_group, key)
            return index_dataset.shape[0]  # Access shape directly


def read_h5ad_gcs(filename: str, storage_client: Client | None = None) -> AnnData:
    r"""
    Read ``.h5ad``-formatted hdf5 file from the Google Cloud Storage.

    Example::

        >>> adata = read_h5ad_gcs("gs://dsp-cellarium-cas-public/test-data/test_0.h5ad")

    Args:
        filename: Path to the data file in Cloud Storage.
    """
    blob = _gcs_blob(filename=filename, storage_client=storage_client)

    with blob.open("rb") as f:
        return read_h5ad(f)


def _gcs_blob(filename: str, storage_client: Client | None = None) -> Blob:
    r"""
    Return the Google Cloud Storage :class:`~google.cloud.storage.Blob` object based on the filename.

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
    return blob


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


def read_n_cells_h5ad_local(filename: str) -> AnnData:
    r"""
    Read the number of cells from an ``.h5ad``-formatted hdf5 file from the local disk.

    Args:
        filename: Path to the local data file.
    """
    if not filename.startswith("file:"):
        raise ValueError("The filename must start with 'file:' protocol name.")
    filename = re.sub(r"^file://?", "", filename)

    with h5py.File(filename, "r") as f:
        key = "index" if "index" in f["obs"].keys() else "_index"
        return f[f"/obs/{key}"].shape[0]


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
