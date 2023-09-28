# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import fsspec
from anndata import AnnData, read_h5ad


def read_h5ad_file(filename: str, storage_kwargs: dict | None = None) -> AnnData:
    r"""
    Read ``.h5ad``-formatted hdf5 file from a filename.

    Example 1::

        >>> adata = read_h5ad_file("test_0.h5ad")

    Example 2::

        >>> adata = read_h5ad_file("gs://dsp-cellarium-cas-public/test-data/test_0.h5ad")

    Example 3::

        >>> adata = read_h5ad_file(
        ...     "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad"
        ... )

    .. note::

        For a full list of the available protocols and the implementations that
        they map across to see the latest online documentation:

        - For implementations built into ``fsspec`` see
          https://filesystem-spec.readthedocs.io/en/latest/api.html#built-in-implementations
        - For implementations in separate packages see
          https://filesystem-spec.readthedocs.io/en/latest/api.html#other-known-implementations

    Args:
        filename:
            Absolute or relative path to the data file. Prefix with a protocol like ``gs://`` to read from alternative
            filesystems.
        storage_kwargs:
            Extra options that make sense to a particular storage connection, e.g. host, port, username, password, etc.
    """
    storage_kwargs = storage_kwargs or {}
    # reset_lock()  # see https://github.com/fsspec/gcsfs/issues/379
    with fsspec.open(filename, "rb", **storage_kwargs) as f:
        return read_h5ad(f)
