# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import fsspec
from anndata import AnnData, read_h5ad


def read_h5ad_file(filename: str, **kwargs) -> AnnData:
    r"""
    Read ``.h5ad``-formatted hdf5 file from a filename.

    Example 1::

        >>> adata = read_h5ad_file(
        ...     "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad"
        ... )

    Example 2::

        >>> adata = read_h5ad_file("gs://dsp-cellarium-cas-public/test-data/test_0.h5ad")

    Args:
        filename:
            Path to the data file.
        **kwargs:
            Extra options that make sense to a particular storage connection, e.g. host, port, username, password, etc.
    """
    with fsspec.open(filename, "rb", **kwargs) as f:
        return read_h5ad(f)
