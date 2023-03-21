# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from scvid.data import read_h5ad_file


def test_read_h5ad_file():
    filename = "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/benchmark_v1.000.h5ad"
    read_h5ad_file(filename)
