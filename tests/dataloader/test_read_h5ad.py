# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from cellarium.ml.data import read_h5ad_file


def test_read_h5ad_file():
    filename = "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad"
    read_h5ad_file(filename)
