# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from cellarium.ml.data import read_h5ad_file


@pytest.mark.parametrize(
    "filename",
    [
        "gs://dsp-cellarium-cas-public/test-data/test_0.h5ad",
        "https://storage.googleapis.com/dsp-cellarium-cas-public/test-data/test_0.h5ad",
    ],
)
def test_read_h5ad_file(filename: str):
    read_h5ad_file(filename)
