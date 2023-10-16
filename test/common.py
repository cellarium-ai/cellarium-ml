# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch


class TestDataset(torch.utils.data.Dataset):
    """A simple dataset for testing purposes."""

    def __init__(self, data, var_names=None):
        self.data = data
        self.var_names = var_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = {"X": self.data[idx, None]}
        if self.var_names is not None:
            data["var_names"] = self.var_names
        return data


try:
    import crick
except ImportError:
    crick = None
requires_crick = pytest.mark.skipif(crick is None, reason="crick is not available")
