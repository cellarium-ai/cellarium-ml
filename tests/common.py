# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

from cellarium.ml.models import CellariumModel
from cellarium.ml.utilities.distributed import GatherLayer, get_rank_and_num_replicas

USE_CUDA = torch.cuda.is_available()


class BoringDataset(torch.utils.data.Dataset):
    """A simple dataset for testing purposes."""

    def __init__(self, data: np.ndarray, var_names: np.ndarray | None = None) -> None:
        self.data = data
        self.var_names = var_names

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        data = {"x_ng": self.data[idx, None]}
        if self.var_names is not None:
            data["var_names_g"] = self.var_names
        return data


class BoringModel(CellariumModel):
    """
    This model appends a batch input to an :attr:`iter_data` list at each iteration.
    Its intended use is for testing purposes where batch inputs can be inspected after
    iteration over the dataset with ``Trainer.fit()``. Batch input would typically contain
    feature counts, worker info, torch.distributed info, cache info, etc.
    """

    def __init__(self) -> None:
        super().__init__()
        self.iter_data: list = []
        self._dummy_param = torch.nn.Parameter(torch.empty(()))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self._dummy_param.data.zero_()

    def forward(self, **kwargs: torch.Tensor) -> dict:
        _, num_replicas = get_rank_and_num_replicas()
        if num_replicas > 1:
            for key, value in sorted(kwargs.items()):
                kwargs[key] = torch.cat(GatherLayer.apply(value), dim=0)
        self.iter_data.append(kwargs)
        return {}
