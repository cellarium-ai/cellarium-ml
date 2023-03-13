from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from scvid.data.util import get_rank_and_num_replicas
from scvid.module import GatherLayer


class OnePassMeanVarStd(nn.Module):
    """
    Calculate the mean, variance, and standard deviation of the data in one pass (epoch)
    using running sums and running squared sums.
    """

    def __init__(self, transform: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.transform = transform
        self.x_sums = 0
        self.x_squared_sums = 0
        self.x_size = 0

    @staticmethod
    def _get_fn_args_from_batch(
        tensor_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Iterable, dict]:
        x = tensor_dict["X"]
        return (x,), {}

    def forward(self, x_ng: torch.Tensor) -> None:
        if self.transform is not None:
            x_ng = self.transform(x_ng)
        num_replicas = get_rank_and_num_replicas()[1]
        if num_replicas > 1:
            x_ng = torch.cat(GatherLayer.apply(x_ng), dim=0)
        self.x_sums += x_ng.sum(dim=0)
        self.x_squared_sums += (x_ng**2).sum(dim=0)
        self.x_size += x_ng.shape[0]

    @property
    def mean(self) -> torch.Tensor:
        return self.x_sums / self.x_size

    @property
    def var(self) -> torch.Tensor:
        return self.x_squared_sums / self.x_size - self.mean**2

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var)
