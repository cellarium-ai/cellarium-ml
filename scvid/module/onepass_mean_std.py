import torch
import torch.nn as nn


class OnePassMeanStd(nn.Module):
    def __init__(self, transform) -> None:
        super().__init__()
        self.transform = transform
        self.running_sums = 0
        self.running_squared_sums = 0
        self.running_size = 0

    def forward(self, x_ng: torch.Tensor) -> None:
        if self.transform is not None:
            x_ng = self.transform(x_ng)
        self.running_sums += x_ng.sum(dim=0)
        self.running_squared_sums += (x_ng**2).sum(dim=0)
        self.running_size += x_ng.shape[0]

    @property
    def mean(self):
        return self.running_sums / self.running_size

    @property
    def var(self):
        return self.running_squared_sums / self.running_size - self.mean**2

    @property
    def std(self):
        return torch.sqrt(self.var)
