import torch
from torch import nn


class LogNormalize(nn.Module):
    """
    LogNormalize gene counts with mean, standard deviation, and pseudo-count.

    Args:
        mean_g: Means for each gene.
        std_g: Standard deviations for each gene.
        C: Gene epxression pseudo-count.
    """

    def __init__(self, mean_g: torch.Tensor, std_g: torch.Tensor, C: int = 10_000):
        super().__init__()
        self.mean_g = mean_g
        self.std_g = std_g
        self.C = C

    def forward(self, x_ng: torch.Tensor) -> torch.Tensor:
        l_n1 = x_ng.sum(axis=-1, keepdim=True)
        y_ng = torch.log1p(self.C * x_ng / l_n1)
        z_ng = (y_ng - self.mean_g) / self.std_g
        return z_ng

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean_g={self.mean_g}, std_g={self.std_g}, C={self.C})"
