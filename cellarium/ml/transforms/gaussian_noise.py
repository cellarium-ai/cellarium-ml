# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


from collections.abc import Sequence
from typing import ClassVar

import torch

from .base import CellariumTransform, Stage
from .randomize import Randomize


class GaussianNoise(CellariumTransform):
    """
    Adds Gaussian noise to gene counts.

    For each count, Gaussian sigma is independently
    and uniformly sampled according to the bounding parameters,
    yielding the sigma matrix sigma_ng.

    .. math::

        y_{ng} = x_{ng} + N(0, \\sigma_{ng})

    By default the transform is active only in the ``"train"`` stage of a
    :class:`~cellarium.ml.core.CellariumPipeline`.

    Args:
        sigma_min:
            Lower bound on Gaussian sigma parameter.
        sigma_max:
            Upper bound on Gaussian sigma parameter.
        p_apply:
            Probability of applying transform to each sample.
        stages:
            The stages in which the transform is active. If ``None`` (default), it is active only in
            ``"train"``. Otherwise this replaces that default.
    """

    default_stages: ClassVar[frozenset[Stage]] = frozenset({"train"})

    def __init__(
        self,
        sigma_min,
        sigma_max,
        p_apply,
        stages: Sequence[Stage] | None = None,
    ):
        super().__init__(stages)

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.randomize = Randomize(p_apply)

    def forward(self, x_ng: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x_ng: Gene counts (log-transformed).

        Returns:
            Gene counts with added Gaussian noise.
        """
        sigma_ng = torch.empty_like(x_ng).uniform_(self.sigma_min, self.sigma_max)
        x_aug = x_ng + torch.normal(mean=0, std=sigma_ng)

        x_ng = self.randomize(x_aug, x_ng)
        return {"x_ng": x_ng}
