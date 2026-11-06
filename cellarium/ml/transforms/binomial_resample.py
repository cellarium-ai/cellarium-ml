# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


from collections.abc import Sequence
from typing import ClassVar

import torch
from torch.distributions import Binomial

from .base import CellariumTransform, Stage
from .randomize import Randomize


class BinomialResample(CellariumTransform):
    """
    Binomial resampling of gene counts.

    For each count, the parameter to the binomial distribution
    is independently and uniformly sampled according to the
    bounding parameters, yielding the parameter matrix p_ng.

    .. math::

        y_{ng} = Binomial(n=x_{ng}, p=p_{ng})

    If ``total_mrna_umis_n`` is given, it is reduced by the counts that the resampling removed,
    so that it remains consistent with the resampled gene counts. Only the counts of the genes in
    ``x_ng`` change, so counts of genes that are not in ``x_ng`` stay in the total.

    By default the transform is active only in the ``"train"`` stage of a
    :class:`~cellarium.ml.core.CellariumPipeline`.

    Args:
        p_binom_min:
            Lower bound on binomial distribution parameter.
        p_binom_max:
            Upper bound on binomial distribution parameter.
        p_apply:
            Probability of applying transform to each sample.
        stages:
            The stages in which the transform is active. If ``None`` (default), it is active only in
            ``"train"``. Otherwise this replaces that default.
    """

    default_stages: ClassVar[frozenset[Stage]] = frozenset({"train"})

    def __init__(
        self,
        p_binom_min: float,
        p_binom_max: float,
        p_apply: float,
        stages: Sequence[Stage] | None = None,
    ):
        super().__init__(stages)

        self.p_binom_min = p_binom_min
        self.p_binom_max = p_binom_max
        self.randomize = Randomize(p_apply)

    def forward(self, x_ng: torch.Tensor, total_mrna_umis_n: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """
        Args:
            x_ng: Gene counts.
            total_mrna_umis_n: Total mRNA UMIs per cell (optional).

        Returns:
            Binomially resampled gene counts, and, if ``total_mrna_umis_n`` was given,
            the total mRNA UMIs per cell reduced by the removed counts.
        """
        p_binom_ng = torch.empty_like(x_ng).uniform_(self.p_binom_min, self.p_binom_max)
        x_aug = Binomial(total_count=x_ng, probs=p_binom_ng).sample()

        y_ng = self.randomize(x_aug, x_ng)
        output = {"x_ng": y_ng}
        if total_mrna_umis_n is not None:
            removed_n = (x_ng - y_ng).sum(dim=-1)
            output["total_mrna_umis_n"] = total_mrna_umis_n - removed_n.to(total_mrna_umis_n.dtype)
        return output
