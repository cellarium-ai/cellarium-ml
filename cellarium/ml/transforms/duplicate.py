# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause


from collections.abc import Sequence
from typing import ClassVar

import torch

from .base import CellariumTransform, Stage


class Duplicate(CellariumTransform):
    """
    Duplicates every row of the input tensor,
    used for contrastive augmentations.

    By default the transform is active in the ``"train"``, ``"validate"`` and ``"test"`` stages of a
    :class:`~cellarium.ml.core.CellariumPipeline` and skipped in ``"predict"``, so that predictions
    are returned for the original rows only.

    Args:
        stages:
            The stages in which the transform is active. If ``None`` (default), it is active in
            ``"train"``, ``"validate"`` and ``"test"``. Otherwise this replaces that default.
    """

    default_stages: ClassVar[frozenset[Stage]] = frozenset({"train", "validate", "test"})

    def __init__(self, stages: Sequence[Stage] | None = None) -> None:
        super().__init__(stages)

    def forward(self, x_ng: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x_ng: Gene counts.

        Returns:
            Duplicated counts.
        """
        return {"x_ng": x_ng.repeat((2, 1))}
