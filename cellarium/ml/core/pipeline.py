# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from itertools import chain
from typing import Any

import numpy as np
import torch
from typing_extensions import Self

from cellarium.ml.models import PredictMixin, TestMixin, ValidateMixin
from cellarium.ml.transforms.base import CellariumTransform, Stage
from cellarium.ml.utilities.core import call_func_with_batch


class CellariumPipeline(torch.nn.ModuleList):
    """
    A pipeline of modules. Modules are expected to return a dictionary. The input dictionary is sequentially passed to
    (piped through) each module and updated with its output dictionary.

    When used within :class:`cellarium.ml.core.CellariumModule`, the last module in the pipeline is expected to be
    a model (:class:`cellarium.ml.models.CellariumModel`) and any preceding modules are expected to be data transforms.

    Each method runs the transforms for one stage: ``forward`` for ``"train"``, and ``validate``, ``test`` and
    ``predict`` for their own. A :class:`~cellarium.ml.transforms.base.CellariumTransform` is skipped in the stages
    it is not active in (e.g. data augmentations are by default active only in ``"train"``), and any other module
    is run in every stage.

    Example:

        >>> from cellarium.ml import CellariumPipeline
        >>> from cellarium.ml.transforms import NormalizeTotal, Log1p
        >>> from cellarium.ml.models import IncrementalPCA
        >>> pipeline = CellariumPipeline([
        ...     NormalizeTotal(),
        ...     Log1p(),
        ...     IncrementalPCA(var_names_g=[f"gene_{i}" for i in range(20)], n_components=10),
        ... ])
        >>> batch = {"x_ng": x_ng, "total_mrna_umis_n": total_mrna_umis_n, "var_names_g": var_names_g}
        >>> output = pipeline(batch)  # or pipeline.predict(batch)

    Args:
        modules:
            Modules to be executed sequentially.
    """

    def __add__(self, other: Iterable[torch.nn.Module]) -> Self:
        return self.__class__(chain(self, other))

    def transform(
        self,
        batch: dict[str, dict[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor],
        stage: Stage,
    ) -> dict[str, dict[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor]:
        """
        Pass the batch through every module of the pipeline that is active in ``stage``,
        treating all of them as transforms (a model is not treated specially).

        Args:
            batch: The batch dictionary.
            stage: The stage whose transforms to run.

        Returns:
            The batch dictionary, updated with the output of each transform.
        """
        self._apply_transforms(self, batch, stage)
        return batch

    @staticmethod
    def _apply_transforms(modules: Iterable[torch.nn.Module], batch: dict[str, Any], stage: Stage) -> None:
        """Update ``batch`` in place with the output of each of the ``modules`` that is active in ``stage``."""
        for module in modules:
            if isinstance(module, CellariumTransform) and not module.is_active(stage):
                continue
            batch |= call_func_with_batch(module.forward, batch)

    def forward(
        self, batch: dict[str, dict[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor]
    ) -> dict[str, dict[str, np.ndarray | torch.Tensor] | torch.Tensor | np.ndarray]:
        return self.transform(batch, "train")

    def predict(
        self, batch: dict[str, dict[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor]
    ) -> dict[str, dict[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor]:
        self._apply_transforms(list(self)[:-1], batch, "predict")

        model = self[-1]
        if not isinstance(model, PredictMixin):
            raise TypeError(f"The last module in the pipeline must be an instance of {PredictMixin}. Got {model}")
        batch |= call_func_with_batch(model.predict, batch)

        return batch

    def validate(self, batch: dict[str, dict[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor]) -> None:
        self._apply_transforms(list(self)[:-1], batch, "validate")

        model = self[-1]
        if not isinstance(model, ValidateMixin):
            raise TypeError(f"The last module in the pipeline must be an instance of {ValidateMixin}. Got {model}")
        call_func_with_batch(model.validate, batch)

    def test(self, batch: dict[str, np.ndarray | torch.Tensor]) -> None:
        self._apply_transforms(list(self)[:-1], batch, "test")

        model = self[-1]
        if not isinstance(model, TestMixin):
            raise TypeError(f"The last module in the pipeline must be an instance of {TestMixin}. Got {model}")
        call_func_with_batch(model.test, batch)
