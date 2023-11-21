# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from cellarium.ml.models import PredictMixin
from cellarium.ml.utilities.testing import assert_only_allowed_args, assert_return_type
from cellarium.ml.utilities.types import BatchDict


class CellariumPipeline(torch.nn.ModuleList):
    """
    A pipeline of modules.

    Example:

        >>> from cellarium.ml import CellariumPipeline
        >>> from cellarium.ml.transforms import NormalizeTotal, Log1p
        >>> from cellarium.ml.models import IncrementalPCA
        >>> pipeline = CellariumPipeline([
        ...     NormalizeTotal(),
        ...     Log1p(),
        ...     IncrementalPCA(feature_schema=[f"gene_{i}" for i in range(20)], k_components=10),
        ... ])
        >>> batch = {"x_ng": x_ng, "total_mrna_umis_n": total_mrna_umis_n, "feature_g": feature_g}
        >>> output = pipeline(batch)  # or pipeline.predict(batch)

    Args:
        modules:
            Modules to be executed sequentially.
    """

    def add_module(self, name: str, module: torch.nn.Module | None) -> None:
        if module is not None:
            allowed_args = set(BatchDict.__optional_keys__)
            assert_only_allowed_args(module.forward, allowed_args)
            assert_return_type(module.forward, BatchDict)
            if isinstance(module, PredictMixin):
                assert_only_allowed_args(module.predict, allowed_args)
                assert_return_type(module.predict, BatchDict)
        super().add_module(name, module)

    def forward(self, batch: BatchDict) -> BatchDict:
        for module in self:
            ann = module.forward.__annotations__
            input_keys = {key for key in ann if key != "return" and key in batch}
            if "kwargs" in ann:
                input_keys |= batch.keys()
            batch |= module(**{key: batch[key] for key in input_keys})  # type: ignore[literal-required]

        return batch

    def predict(self, batch: BatchDict) -> BatchDict:
        model = self[-1]
        assert isinstance(model, PredictMixin)

        for module in self[:-1]:
            ann = module.forward.__annotations__
            input_keys = {key for key in ann if key != "return" and key in batch}
            if "kwargs" in ann:
                input_keys |= batch.keys()
            batch |= module(**{key: batch[key] for key in input_keys})  # type: ignore[literal-required]

        ann = model.predict.__annotations__
        input_keys = {key for key in ann if key != "return" and key in batch}
        if "kwargs" in ann:
            input_keys |= batch.keys()
        batch |= model.predict(**{key: batch[key] for key in input_keys})  # type: ignore[literal-required]

        return batch
