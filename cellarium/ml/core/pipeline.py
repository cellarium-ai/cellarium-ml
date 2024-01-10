# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

from cellarium.ml.models import PredictMixin


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

    def forward(self, batch: dict[str, np.ndarray | torch.Tensor]) -> dict[str, torch.Tensor | np.ndarray]:
        for module in self:
            # get the module input keys
            ann = module.forward.__annotations__
            input_keys = {key for key in ann if key != "return" and key in batch}
            # allow all keys to be passed to the module
            if "kwargs" in ann:
                input_keys |= batch.keys()
            batch |= module(**{key: batch[key] for key in input_keys})

        return batch

    def predict(self, batch: dict[str, np.ndarray | torch.Tensor]) -> dict[str, np.ndarray | torch.Tensor]:
        model = self[-1]
        assert isinstance(model, PredictMixin)

        for module in self[:-1]:
            # get the module input keys
            ann = module.forward.__annotations__
            input_keys = {key for key in ann if key != "return" and key in batch}
            # allow all keys to be passed to the module
            if "kwargs" in ann:
                input_keys |= batch.keys()
            batch |= module(**{key: batch[key] for key in input_keys})

        # get the model predict input keys
        ann = model.predict.__annotations__
        input_keys = {key for key in ann if key != "return" and key in batch}
        # allow all keys to be passed to the predict method
        if "kwargs" in ann:
            input_keys |= batch.keys()
        batch |= model.predict(**{key: batch[key] for key in input_keys})

        return batch
