# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

from cellarium.ml.models import CellariumPipelineUpdatable, PredictMixin


class CellariumPipeline(torch.nn.ModuleList):
    def __init__(self, modules: list[CellariumPipelineUpdatable | torch.nn.Module]) -> None:
        """
        A pipeline of modules. All modules must be signed with :class:`CellariumPipelineProtocol`.

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
                Modules to be executed sequentially. Must be signed with :class:`CellariumPipelineProtocol`.
        """

        self.post_init_done = False

        super().__init__(modules=modules)  # type: ignore

    def _forward_batch_through_modules(
        self, batch: dict[str, np.ndarray | torch.Tensor], modules: torch.nn.ModuleList
    ) -> None:
        """
        Forward the batch through the modules.

        Args:
            batch: The batch to forward.
            modules: The modules to forward the batch through.
        """
        for module in modules:
            if not self.post_init_done and isinstance(module, CellariumPipelineUpdatable):
                module.update_input_tensors_from_previous_module(batch=batch)

            # get the module input keys
            ann = module.forward.__annotations__
            input_keys = {key for key in ann if key != "return" and key in batch}
            # allow all keys to be passed to the module
            if "kwargs" in ann:
                input_keys |= batch.keys()
            batch |= module(**{key: batch[key] for key in input_keys})

        self.post_init_done = True

    def forward(self, batch: dict[str, np.ndarray | torch.Tensor]) -> dict[str, torch.Tensor | np.ndarray]:
        """
        Forward the batch through the pipeline.

        Args:
            batch: The batch to forward.

        Returns:
            The batch after it has been forwarded through the pipeline.
        """
        self._forward_batch_through_modules(batch=batch, modules=self)
        return batch

    def predict(self, batch: dict[str, np.ndarray | torch.Tensor]) -> dict[str, np.ndarray | torch.Tensor]:
        model = self[-1]
        assert isinstance(model, PredictMixin)

        self._forward_batch_through_modules(batch=batch, modules=self[:-1])

        # get the model predict input keys
        ann = model.predict.__annotations__
        input_keys = {key for key in ann if key != "return" and key in batch}
        # allow all keys to be passed to the predict method
        if "kwargs" in ann:
            input_keys |= batch.keys()
        batch |= model.predict(**{key: batch[key] for key in input_keys})

        return batch
