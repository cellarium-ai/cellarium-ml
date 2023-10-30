# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from cellarium.ml.utilities.types import OutputDict


class CellariumPipeline(torch.nn.Module):
    def __init__(
        self,
        pipeline: list[torch.nn.Module],
    ):
        super().__init__()
        self.pipeline = torch.nn.ModuleList(pipeline)

    def forward(
        self,
        batch: OutputDict,
    ) -> OutputDict:
        for model in self.pipeline:
            ann = model.forward.__annotations__
            input_keys = tuple(key for key in ann if key != "return")

            output = model(**{key: batch[key] for key in input_keys})
            batch |= output
        return batch
