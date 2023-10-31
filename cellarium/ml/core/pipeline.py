# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from cellarium.ml.models import PredictMixin
from cellarium.ml.utilities.testing import assert_only_allowed_args
from cellarium.ml.utilities.types import BatchDict


class CellariumPipeline(torch.nn.ModuleList):
    def add_module(self, name: str, module: torch.nn.Module | None) -> None:
        if module is not None:
            allowed_args = set(BatchDict.__optional_keys__)
            assert_only_allowed_args(module.forward, allowed_args)
            if isinstance(module, PredictMixin):
                assert_only_allowed_args(module.predict, allowed_args)
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
